import gpboost as gpb
import numpy as np
import scipy
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
#from merf import MERF
from sklearn.datasets import make_friedman3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from tqdm.notebook import tqdm, tnrange
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=2, bar_length=100):

    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    import sys
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def linear_func(n_samples, random_state=1):
    np.random.seed(random_state)
    X = np.random.uniform(size=(n_samples, 4))
    F = 1 + np.sum(X, axis=1)

    return X, F

def exponential_kernel(x1, x2, sigma2, rho):
    '''
    Computes the exponential kernel matrix between two vectors.
    '''
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)

    return sigma2 * np.exp(-scipy.spatial.distance.cdist(x1, x2, 'sqeuclidean') / rho)


def generate_data(n, m, p, func, shared_gp=True, random_state=1):
    # Features
    if func == 'make_friedman3':
        X, F = make_friedman3(n_samples=n, random_state=random_state)
        C = 1 / np.sqrt(np.var(F)) # set C such^ that Var(F) is approx. 1 
        F *= C
        #F *= 10**0.5
    elif func == 'linear_func':
        X, F = linear_func(n_samples=n, random_state=random_state)
        C = 1 / np.sqrt(np.var(F)) # set C such that Var(F) is approx. 1 
        F *= C
    else:
        raise Exception('function is invalid') 

    # Create groups
    group = np.arange(n) # Variable that stores group IDs
    for i in range(m):
        group[i*p:(i+1)*p] = i

    # Incidence matrix relating grouped random effects to observations
    Z1 = np.zeros((n, m))
    for i in range(m):
        Z1[np.where(group==i), i] = 1

    # Simulate random (sorted) observation times for each group
    rng = np.random.default_rng(seed=random_state)
    times_arrays = [np.sort(rng.choice(700, size=p, replace=False, shuffle=False)) for i in range(m)] # was 1000
    times = np.concatenate(times_arrays)

    # Simulate GPs for the random effects part 
    GP_list = []
    np.random.seed(random_state)
    if shared_gp:
        # Simulate GPs with same parameters
        sigma2_2 = 1 ** 2  # Marginal variance of GP
        rho = 0.1  # GP Range parameter
        for arr in times_arrays:
            K = exponential_kernel(arr, arr, sigma2_2, rho)
            g = np.random.multivariate_normal(mean=np.zeros(p), cov=K)
            GP_list.append(g)
    else:
        # Simulate GPs with different, random parameters
        for arr in times_arrays:
            sigma2_2 = scipy.stats.invgamma.rvs(1, loc=0, scale=10)
            rho = np.random.uniform(0.01, 1000)
            #sigma2_2 = scipy.stats.invgamma.rvs(1, loc=0, scale=1)
            #rho = np.random.uniform(0.1, 1.5)
            K = exponential_kernel(arr, arr, sigma2_2, rho)
            g = np.random.multivariate_normal(mean=np.zeros(p), cov=K)
            GP_list.append(g)

    # Simulate outcome variable
    np.random.seed(random_state)
    sigma2 = 0.5 ** 2  # Error variance
    b = np.random.normal(size=m) # Simulate random effect intercept
    G = np.dot(Z1, b) + np.concatenate(GP_list) + 0.005 * np.concatenate(times_arrays) # Combine random effect intercept with linear trend GP
    epsilon = np.sqrt(sigma2) * np.random.normal(size=n) # Simulate error term
    y = F + G + epsilon

    # Create dataframes
    X = pd.concat([pd.DataFrame(group, columns=['group']), pd.DataFrame(times, columns=['times']), pd.DataFrame(X,columns=['feature_1','feature_2','feature_3','feature_4'])], axis=1)
    data = pd.concat([X, pd.DataFrame(y, columns=['y'])], axis=1)

    return data, X, y, F


def train_test_split_grouped(X, y, F, group_sizes, test_size=0.2, random_state=1):

    '''
    Train/test dataset split. The test set ***may contain*** observations from groups also seen in the training set, or from entirely new groups not seen during training.
    Since we are dealing with temporal data, the test set only contains future observations from groups in the training set if the same group appears in both the training and test sets.
    '''

    assert len(X) == len(y), "lengths of input data array X and labels array y must be the same"
    assert len(y) == len(F), "lengths of F and y must be the same"
    assert 0 < test_size < 1, "Test size must be strictly between 0 and 1"
    assert np.sum(group_sizes) == len(y), "Sum of group_sizes must be equal to length of labels array y"
    assert group_sizes.all() > 0, "Group sizes should be non-negative"

    np.random.seed(random_state)
    y_len = len(y)
    test_len = int(test_size * y_len)
    sample_len = 0
    no_groups = len(group_sizes)
    test_idx = []
    n_samples_chosen_per_group = np.zeros_like(group_sizes)
    last_idx_arr = np.cumsum(group_sizes)-1 # Array of index of the last observation in each group within the overall dataset

    # Keep picking observations until the required number of test observations has been picked
    while sample_len < test_len:
        group_idx = np.random.randint(no_groups) # Pick a random group
        if group_sizes[group_idx] > 0:
            n = np.random.randint(min([group_sizes[group_idx], test_len-sample_len])) + 1 # Pick a random sample of size n>=1 from the chosen group
            last_idx = last_idx_arr[group_idx]-n_samples_chosen_per_group[group_idx] # Index of the last observation remaining in each group within the overall dataset
            test_idx += [last_idx-i for i in range(n)]
            n_samples_chosen_per_group[group_idx] += n # Update number of samples chosen from the group
            group_sizes[group_idx] -= n # Update current group sizes
            sample_len += n

    mask = np.ones(y_len, dtype=bool)
    mask[test_idx] = False
    X_train, X_test = X[mask], X[~mask]
    y_train, y_test = y[mask], y[~mask]
    F_train, F_test = F[mask], F[~mask]

    return X_train, X_test, y_train, y_test, F_train, F_test


def train_test_split_grouped_interpolation(X, y, F, group_sizes, test_size=0.2, random_state=1):
    '''
    Train/test split, but test set contains at least one observation from each group in the training set, and contains no unseen groups.
    '''
    assert len(X) == len(y), "lengths of input data array X and labels array y must be the same"
    assert len(y) == len(F), "lengths of F and y must be the same"
    assert 0 < test_size < 1, "Test size must be strictly between 0 and 1"
    assert np.sum(group_sizes) == len(y), "Sum of group_sizes must be equal to length of labels array y"
    assert group_sizes.all() > 0, "Group sizes should be non-negative"
    assert len(group_sizes) < len(X), "Number of groups should be less than number of observations"

    np.random.seed(random_state)
    y_len = len(y)
    test_len = int(test_size * y_len)
    no_groups = len(group_sizes)

    # Pick one observation from all groups
    sample_len = no_groups
    n_samples_chosen_per_group = np.ones_like(group_sizes)
    last_idx_arr = np.cumsum(group_sizes)-1 # Array of index of the last observation in each group within the overall dataset
    test_idx = [last_idx_arr[i] for i in range(no_groups)]
    group_sizes_new = group_sizes.copy()
    group_sizes_new -= 1

    # Keep picking more observations until the required number of test observations has been picked
    while sample_len < test_len:
        group_idx = np.random.randint(no_groups) # Pick a random group
        if group_sizes_new[group_idx] > 1:
            if test_len - sample_len > 1:
                n = np.random.randint(1, min([group_sizes_new[group_idx], test_len-sample_len])) # Pick a random sample of size 1<=n<group_size from the chosen group
            else:
                n = 1
            last_idx = last_idx_arr[group_idx]-n_samples_chosen_per_group[group_idx] # Index of the last observation remaining in each group within the overall dataset
            test_idx += [last_idx-i for i in range(n)]
            n_samples_chosen_per_group[group_idx] += n # Update number of samples chosen from the group
            group_sizes_new[group_idx] -= n # Update current group sizes
            sample_len += n

    mask = np.ones(y_len, dtype=bool)
    mask[test_idx] = False
    X_train, X_test = X[mask], X[~mask]
    y_train, y_test = y[mask], y[~mask]
    F_train, F_test = F[mask], F[~mask]

    return X_train, X_test, y_train, y_test, F_train, F_test

def train_test_split_grouped_extrapolation(X, y, F, groups, test_size=0.2, random_state=1):
    '''
    Train/test split, but test set only contains only unseen groups.
    Note: X should be a dataframe of observations.
    ``test_size`` represents the proportion of groups to include in the test split (rounded up).
    '''
    train_idx, test_idx = next(GroupShuffleSplit(test_size=test_size, random_state=random_state).split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    F_train, F_test = F[train_idx], F[test_idx]

    return X_train, X_test, y_train, y_test, F_train, F_test


def generate_datasets(n, m, p, n_datasets, n_valid, func, shared_gp=True, random_state=1, include_cat_feature=False):

    '''
    Returns datasets as generated by the function generate_data above. 1st and 2nd test sets refer to the Extrapolation and Interpolation test sets respectively. 
    ''' 

    datasets = {}
    datasets['data'] = []
    datasets['X_train'] = []
    datasets['X_test1'] = []
    datasets['X_test2'] = []
    datasets['y_train'] = []
    datasets['y_test1'] = []
    datasets['y_test2'] = []
    datasets['data_train'] = []
    datasets['F_test1'] = []
    datasets['F_test2'] = []

    # Create training and test sets
    for i in range(n_datasets):
        data, X, y, F = generate_data(n, m, p, func, shared_gp, random_state=i+random_state)
        datasets['data'].append(data)

        # Extrapolation test set
        group_sizes = data.groupby(['group']).size().to_numpy()
        X_new, X_test1, y_new, y_test1, F_new, F_test1 = train_test_split_grouped_extrapolation(X, y, F, X['group'], test_size=0.2, random_state=i)

        # Interpolation test set
        group_sizes = X_new.groupby(['group']).size().to_numpy()
        X_train, X_test2, y_train, y_test2, _, F_test2 = train_test_split_grouped_interpolation(X_new, y_new, F_new, group_sizes, test_size=0.25, random_state=i)

        # Create dataset object
        if include_cat_feature: # This is for plain GPBoost model without any random effects
            data_train = gpb.Dataset(data=X_train[['group', 'times', 'feature_1','feature_2','feature_3','feature_4']], label=y_train, categorical_feature=['group'])
            #X_test1, X_test2 = X_test1.drop('times', axis=1, inplace=False), X_test2.drop('times', axis=1, inplace=False)
        else:
            data_train = gpb.Dataset(data=X_train[['feature_1','feature_2','feature_3','feature_4']], label=y_train)

        datasets['X_train'].append(X_train)
        datasets['X_test1'].append(X_test1)
        datasets['X_test2'].append(X_test2)
        datasets['y_train'].append(y_train)
        datasets['y_test1'].append(y_test1)
        datasets['y_test2'].append(y_test2)
        datasets['F_test1'].append(F_test1)
        datasets['F_test2'].append(F_test2)
        datasets['data_train'].append(data_train)
    
    # Create validation datasets
    validation_datasets = {}
    validation_datasets['X'] = []
    validation_datasets['y'] = []
    validation_datasets['data_train'] = []

    for i in range(n_valid):
        data, X, y, _ = generate_data(n, m, p, func, shared_gp, random_state=i+n_datasets+random_state)
        X['times'] = pd.Series(np.tile(np.arange(p)*10+10, m)) # Fix times across validation datasets for grid search
        if include_cat_feature:
            data_train = gpb.Dataset(data=X[['group', 'times', 'feature_1','feature_2','feature_3','feature_4']], label=y, categorical_feature=['group'])
        else:
            data_train = gpb.Dataset(data=X[['feature_1','feature_2','feature_3','feature_4']], label=y)
        validation_datasets['X'].append(X)
        validation_datasets['y'].append(y)
        validation_datasets['data_train'].append(data_train)

    return datasets, validation_datasets



def grid_search_tune_parameters_multiple(param_grid, train_sets, params=None, num_try_random=None,
                                         num_boost_round=100, gp_model=None,
                                         use_gp_model_for_validation=True, train_gp_model_cov_pars=True,
                                         folds=None, nfold=5, stratified=False, shuffle=True,
                                         metrics=None, fobj=None, feval=None, init_model=None,
                                         feature_name='auto', categorical_feature='auto',
                                         early_stopping_rounds=None, fpreproc=None,
                                         verbose_eval=1, seed=0, callbacks=None):
    '''
    Conducts grid search over a list of training sets.
    '''
    # Check correct format
    if not isinstance(param_grid, dict):
        raise ValueError('param_grid needs to be a dict')
    if verbose_eval is None:
        verbose_eval = 0
    else:
        if not isinstance(verbose_eval, int):
            raise ValueError('verbose_eval needs to be int')
    if params is None:
        params = {}
    else:
        params = copy.deepcopy(params)
    param_grid = copy.deepcopy(param_grid)

    for param in param_grid:
        if gpb.basic.is_numeric(param_grid[param]):
            param_grid[param] = [param_grid[param]]
        param_grid[param] = gpb.basic._format_check_1D_data(param_grid[param],
                                                  data_name=param, check_data_type=False,
                                                  check_must_be_int=False, convert_to_type=None)

    # Determine combinations of parameter values that should be tried out
    grid_size = gpb.engine._get_grid_size(param_grid)

    if num_try_random is not None: # Random grid search 
        if num_try_random > grid_size:
            raise ValueError('num_try_random is larger than the number of all possible combinations of parameters in param_grid')
        try_param_combs = np.random.RandomState(seed).choice(a=grid_size, size=num_try_random, replace=False)
        print('Starting random grid search with ' + str(num_try_random) + ' trials out of ' + str(grid_size) + ' parameter combinations...')
    else:
        try_param_combs = range(grid_size)
        print('Starting deterministic grid search with ' + str(grid_size) + ' parameter combinations...')
    if verbose_eval < 2:
        verbose_eval_cv = False
    else:
        verbose_eval_cv = True

    best_score = 1e99
    current_score = 1e99
    best_params = {}
    best_num_boost_round = num_boost_round
    counter_num_comb = 1
    counter = 0 

    for param_comb_number in try_param_combs: # each number here corresponds to a different choice of parameters

        counter += 1
        print('\n')
        print_progress(iteration = counter, total = grid_size, prefix = '', suffix = '', decimals = 4, bar_length = 50)
        print('\n')

        cvbst_list = []
        param_comb = gpb.engine._get_param_combination(param_comb_number=param_comb_number, param_grid=param_grid)
        for param in param_comb:
            params[param] = param_comb[param]
        if verbose_eval >= 1:
            print('Trying parameter combination ' + str(counter_num_comb) +
                  ' of ' + str(len(try_param_combs)) + ': ' + str(param_comb) + ' ...')

        for train_set in train_sets: 

            # Perform cross-validation for choosing number of boosting iterations
            cvbst = gpb.cv(params=params, train_set=train_set, num_boost_round=num_boost_round,
                        gp_model=gp_model, use_gp_model_for_validation=use_gp_model_for_validation,
                        train_gp_model_cov_pars=train_gp_model_cov_pars,
                        folds=folds, nfold=nfold, stratified=stratified, shuffle=shuffle,
                        metrics=metrics, fobj=fobj, feval=feval, init_model=init_model,
                        feature_name=feature_name, categorical_feature=categorical_feature,
                        early_stopping_rounds=early_stopping_rounds, fpreproc=fpreproc,
                        verbose_eval=verbose_eval_cv, seed=seed, callbacks=callbacks,
                        eval_train_metric=False, return_cvbooster=False)
            cvbst_list.append(cvbst)

        current_score_is_better = False
        current_score = np.mean([np.min(cvbst[next(iter(cvbst))]) for cvbst in cvbst_list])

        if current_score < best_score:
            current_score_is_better = True
        if current_score_is_better:
            best_score = current_score
            best_params = param_comb
            best_num_boost_round = np.mean([np.argmin(cvbst[next(iter(cvbst))]) for cvbst in cvbst_list])

            if verbose_eval >= 1:
                print('***** New best score (' + str(best_score) + ') found for the following parameter combination:')
                best_params_print = copy.deepcopy(best_params)
                best_params_print['num_boost_round'] = best_num_boost_round
                print(best_params_print)

        counter_num_comb = counter_num_comb + 1

    return {'best_params': best_params, 'best_iter': best_num_boost_round, 'best_score': best_score}


def train_and_test(datasets, n_datasets, num_boost_round=0, params={}, merf=False, GPBoost_cat=False, linear=False, GP=False, shared=False, 
                                                                                no_features=False, kernel='exponential', matern_param=2.5):

    np.random.seed(1)
    RMSE_list1 = []
    RMSE_list2 = []
    F_list = []
    time_list = []

    for i in tnrange(n_datasets):

        data_train = datasets['data_train'][i]
        X_train = datasets['X_train'][i]
        X_test1 = datasets['X_test1'][i]
        X_test2 = datasets['X_test2'][i]
        y_train = datasets['y_train'][i]
        y_test1 = datasets['y_test1'][i]
        y_test2 = datasets['y_test2'][i]
        F_test1 = datasets['F_test1'][i]
        F_test2 = datasets['F_test2'][i]


        #if not GPBoost_cat:
        # Random effects covariates
        groups_train = X_train['group']
        groups_test1 = X_test1['group']
        groups_test2 = X_test2['group']
        times_train = X_train['times']
        times_test1 = X_test1['times']
        times_test2 = X_test2['times']


        ########### Linear mixed effects model ##############
        if linear:
            if no_features:
                X_train_linear = np.ones(len(X_train))
                X_test1_linear = np.ones(len(X_test1))
                X_test2_linear = np.ones(len(X_test2))
            else:
                X_train_linear = np.column_stack((np.ones(len(X_train)), X_train[['feature_1','feature_2','feature_3','feature_4']])) # add column of ones to get a global intercept 
                X_test1_linear = np.column_stack((np.ones(len(X_test1)), X_test1[['feature_1','feature_2','feature_3','feature_4']]))
                X_test2_linear = np.column_stack((np.ones(len(X_test2)), X_test2[['feature_1','feature_2','feature_3','feature_4']]))
        #########################################################


        # Define the model
        if merf:
            Z_train = np.column_stack((np.ones(len(X_train)), times_train))
            Z_test1 = np.column_stack((np.ones(len(X_test1)), times_test1))
            Z_test2 = np.column_stack((np.ones(len(X_test2)), times_test2))
            fixed_effects_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=1)
            merf_model = MERF(fixed_effects_model, max_iterations=100)

        elif GPBoost_cat:
            pass 


        else:
            # Models with GP components for the observation times (includes both linear models and Boosted-tree fixed effect models)
            if GP:

                if kernel == 'matern':
                    if shared:
                        gp_model = gpb.GPModel(group_data=groups_train, gp_coords=times_train, cov_function=kernel, cov_fct_shape=matern_param)
                    else:
                        gp_model = gpb.GPModel(group_data=groups_train, gp_coords=times_train, cluster_ids=groups_train, cov_function=kernel, cov_fct_shape=matern_param)

                else:
                    if shared:
                        gp_model = gpb.GPModel(group_data=groups_train, gp_coords=times_train, cov_function=kernel)
                    else:
                        gp_model = gpb.GPModel(group_data=groups_train, gp_coords=times_train, cluster_ids=groups_train, cov_function=kernel)

            # LME models without GP components
            else:
                gp_model = gpb.GPModel(group_data=np.column_stack((groups_train, times_train)))
            gp_model.set_optim_params(params={'optimizer_cov': 'gradient_descent', 'use_nesterov_acc': True}) # these are the default parameters


        #################### Training ####################
        start_time = time.time()

        try:
            if merf:
                merf_model.fit(X_train[['feature_1','feature_2','feature_3','feature_4']], Z_train, groups_train, y_train)

            elif GPBoost_cat:
                bst = gpb.train(params=params, train_set=data_train, num_boost_round=num_boost_round, categorical_feature=['group'])
                #bst = gpb.train(params=params, train_set=data_train, num_boost_round=num_boost_round, categorical_feature=['group'], feature_name=[col_name for col_name in datasets['X_train'][0].columns.values])

            else:
                if linear: # LME models
                    gp_model.fit(y=y_train, X=X_train_linear)
                else: # GPBoost models
                    bst = gpb.train(params=params, train_set=data_train, gp_model=gp_model, num_boost_round=num_boost_round)

            time_list.append(time.time() - start_time)


            #################### Prediction ####################
            if merf:
                y_pred1 = merf_model.predict(X_test1[['feature_1','feature_2','feature_3','feature_4']], Z_test1, groups_test1)
                y_pred2 = merf_model.predict(X_test2[['feature_1','feature_2','feature_3','feature_4']], Z_test2, groups_test2)
                F_pred1 = merf_model.trained_fe_model.predict(X_test1[['feature_1','feature_2','feature_3','feature_4']])
                F_pred2 = merf_model.trained_fe_model.predict(X_test2[['feature_1','feature_2','feature_3','feature_4']])

            elif GPBoost_cat:
                y_pred1 = bst.predict(data=X_test1)
                y_pred2 = bst.predict(data=X_test2)

            else:

                if linear: # LME models
                    if GP:
                        if shared:
                            y_pred1 = gp_model.predict(group_data_pred=groups_test1, gp_coords_pred=times_test1, X_pred=X_test1_linear)['mu']
                            y_pred2 = gp_model.predict(group_data_pred=groups_test2, gp_coords_pred=times_test2, X_pred=X_test2_linear)['mu']
                        else:
                            y_pred1 = gp_model.predict(group_data_pred=groups_test1, gp_coords_pred=times_test1, cluster_ids_pred=groups_test1, X_pred=X_test1_linear)['mu']
                            y_pred2 = gp_model.predict(group_data_pred=groups_test2, gp_coords_pred=times_test2, cluster_ids_pred=groups_test2, X_pred=X_test2_linear)['mu']
                    else:
                        y_pred1 = gp_model.predict(group_data_pred=np.column_stack((groups_test1, times_test1)), X_pred=X_test1_linear)['mu']
                        y_pred2 = gp_model.predict(group_data_pred=np.column_stack((groups_test2, times_test2)), X_pred=X_test2_linear)['mu']

                        

                    if not no_features: # Linear models Fixed effects part prediction, i.e. F(X) = Xβ for LME models
                        F_pred1 = X_test1_linear.dot(gp_model.get_coef().T)
                        F_pred2 = X_test2_linear.dot(gp_model.get_coef().T)

                else: # GPBoost models
                    if GP:
                        if shared:
                            pred1 = bst.predict(data=X_test1[['feature_1','feature_2','feature_3','feature_4']], group_data_pred=groups_test1, gp_coords_pred=times_test1, pred_latent=True)
                            pred2 = bst.predict(data=X_test2[['feature_1','feature_2','feature_3','feature_4']], group_data_pred=groups_test2, gp_coords_pred=times_test2, pred_latent=True)
                        else:
                            pred1 = bst.predict(data=X_test1[['feature_1','feature_2','feature_3','feature_4']], group_data_pred=groups_test1, gp_coords_pred=times_test1, cluster_ids_pred=groups_test1, pred_latent=True)
                            pred2 = bst.predict(data=X_test2[['feature_1','feature_2','feature_3','feature_4']], group_data_pred=groups_test2, gp_coords_pred=times_test2, cluster_ids_pred=groups_test2, pred_latent=True)
                    else:
                        pred1 = bst.predict(data=X_test1[['feature_1','feature_2','feature_3','feature_4']], group_data_pred=np.column_stack((groups_test1, times_test1)), pred_latent=True)
                        pred2 = bst.predict(data=X_test2[['feature_1','feature_2','feature_3','feature_4']], group_data_pred=np.column_stack((groups_test2, times_test2)), pred_latent=True)

                        

                    y_pred1 = pred1['fixed_effect'] + pred1['random_effect_mean']
                    y_pred2 = pred2['fixed_effect'] + pred2['random_effect_mean']


                    if not no_features: # Fixed effects part prediction
                        F_pred1, F_pred2 = pred1['fixed_effect'], pred2['fixed_effect']



            RMSE_list1.append(np.sqrt(np.mean((y_test1-y_pred1)**2)))
            RMSE_list2.append(np.sqrt(np.mean((y_test2-y_pred2)**2)))

            if not no_features:
                F_list.append(np.sqrt(np.mean((F_test1-F_pred1)**2)))
                F_list.append(np.sqrt(np.mean((F_test2-F_pred2)**2))) # RMSE for F is given by both interpolation and extrapolation

        except:
            print('Error encountered')
            continue  

    if no_features: # fixed part prediction is just the global average target value
        return time_list, RMSE_list1, RMSE_list2
    else:
        return time_list, RMSE_list1, RMSE_list2, F_list




