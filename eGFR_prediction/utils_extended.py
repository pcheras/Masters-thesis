import gpboost as gpb
import numpy as np
import scipy
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman3
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from tqdm.notebook import tqdm, tnrange
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pystan


def encode(df, linear=False):
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    if linear:
        OH_encoder = OneHotEncoder(drop='first', dtype=int)
    else:
        OH_encoder = OneHotEncoder(dtype=int)
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(df[object_cols]).toarray(), columns=OH_encoder.get_feature_names(object_cols))
    num_data = df.drop(object_cols, axis=1)
    data_new = pd.concat([num_data, OH_cols], axis=1)
    return data_new


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


def generate_data(n, m, p, func, shared_gp=True, random_state=1, rho=0.1, sigma2_2=1):
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

        #sigma2_2 = 1 ** 2  # Marginal variance of GP
        #rho = 0.1  # GP Range parameter

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


def generate_datasets(n, m, p, n_datasets, n_valid, func, shared_gp=True, random_state=1, include_cat_feature=False, rho=0.1, sigma2_2=1, stan_GP=False):

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
    datasets['F_train'] = []
    datasets['F_test1'] = []
    datasets['F_test2'] = []

    if stan_GP: # save F_train to fit a GP on the random effects
        datasets['F_train'] = []

    # Create training and test sets
    for i in range(n_datasets):
        data, X, y, F = generate_data(n, m, p, func, shared_gp, random_state=i+random_state, rho=rho, sigma2_2=sigma2_2)
        datasets['data'].append(data)

        # Extrapolation test set
        group_sizes = data.groupby(['group']).size().to_numpy()
        X_new, X_test1, y_new, y_test1, F_new, F_test1 = train_test_split_grouped_extrapolation(X, y, F, X['group'], test_size=0.2, random_state=i)

        # Interpolation test set
        group_sizes = X_new.groupby(['group']).size().to_numpy()
        X_train, X_test2, y_train, y_test2, F_train, F_test2 = train_test_split_grouped_interpolation(X_new, y_new, F_new, group_sizes, test_size=0.25, random_state=i)

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
        datasets['F_train'].append(F_train)
        datasets['F_test1'].append(F_test1)
        datasets['F_test2'].append(F_test2)
        datasets['data_train'].append(data_train)

        if stan_GP:
            datasets['F_train'].append(F_train)
    
    # Create validation datasets
    validation_datasets = {}
    validation_datasets['X'] = []
    validation_datasets['y'] = []
    validation_datasets['data_train'] = []

    for i in range(n_valid):
        data, X, y, _ = generate_data(n, m, p, func, shared_gp, random_state=i+n_datasets+random_state, rho=rho, sigma2_2=sigma2_2)
        X['times'] = pd.Series(np.tile(np.arange(p)*10+10, m)) # Fix times across validation datasets for grid search

        if include_cat_feature:
            data_train = gpb.Dataset(data=X[['group', 'times', 'feature_1','feature_2','feature_3','feature_4']], label=y, categorical_feature=['group'])
        else:
            data_train = gpb.Dataset(data=X[['feature_1','feature_2','feature_3','feature_4']], label=y)

        validation_datasets['X'].append(X)
        validation_datasets['y'].append(y)
        validation_datasets['data_train'].append(data_train)

    return datasets, validation_datasets



def grid_search_tune_parameters_multiple(data_cv_list, dataset_list, param_grid, params=None, num_try_random=None,
                                         num_boost_round=100,
                                         use_gp_model_for_validation=True, train_gp_model_cov_pars=True,
                                         folds_list=None, nfold=5, stratified=False, shuffle=True,
                                         metrics='rmse', fobj=None, feval=None, init_model=None,
                                         feature_name='auto', categorical_feature='auto',
                                         early_stopping_rounds=None, fpreproc=None,
                                         verbose_eval=1, seed=0, callbacks=None,
                                         gp_model_type='Random Intercept', 
                                         num_neighbors=30, 
                                         vecchia_ordering='none', 
                                         vecchia_pred_type='order_pred_first', 
                                         num_neighbors_pred=30):
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
    if num_try_random is not None:
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


    for param_comb_number in try_param_combs:
        cvbst_list = []
        param_comb = gpb.engine._get_param_combination(param_comb_number=param_comb_number, param_grid=param_grid)
        for param in param_comb:
            params[param] = param_comb[param]
        if verbose_eval >= 1:
            print('Trying parameter combination ' + str(counter_num_comb) +
                  ' of ' + str(len(try_param_combs)) + ': ' + str(param_comb) + ' ...')

        for i, dataset in enumerate(dataset_list):
            folds = [folds_list[i]]
            if gp_model_type == 'Random Intercept':
                gp_model = gpb.GPModel(group_data=np.column_stack((data_cv_list[i]['ID'], data_cv_list[i]['times'])))

            elif gp_model_type == 'Individual GP':
                gp_model = gpb.GPModel(group_data=data_cv_list[i]['ID'], gp_coords=data_cv_list[i]['times'], cluster_ids=data_cv_list[i]['ID'], cov_function='exponential')

            elif gp_model_type == 'Shared GP': # Vecchia approximation in GPBoost model with Shared GP
                gp_model = gpb.GPModel(gp_coords=data_cv_list[i]['times'], cov_function='exponential', vecchia_approx=True, num_neighbors=num_neighbors, 
                                                            vecchia_ordering=vecchia_ordering, vecchia_pred_type=vecchia_pred_type, num_neighbors_pred=num_neighbors_pred)

            elif gp_model_type is None: # CatBoost model
                gp_model = None

            else:
                raise Exception('gp_model_type is invalid')
                
            if gp_model_type is not None:
                gp_model.set_optim_params(params={'optimizer_cov': 'gradient_descent', 'use_nesterov_acc': True})
                
            cvbst = gpb.cv(params=params, train_set=dataset, num_boost_round=num_boost_round,
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
    
    #print(gp_model.get_cov_pars().values.squeeze())
    if no_features: # fixed part prediction is just the global average target value
        return time_list, RMSE_list1, RMSE_list2
    else:
        return time_list, RMSE_list1, RMSE_list2, F_list




def train_and_test_customGP(datasets, n_datasets, num_boost_round=0, params={}, linear=False, no_features=False, kernel='exponential', custom_cov_pars=None, stan_GP_code=None, no_chains=1):

    """ 

    This function is to be used for GPBoost or LME models with SHARED GP and not individual GPs

    """

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
        F_train = datasets['F_train'][i]
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


        if custom_cov_pars is None: # Estimate kernel hyperparams

            # Use the Stan model for posterior inference over the kernel hyperparameters
            gp_data = {'N': len(X_train), 'y': y_train - F_train, 'x': X_train['times']}
            gp_fit = pystan.stan(model_code=stan_GP_code, data=gp_data, iter=3000, chains=no_chains, warmup=1000, n_jobs=-1, seed=i)
            rho_post_mean = gp_fit.to_dataframe()['rho'].mean()
            sigma_sq_post_mean = gp_fit.to_dataframe()['sigma_sq'].mean()


        ########### Linear mixed effects model ##############
        if linear:
            X_train_linear = np.column_stack((np.ones(len(X_train)), X_train[['feature_1','feature_2','feature_3','feature_4']])) # add column of ones to get a global intercept 
            X_test1_linear = np.column_stack((np.ones(len(X_test1)), X_test1[['feature_1','feature_2','feature_3','feature_4']]))
            X_test2_linear = np.column_stack((np.ones(len(X_test2)), X_test2[['feature_1','feature_2','feature_3','feature_4']]))
        #########################################################


        # Define the model
        
        # Models with GP component for the observation times
        gp_model = gpb.GPModel(group_data=groups_train, gp_coords=times_train, cov_function=kernel)


        #################### Training ####################
        start_time = time.time()

        try:
        
            if linear: # LME models
                gp_model.fit(y=y_train, X=X_train_linear)
            else: # GPBoost models
                bst = gpb.train(params=params, train_set=data_train, gp_model=gp_model, num_boost_round=num_boost_round)

            time_list.append(time.time() - start_time)


            #################### Prediction ####################


            #print(gp_model.get_cov_pars().values.squeeze()) # these are the learned kernel hyperparameters

            # Use the kernel variance and lengthscale learned from the posterior 
            if custom_cov_pars is None:
                custom_cov_pars_new = np.concatenate([gp_model.get_cov_pars().values.squeeze()[:2] , np.array([sigma_sq_post_mean, rho_post_mean])])
            else:
                custom_cov_pars_new = np.concatenate([gp_model.get_cov_pars().values.squeeze()[:2] , custom_cov_pars])
            


            if linear: # LME models with shared GP component
                y_pred1 = gp_model.predict(group_data_pred=groups_test1, gp_coords_pred=times_test1, X_pred=X_test1_linear, predict_var=True, cov_pars=custom_cov_pars_new)['mu']
                y_pred2 = gp_model.predict(group_data_pred=groups_test2, gp_coords_pred=times_test2, X_pred=X_test2_linear, predict_var=True, cov_pars=custom_cov_pars_new)['mu']
                    

                if not no_features: # Linear models Fixed effects part prediction, i.e. F(X) = Xβ for LME models
                    F_pred1 = X_test1_linear.dot(gp_model.get_coef().T)
                    F_pred2 = X_test2_linear.dot(gp_model.get_coef().T)


            else: # GPBoost models with shared GP component
                pred1 = bst.predict(data=X_test1[['feature_1','feature_2','feature_3','feature_4']], gp_coords_pred=times_test1, group_data_pred=groups_test1, pred_latent=True)
                pred2 = bst.predict(data=X_test2[['feature_1','feature_2','feature_3','feature_4']], gp_coords_pred=times_test2, group_data_pred=groups_test2, pred_latent=True)
                    

                # Prediction here is fixed effect component of GPBoost and GP posterior mean with custom kernel params
                y_pred1 = pred1['fixed_effect'] + gp_model.predict(group_data_pred=groups_test1, gp_coords_pred=times_test1, predict_var=True, cov_pars=custom_cov_pars_new)['mu']
                y_pred2 = pred2['fixed_effect'] + gp_model.predict(group_data_pred=groups_test2, gp_coords_pred=times_test2, predict_var=True, cov_pars=custom_cov_pars_new)['mu']


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


def train_and_test_Vecchia(datasets, n_datasets, num_boost_round=0, params={}, merf=False, GPBoost_cat=False, linear=False, GP=False, shared=False, no_features=False, kernel='exponential', 
                                                            matern_param=2.5, vecchia_approx=True, num_neighbors=5, vecchia_ordering='none', vecchia_pred_type='order_pred_first', num_neighbors_pred=5):

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
                        gp_model = gpb.GPModel(gp_coords=X_train['times'], cov_function='exponential', vecchia_approx=vecchia_approx, num_neighbors=num_neighbors, 
                                                                                vecchia_ordering=vecchia_ordering, vecchia_pred_type=vecchia_pred_type, num_neighbors_pred=num_neighbors_pred)
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
                            pred1 = bst.predict(data=X_test1[['feature_1','feature_2','feature_3','feature_4']], gp_coords_pred=X_test1['times'], pred_latent=True)
                            pred2 = bst.predict(data=X_test2[['feature_1','feature_2','feature_3','feature_4']], gp_coords_pred=X_test2['times'], pred_latent=True)
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
    
    #print(gp_model.get_cov_pars().values.squeeze())
    if no_features: # fixed part prediction is just the global average target value
        return time_list, RMSE_list1, RMSE_list2
    else:
        return time_list, RMSE_list1, RMSE_list2, F_list



def get_exp_kernel(rho, sigma2):
    return lambda a, b: sigma2 * np.exp(- (a.reshape(-1, 1) - b.reshape(1, -1))**2 / (rho))

    

def plot_conditional_functions(domain, kernel, X, y, sigma_n=0.25, posterior_samples=False, N_samples = 3):

    # The following expressions are obtained from slide 4 but using y instead of f (i.e. noisy observations case)
    Kxx = kernel(X, X) + sigma_n**2 * np.eye(X.shape[0])
    Kx = kernel(domain, X)
    K = kernel(domain, domain)
    
    Kxx_inv = np.linalg.inv(Kxx)
    
    mean = Kx @ (Kxx_inv @ y) # A * C^-1 * y from slide 4
    Cov = K - Kx @ Kxx_inv @ Kx.T #  D - A * C^-1 * A' from the expression (slide 4) for the cond. distr --> f* | x*, x, y
    
    d, V = np.linalg.eigh(Cov)
    L = (np.clip(d, a_min=0.0, a_max=None))**0.5 * V
    f = np.expand_dims(mean, axis=1) + L @ np.random.randn(L.shape[0], N_samples) # sample points f* from the above conditional distr.
                                                                   # by using f* = μ + Σ^0.5 * ε , where ε ~ N(0, I)
                                                                   # with μ and Σ specified as mean, Cov above
    
    std = np.diagonal(Cov)**0.5
    #plt.fill_between(domain, mean-std, mean+std, color='k', alpha=0.1)
    plt.fill_between(domain, mean-1.96*std, mean+1.96*std, alpha=0.25, fc='b', label='95% confidence interval')
    plt.plot(domain, mean, label='Posterior mean');
    
    if posterior_samples:
        plt.plot(domain, f);


def get_indGP_plot(train_df, interp_test_df, bst, rho=None, sigma2=None, group_no=None, sigma_n=2):

    assert (rho is not None) and (sigma2 is not None) and (group_no is not None)

    group_no , n_training_pts = group_no , train_df[train_df['ID'] == group_no].shape[0]


    single_group_data = pd.concat([train_df[train_df['ID'] == group_no] , interp_test_df[interp_test_df['ID'] == group_no]], axis=0, ignore_index=True)
    y_true = single_group_data['egfr'].to_numpy()

    predictions = bst.predict(data=single_group_data.drop(columns=['egfr', 'ID', 'times']), 
                                group_data_pred=single_group_data['ID'],
                                gp_coords_pred=single_group_data['times'],
                                cluster_ids_pred=single_group_data['ID'].values.astype(int),
                                predict_var=True, pred_latent=True)

    domain = np.linspace(0, single_group_data['times'].values.max() + 100, 1000)

    # Condition on the datapoints which have been observed in the training set
    x_obs = single_group_data['times'].values[:n_training_pts]
    x_test = single_group_data['times'].values[n_training_pts:]

    y_vals = single_group_data['egfr'].values - predictions['fixed_effect']
    y_obs = y_vals[:n_training_pts]
    y_test = y_vals[n_training_pts:]

    # Show the learned posterior over group data
    plt.figure(figsize=(15,6))
    plot_conditional_functions(domain, get_exp_kernel(rho=rho, sigma2=sigma2), x_obs, y_obs, sigma_n=sigma_n);
    plt.plot(x_obs, y_obs, 'r.', markersize=10, label='Training points');
    plt.plot(x_test, y_test, 'k.', markersize=10, label='Test set 1 points');

    plt.title(f'Patient {group_no} random effects', fontsize=24)
    plt.xlabel('$t$', fontsize=18)
    plt.ylabel('random effects', fontsize=18)
    plt.legend(loc='upper left');
    plt.show()




