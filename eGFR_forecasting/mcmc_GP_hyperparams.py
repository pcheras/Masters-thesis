if __name__ == '__main__':

    import numpy as np
    import scipy
    import time
    import copy
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    from utils_extended import *
    import gpboost as gpb
    from pdpbox import pdp
    import pystan 
    import arviz as az

    site_1 = 'Sheffield'
    site_2 = 'sheffield'

    df_1 = pd.DataFrame(data=np.zeros(shape=(5, 2)), columns=['RMSE_interpolation', 'RMSE_extrapolation'], index=np.arange(1, 6))
    df_2 = pd.DataFrame(data=np.zeros(shape=(5, 2)), columns=['RMSE_interpolation', 'RMSE_extrapolation'], index=np.arange(1, 6))
    posterior_params = pd.DataFrame(data=np.zeros(shape=(5, 2)), columns=['rho', 'sigma_sq'], index=np.arange(1, 6))

    for i in range(1, 6):

        run_number = i


        data_train_full_raw = pd.read_csv(f'{site_1}/train_test_sets/data_train_full_{site_2}_{run_number}.csv').drop(columns=['site'])
        test_extrapolation_raw = pd.read_csv(f'{site_1}/train_test_sets/test_extrapolation_{site_2}_{run_number}.csv').drop(columns=['site'])
        test_interpolation_raw = pd.read_csv(f'{site_1}/train_test_sets/test_interpolation_{site_2}_{run_number}.csv').drop(columns=['site'])

        # One-hot encode categorical features
        data_train_full_raw2 = encode(data_train_full_raw)
        test_extrapolation = encode(test_extrapolation_raw)
        test_interpolation_raw2 = encode(test_interpolation_raw)

        # Dummy encode categorical features for linear model
        data_train_full2_raw2 = encode(data_train_full_raw, linear=True)
        test_extrapolation2 = encode(test_extrapolation_raw, linear=True)
        test_interpolation2_raw2 = encode(test_interpolation_raw, linear=True)

        # Fix missing columns
        test_extrapolation['disease_Obstructive'] = 0
        data_train_full_raw2['gender_Unknown'] = 0
        data_train_full_raw2['ethnicity_Asian'] = 0
        data_train_full_raw2['disease_Vascular'] = 0
        data_train_full = data_train_full_raw2[list(test_extrapolation.columns.values)]

        test_interpolation_raw2['gender_Unknown'] = 0
        test_interpolation_raw2['ethnicity_Asian'] = 0
        test_interpolation_raw2['disease_Vascular'] = 0
        test_interpolation = test_interpolation_raw2[list(test_extrapolation.columns.values)]

        test_extrapolation2['disease_Obstructive'] = 0
        data_train_full2_raw2['gender_Unknown'] = 0
        data_train_full2_raw2['ethnicity_Caucasian'] = 0
        data_train_full2_raw2['disease_Vascular'] = 0
        data_train_full2 = data_train_full2_raw2[list(test_extrapolation2.columns.values)]

        test_interpolation2_raw2['gender_Unknown'] = 0
        test_interpolation2_raw2['ethnicity_Caucasian'] = 0
        test_interpolation2_raw2['disease_Vascular'] = 0
        test_interpolation2 = test_interpolation2_raw2[list(test_extrapolation2.columns.values)]

        data_train_list, data_val_list = [], []

        for i in range(5):
            train_raw = pd.read_csv('Patras/tuning_data/data_cv_train_patras_' + str(i+1) + '.csv').drop(columns=['site'])
            train = encode(train_raw)
            val_raw = pd.read_csv('Patras/tuning_data/data_cv_val_patras_' + str(i+1) + '.csv').drop(columns=['site'])
            val = encode(val_raw)
            data_train_list.append(train)
            data_val_list.append(val)

        data_cv_list = [pd.concat([data_train_list[i], data_val_list[i]], ignore_index=True).reset_index(drop=True).fillna(0) for i in range(5)]

        # Get indices of training and validation sets for cross-validation
        data_cv_indices_list = []
        for i in range(5):
            data_cv_indices_list.append((np.arange(len(data_train_list[i])), np.arange(len(data_train_list[i]), len(data_train_list[i])+len(data_val_list[i]))))

        # Train/test split
        train_X, train_y = data_train_full.drop(columns=['egfr']), data_train_full['egfr'].to_numpy(dtype=np.float64)
        test_extrapolation_X, test_extrapolation_y = test_extrapolation.drop(columns=['egfr']), test_extrapolation['egfr'].to_numpy(dtype=np.float64)
        test_interpolation_X, test_interpolation_y = test_interpolation.drop(columns=['egfr']), test_interpolation['egfr'].to_numpy(dtype=np.float64)
        data_train_full_gpb = gpb.Dataset(data=train_X.drop(columns=['ID', 'times']), label=train_y)
        data_train_full_gpb_cat = gpb.Dataset(data=train_X, label=train_y, categorical_feature=['ID'])


        # For linear
        train_X2 = data_train_full2.drop(columns=['egfr'])
        test_extrapolation_X2 = test_extrapolation2.drop(columns=['egfr'])
        test_interpolation_X2 = test_interpolation2.drop(columns=['egfr'])
        data_train_full2_gpb = gpb.Dataset(data=train_X2.drop(columns=['ID', 'times']), label=train_y)

        train_X_features = train_X.drop(columns=['ID', 'times'])
        test_extrapolation_X_features = test_extrapolation_X.drop(columns=['ID', 'times'])
        test_interpolation_X_features = test_interpolation_X.drop(columns=['ID', 'times'])

        train_X2_features = train_X2.drop(columns=['ID', 'times'])
        test_extrapolation_X2_features = test_extrapolation_X2.drop(columns=['ID', 'times'])
        test_interpolation_X2_features = test_interpolation_X2.drop(columns=['ID', 'times'])
        train_X2_linear = np.column_stack((np.ones(len(train_X2_features)), train_X2_features))
        test_extrapolation_X2_linear = np.column_stack((np.ones(len(test_extrapolation_X2_features)), test_extrapolation_X2_features))
        test_interpolation_X2_linear = np.column_stack((np.ones(len(test_interpolation_X2_features)), test_interpolation_X2_features))

        # 8. GPBoost with Independent Gaussian Process
        np.random.seed(1)
        gp_model = gpb.GPModel(group_data=train_X['ID'], gp_coords=train_X['times'], cluster_ids=train_X['ID'].values.astype(int))
        gp_model.set_optim_params(params={'optimizer_cov': 'gradient_descent', 'use_nesterov_acc': True, 'init_cov_pars' : np.array([10, 10, 10, 10])})
        params = {'objective': 'regression_l2',
                'learning_rate': 1.5,
                'max_depth': 10,
                'min_data_in_leaf': 120,
                'verbose': 0}

        start_time = time.time()
        bst = gpb.train(params=params,
                        train_set=data_train_full_gpb,
                        gp_model=gp_model,
                        num_boost_round=1238)

        pred_training = bst.predict(data=train_X_features,
                                        group_data_pred=train_X['ID'],
                                        gp_coords_pred=train_X['times'],
                                        cluster_ids_pred=train_X['ID'].values.astype(int), pred_latent=True, predict_var=True)

        y_pred_training = pred_training['fixed_effect'] + pred_training['random_effect_mean']


        gp_code = """
        // Fit a Gaussian process's hyperparameters
        // for squared exponential prior

        data {
        int<lower=1> N;
        vector[N] x;
        vector[N] y;
        }
        transformed data {
        vector[N] mu;
        for (i in 1:N) 
            mu[i] <- 0;

        }
        parameters {

        real<lower=3500> rho; // kernel lengthscale
        real<lower=0> sigma_sq; // kernel marginal variance
        
        }
        model {
        matrix[N, N] L_K;
        matrix[N,N] K;

        // off-diagonal elements
        for (i in 1:(N-1)) {
            for (j in (i+1):N) {
            K[i,j] <- sigma_sq * exp(- 1/rho * pow(x[i] - x[j],2));
            K[j,i] <- K[i,j];
            }
        }

        // diagonal elements
        for (k in 1:N)
            K[k,k] <- 0.25 + sigma_sq; // assume known error term variance is 0.25 (then K[k, k] = sigma_sq + error term variance)
        
        L_K = cholesky_decompose(K);
        
            // Sigma[k,k] <- eta_sq + sigma_sq; // + jitter

        
        rho ~ normal(5000, 1000);
        sigma_sq ~ normal(150, 100);

        y ~ multi_normal_cholesky(mu, L_K);
        }
        """

        # Obtain the posterior of the kernel hyperparameters using MCMC over a subset of the training data for computational reasons
        np.random.seed(1)
        subset_index = np.random.choice(a=len(train_y), size=1000, replace=False)
        y_mcmc = train_y[subset_index] - pred_training['fixed_effect'][subset_index]
        x_mcmc = train_X['times'][subset_index].values

        gp_data = {'N': len(subset_index), 'y': y_mcmc, 'x': x_mcmc}

        gp_fit = pystan.stan(model_code=gp_code, data=gp_data,
                    iter=1500, chains=2, warmup=300, n_jobs=-1, seed=1) 

        # Print a summary 
        print(gp_fit.stansummary(pars=['rho', 'sigma_sq'], probs = (0.025, 0.975)))



        #################### LETS SEE IF POSTERIOR MEAN USED AS CUSTOM GP COV PARAMS IMPROVE SCORES ###########################
        np.random.seed(1)
        rho_post_mean = gp_fit.to_dataframe()['rho'].mean()
        sigma_sq_post_mean = gp_fit.to_dataframe()['sigma_sq'].mean()
        rho_post_std = gp_fit.to_dataframe()['rho'].std()
        sigma_sq_post_std = gp_fit.to_dataframe()['sigma_sq'].std()

        posterior_params.iloc[i-1, 0] = f'{rho_post_mean} +\- {rho_post_std}'
        posterior_params.iloc[i-1, 1] = f'{sigma_sq_post_mean} +\- {sigma_sq_post_std}'

        gp_model = gpb.GPModel(group_data=train_X['ID'], gp_coords=train_X['times'], cluster_ids=train_X['ID'].values.astype(int))
        gp_model.set_optim_params(params={'optimizer_cov': 'gradient_descent', 'use_nesterov_acc': True, 'init_cov_pars' : np.array([1, 1, 1, 1])})

        params = {'objective': 'regression_l2',
                'learning_rate': 1.5,
                'max_depth': 5,
                'min_data_in_leaf': 100,
                'verbose': 0}

        bst = gpb.train(params=params,
                        train_set=data_train_full_gpb,
                        gp_model=gp_model,
                        num_boost_round=1350,
                        train_gp_model_cov_pars=True)

        # Use posterior means for rho and sigma_sq during prediction time
        gp_params = np.concatenate([gp_model.get_cov_pars().values[0][:2] , np.array([sigma_sq_post_mean, rho_post_mean])])


        pred_interpolation = bst.predict(data=test_interpolation_X_features,
                                        group_data_pred=test_interpolation_X['ID'],
                                        gp_coords_pred=test_interpolation_X['times'],
                                        cluster_ids_pred=test_interpolation_X['ID'].values.astype(int), pred_latent=True)

        y_pred_interpolation = pred_interpolation['fixed_effect'] + gp_model.predict(group_data_pred=test_interpolation_X['ID'], gp_coords_pred=test_interpolation_X['times'], 
                                                                                                    predict_var=True, cov_pars=gp_params, cluster_ids_pred=test_interpolation_X['ID'].values.astype(int))['mu']

        int_score = np.sqrt(np.mean((test_interpolation_y-y_pred_interpolation)**2))

        pred_extrapolation = bst.predict(data=test_extrapolation_X_features,
                                        group_data_pred=test_extrapolation_X['ID'],
                                        gp_coords_pred=test_extrapolation_X['times'],
                                        cluster_ids_pred=test_extrapolation_X['ID'].values.astype(int), pred_latent=True)

        y_pred_extrapolation = pred_extrapolation['fixed_effect'] + gp_model.predict(group_data_pred=test_extrapolation_X['ID'], gp_coords_pred=test_extrapolation_X['times'], 
                                                                                                    predict_var=True, cov_pars=gp_params, cluster_ids_pred=test_extrapolation_X['ID'].values.astype(int))['mu']

        ext_score = np.sqrt(np.mean((test_extrapolation_y-y_pred_extrapolation)**2)) 

        df_1.iloc[i-1, 0] = int_score
        df_1.iloc[i-1, 1] = ext_score


        #################################### POSTERIOR MEAN AS INIT COV PARAM #########################################################

        # 8. GPBoost with Independent Gaussian Process
        np.random.seed(1)
        gp_model = gpb.GPModel(group_data=train_X['ID'], gp_coords=train_X['times'], cluster_ids=train_X['ID'].values.astype(int))
        gp_model.set_optim_params(params={'optimizer_cov': 'gradient_descent', 'use_nesterov_acc': True, 'init_cov_pars':np.array([1, 1, sigma_sq_post_mean, rho_post_mean])})

        params = {'objective': 'regression_l2',
                'learning_rate': 1.5,
                'max_depth': 5,
                'min_data_in_leaf': 100,
                'verbose': 0}

        start_time = time.time()
        bst = gpb.train(params=params,
                        train_set=data_train_full_gpb,
                        gp_model=gp_model,
                        num_boost_round=1350,
                        train_gp_model_cov_pars=True)

        pred_interpolation = bst.predict(data=test_interpolation_X_features,
                                        group_data_pred=test_interpolation_X['ID'],
                                        gp_coords_pred=test_interpolation_X['times'],
                                        cluster_ids_pred=test_interpolation_X['ID'].values.astype(int), pred_latent=True)

        y_pred_interpolation = pred_interpolation['fixed_effect'] + pred_interpolation['random_effect_mean']
        int_score_2 = np.sqrt(np.mean((test_interpolation_y-y_pred_interpolation)**2))


        pred_extrapolation = bst.predict(data=test_extrapolation_X_features,
                                        group_data_pred=test_extrapolation_X['ID'],
                                        gp_coords_pred=test_extrapolation_X['times'],
                                        cluster_ids_pred=test_extrapolation_X['ID'].values.astype(int), pred_latent=True)

        y_pred_extrapolation = pred_extrapolation['fixed_effect'] + pred_extrapolation['random_effect_mean']
        ext_score_2 = np.sqrt(np.mean((test_extrapolation_y-y_pred_extrapolation)**2))

        df_2.iloc[i-1, 0] = int_score_2
        df_2.iloc[i-1, 1] = ext_score_2


    df_1.to_csv('post_mean_as_params.csv')
    df_2.to_csv('post_mean_init_params.csv')
    posterior_params.to_csv('posterior_params_summary.csv')

