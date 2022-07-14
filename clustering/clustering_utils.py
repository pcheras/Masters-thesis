import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram



def generate_ts_data(funcs, n_ts, rand_seed=1):
    
    #labels_dict = {'increasing' : 0, 'decreasing' : 1, 'other' : 2}
    sampled_times = []
    true_labels = []
    y_series = []

    for j in range(n_ts):
        
        np.random.seed(j * rand_seed)
        n_time_points = np.random.choice(17, 1) + 2 # n_time_points is in [3, 20]
        t = np.sort(np.random.choice(1000, size=n_time_points, replace=False)) # time interval considered is (0, 1000) 
        t = np.concatenate([np.array([0]), t]) # all series should start at 0 
        sampled_times.append(t)
        func_type = np.random.choice(['increasing', 'decreasing', 'other'], size=1)[0] # choose a function number at random
        true_labels.append(func_type)
        noise_val = np.random.rand(1) * 500 # sample noise value from U(0, 500)

        if func_type == 'increasing':

            f = np.random.choice(['linear', 'quadratic'], size=1)[0]
            if f == 'linear':
                slope = np.random.rand(1) * 20 # sample slope coefficient from U(0, 20)
                y = funcs['linear'](t, coef=slope, noise=noise_val)
            else: # quadratic
                quadratic_coef = np.random.rand(1) * 0.01  # sample slope coefficient from U(0, 0.01)
                y = funcs['quadratic'](t, coef=quadratic_coef, noise=noise_val)

            y_series.append(y)


        elif func_type == 'decreasing':

            f = np.random.choice(['linear', 'quadratic'], size=1)
            if f == 'linear':
                slope = - np.random.rand(1) * 20 # sample slope coefficient from U(-20, 0)
                y = funcs['linear'](t, coef=slope, noise=noise_val)
            else: # quadratic
                quadratic_coef = - np.random.rand(1) * 0.01  # sample slope coefficient from U(-0.01, 0)
                y = funcs['quadratic'](t, coef=quadratic_coef, noise=noise_val)
                
            y_series.append(y)

        else: # other function type 
            y=funcs['random'](t, noise=noise_val)
            y_series.append(y)

    return sampled_times, y_series, np.array(true_labels)


def plot_data(sampled_times, y_series, labels):

    fig, axs = plt.subplots(1, 3, figsize=(40, 10))

    for n in range(len(labels)):
        if labels[n] == 'increasing':
            axs[0].plot(sampled_times[n], y_series[n], '-o', label=f'{n}')
        elif labels[n] == 'decreasing':
            axs[1].plot(sampled_times[n], y_series[n], '-o', label=f'{n}')
        else:
            axs[2].plot(sampled_times[n], y_series[n], '-o', label=f'{n}')

    axs[0].set_title('Increasing trend', fontsize=25)
    axs[0].set_ylabel('y', fontsize = 25)
    axs[0].set_xlabel('Time', fontsize = 25)
    axs[1].set_title('Decreasing trend', fontsize=25)
    axs[1].set_ylabel('y', fontsize = 25)
    axs[1].set_xlabel('Time', fontsize = 25)
    axs[2].set_title('Other trend', fontsize=25);
    axs[2].set_ylabel('y', fontsize = 25)
    axs[2].set_xlabel('Time', fontsize = 25)
    axs[0].legend(frameon = False);
    axs[1].legend(frameon = False);
    axs[2].legend(frameon = False);
    fig.suptitle('Ground truth cluster assignments',  fontsize = 40);
    plt.show();

def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def hierarchical_clustering(dist_mat, method='average', plot_dendrogram=True):
    
    if method == 'complete':
        Z = complete(dist_mat)
    if method == 'single':
        Z = single(dist_mat)
    if method == 'average':
        Z = average(dist_mat)
    if method == 'ward':
        Z = ward(dist_mat)
    
    if plot_dendrogram:
        fig = plt.figure(figsize=(16, 8))
        dn = dendrogram(Z)
        plt.title(f"Dendrogram for {method}-linkage with DTW distance", fontsize=18)
        plt.xlabel('Time series ID', fontsize=14)
        #plt.xticks(rotation=90)
        plt.show()
    
    return Z
    

def plot_subset_from_pred_clusters(pred_labels, ids, times, egfr_vals, subset_size, seed=1, plot_smoothed=False, alpha=None):
    
    """
    Plots a random subset of 'subset_size' eGFR trajectories based on their assigned cluster labels
    """

    if plot_smoothed:
        assert alpha is not None , 'Provide exponential smoothing constant'
    
    unique_labels = np.unique(pred_labels)
    fig, ax = plt.subplots(1, len(unique_labels), figsize=(60, 20))
    np.random.seed(seed)

    for j , label in enumerate(unique_labels):
        
        subset_ind = np.where(pred_labels == label)[0]
        cluster_subset_ind = np.random.choice(subset_ind, size=subset_size, replace=False) # selected a subset of the time series which were assigned to cluster 'label'

        for i in cluster_subset_ind:
           
            if plot_smoothed:
                ax[j].plot(times[i]['times'], exponential_smoothing(egfr_vals[i]['egfr'].values, alpha=alpha), '-o', label=f'ID: {ids[i]}')
            else:
                ax[j].plot(times[i]['times'], egfr_vals[i]['egfr'], '-o', label=f'ID: {ids[i]}')

            ax[j].set_xlabel('Time', size=18)
            ax[j].set_ylabel('eGFR', size=18)
            ax[j].set_title(f'Cluster {label}', fontsize=30)
            ax[j].legend(frameon = False);
            fig.suptitle('Cluster assignments',  fontsize = 40);

    return 