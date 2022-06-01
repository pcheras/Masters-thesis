import random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import linregress


def imputer(df, lookup_table, median_val, variable):
    '''
    If zero height value, impute mean height value depending on patient's gender/ethnicity combination.
    If patient's gender/ethnicity combination has no non-zero mean height value, impute the mean height value of all patients with non-zero heights.
    '''
    df2 = df.copy()
    for i in range(len(df)):
        if df2.at[i, variable] == 0:
            for j in range(len(lookup_table)):
                if df2.at[i, 'gender'] == lookup_table.at[j, 'gender'] and df2.at[i, 'ethnicity'] == lookup_table.at[j, 'ethnicity']:
                    df2.at[i, variable] = lookup_table.at[j, variable]
                    break
            else:
                df2.at[i, variable] = median_val
    
    return df2

def shuffle_data_by_group(df, group_column_name, random_state=1):
    random.seed(random_state)
    group_object = df.groupby(group_column_name)
    groups = [group_object.get_group(x) for x in group_object.groups]
    random.shuffle(groups)
    for i in range(len(groups)):
        groups[i][group_column_name] = i
    return pd.concat(groups).reset_index(drop=True)


def reset_group_id(df, group_column_name):
    group_object = df.groupby(group_column_name)
    groups = [group_object.get_group(x) for x in group_object.groups]
    for i in range(len(groups)):
        groups[i][group_column_name] = i
    return pd.concat(groups).reset_index(drop=True)


def datetime_to_days_diff(df, group_column_name, time_column_name):
    group_object = df.groupby(group_column_name)
    grouped_data = [group_object.get_group(x) for x in group_object.groups]
    new_group_list = []
    for group in grouped_data:
        group['times'] = None
        ref = group[time_column_name].iloc[0]
        for i in range(len(group)):
            group['times'].iloc[i] = (group[time_column_name].iloc[i] - ref) / np.timedelta64(1, 'D')
        group['times'] = pd.to_numeric(group['times'], downcast="integer")
        new_group_list.append(group)
    df_new = pd.concat(new_group_list, axis=0, ignore_index=True)
    return df_new


def train_test_split_grouped_interpolation(df, group_sizes, test_size=0.2, random_state=1):
    '''
    Train/test split, but test set contains at least one observation from each group in the training set, and contains no unseen groups.
    '''
    assert 0 < test_size < 1, "Test size must be strictly between 0 and 1"
    assert np.sum(group_sizes) == len(df), "Sum of group_sizes must be equal to length dataframe"
    assert group_sizes.all() > 0, "Group sizes should be non-negative"
    assert len(group_sizes) < len(df), "Number of groups should be less than number of observations"

    np.random.seed(random_state)
    df_len = len(df)
    test_len = int(test_size * df_len)
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

    mask = np.ones(df_len, dtype=bool)
    mask[test_idx] = False
    df_train, df_test = df[mask], df[~mask]

    return df_train, df_test, n_samples_chosen_per_group


def train_test_split_grouped_extrapolation(df, groups, test_size=0.2, random_state=1):
    '''
    Train/test split for a dataframe, but test set contains only unseen groups.
    ``test_size`` represents the proportion of groups to include in the test split (rounded up).
    '''
    train_idx, test_idx = next(GroupShuffleSplit(test_size=test_size, random_state=random_state).split(df, groups=groups))
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

    return df_train, df_test


def compute_slope(df):
    '''
    Converts a longitudinal dataset to a cross-sectional dataset by taking the slope of the regression line between egfr and times.
    '''
    group_object = df.groupby(['ID'])
    grouped_data = [group_object.get_group(x) for x in group_object.groups]
    df_list = []
    for group in grouped_data:
        # Ignore patients with only 1 observation
        if len(group) == 1:
            continue
        ref = group.iloc[[0]]
        X = group['times'].to_numpy()
        y = group['egfr'].to_numpy()
        result = linregress(X, y)
        ref['slope'] = result.slope
        ref['r2'] = (result.rvalue)**2
        df_list.append(ref)
    df_new = pd.concat(df_list, ignore_index=True)
    df_new.drop(columns=['times'], inplace=True)

    return df_new

def create_response_variable(df, ref_df, top_percentile):
    p = np.percentile(ref_df['slope'], top_percentile)
    df['category'] = None
    for i in range(len(df)):
        if df.at[i, 'slope'] >= 0:
            df.at[i, 'category'] = 'Other'
        elif df.at[i, 'slope'] < p:
            df.at[i, 'category'] = 'Fast'
        else:
            df.at[i, 'category'] = 'Slow'

    return df, p



def create_response_variable_test(df, ref_df, p):
    df['category'] = None
    for i in range(len(df)):
        if df.at[i, 'slope'] >= 0:
            df.at[i, 'category'] = 'Other'
        elif df.at[i, 'slope'] < p:
            df.at[i, 'category'] = 'Fast'
        else:
            df.at[i, 'category'] = 'Slow'

    return df




def make_train_test_sets(train_df, test_df, top_percentile):
    train_df2 = compute_slope(train_df)
    test_df2 = compute_slope(test_df)

    train_df2_highr2 = train_df2.loc[train_df2['r2']>0.3].reset_index(drop=True)
    train_df2_highr2_negative_slope = train_df2.loc[(train_df2['slope']<0) & (train_df2['r2']>0.3)].reset_index(drop=True)
    train_df3, p = create_response_variable(train_df2_highr2, train_df2_highr2_negative_slope, top_percentile)

    test_df2_highr2 = test_df2.loc[test_df2['r2']>0.3].reset_index(drop=True)
    test_df2_highr2_negative_slope = test_df2.loc[(test_df2['slope']<0) & (test_df2['r2']>0.3)].reset_index(drop=True)
    test_df3 = create_response_variable_test(test_df2_highr2, test_df2_highr2_negative_slope, p)

    return train_df3, test_df3





