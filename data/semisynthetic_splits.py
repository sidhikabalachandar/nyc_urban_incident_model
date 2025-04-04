import pandas as pd
import numpy as np
from scipy.special import logit
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_index', type=int)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--split_week', type=int, default=130) # default parameter for 75/25 train/test split for 3 years of data (from 2021-2023)
    args = parser.parse_args()
    return args

# to make semisynthetic data well specified, we do not want any node/type pairs where P(T) = 0 or P(T) = 1
# thus we randomly bitflip one datapoint for these sets
# for this, we define helper functions

# Group by 'node' and 'type', and apply a lambda to set one random 'T' to 1 per group
def set_random_one(group):
    # Choosing a random index from the group's indices
    random_index = np.random.choice(group.index)
    # Setting 'T' to 1 at the chosen index
    group.at[random_index, 'bitflip_reported'] = 1
    return group
    
# Group by 'node' and 'type', and apply a lambda to set one random 'T' to 0 per group
def set_random_zero(group):
    # Choosing a random index from the group's indices
    random_index = np.random.choice(group.index)
    # Setting 'T' to 1 at the chosen index
    group.at[random_index, 'bitflip_reported'] = 0
    return group

def generate_ratings(df, split, args):
    # define covariates
    normalized_covariates = ['normalized_log_population_density',
                             'normalized_log_income_median',
                             'normalized_education_bachelors_pct',
                             'normalized_race_white_nh_pct',
                             'normalized_age_median',
                             'normalized_households_renteroccupied_pct',
                             'normalized_rating']
    thetas = ['theta.{}_{}'.format(i, args.set_index) for i in range(1, len(normalized_covariates))]
    theta_0 = 'theta.0_{}'.format(args.set_index)
    theta_7 = 'theta.7_{}'.format(args.set_index) # theta_7 = alpha
    
    # get empirical E_t[T_{ikt}] or P(T) estimates from df
    # E_t[T_{ikt}] is the mean value of T across time (t) for each node, type pair (i,k)
    empirical_pt = df.groupby(['type_idxs', 'node_idxs'])['finegrained_reported'].mean()
    empirical_pt_df = empirical_pt.reset_index()
    empirical_pt_df = empirical_pt_df.rename(columns={'finegrained_reported': '{}_P(T)'.format(split)})
    # map to full dataset
    df = pd.merge(df, empirical_pt_df, on=['type_idxs', 'node_idxs'])
    
    # get number of data points for each node/type pair
    num_entries = df.groupby(['type_idxs', 'node_idxs']).size().reset_index()
    num_entries = num_entries.rename(columns={0: '{}_num_rows'.format(split)})
    df = pd.merge(df, num_entries, on=['type_idxs', 'node_idxs'])
    
    # bitflip node/type pairs with P(T) = 0
    pt0_df = df[df['{}_P(T)'.format(split)] == 0].copy()
    pt0_df['bitflip_reported'] = pt0_df['finegrained_reported']

    # Apply the function across the DataFrame grouped by 'node' and 'type'
    pt0_df = pt0_df.groupby(['node_idxs', 'type_idxs']).apply(set_random_one).reset_index(drop=True)
    pt0_df['bitflip_{}_P(T)'.format(split)] = 1 / pt0_df['{}_num_rows'.format(split)]
    
    # bitflip node/type pairs with P(T) = 1
    pt1_df = df[df['{}_P(T)'.format(split)] == 1].copy()
    pt1_df['bitflip_reported'] = pt1_df['finegrained_reported']

    # Apply the function across the DataFrame grouped by 'node' and 'type'
    pt1_df = pt1_df.groupby(['node_idxs', 'type_idxs']).apply(set_random_zero).reset_index(drop=True)
    pt1_df['bitflip_{}_P(T)'.format(split)] = (pt1_df['{}_num_rows'.format(split)] - 1) / pt1_df['{}_num_rows'.format(split)]
    
    # add in bitflipped information
    df = df.drop(columns=['bitflip_reported', 'bitflip_P(T)'])
    no_bitflip = df[(df['{}_P(T)'.format(split)] != 0) & (df['{}_P(T)'.format(split)] != 1)].copy()
    no_bitflip['bitflip_reported'] = no_bitflip['finegrained_reported']
    no_bitflip['bitflip_{}_P(T)'.format(split)] = no_bitflip['{}_P(T)'.format(split)]
    df = pd.concat([no_bitflip, pt0_df, pt1_df])
    
    assert(len(df[['type_idxs', 'node_idxs']].drop_duplicates()) == len(df[['type_idxs', 'node_idxs', 'bitflip_{}_P(T)'.format(split)]].drop_duplicates()))
    
    # get synthetic rating
    # r_{ikt} = (1 / alpha_k) * (logit(E_t[T_{ikt}]) - theta_k X_i)
    thetax = (df[normalized_covariates[:-1]].to_numpy() * df[thetas].to_numpy()).sum(axis=1)
    rating = (1 / df[theta_7]) * (logit(df['bitflip_{}_P(T)'.format(split)].values) - thetax - df[theta_0])
    df['semisynthetic_rating'] = rating
    
    # save data
    df['tonly_real_rating_observed'] = 0
    # remove non c-type columns for fast hdf upload and download
    df = df.drop(['GEOID', 'typeagency', 'finegrained_id', 'bitflip_P(T)'], axis=1)
    assert(len(df[df.isna().any(axis=1)]) == 0)
    df.to_hdf('/share/garg/311_data/sb2377/clean_codebase/semisynthetic/{}_{}.h5'.format(split, args.set_index), key='df', mode='w')
    return df

def main():
    args = get_args()
    
    # load files
    data_df = pd.read_hdf('{}/full_{}.h5'.format(args.data_dir, args.set_index), 'df')
    
    # split data into train and test sets using a time based split
    train_df = data_df[data_df['report_week'] < args.split_week]
    test_df = data_df[data_df['report_week'] >= args.split_week]
    
    # generate synthetic ratings for train and test sets
    # formula to create synthetic ratings
    # r_{ikt} = (1 / alpha_k) * (logit(E_t[T_{ikt}]) - theta_k X_i)
    
    # in order to generate the ratings we need
    # (i) synthetic values for theta_k
    # (ii) an estimate of E_t[T_{ikt}]
    
    # we have already generated (i) and stored these values in coeffs_df
    # here we generate (ii)
    
    # NOTE: train set ratings should only use E_t[T_{ikt}] estimates from the train set (and vice versa for the test set)
    # thus we cannot use the estimates of E_t[T_{ikt}] generated in semisynthetic_coeffs.ipynb 
    # these estimates are generated across both the train and test set
    
    # generate ratings
    train_df = generate_ratings(train_df, 'train', args)
    test_df = generate_ratings(test_df, 'test', args)
    
    # filter out node/type pairs with only one data point in test set
    filtered_test_df = test_df[(test_df['bitflip_test_P(T)'] != 0) & (test_df['bitflip_test_P(T)'] != 1)]
    assert(len(filtered_test_df[filtered_test_df.isna().any(axis=1)]) == 0)
    filtered_test_df.to_hdf('/share/garg/311_data/sb2377/clean_codebase/semisynthetic/filtered_test_{}.h5'.format(args.set_index), key='df', mode='w')
    
    
if __name__ == "__main__":
    main()