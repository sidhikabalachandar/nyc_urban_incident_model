import pandas as pd
from scipy.special import logit
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_index', type=int)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--coeffs_file', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    # load files
    data_df = pd.read_hdf(args.data_file, 'df')
    coeffs_df = pd.read_csv(args.coeffs_file)
    
    # define covariates
    normalized_covariates = ['normalized_log_population_density',
                             'normalized_log_income_median',
                             'normalized_education_bachelors_pct',
                             'normalized_race_white_nh_pct',
                             'normalized_age_median',
                             'normalized_households_renteroccupied_pct']
    thetas = ['theta.{}_{}'.format(i, args.set_index) for i in range(1, len(normalized_covariates) + 2)]
    set_index = args.set_index
    
    # in this file we create synthetic intercepts such that the generated ratings are mean 0 
    # formula to create synthetic intercepts: 
    # theta_k[0] = logit(E_t[T_{ikt}]) - theta_k[1:] X_i[1:]
    
    # get synthetic coefficients theta_k[1:]
    data_df = pd.merge(data_df, coeffs_df[['type_idxs'] + thetas], on='type_idxs')
    
    # theta_k[0] = logit(E_t[T_{ikt}]) - theta_k[1:] X_i[1:]
    thetax = (data_df[normalized_covariates].to_numpy() * data_df[thetas[:-1]].to_numpy()).sum(axis=1)
    intercept = logit(data_df['bitflip_P(T)'].values) - thetax
    
    # get mean intercept for each type
    df = pd.DataFrame()
    df['type_idxs'] = data_df['type_idxs']
    df['theta.0_{}'.format(set_index)] = intercept
    mean_intercepts = df.groupby('type_idxs')['theta.0_{}'.format(set_index)].mean().reset_index()
    
    # add to mean type specific intercept to data_df
    data_df = pd.merge(data_df, mean_intercepts, on='type_idxs')
    
    # save mean intercept
    mean_intercept = mean_intercepts['theta.0_{}'.format(set_index)].mean()
    data_df['mean_theta.0_{}'.format(set_index)] = mean_intercept
    
    assert(len(data_df[data_df.isna().any(axis=1)]) == 0)
    
    # save data
    data_df.to_hdf('{}/full_{}.h5'.format(args.save_dir, set_index), key='df', mode='w')
    
if __name__ == "__main__":
    main()