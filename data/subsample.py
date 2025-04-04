import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--base_data_file', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--type_label', type=str)
    parser.add_argument('--type_label_idx', type=int)
    parser.add_argument('--data_idx', type=int)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    # load files
    base_df = pd.read_csv(args.base_data_file)
    train_df = pd.read_hdf('{}/train_{}.h5'.format(args.data_dir, args.data_idx), 'df')
    train_df = train_df.reset_index(drop=True)
    
    # number of subsampled ratings
    subsampled_amounts = [100, 500, 1000, 10000, 100000]
    subsampled_amount_labels = ['100', '500', '1k', '10k', '100k']
    
    # get type indices
    type_df = base_df[['typeagency', 'type_idxs']].drop_duplicates()
    street_idx = type_df[type_df['typeagency'] == 'StreetConditionDOT']['type_idxs'].iloc[0]
    park_idx = type_df[type_df['typeagency'] == 'MaintenanceorFacilityDPR']['type_idxs'].iloc[0]
    rodent_idx = type_df[type_df['typeagency'] == 'RodentDOHMH']['type_idxs'].iloc[0]
    restaurant_idx = type_df[type_df['typeagency'] == 'FoodDOHMH']['type_idxs'].iloc[0]
    dcwp_idx = type_df[type_df['typeagency'] == 'ConsumerComplaintDCWP']['type_idxs'].iloc[0]
    indices = [street_idx, park_idx, rodent_idx, restaurant_idx, dcwp_idx]
    
    # get datapoints of type args.type_label with observed ratings
    mask_label = 'real_rating_observed'
    type_label = args.type_label
    idx = indices[args.type_label_idx]
    observed_df = train_df[(train_df['type_idxs'] == idx) & (train_df[mask_label] == 1)]
    total_num_rated = len(observed_df[mask_label])
    
    # subsample observed ratings for each subsampled amount in subsampled_amounts
    for i in range(len(subsampled_amounts)):
        subsampled_amount = subsampled_amounts[i]
        amount_label = subsampled_amount_labels[i]
        subsampled_mask_label = 'subsampled_{}_rating_observed_{}'.format(type_label, amount_label)
        train_df[subsampled_mask_label] = train_df[mask_label]
        
        # sample total - subsampled amount ratings, these will be masked out
        sampled_indices = observed_df.sample(n=total_num_rated - subsampled_amount, replace=False).index
        train_df.loc[sampled_indices, subsampled_mask_label] = 0
    
    assert(len(train_df[train_df.isna().any(axis=1)]) == 0)
    train_df.to_hdf('{}/train_{}_{}_subsampled.h5'.format(args.save_dir, args.data_idx, args.label), key='df', mode='w')
    
if __name__ == "__main__":
    
    main()