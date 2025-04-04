import torch
import argparse
import pytorch_lightning as pl
import pandas as pd
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from data import *
from torch.utils.data import DataLoader
from model import *
import pickle
import numpy as np
import torch.nn as nn

torch.set_float32_matmul_precision('medium')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0) # random seed
    parser.add_argument('--base_data_file', type=str) # path to base data (contains information on type and node idxs and mapping)
    parser.add_argument('--intermediate_embedding_dim', type=int, default=2292) # size of intermediate node embedding
    parser.add_argument('--final_embedding_dim', type=int, default=50) # size of final node embedding
    parser.add_argument('--graph_file', type=str) # path to graph file (contains information on nodes and edges in graph)
    parser.add_argument('--results_dir', type=str) # path to graph file (contains information on nodes and edges in graph)
    parser.add_argument('--job_id', type=str, default=0)
    parser.add_argument('--covariates', # list of covariates to use for reporting regression model
                        nargs='+',
                        default=['normalized_log_population_density',
                                 'normalized_log_income_median',
                                 'normalized_education_bachelors_pct',
                                 'normalized_race_white_nh_pct',
                                 'normalized_age_median',
                                 'normalized_households_renteroccupied_pct'])
    parser.add_argument('--type', type=str) # type of data used (e.g., semisynthetic or real)
    parser.add_argument('--complaint_type', choices=['street', 'park', 'rodent', 'restaurant', 'dcwp']) # if using subsampled data, which complaint type is subsampled
    parser.add_argument('--num_ratings_observed', type=str, default='all') # how many ratings of args.complaint_type type are observed
    parser.add_argument('--train_data_file', type=str) # path to train data
    parser.add_argument('--batch_size', type=int, default=16000) # batch size
    parser.add_argument('--num_workers', type=int, default=4) # number of dataset workers
    
    parser.add_argument('--seed', type=int, default=0) # random seed
    parser.add_argument('--base_data_file', type=str) # path to base data (contains information on type and node idxs and mapping)
    parser.add_argument('--train_data_file', type=str) # path to train data
    
    parser.add_argument('--base_data_file', type=str) # path to base data (contains type/node mapping information)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--intermediate_embedding_dim', type=int, default=2292)
    parser.add_argument('--final_embedding_dim', type=int, default=50)
    parser.add_argument('--covariates_type', type=str, default='all')
    parser.add_argument('--type', type=str)
    parser.add_argument('--batch_size', type=int, default=16000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save_frequency', type=int, default=1)
    parser.add_argument('--T_loss_has_rating_weight', type=float, default=1)
    parser.add_argument('--T_loss_not_has_rating_weight', type=float, default=1)
    parser.add_argument('--rating_loss_weight', type=float, default=1)
    parser.add_argument('--reg_loss_weight', type=float, default=0)
    
    parser.add_argument('--epoch', type=str)
    parser.add_argument('--file_type', type=str, default='')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    pl.seed_everything(args.seed)
    if torch.cuda.is_available():
        print('Using cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get type and node information
    base_data_df = pd.read_csv(args.base_data_file)
    
    nodes = base_data_df['GEOID'].unique()
    num_nodes = len(nodes)
    types = base_data_df['typeagency'].unique()
    num_types = len(types)
    print('num_nodes:', num_nodes)
    print('num_types:', num_types)

    # get gnn hyperparameters
    node_feature_dim = num_nodes
    intermediate_embedding_dim = args.intermediate_embedding_dim
    final_embedding_dim = args.final_embedding_dim
    
    # Make the networkx graph
    my_G = nx.read_graphml(args.graph_file) 
    G = nx.DiGraph()
    G.add_nodes_from(my_G.nodes())
    G.add_edges_from(my_G.edges())
    x = pickle.load(open('{}/job{}/x.pkl'.format(args.results_dir, args.job_id), 'rb'))
    data = from_networkx(G)
    data.x = x.to(device)
    data.edge_index = data.edge_index.to(device)
        
    # get demographic data
    num_demographic_covar = len(args.covariates)
    
    # get observed type indices
    type_df = base_data_df[['typeagency', 'type_idxs']].drop_duplicates()
    street_idx = type_df[type_df['typeagency'] == 'StreetConditionDOT']['type_idxs'].iloc[0]
    park_idx = type_df[type_df['typeagency'] == 'MaintenanceorFacilityDPR']['type_idxs'].iloc[0]
    rodent_idx = type_df[type_df['typeagency'] == 'RodentDOHMH']['type_idxs'].iloc[0]
    restaurant_idx = type_df[type_df['typeagency'] == 'FoodDOHMH']['type_idxs'].iloc[0]
    dcwp_idx = type_df[type_df['typeagency'] == 'ConsumerComplaintDCWP']['type_idxs'].iloc[0]
    observed_type_idxs = np.array([street_idx, park_idx, rodent_idx, restaurant_idx, dcwp_idx])
    
    # get column names for data from df
    if args.type == 'semisynthetic_test':
        t_label = 'bitflip_reported'
        rating_label = 'semisynthetic_rating'
        type_has_rating_mask_label = 'type_rating_observed'
        has_rating_mask_label = 'tonly_real_rating_observed'
    elif args.type == 'real_time_split_test':
        t_label = 'finegrained_reported'
        rating_label = 'normalized_rating'
        type_has_rating_mask_label = 'type_rating_observed'
        has_rating_mask_label = 'tonly_real_rating_observed'
    elif args.type == 'real_random_split_test':
        t_label = 'finegrained_reported'
        rating_label = 'normalized_rating'
        type_has_rating_mask_label = 'type_rating_observed'
        has_rating_mask_label = 'tonly_real_rating_observed'
    
    # get data
    train_dataset = RatingDataset(file=args.train_data_file,
                                  t_label=t_label,
                                  rating_label=rating_label,
                                  has_rating_mask_label=has_rating_mask_label,
                                  type_has_rating_mask_label=type_has_rating_mask_label,
                                  covariates=args.covariates)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    # get model
    model = GNN_model(num_types, 
                       num_nodes, 
                       num_demographic_covar,
                       node_feature_dim,
                       intermediate_embedding_dim,
                       final_embedding_dim,
                       data, 
                       device, 
                       args.lr, 
                       args.save_frequency,
                       args.T_loss_has_rating_weight, 
                       args.T_loss_not_has_rating_weight, 
                       args.rating_loss_weight,
                       args.reg_loss_weight,
                       observed_type_idxs
                      ).to(device)

    checkpoint_path = '/share/garg/311_data/sb2377/results/job{}/model-epoch={}.ckpt'.format(args.job_id, args.epoch)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        for batch in train_loader:
            logit_pt, pred_rating, node_embedding, type_embedding = model(batch['node_idxs'].to(device), 
                                                                          batch['type_idxs'].to(device), 
                                                                          batch['demographics'].to(device),
                                                                         batch['rating'].to(device),
                                                                         batch['has_rating_mask'].to(device),
                                                                          batch['type_has_rating_mask'].to(device),
                                                                         )
            true_rating = batch['rating']
            mask = batch['has_rating_mask']
            node_idxs = batch['node_idxs']
            type_idxs = batch['type_idxs']
            demographics = batch['demographics']
            pred_pt = sigmoid(logit_pt)
            true_t = batch['T']
            break
            
    pickle.dump((pred_rating.cpu(), 
                 true_rating.cpu(), 
                 mask.cpu(),
                 node_embedding.cpu(),
                 type_embedding.cpu(),
                 node_idxs.cpu(),
                 type_idxs.cpu(),
                 demographics.cpu(),
                 pred_pt.cpu(),
                 true_t.cpu()
                ), open('/share/garg/311_data/sb2377/results/job{}/epoch={}_test{}.pkl'.format(args.job_id, args.epoch, args.file_type), "wb"))
    
if __name__ == "__main__":
    
    main()
              