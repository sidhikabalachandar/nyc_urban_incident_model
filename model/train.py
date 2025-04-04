import torch
import argparse
import os
import pytorch_lightning as pl
import pandas as pd
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from data import *
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from model import *
import pickle
from pytorch_lightning.callbacks import ModelCheckpoint
import glob 

torch.set_float32_matmul_precision('medium')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0) # random seed
    parser.add_argument('--base_data_file', type=str) # path to base data (contains information on type and node idxs and mapping)
    parser.add_argument('--intermediate_embedding_dim', type=int, default=2292) # size of intermediate node embedding
    parser.add_argument('--final_embedding_dim', type=int, default=50) # size of final node embedding
    parser.add_argument('--graph_file', type=str) # path to graph file (contains information on nodes and edges in graph)
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
    parser.add_argument('--results_dir', type=str) # path to directory where results will be logged
    parser.add_argument('--lr', type=float, default=0.01) # learning rate
    parser.add_argument('--save_frequency', type=int, default=1) # results will be saved every args.save_frequency epochs
    parser.add_argument('--T_loss_has_rating_weight', type=float, default=1) # weight on report loss for data points with observed ratings
    parser.add_argument('--T_loss_not_has_rating_weight', type=float, default=1) # weight on report loss for data points with unobserved ratings
    parser.add_argument('--rating_loss_weight', type=float, default=1) # weight on rating loss
    parser.add_argument('--reg_loss_weight', type=float, default=0) # weight on regularization loss
    parser.add_argument('--job_id', type=str, default=0) # job id
    parser.add_argument('--num_epochs', type=int)
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
    x = torch.eye(num_nodes) 
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
    if args.type == 'semisynthetic':
        t_label = 'bitflip_reported'
        rating_label = 'semisynthetic_rating'
        type_has_rating_mask_label = 'type_rating_observed'
        if args.num_ratings_observed == 'all':
            has_rating_mask_label = 'real_rating_observed'
        else:
            has_rating_mask_label = 'few_{}_rating_observed_{}'.format(args.complaint_type, args.num_ratings_observed)
    elif args.type == 'semisynthetic_tonly':
        t_label = 'bitflip_reported'
        rating_label = 'semisynthetic_rating'
        has_rating_mask_label = 'tonly_real_rating_observed'
        type_has_rating_mask_label = 'tonly_real_rating_observed'
    elif args.type == 'real_time_split':
        t_label = 'finegrained_reported'
        rating_label = 'normalized_rating'
        type_has_rating_mask_label = 'type_rating_observed'
        if args.num_ratings_observed == 'all':
            has_rating_mask_label = 'time_split_real_rating_observed'
        else:
            has_rating_mask_label = 'few_time_split_{}_rating_observed_{}'.format(args.complaint_type, args.num_ratings_observed)
    elif args.type == 'real_time_split_tonly':
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
    train_sampler = RatingSampler(train_dataset, args.batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers)
    
    # set up wandb
    wandb_logger = WandbLogger(project='gnns_underreporting', save_dir=args.results_dir) 
    wandb_logger.experiment.config["base_data_file"] = args.base_data_file
    wandb_logger.experiment.config["intermediate_embedding_dim"] = args.intermediate_embedding_dim
    wandb_logger.experiment.config["final_embedding_dim"] = args.final_embedding_dim
    wandb_logger.experiment.config["type"] = args.type
    wandb_logger.experiment.config["complaint_type"] = args.complaint_type
    wandb_logger.experiment.config["num_ratings_observed"] = args.num_ratings_observed
    wandb_logger.experiment.config["train_data_file"] = args.train_data_file
    wandb_logger.experiment.config["batch_size"] = args.batch_size
    wandb_logger.experiment.config["num_workers"] = args.num_workers
    wandb_logger.experiment.config["lr"] = args.lr
    wandb_logger.experiment.config["save_frequency"] = args.save_frequency
    wandb_logger.experiment.config["T_loss_has_rating_weight"] = args.T_loss_has_rating_weight
    wandb_logger.experiment.config["T_loss_not_has_rating_weight"] = args.T_loss_not_has_rating_weight
    wandb_logger.experiment.config["rating_loss_weight"] = args.rating_loss_weight
    wandb_logger.experiment.config["reg_loss_weight"] = args.reg_loss_weight
    wandb_logger.experiment.config["job_id"] = args.job_id
    
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
    wandb_logger.watch(model, log_freq=1)

    # logging
    log_dir = '{}/job{}'.format(args.results_dir, args.job_id)

    # save node features
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    pickle.dump(x, open('{}/x.pkl'.format(log_dir), 'wb'))

    # set up model saving
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,  # Directory where checkpoints will be saved
        filename='model-{epoch:02d}',  # Naming pattern for saved checkpoints
        every_n_epochs=args.save_frequency,  # Save every args.save_frequency epochs
        save_top_k=-1 # Save all epochs
    )
    
    # check if job has already been run and if it should be resumed
    checkpoint_files = glob.glob(os.path.join(log_dir, "model-epoch=*.ckpt"))

    epochs = []
    for file in checkpoint_files:
        filename = os.path.basename(file)
        try:
            epoch = filename.split("model-epoch=")[1].split(".ckpt")[0]  # Extract epoch number
            epochs.append(int(epoch))
        except ValueError:
            continue  # Skip invalid filenames

    start_epoch = max(epochs) if epochs else -1

    # train
    trainer = pl.Trainer(max_epochs=args.num_epochs, 
                         logger=wandb_logger, 
                         callbacks=[checkpoint_callback]
                        )
    if start_epoch == -1:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, 
                    train_loader,
                    ckpt_path="{}/job{}/model-epoch={}.ckpt".format(args.results_dir, args.job_id, start_epoch)
                   )
    
if __name__ == "__main__":
    
    main()
              