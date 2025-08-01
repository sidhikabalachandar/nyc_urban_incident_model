import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, BatchNorm
from torch.nn import Linear
import torch.nn as nn
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, mean_squared_error

class GNN_model(pl.LightningModule):
    def __init__(self, 
                 num_types, 
                 num_nodes, 
                 num_demographic_covar, 
                 node_feature_dim,
                 intermediate_embedding_dim,
                 final_embedding_dim,
                 data, 
                 device, 
                 lr, 
                 save_frequency,
                 T_loss_has_rating_weight,
                 T_loss_not_has_rating_weight, 
                 rating_loss_weight,
                 reg_loss_weight,
                 reg_loss_reporting_weight,
                 rating_coeff_loss_weight,
                 scaling_factor,
                 observed_type_idxs,
                 complaint_type_idx
                ):
        super().__init__()
        
        self.automatic_optimization = False
        
        # layers
        self.conv1 = GCNConv(node_feature_dim, intermediate_embedding_dim)
        self.batch_norm1 = BatchNorm(intermediate_embedding_dim)
        self.conv2 = GCNConv(intermediate_embedding_dim, final_embedding_dim)
        self.batch_norm2 = BatchNorm(final_embedding_dim)
        self.type_layer = Linear(num_types, final_embedding_dim)
        self.rating_layer = Linear(final_embedding_dim * final_embedding_dim, 1)
        self.pt_layer = nn.Parameter(torch.zeros(num_types, num_demographic_covar + 1))
        self.pt_layer_bias = nn.Parameter(torch.zeros(num_types, 1))
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        
        # losses
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        # weights
        self.T_loss_has_rating_weight = T_loss_has_rating_weight
        self.T_loss_not_has_rating_weight = T_loss_not_has_rating_weight
        self.rating_loss_weight = rating_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.reg_loss_reporting_weight = reg_loss_reporting_weight
        self.rating_coeff_loss_weight = rating_coeff_loss_weight
        self.scaling_factor = scaling_factor
        if self.rating_loss_weight == 0:
            self.T_only = True
        else:
            self.T_only = False
        
        # dimensions
        self.node_feature_dim = node_feature_dim
        self.intermediate_embedding_dim = intermediate_embedding_dim
        self.final_embedding_dim = final_embedding_dim
        
        # data
        self.num_types = num_types
        self.num_nodes = num_nodes
        self.data = data.to(device)
        self.one_hot_type = torch.eye(num_types).to(device)
        self.observed_type_idxs = observed_type_idxs
        self.complaint_type_idx = complaint_type_idx
        
        # training info
        self.d = device
        self.lr = lr
        self.save_frequency = save_frequency
        
        # information to save
        self.training_step_outputs = []
        self.val_step_outputs = []
        self.train_type_idxs = []
        self.val_type_idxs = []
        self.train_node_idxs = []
        self.val_node_idxs = []

    def forward(self, 
                node_idxs, 
                type_idxs, 
                demographic_covars, 
                true_rating, 
                has_rating_mask,
                type_has_rating_mask
               ):
        batch_size = demographic_covars.size()[0]
        
        # get node embedding
        # gcn layer 1
        node_features = self.data.x
        edge_index = self.data.edge_index
        assert(node_features.size() == (self.num_nodes, self.node_feature_dim))
        node_embedding_0 = self.conv1(node_features, edge_index)
        node_embedding_0 = self.leaky_relu(node_embedding_0)
        node_embedding_0 = self.batch_norm1(node_embedding_0)
        assert(node_embedding_0.size() == (self.num_nodes, self.intermediate_embedding_dim))
        assert not torch.isnan(node_embedding_0).any(), 'Output of GCN layer 1 contains NaN values'
        
        # gcn layer 2
        node_embedding = self.conv2(node_embedding_0, edge_index)
        node_embedding = self.batch_norm2(node_embedding)
        assert(node_embedding.size() == (self.num_nodes, self.final_embedding_dim))
        assert not torch.isnan(node_embedding).any(), 'Node embedding contains NaN values'
        
        # get type embedding
        type_features = self.one_hot_type
        type_embedding = self.type_layer(type_features)
        assert(type_embedding.size() == (self.num_types, self.final_embedding_dim))
        assert not torch.isnan(type_embedding).any(), 'Type embedding contains NaN values'
        
        # get rating (apply linear layer to node and type embeddings)
        # rating = dot_prod(node_embedding[node_idx], type_embedding[type_idx])
        pred_rating = torch.einsum('ij,ij->i', node_embedding[node_idxs], type_embedding[type_idxs])
        assert(pred_rating.size() == (batch_size,))
        assert not torch.isnan(pred_rating).any(), 'Rating contains NaN values'
        
        # get P(T) ~ demographics + rating
        # for types with true rating, use true rating
        # else, use predicted rating
        not_has_rating_mask = torch.logical_not(has_rating_mask)
        masked_rating = pred_rating * not_has_rating_mask + true_rating * has_rating_mask
        
        if not self.T_only:
            # Determine the weights based on observed/unobserved types
            # each observed types learns type specific weights
            # the unobserved types use the average learned weights across the observed types
            type_not_has_rating_mask = torch.logical_not(type_has_rating_mask)
            type_specific_coeffs = self.pt_layer[type_idxs]
            mean_coeffs = self.pt_layer[self.observed_type_idxs].mean(dim=0)
            weights = type_specific_coeffs * type_has_rating_mask.unsqueeze(-1).float() + mean_coeffs * type_not_has_rating_mask.unsqueeze(-1).float()

            type_specific_bias = self.pt_layer_bias[type_idxs]
            mean_bias = self.pt_layer_bias[self.observed_type_idxs].mean(dim=0)
            bias = type_specific_bias * type_has_rating_mask.unsqueeze(-1).float() + mean_bias * type_not_has_rating_mask.unsqueeze(-1).float()
        else:
            weights = self.pt_layer[type_idxs]
            bias = self.pt_layer_bias[type_idxs]
        
        # calculate logit(P(T))
        covars = torch.cat([demographic_covars, masked_rating.unsqueeze(dim=1)], dim=1)
        logit_pt =  torch.einsum('ij,ij->i', covars, weights)
        logit_pt = logit_pt + bias.squeeze(dim=1)        
        assert(logit_pt.size() == (batch_size,))
        assert not torch.isnan(logit_pt).any(), 'logit(P(T)) contains NaN values'
        
        return logit_pt, pred_rating, node_embedding, type_embedding
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        
        # Scale down gradients instead of clearing them completely
        if not self.T_only and batch['has_rating_mask'].sum() == 0:
            # Scale the gradients instead of setting to None
            if self.pt_layer.grad is not None:
                self.pt_layer.grad *= self.scaling_factor
            
            if self.pt_layer_bias.grad is not None:
                self.pt_layer_bias.grad *= self.scaling_factor
        
        opt.step()
    
    def compute_loss(self, batch):
        t_labels = batch['T']
        true_rating = batch['rating']
        has_rating_mask = batch['has_rating_mask']
        type_has_rating_mask = batch['type_has_rating_mask']
        not_has_rating_mask = torch.logical_not(has_rating_mask)
        logit_pt, pred_rating, _, _ = self(batch['node_idxs'].to(self.d), 
                                           batch['type_idxs'].to(self.d), 
                                           batch['demographics'].to(self.d), 
                                           true_rating.to(self.d), 
                                           has_rating_mask.to(self.d),
                                           type_has_rating_mask.to(self.d)
                                          )
        
        # calculate P(T)
        pt = self.sigmoid(logit_pt)
        
        # loss (a): BCE loss between predicted P(T) and true T for data points with unobserved rating
        if not_has_rating_mask.sum() == 0:
            loss_t_not_has_rating = torch.zeros((1,)).to(self.d)
        else:
            loss_t_not_has_rating = self.bce_loss(logit_pt[not_has_rating_mask], t_labels[not_has_rating_mask])
        
        # loss (b): BCE loss between predicted P(T) and true T for data points with observed rating
        if has_rating_mask.sum() == 0:
            loss_t_has_rating = torch.zeros((1,)).to(self.d)
        else:
            loss_t_has_rating = self.bce_loss(logit_pt[has_rating_mask], t_labels[has_rating_mask])
            
        # loss (c): MSE loss between predicted rating and true rating for data points with observed rating
        if has_rating_mask.sum() == 0:
            observed_loss_rating = torch.zeros((1,)).to(self.d)
        else:
            observed_loss_rating = self.mse_loss(pred_rating[has_rating_mask], true_rating[has_rating_mask])
            
        # loss (d): L2 regularization loss for data points with unobserved ratings
        if not_has_rating_mask.sum() == 0:
            reg_loss = torch.zeros((1,)).to(self.d)
        else:
            reg_loss = torch.norm(pred_rating[not_has_rating_mask], p=2) ** 2

        # loss (e): L2 regularization loss for reporting coefficients other than the intercept and the rating coefficient
        reg_loss_reporting = torch.norm(self.pt_layer[self.complaint_type_idx, :-1], p=2) ** 2

        # loss (f): relu penalty on rating coefficient ()
        rating_coeff = self.pt_layer[self.complaint_type_idx, -1]
        rating_coeff_loss = torch.nn.functional.relu(rating_coeff)
        
        # combined weighted loss
        combined_loss = (self.T_loss_not_has_rating_weight * loss_t_not_has_rating + 
                         self.T_loss_has_rating_weight * loss_t_has_rating + 
                         self.rating_loss_weight * observed_loss_rating + 
                         self.reg_loss_weight * reg_loss + 
                         self.reg_loss_reporting_weight * reg_loss_reporting +
                         self.rating_coeff_loss_weight * rating_coeff_loss)
        
        # log results
        self.training_step_outputs.append({'loss': combined_loss.item(),
                                           't_loss_not_has_rating': loss_t_not_has_rating.item(),
                                           't_loss_has_rating': loss_t_has_rating.item(),
                                           'observed_rating_loss': observed_loss_rating.item(),
                                           'reg_loss': reg_loss.item(),
                                           'reg_loss_reporting': reg_loss_reporting.item(),
                                           'rating_coeff_loss': rating_coeff_loss.item(),
                                           'P(T)': pt.detach(),
                                           'true_t': t_labels.detach(),
                                           'pred_rating': pred_rating.detach(),
                                           'true_rating': true_rating.detach(),
                                           'has_rating_mask': has_rating_mask.detach(),
                                           'type_has_rating_mask': type_has_rating_mask.detach()
                                          })
        self.train_type_idxs.append(batch['type_idxs'].detach())
        self.train_node_idxs.append(batch['node_idxs'].detach())
        return combined_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def on_train_epoch_end(self):
        # Calculate and log training metrics
        # log losses
        # combined loss
        avg_train_loss = torch.tensor([x['loss'] for x in self.training_step_outputs]).mean()
        
        # loss (a)
        avg_train_loss_t_not_has_rating = torch.tensor([x['t_loss_not_has_rating'] for x in self.training_step_outputs]).mean()
        
        # loss (b)
        avg_train_loss_t_has_rating = torch.tensor([x['t_loss_has_rating'] for x in self.training_step_outputs]).mean()
        
        # loss (c)
        avg_train_observed_loss_rating = torch.tensor([x['observed_rating_loss'] for x in self.training_step_outputs]).mean()
        
        # loss (d)
        avg_train_reg_loss = torch.tensor([x['reg_loss'] for x in self.training_step_outputs]).mean()
        
        # loss (e)
        avg_train_reg_loss_reporting = torch.tensor([x['reg_loss_reporting'] for x in self.training_step_outputs]).mean()
        
        # loss (f)
        avg_train_rating_coeff_loss = torch.tensor([x['rating_coeff_loss'] for x in self.training_step_outputs]).mean()
        
        # get variables
        pred_rating = torch.cat([x['pred_rating'] for x in self.training_step_outputs]).cpu().detach().numpy()
        true_rating = torch.cat([x['true_rating'] for x in self.training_step_outputs]).cpu().detach().numpy()
        
        # calculate correlation
        full_corr, _ = pearsonr(pred_rating, true_rating)
        
        # calculate mse
        full_mse = mean_squared_error(pred_rating, true_rating)
        
        # calculate auroc and auprc
        pred_t = torch.cat([x['P(T)'] for x in self.training_step_outputs]).cpu().detach().numpy()
        true_t = torch.cat([x['true_t'] for x in self.training_step_outputs]).cpu().detach().numpy()
        auroc_t = roc_auc_score(true_t, pred_t)
        
        self.log('train_auroc_T_full', auroc_t) 
        
        self.log('train_mse_r_full', full_mse)
        
        self.log('train_corr_r_full', full_corr)
        
        self.log('train_loss', avg_train_loss)
        self.log('train_loss_rating', avg_train_observed_loss_rating)
        self.log('train_loss_T_has_rating', avg_train_loss_t_has_rating)
        self.log('train_loss_T_not_has_rating', avg_train_loss_t_not_has_rating)
        self.log('train_reg_loss', avg_train_reg_loss)
        self.log('train_reg_loss_reporting', avg_train_reg_loss_reporting)
        self.log('train_rating_coeff_loss', avg_train_rating_coeff_loss)
        self.training_step_outputs = []
        self.train_type_idxs = []
        self.train_node_idxs = []