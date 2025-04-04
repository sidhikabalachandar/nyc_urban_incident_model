from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.utils.data import Sampler
import random

# dataset
class RatingDataset(Dataset):
    def __init__(self, 
                 file,  
                 t_label, 
                 rating_label,
                 has_rating_mask_label, 
                 type_has_rating_mask_label,
                 covariates
                ):
        df = pd.read_hdf(file, 'df')
        self.node_idxs = torch.tensor(df['node_idxs'].values, dtype=torch.long)
        self.type_idxs = torch.tensor(df['type_idxs'].values, dtype=torch.long)
        self.T = torch.tensor(df[t_label].values, dtype=torch.float)
        self.rating = torch.tensor(df[rating_label].values, dtype=torch.float)
        self.has_rating_mask = torch.tensor(df[has_rating_mask_label].values, dtype=torch.bool)
        self.type_has_rating_mask = torch.tensor(df[type_has_rating_mask_label].values, dtype=torch.bool)
        self.demographics = torch.tensor(df[covariates].values, dtype=torch.float) 
        
    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return {
            'node_idxs': self.node_idxs[idx],
            'type_idxs': self.type_idxs[idx],
            'T': self.T[idx],
            'rating': self.rating[idx],
            'has_rating_mask': self.has_rating_mask[idx],
            'type_has_rating_mask': self.type_has_rating_mask[idx],
            'demographics': self.demographics[idx]
        }

# sampler that separates data with observed ratings and unobserved ratings in separate batches 
class RatingSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Separate indices into two groups: those that have observed ratings and those that don't
        grad_tensor = dataset.has_rating_mask
        self.indices_by_grad = {
            False: torch.nonzero(grad_tensor == False, as_tuple=False).squeeze().tolist(),
            True: torch.nonzero(grad_tensor == True, as_tuple=False).squeeze().tolist()
        }

    def __iter__(self):
        # Shuffle the indices within each group using Python's random.shuffle
        for grad_value in [False, True]:
            indices = self.indices_by_grad[grad_value]
            random.shuffle(indices)

        # Combine the batches from both groups and interleave them
        all_batches = []
        for grad_value in [False, True]:
            indices = self.indices_by_grad[grad_value]
            batch = []
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    all_batches.append(batch)
                    batch = []
            if batch:  # Add any remaining data points that don't fit in a full batch
                all_batches.append(batch)

        # Shuffle all batches from both groups
        random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch

    def __len__(self):
        # Calculate the total number of batches
        total_batches = 0
        for grad_value in [False, True]:
            total_batches += (len(self.indices_by_grad[grad_value]) + self.batch_size - 1) // self.batch_size
        return total_batches