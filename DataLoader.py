#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:11:52 2023

"""

# =============================================================================
# Step 0: define a function to read a graph from a single txt file
# =============================================================================

import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, remove_self_loops


##define a function to read file
def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    print(path)
    return read_txt_array(path, sep=',', dtype=dtype)


##define a function to combine items into sequences
def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None

##define a funtion to split data into batches
def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    print("----row----")
    print(row)
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    
    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    ##define a slices
    slices = {'edge_index': edge_slice}
        
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
        
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
            
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    ##for second-order index
    if data.edge_index2 is not None:
        row2, _ = data.edge_index2
        edge_slice2 = torch.cumsum(torch.from_numpy(np.bincount(batch[row2])), 0)
        edge_slice2 = torch.cat([torch.tensor([0]), edge_slice2])
        
        # Edge indices should start at zero for every graph.
        data.edge_index2 -= node_slice[batch[row2]].unsqueeze(0)
        
        ##define a slices
        slices['edge_index2'] = edge_slice2
        slices['edge_attr2'] = edge_slice2

        
    return data, slices

##IMPORTANT function 1: define a function to read data from text files
def read_tu_data(folder, prefix):
    
    # =============================================================================
    # read edge index from adj matrix
    # =============================================================================
    ##first order adj matrix
    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1 
    
    ##second order adj matrix
    edge_index2 = read_file(folder, prefix, 'A2', torch.long).t() - 1 

    # =============================================================================
    # read graph index
    # =============================================================================    
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1
    print(batch)
    print(batch.size(0))

    # =============================================================================
    # read node attributes
    # =============================================================================
    node_attributes = torch.empty((batch.size(0), 0))

    node_attributes = read_file(folder, prefix, 'node_attributes', torch.float32)
    
    if node_attributes.dim() == 1:
        node_attributes = node_attributes.unsqueeze(-1)

    # =============================================================================
    # read edge attributes
    # =============================================================================
    ##first-order edge attributes
    edge_attributes = torch.empty((edge_index.size(1), 0))
    edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if edge_attributes.dim() == 1:
        edge_attributes = edge_attributes.unsqueeze(-1)
        
    ##second-order edge attributes
    edge_attributes2 = torch.empty((edge_index2.size(1), 0))
    edge_attributes2 = read_file(folder, prefix, 'edge_attributes2')
    if edge_attributes2.dim() == 1:
        edge_attributes2 = edge_attributes2.unsqueeze(-1)

    # =============================================================================
    # concategate node attributes
    # =============================================================================
    x = cat([node_attributes])
    
    print("-------------x------------")
    print(x)
    # =============================================================================
    # concategate edge attributes and edge lables
    # =============================================================================
    ##first-order edge attributes
    edge_attr = cat([edge_attributes])

    ##second-order edge attributes
    edge_attr2 = cat([edge_attributes2])
  
    # =============================================================================
    # read graph attributes or graph labels
    # =============================================================================
    y = read_file(folder, prefix, 'graph_labels', torch.long)
    
    # =============================================================================
    # get total number of nodes for all graphs
    # =============================================================================
    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)

    # =============================================================================
    # remove self-loops: we should not remove selfloops
    # =============================================================================
    ##first-order    
    # print(edge_index)
    # print(edge_index.size(1))
    # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr) 
    # print(edge_index)
    # print(edge_index.size(1))

    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
    
    # print(edge_index)
    # print(edge_index.size(1))

        
    ##second-order   
    # edge_index2, edge_attr2 = remove_self_loops(edge_index2, edge_attr2) 
    edge_index2, edge_attr2 = coalesce(edge_index2, edge_attr2, num_nodes)

    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.edge_index2 = edge_index2
    data.edge_attr2 = edge_attr2
        
    # print("_______________important info 0_______________")
    # print(data.x)
    # print(data.edge_index)
    # print(data.edge_index2)
    # print(data.edge_attr)
    # print(data.edge_attr2)
    # print("____________________________________________")
    
    data, slices = split(data, batch)

    sizes = {
        'num_node_attributes': node_attributes.size(-1),
        'num_edge_attributes': edge_attributes.size(-1),
        'num_edge_attributes2': edge_attributes2.size(-1)
    }

    
    # print("_______________important info 1_______________")
    # print(data.x)
    # print(data.edge_index)
    # print(data.edge_index2)
    # print(data.edge_attr)
    # print(data.edge_attr2)
    # print("____________________________________________")
    
    return data, slices, sizes


# =============================================================================
# Step 1: define a class to read all text file based on read_tu_data() function
# =============================================================================
from typing import Callable, List, Optional
from torch_geometric.data import InMemoryDataset


class ParseDataset(InMemoryDataset):
    
    def __init__(self,
                 root: str,
                 name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 cleaned: bool = False):
        
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        load_data = torch.load(self.processed_paths[0])
              
        self.data, self.slices, self.sizes = load_data
        
        num_node_attributes = self.num_node_attributes
        self.data.x = self.data.x[:, :num_node_attributes]
        
        num_edge_attrs = self.num_edge_attributes
        self.data.edge_attr = self.data.edge_attr[:, :num_edge_attrs]
        
        num_edge_attrs2 = self.num_edge_attributes2
        self.data.edge_attr2 = self.data.edge_attr2[:, :num_edge_attrs2]
        
        # print("_______________important info 3_______________")
        # print(self.data.x)
        # print(self.data.edge_index)
        # print(self.data.edge_index2)
        # print(self.data.edge_attr)
        # print(self.data.edge_attr2)
        # print("____________________________________________")

    @property
    def raw_dir(self) -> str:
        name = f'Raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']
    
    @property
    def num_edge_attributes2(self) -> int:
        return self.sizes['num_edge_attributes2']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    ##renove ~/processed/ directory to run this 
    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
 
        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self.data, self.slices, sizes), self.processed_paths[0])
        # print("_______________important info 2_______________")
        # print(self.data.x)
        # print(self.data.edge_index)
        # print(self.data.edge_index2)
        # print(self.data.edge_attr)
        # print(self.data.edge_attr2)
        # print("____________________________________________")

        
    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
   

# =============================================================================
# Step 2: define a function to create data loader based on ParseDataset class
# =============================================================================

from torch_geometric.data import DataLoader, DenseDataLoader

DATA_PATH = 'Data'

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)


##IMPORTANT function: define a function to load dataset
def load_data(data_name, 
              dense=False, 
              seed=1213, 
              save_indices_to_disk=True):
    
    np.random.seed(seed)
    newcoin = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    ##get raw dataset if it already exists
    print(DATA_PATH + "/" + data_name + "/Raw/")
    if os.path.exists(DATA_PATH + "/" + data_name + "/Raw/"):
        
        print("++++++++find dataset++++++++++++")
        
        dataset_raw = ParseDataset(root=DATA_PATH, name=data_name)
    else:
        raise NotImplementedError

    dataset = dataset_raw
    
    dataset_list = [data for data in dataset]
    

    ##genrate training indices if not pre-defined
    train_indices = [i for i, data in enumerate(dataset_list) if data.y.item()==0 and newcoin.random()<0.7]
    
    test_normal_indices = [i for i, data in enumerate(dataset_list) if i not in train_indices and data.y.item()==0 ]
    
    test_abnormal_indices = [i for i, data in enumerate(dataset_list) if i not in train_indices and data.y.item()==1 ]
    
    # =============================================================================
    # only normal samples in training dataset   
    # =============================================================================
    ##generate training dataset: only keep normal class left
    # train_dataset = [dataset_list[idx] for idx in train_indices] 
    
    # ##generate testing dataset: normal class and abnormal class
    # test_dataset = [dataset_list[idx] for idx in range(len(dataset_list)) if idx not in train_indices]
    
    # =============================================================================
    #  normal samples + abnormal datasets in training dataset: todo to test robustness   
    # =============================================================================
    # train_indices = train_indices + test_abnormal_indices[:40]
    
    # test_abnormal_indices = test_abnormal_indices[40:]
    
    train_dataset = [dataset_list[idx] for idx in train_indices]
    
    test_dataset = [dataset_list[idx] for idx in range(len(dataset_list)) if idx not in train_indices]

    return train_dataset, test_dataset, dataset_raw


##define a function as dataloader
def create_loaders(data_name, 
                   batch_size=64, 
                   dense=False, 
                   data_seed=1213):

    ##generate training dataset and testing dataset using predefined function 
    train_dataset, test_dataset, dataset_raw= load_data(data_name, 
                                                        dense=dense, 
                                                        seed=data_seed)



    print("After downsampling and test-train splitting, distribution of classes:")
    labels = np.array([data.y.item() for data in train_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("TRAIN: Number of graphs: %d, Class distribution %s"%(len(train_dataset), label_dist))
    
    labels = np.array([data.y.item() for data in test_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("TEST: Number of graphs: %d, Class distribution %s"%(len(test_dataset), label_dist))

    Loader = DenseDataLoader if dense else DataLoader
    
    num_workers = 0
    
    ##----create a batch-based training dataset loader----##
    train_loader = Loader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    ##----create a batch-based testing dataset loader----##
    test_loader = Loader(test_dataset, batch_size=batch_size, shuffle=False,  pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader, train_dataset[0].num_features, train_dataset, test_dataset, dataset_raw


# =============================================================================
# Step 3: create a trainer class to train NN, including train() and test()
# =============================================================================
from sklearn.metrics import average_precision_score, roc_auc_score

class MeanTrainer:
    
    # =============================================================================
    # Step1. initialise the trainer with given hyperparameters
    # =============================================================================
    def __init__(self, model, optimizer, alpha=1.0, beta=0.0, device=torch.device("cpu"), regularizer="variance"):
        
        self.device = device

        self.model = model
        self.optimizer = optimizer

        ##--parameters for OCSVDD objectives----##
        self.center = None
        self.reg_weight = 0
        self.alpha = alpha
        self.beta = beta
        self.regularizer = regularizer   
    
    # =============================================================================
    # Step2. define the train funtion
    # =============================================================================
    def train(self, train_loader):
        print("\n++++++++++++++++trainers.py++++++++++++++++")
        print("----------train()----------")
        self.model.train()
        
        ##----first iteration, define s list to store vectors for computing SVDD center---##
        if self.center == None:
            F_list = []

        svdd_loss_accum = 0
        total_iters = 0

        for batch in train_loader:
            print("\n++++++++++++++++trainers.py++++++++++++++++")
            print("----------batch training start----------")
            
            ##----use GIN model to obatin node embeddings----##
            ##be careful: we use full batch training (for each graph) in DiGCN
            ##However, for the set of graphs, we can use batch training (for graph database) as here
            train_embeddings = self.model(batch)
            
            print("----------batch training end----------")
            
            ##----use mean Readout to obtain graph embeddings----##
            # sum_train_embeddings = [(torch.sum(emb, dim=0)) for emb in train_embeddings]
            # max_train_embeddings = [(torch.max(emb, dim=0))[0] for emb in train_embeddings]
            mean_train_embeddings = [torch.mean(emb, dim=0) for emb in train_embeddings] # Mean-ggregation: G_emb = mean(v_emb for v in G)
            F_train = torch.stack(mean_train_embeddings)
                
            ##----if first iteration, store vectors for computing SVDD center, and do not perform any backprop----##
            if self.center == None:
                F_list.append(F_train)
            
            ##----if not first iteration, perform backprop----##
            else:
                train_scores = torch.sum((F_train - self.center)**2, dim=1).cpu() 
                ##the second term in SVDD objective is controled by regularizer automatically
                
                svdd_loss = torch.mean(train_scores)
                
                #backpropagate
                self.optimizer.zero_grad()
                svdd_loss.backward()    
                self.optimizer.step()
                
                svdd_loss_accum += svdd_loss.detach().cpu().numpy()
                total_iters += 1

        ##----first epoch only, compute SVDD center----##
        if self.center == None: ##first epoch only
            full_F_list = torch.cat(F_list)
            self.center = torch.mean(full_F_list, dim=0).detach() #no backpropagation for center

            average_svdd_loss = -1
        ##----if not first epoch, compute averaged SVDD loss ----##
        else:
            average_svdd_loss = svdd_loss_accum/total_iters

        return average_svdd_loss


    # =============================================================================
    # Step3. define the test funtion
    # =============================================================================
    def test(self, test_loader):
        print("\n++++++++++++++++trainers.py++++++++++++++++")
        print("----------test()----------")
        self.model.eval()
        
        with torch.no_grad():

            dists_list = []
            for batch in test_loader:

                test_embeddings = self.model(batch)
                # mean_test_embeddings = [(torch.max(emb, dim=0))[0] for emb in test_embeddings]
                mean_test_embeddings = [torch.mean(emb, dim=0) for emb in test_embeddings] # Mean-aggregation: G_emb = mean(v_emb for v in G)
                F_test = torch.stack(mean_test_embeddings)
                
                batch_dists = torch.sum((F_test - self.center)**2, dim=1).cpu()
                dists_list.append(batch_dists)
            
            labels = torch.cat([batch.y for batch in test_loader])
            dists = torch.cat(dists_list)

            ap = average_precision_score(y_true= labels, y_score= dists, average = None, pos_label= 1, sample_weight= None)
            roc_auc = roc_auc_score(y_true= labels, y_score= dists, average = None,
                                    sample_weight= None, max_fpr = None, 
                                    multi_class = 'raise', labels =None)

            return ap, roc_auc, dists, labels


# =============================================================================
# Step 4: Define three GNN models
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool



#####build GIN model

class GIN(nn.Module):
    """
    Note: batch normalization can prevent divergence maybe, take care of this later. 
    """
    def __init__(self,  nfeat, nhid, nlayer, dropout=0, act=ReLU(), bias=False, **kwargs):
        super(GIN, self).__init__()
        self.norm = BatchNorm1d
        self.nlayer = nlayer
        self.act = act
        self.transform = Sequential(Linear(nfeat, nhid), self.norm(nhid))
        self.pooling = global_mean_pool
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.nns = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(nlayer):
            self.nns.append(Sequential(Linear(nhid, nhid, bias=bias), 
                                       act, Linear(nhid, nhid, bias=bias)))
            self.convs.append(GINConv(self.nns[-1]))
            self.bns.append(self.norm(nhid))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.transform(x) # weird as normalization is applying to all ndoes in database
        
        # can I also record the distance to center, which is the variance?
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            x = self.bns[i](x)

        emb_list = []
        for g in range(data.num_graphs):
            emb = x[data.batch==g]
            emb_list.append(emb)
        #graph_embeds = torch.stack(graph_embeds)

        return emb_list


##### build DiGCN model

from DIGCNConv import DIGCNConv


class DiGCN(nn.Module):
    
    def __init__(self, nfeat, nhid, nlayer, dropout=0, bias=False, **kwargs):
        ##two layers
        super(DiGCN, self).__init__()
        self.conv1 = DIGCNConv(nfeat, nhid)
        self.conv2 = DIGCNConv(nhid, nhid)
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)

        
        emb_list = []
        for g in range(data.num_graphs):
            emb = x[data.batch==g]
            emb_list.append(emb)
    
        return emb_list        

##### build  Inception Block for DiGCN model (Experiments show that InceptionBlock is not needed) 
class InceptionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InceptionBlock, self).__init__()
        self.ln = Linear(in_dim, out_dim)
        self.conv1 = DIGCNConv(in_dim, out_dim)
        self.conv2 = DIGCNConv(in_dim, out_dim)
        
    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    
    ##we need edge_index2 and  edge_attr2
    def forward(self, x, edge_index, edge_attr, edge_index2, edge_attr2):
        x0 = self.ln(x)
        x1 = self.conv1(x, edge_index, edge_attr)
        x2 = self.conv2(x, edge_index2, edge_attr2)
        
        return x0, x1, x2
    
    
class DiGCN_IB_Sum(nn.Module):
    def __init__(self, nfeat, nhid, nlayer, bias=False, **kwargs):
        super(DiGCN_IB_Sum, self).__init__()
        self.ib1 = InceptionBlock(nfeat, nhid)
        self.ib2 = InceptionBlock(nhid, nhid)
        self.ib3 = InceptionBlock(nhid, nhid)

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()
        self.ib3.reset_parameters()

    def forward(self, data, dropout_v = 0.1):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index2, edge_attr2 = data.edge_index2, data.edge_attr2
        
        print("_______________-------------______________-------------")
        print(x)
        # print(edge_index)
        # print(edge_attr)
        # print(edge_index2)
        # print(edge_attr2)
        
        
        x0,x1,x2 = self.ib1(x, edge_index, edge_attr, edge_index2, edge_attr2)
        x0 = F.dropout(x0, p=dropout_v, training=self.training)
        x1 = F.dropout(x1, p=dropout_v, training=self.training)
        x2 = F.dropout(x2, p=dropout_v, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=dropout_v, training=self.training)

        x0,x1,x2 = self.ib2(x, edge_index, edge_attr, edge_index2, edge_attr2)
        x0 = F.dropout(x0, p=dropout_v, training=self.training)
        x1 = F.dropout(x1, p=dropout_v, training=self.training)
        x2 = F.dropout(x2, p=dropout_v, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=dropout_v, training=self.training)

        x0,x1,x2 = self.ib3(x, edge_index, edge_attr, edge_index2, edge_attr2)
        x0 = F.dropout(x0, p=dropout_v, training=self.training)
        x1 = F.dropout(x1, p=dropout_v, training=self.training)
        x2 = F.dropout(x2, p=dropout_v, training=self.training)
        x = x0+x1+x2

        ## return embedding of graphs
        emb_list = []
        for g in range(data.num_graphs):
            emb = x[data.batch==g]
            emb_list.append(emb)
    
        return emb_list     
