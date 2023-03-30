"""
#Repurposed from the GLAM paper https://github.com/sawlani/GLAM
#Date: 01 Jan 2023

"""
# the absolute path of the Logs2Graph project
root_path = r'/home/SteveJobs/Logs2Graph'


import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import pickle
import argparse
from types import SimpleNamespace

from matplotlib import rcParams
rcParams.update({'figure.autolayout': False})

from DataLoader import create_loaders, MeanTrainer, GIN, DiGCN, DiGCN_IB_Sum



##--------------------------------------------
##Step 1. first clear all files under the /processed/~ directory
##--------------------------------------------

import os, shutil
folder = root_path + '/Data/Thunderbird/processed'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
        

folder = root_path + '/Data/Thunderbird/Raw'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
        
##--------------------------------------------
##Step 2. copy all files from a directory to another
##--------------------------------------------       

import shutil
import os
 
# path to source directory
src_dir = root_path + '/Data/Thunderbird/Graph/Raw/'
 
# path to destination directory
dest_dir = root_path + '/Data/Thunderbird/Raw/'
 
# getting all the files in the source directory
my_files = os.listdir(src_dir)
 
for file_name in my_files:
    print(file_name)
    print(type(dest_dir))
    src_file_name = src_dir + file_name
    dest_file_name = dest_dir + file_name
    shutil.copy(src_file_name, dest_file_name)
     
    
        
##--------------------------------------------
##Step 3. define a function to run experiments
##--------------------------------------------          
        
def run_experiment(
    data = "HDFS", #data_name to use
    data_seed=1213, 
    alpha=1.0, 
    beta=0.0,
    epochs=150, 
    model_seed=0, 
    num_layers=1, 
    device=0,
    aggregation="Mean", #We can choose it from {"Mean", "Max", "Sum"}
    bias=False,
    hidden_dim=64,
    lr=0.1,
    weight_decay=1e-5,
    batch = 64
    ):

    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    
    # =============================================================================
    # Step1. load data using predefined script dataloader.py
    # we should define this function by ourself
    # =============================================================================
    
    train_loader, test_loader, num_features, train_dataset, test_dataset, raw_dataset = create_loaders(data_name=data, 
                                                                                                       batch_size=batch,  
                                                                                                       dense=False,
                                                                                                       data_seed=data_seed)

    # print("-------main.py-----")
    # print(train_dataset[0].edge_attr)
    
    ##----set seeds for cuda----
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
        
    # =============================================================================
    # Step2. train a GIN model with given parameters
    # =============================================================================
    
    ##----setting paramters----
    # model = GIN(nfeat = num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias) ##this one can only handle undirected graphs
    model = DiGCN(nfeat = num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias)
    # model = DiGCN_IB_Sum(nfeat = num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias)

    ##----important paramter 0----##
    ##the learning rate, weight decay hyperparameter are given here
    ##In GLAM they use SGD, however, we will use Adam in our paper
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay) 
    
    if aggregation=="Mean":
        trainer = MeanTrainer(
            model=model,
            optimizer=optimizer,
            alpha=alpha,
            beta=beta,
            device=device
            )
    
    epochinfo = []

    ##----starting training----
    for epoch in range(epochs+1):

        print("\n+++++++++++++++++++main.py++++++++++++++++++++++")
        print("Epoch %3d" % (epoch), end="\t")
        
        print("\n+++++++++++++++++++main.py++++++++++++++++++++++")
        print("\n---------epoch train start-------------")
        svdd_loss = trainer.train(train_loader=train_loader)
        print("SVDD loss: %f" % (svdd_loss), end="\t")
        print("\n---------epoch train end-------------")
        
        print("\n+++++++++++++++++++main.py++++++++++++++++++++++")
        print("\n---------epoch test start-------------")
        ap, roc_auc, dists, labels = trainer.test(test_loader=test_loader)
        #print("AP: %f" % ap, end="\t")
        print("ROC-AUC: %f" % roc_auc)
        print("\n---------epoch test end-------------")

        
        ##----set a temporary object to store important information----
        TEMP = SimpleNamespace()
        TEMP.epoch_no = epoch
        TEMP.dists = dists
        TEMP.labels = labels
        TEMP.ap = ap
        TEMP.roc_auc = roc_auc
        TEMP.svdd_loss = svdd_loss

        epochinfo.append(TEMP)  



    best_svdd_idx = np.argmin([e.svdd_loss for e in epochinfo[1:]])+1
    
    print("      Min SVDD, at epoch %d, AP: %.3f, ROC-AUC: %.3f" % (best_svdd_idx, epochinfo[best_svdd_idx].ap, epochinfo[best_svdd_idx].roc_auc))
    print("    At the end, at epoch %d, AP: %.3f, ROC-AUC: %.3f" % (args.epochs, epochinfo[-1].ap, epochinfo[-1].roc_auc))

    ##----record the best epoch's information----

    important_epoch_info = {}
    important_epoch_info['svdd'] = epochinfo[best_svdd_idx]
    important_epoch_info['last'] = epochinfo[-1]
    
    return important_epoch_info, train_dataset, test_dataset, raw_dataset


# =============================================================================
# Step 4: define a parser
# The argparse module makes it easy to write user-friendly command-line interfaces. 
# The program defines what arguments it requires, and argparse will figure out 
# how to parse those out of sys.argv
# =============================================================================

parser = argparse.ArgumentParser(description='OCDiGCN:')


##----important paramter 1----##
parser.add_argument('--data', default='Thunderbird',
                    help='dataset name (default: HDFS)') 

parser.add_argument('--batch', type=int, default=2000,
                    help='batch size (default: 64)')
parser.add_argument('--data_seed', type=int, default=421,
                    help='seed to split the inlier set into train and test (default: 1213)')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')

parser.add_argument('--epochs', type=int, default=150, ##150 is good for Thunderbird
                    help='number of epochs to train (default: 150)')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='number of hidden units (default: 64)')
parser.add_argument('--layers', type=int, default=2,
                    help='number of hidden layers (default: 2)')

##----important paramter 2----##
parser.add_argument('--bias', action="store_true", default = False,
                                    help='Whether to use bias terms in the GNN.')

parser.add_argument('--aggregation', type=str, default="Mean", choices=["Max", "Mean", "Sum"],
                    help='Type of graph level aggregation (default: Mean)')

parser.add_argument('--use_config', action="store_true",
                                    help='Whether to use configuration from a file')
parser.add_argument('--config_file', type=str, default="configs/config.txt",
                    help='Name of configuration file (default: configs/config.txt)')


parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight_decay constant lambda (default: 1e-4)')
parser.add_argument('--model_seed', type=int, default=0, 
                    help='Model seed (default: 0)')

# =============================================================================
# Step 5:  configure paramters
# store each paramter as an individual list since we want to do model selection
# =============================================================================
args = parser.parse_args()

lrs = [args.lr]
weight_decays = [args.weight_decay]
layercounts = [args.layers]
model_seeds = [args.model_seed]


##----if we use configuration file to store parameters----##
## this is mainly for unsupervised model selection, we can search parameters
## in a range of values

if args.use_config:

    with open(args.config_file) as f:
        lines = [line.rstrip() for line in f]

    for line in lines:
        words = line.split()
        
        ##Learning Rate
        if words[0] == "LR": 
            lrs = [float(w) for w in words[1:]]
        ##Weight Decay
        elif words[0] == "WD": 
            weight_decays = [float(w) for w in words[1:]]
        ##the number of  hidden layers
        elif words[0] == "layers":
            layercounts = [int(w) for w in words[1:]]
        ##the model seeds
        elif words[0] == "model_seeds":
            model_seeds = [int(w) for w in words[1:]]
        else:
            print("Cannot parse line: ", line)


# =============================================================================
# Step 6. we store all model candidates by traversing all parameter value lists
# =============================================================================

##use a dictionary to store model hyperparameters for different model candidates
MyDict = {}


for lr in lrs:
    for weight_decay in weight_decays:
        for model_seed in model_seeds:
            for layercount in layercounts:
            
                print("Running experiment for LR=%f, weight decay = %.1E, model seed = %d, number of layers = %d" % (lr, weight_decay, model_seed, layercount))
                MyDict[(lr,weight_decay,model_seed, layercount)], my_train, my_test, my_raw_data = run_experiment(
                    data=args.data,
                    data_seed=args.data_seed,
                    epochs=args.epochs,
                    model_seed=model_seed, # SEED
                    num_layers=layercount, # HYPERPARAMETER
                    device=args.device,
                    aggregation=args.aggregation,
                    bias=args.bias,
                    hidden_dim=args.hidden_dim,
                    lr=lr,  # HYPERPARAMETER
                    weight_decay=weight_decay,  # HYPERPARAMETER
                    batch=args.batch
                )

##Store the results in a directory if we use configuration file to run the experiments                     
if args.use_config:
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    with open('outputs/GIN_'+ args.aggregation + '_models_' + args.data + '_' + str(args.data_seed) + '.pkl', 'wb') as f:
        pickle.dump(MyDict, f)


# =============================================================================
# #Visualization of a single graph
# =============================================================================
test1 = my_raw_data[0]
import networkx as nx
import torch_geometric
g = torch_geometric.utils.to_networkx(test1, to_undirected=False)
nx.draw(g, with_labels = True)





