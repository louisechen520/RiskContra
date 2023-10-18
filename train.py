import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import json
import configparser
import pickle as pkl
from time import time
from datetime import datetime
import shutil
import argparse
import random
import math

import sys
import os

import ipdb

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.dataloader import normal_and_generate_dataset_time,get_mask,get_adjacent,get_grid_node_map_maxtrix
from lib.early_stop import EarlyStopping
from RiskContra import RiskContra
from lib.utils import mask_loss,compute_loss,predict_and_evaluate,create_mask,compute_contra_loss,mix_up

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file', default='config/chicago/RiskContra_Chicago_Config.json')
parser.add_argument("--gpus", type=str,help="test program", default='0')
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument("--lr", type=float, default=1e-3) #-4
parser.add_argument("--fusion", type=int, default=8) #12
parser.add_argument("--temp", type=float, default=0.1)
parser.add_argument("--weight", type=float, default=0.1)
parser.add_argument("--K", type=int, default=8)

args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

north_south_map = config['north_south_map']
west_east_map = config['west_east_map']


all_data_filename = config['all_data_filename']
mask_filename = config['mask_filename']

road_adj_filename = config['road_adj_filename']
risk_adj_filename = config['risk_adj_filename']
poi_adj_filename = config['poi_adj_filename']
grid_node_filename = config['grid_node_filename']
grid_node_map = get_grid_node_map_maxtrix(grid_node_filename)
num_of_vertices = grid_node_map.shape[1]


# patience = config['patience']
patience = 10
delta = config['delta']

learning_rate = args.lr
fusion = args.fusion
t = args.temp
wei = args.weight
K = args.K

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)


train_rate = config['train_rate']
valid_rate = config['valid_rate']

recent_prior = config['recent_prior']
week_prior = config['week_prior']
one_day_period = config['one_day_period']
days_of_week = config['days_of_week']
pre_len = config['pre_len']
seq_len = recent_prior + week_prior

training_epoch = config['training_epoch']

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def training(net,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            high_test_loader,
            road_adj,
            risk_adj,
            poi_adj,
            risk_mask,
            trainer,
            early_stop,
            device,
            scaler,
            data_type='nyc'
            ):
    start = time()
    global_step = 1
    log_file = '/data/chachen/year2/RiskContra_final_version/chi_final/log/log'+str(t)
    log = open(log_file, 'w')
    for epoch in range(1,training_epoch+1):
        net.train()
        batch = 1
        for train_feature_new,train_feature,target_time,gragh_feature,train_label_new,train_label in train_loader:
            start_time = time()
            train_feature_new,train_feature,target_time,gragh_feature,train_label_new,train_label = \
            train_feature_new.to(device),train_feature.to(device),\
            target_time.to(device),gragh_feature.to(device),train_label_new.to(device),train_label.to(device)
            # ipdb.set_trace()

            train_feature_mix,train_label_mix,gragh_feature_mix = mix_up(train_feature,train_feature_new,train_label,train_label_new,grid_node_map,wei)
            # ipdb.set_trace()
            mask_matrix_temp, mask_matrix = create_mask(train_feature_mix,train_label_mix)
            # mask_matrix_temp, mask_matrix = create_mask(train_feature,train_label)
            pred, _ = net(train_feature,target_time,gragh_feature,road_adj,risk_adj,poi_adj,grid_node_map)
            _, grid_mlp = net(train_feature_mix,target_time,gragh_feature,road_adj,risk_adj,poi_adj,grid_node_map)
            contra_loss = compute_contra_loss(mask_matrix_temp, mask_matrix, grid_mlp, t)
            orig_loss = mask_loss(pred,train_label,risk_mask,data_type=data_type)#l的shape：(1,)
            # mix_loss = mask_loss(pred_mix,train_label_mix,risk_mask,data_type=data_type)
            l = orig_loss + 0.001*contra_loss
            trainer.zero_grad()
            l.backward()
            trainer.step()
            training_loss_origin = orig_loss.cpu().item()
            training_loss_contra = contra_loss.cpu().item()
            training_loss_total = l.cpu().item()

            print('global step: %s, epoch: %s, batch: %s, orig_loss: %.6f, contra_loss: %.6f, total_loss: %.6f, time: %.2fs'
                % (global_step,epoch, batch, training_loss_origin, training_loss_contra, training_loss_total, time() - start_time),flush=True)
            
            batch+=1
            global_step+=1

        embed = grid_mlp.cpu().detach().numpy()
        # np.save('embed/embed_{}'.format(epoch), embed)
        #compute va/test loss
        val_loss = compute_loss(net,val_loader,risk_mask,road_adj,risk_adj,poi_adj,grid_node_map,global_step-1,device,data_type)
        print('global step: %s, epoch: %s,val loss：%.6f' %(global_step-1,epoch,val_loss),flush=True)
        log_string(log, 'global step: %s, epoch: %s,val loss：%.6f' %(global_step-1,epoch,val_loss))

        if epoch == 1 or val_loss < early_stop.best_score:
            test_rmse,test_recall,test_map,test_inverse_trans_pre,test_inverse_trans_label = \
                        predict_and_evaluate(net,test_loader,risk_mask,road_adj,risk_adj,poi_adj,grid_node_map,global_step-1,scaler,device)
            # np.save('pred', test_inverse_trans_pre)
            # np.save('label', test_inverse_trans_label)
            high_test_rmse,high_test_recall,high_test_map,_,_ = \
                        predict_and_evaluate(net,high_test_loader,risk_mask,road_adj,risk_adj,poi_adj,grid_node_map,global_step-1,scaler,device)

            print('global step: %s, epoch: %s, test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f,hihg test RMSE: %.4f,high test Recall: %.2f%%,high test MAP: %.4f'
                % (global_step-1,epoch, test_rmse,test_recall,test_map,high_test_rmse,high_test_recall,high_test_map),flush=True)
            log_string(log, 'global step: %s, epoch: %s, test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f,hihg test RMSE: %.4f,high test Recall: %.2f%%,high test MAP: %.4f'
                % (global_step-1,epoch, test_rmse,test_recall,test_map,high_test_rmse,high_test_recall,high_test_map))
        
        #early stop according to val loss
        early_stop(val_loss,test_rmse,test_recall,test_map,
                    high_test_rmse,high_test_recall,high_test_map,
                    test_inverse_trans_pre, test_inverse_trans_label)
        if early_stop.early_stop:
            print("Early Stopping in global step: %s, epoch: %s"%(global_step,epoch),flush=True)
            
            print('best test RMSE: %.4f,best test Recall: %.2f%%,best test MAP: %.4f'
                % (early_stop.best_rmse,early_stop.best_recall,early_stop.best_map),flush=True)
            print('best test high RMSE: %.4f,best test high Recall: %.2f%%,best high test MAP: %.4f'
                % (early_stop.best_high_rmse,early_stop.best_high_recall,early_stop.best_high_map),flush=True)
            log_string(log, 'best test RMSE: %.4f,best test Recall: %.2f%%,best test MAP: %.4f'
                % (early_stop.best_rmse,early_stop.best_recall,early_stop.best_map))
            log_string(log, 'best test high RMSE: %.4f,best test high Recall: %.2f%%,best high test MAP: %.4f'
                % (early_stop.best_high_rmse,early_stop.best_high_recall,early_stop.best_high_map))
            
            break
    end = time()
    print("total time: ", (end-start))
    return early_stop.best_rmse,early_stop.best_recall,early_stop.best_map

def main(config):
    batch_size = config['batch_size']
    # learning_rate = config['learning_rate']

    num_of_gru_layers = config['num_of_gru_layers']
    gru_hidden_size = config['gru_hidden_size']
    gcn_num_filter = config['gcn_num_filter']
    
    loaders = []
    scaler = ""
    train_data_shape = ""
    graph_feature_shape = ""
    for idx,(x_new,x,y_new,y,target_times,high_x,high_y,high_target_times,scaler) in enumerate(normal_and_generate_dataset_time(
                                    all_data_filename,
                                    train_rate=train_rate,
                                    valid_rate=valid_rate,
                                    recent_prior = recent_prior,
                                    week_prior = week_prior,
                                    one_day_period = one_day_period,
                                    days_of_week = days_of_week,
                                    pre_len = pre_len)):
        if args.test:
            x = x[:100]
            y = y[:100]
            target_times = target_times[:100]
            high_x = high_x[:100]
            high_y = high_y[:100]
            high_target_times = high_target_times[:100]

        if 'nyc' in all_data_filename:
            graph_x = x[:,:,[0,46,47],:,:].reshape((x.shape[0],x.shape[1],-1,north_south_map*west_east_map))
            high_graph_x = high_x[:,:,[0,46,47],:,:].reshape((high_x.shape[0],high_x.shape[1],-1,north_south_map*west_east_map))
            graph_x = np.dot(graph_x,grid_node_map)
            high_graph_x = np.dot(high_graph_x,grid_node_map)
        if 'chicago' in all_data_filename:
            graph_x = x[:,:,[0,39,40],:,:].reshape((x.shape[0],x.shape[1],-1,north_south_map*west_east_map))
            high_graph_x = high_x[:,:,[0,39,40],:,:].reshape((high_x.shape[0],high_x.shape[1],-1,north_south_map*west_east_map))
            graph_x = np.dot(graph_x,grid_node_map)
            high_graph_x = np.dot(high_graph_x,grid_node_map)

        print("feature:",str(x.shape),"label:",str(y.shape),"time:",str(target_times.shape),
            "high feature:",str(high_x.shape),"high label:",str(high_y.shape))
        print("graph_x:",str(graph_x.shape),"high_graph_x:",str(high_graph_x.shape))
        if idx == 0:
            scaler = scaler
            train_data_shape = x.shape
            time_shape = target_times.shape
            graph_feature_shape = graph_x.shape
            loaders.append(Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(x_new),
                    torch.from_numpy(x),
                    torch.from_numpy(target_times),
                    torch.from_numpy(graph_x),
                    torch.from_numpy(y_new),
                    torch.from_numpy(y)
                    ),
                    batch_size=batch_size,
                    shuffle=(idx == 0)
                ))
        if idx ==1:
            loaders.append(Data.DataLoader(
            Data.TensorDataset(
                torch.from_numpy(x),
                torch.from_numpy(target_times),
                torch.from_numpy(graph_x),
                torch.from_numpy(y)
                ),
                batch_size=batch_size,
                shuffle=(idx == 0)
                ))
        if idx == 2:
            loaders.append(Data.DataLoader(
            Data.TensorDataset(
                torch.from_numpy(x),
                torch.from_numpy(target_times),
                torch.from_numpy(graph_x),
                torch.from_numpy(y)
                ),
                batch_size=batch_size,
                shuffle=(idx == 0)
            ))
            high_test_loader = Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(high_x),
                    torch.from_numpy(high_target_times),
                    torch.from_numpy(high_graph_x),
                    torch.from_numpy(high_y)
                    ),
                    batch_size=batch_size,
                    shuffle=(idx == 0)
                )
    train_loader, val_loader, test_loader = loaders

    nums_of_filter = []
    for _ in range(2):
        nums_of_filter.append(gcn_num_filter)

    Model = RiskContra(train_data_shape[2],num_of_gru_layers,seq_len,pre_len,
                    gru_hidden_size,time_shape[1],graph_feature_shape[2],
                    nums_of_filter,north_south_map,west_east_map,fusion,K)
    #multi gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!",flush=True)
        Model = nn.DataParallel(Model)
    Model.to(device)
    # print(Model)

    num_of_parameters = 0
    for name,parameters in Model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)


    trainer = optim.AdamW(Model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=patience,delta=delta)
    
    risk_mask = get_mask(mask_filename)
    road_adj = get_adjacent(road_adj_filename)
    risk_adj = get_adjacent(risk_adj_filename)
    if poi_adj_filename == "":
        poi_adj = None
    else:
        poi_adj = get_adjacent(poi_adj_filename)

    best_mae,best_mse,best_rmse = training(
            Model,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            high_test_loader,
            road_adj,
            risk_adj,
            poi_adj,
            risk_mask,
            trainer,
            early_stop,
            device,
            scaler,
            data_type = config['data_type']
            )
    return best_mae,best_mse,best_rmse

if __name__ == "__main__":
    
    main(config)
