import os
import sys
import math
import argparse
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import itertools
import ast
import json
from data_preprocessor import DataPreprocessor
from data_loader import DataLoader
from DiSMoE import *


parser = argparse.ArgumentParser(description="Train DiS-MoE model ")

parser.add_argument("--save_path", type=str, default="./result/demo.json", help="Path to save training results")
parser.add_argument("--data_path", type=str, default="./Data/Beijing", help="Path to data directory")
parser.add_argument("--hidden_dim", type=int, default=16, help="size_of_hidden_dimension")
parser.add_argument("--ffn_hidden_dim", type=int, default=32, help="ffn_hidden_dim")
parser.add_argument("--num_FHG_layers", type=int,default=1,help="num_of_frequency_heterogeneous_graph_layer")
parser.add_argument("--num_experts", type=int,default=16,help="num_of_experts")
parser.add_argument("--num_heads", type=int,default=2,help="num_of_heads_in_router")
parser.add_argument("--d", type=int,default=32,help="hidden_dimension_of_MLPs_for_conditional_affine_transformation")
parser.add_argument("--topk", type=int,default=4,help="topk_amp")

parser.add_argument("--balance_loss_alpha", type=float, default=0.8, help="Coefficient for the load balancing loss")
parser.add_argument("--diversity_loss_alpha", type=float, default=0.01, help="Coefficient for the diversity loss")
parser.add_argument("--mmd_sigma", type=float, default=1.0, help="sigma for mmd")

parser.add_argument("--context_dim_static", type=int, default=22, help="dimension of static urban context features")

parser.add_argument("--N_aqi", type=int, default=35, help="Number_of_air_quality_stations")
parser.add_argument("--N_meo", type=int, default=18, help="Number_of_meteorological_stations")



parser.add_argument("--T_in", type=int, default=72, help="input_length")
parser.add_argument("--T_out", type=int, default=48, help="output_length")


parser.add_argument("--learning_rate", type=float,default=0.0001,help="learning_rate")
parser.add_argument("--seed",type=int,default=111,help="set the random seed")
parser.add_argument("--batch_size",type=int,default=64,help="batch size")
parser.add_argument("--epochs",type=int,default=100,help="number of epochs")
parser.add_argument("--patience",type=int,default=10,help="patience for early stopping")
parser.add_argument("--test_only", type=bool, default=False ,help="test only, skip training")

args = parser.parse_args()


def loss_mse(y_pred, y_true, y_mask):
    y_mse = torch.sub(y_pred, y_true)
    y_mse = torch.mul(y_mse, y_mse)
    y_mse = torch.mul(y_mse, y_mask)
    y_mse = torch.div(torch.sum(y_mse), torch.sum(y_mask))
    return y_mse

def mae(pred, label, mask):
    mae = np.sum(abs(label-pred)*mask)
    num = np.sum(mask)
    mae = mae/num
    return mae

def smape(pred, label, mask):
    smape = np.sum(2.0*(np.abs(pred - label) / (np.abs(pred) + np.abs(label)))*mask)
    num = np.sum(mask)
    smape = smape/num
    return smape




seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


if __name__ == '__main__':
    preprocessor = DataPreprocessor(T_in=args.T_in, T_out=args.T_out, data_dir=args.data_path)
    if args.test_only == False:
        preprocessor.run()
    

    data_loader = DataLoader(args.data_path, args.batch_size)
    

    generator = DiSMoE(in_aqi_features=6,
                     in_meo_features=4,
                     dropout=0.5, 
                     alpha=0.2, 
                     ffn_hidden_dim=args.ffn_hidden_dim,
                     num_FHG_layers=args.num_FHG_layers,
                     hidden_dim=args.hidden_dim,
                     
                     context_dim_static=args.context_dim_static,
                     mmd_sigma=args.mmd_sigma,

                     num_experts=args.num_experts,
                     num_heads=args.num_heads,
                     d=args.d,
                     topk=args.topk,
                     T_in=args.T_in,
                     T_out=args.T_out,
                     N_aqi = args.N_aqi,
                     N_meo = args.N_meo
                    )
    
    generator.to(device)
    
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)

    best_epoch = 0
    min_val_loss = float('inf')
    epochs_no_improve = 0

    if args.test_only == False:
        for epoch in range(args.epochs):
            st_time = time.time()

            print('Training...')
            generator.train()
            
            data_loader.shuffle_train_data()
            
            train_metrics = {
                'aqi_loss': 0, 'temp_loss': 0, 'humi_loss': 0, 'wind_loss': 0,
                'aqi_mae': 0, 'temp_mae': 0, 'humi_mae': 0, 'wind_mae': 0,
                'aqi_smape': 0, 'temp_smape': 0, 'humi_smape': 0, 'wind_smape': 0
            }
            
            num_batches = data_loader.get_num_batches('train')
            for i in range(num_batches):
                X_batch = data_loader.load_X_batch(
                    data_loader.X_aqi_train, data_loader.X_meo_train,
                    data_loader.X_aqi_ex_train, data_loader.X_meo_ex_train, i
                )
                y_batch = data_loader.load_y_batch(
                    data_loader.y_aqi_train, data_loader.y_meo_train,
                    data_loader.y_aqi_mask_train, data_loader.y_meo_mask_train, i
                )
                
                X_aqi, X_meo, X_aqi_ex, X_meo_ex = X_batch
                y_aqi, y_meo, y_aqi_mask, y_meo_mask = y_batch
                
                out_aqi, out_temp, out_humi, out_wind, balance_loss, diversity_loss = generator(
                    X_aqi, X_meo, X_aqi_ex, X_meo_ex,
                    data_loader.context_feat, data_loader.adj, data_loader.adj_norm,
                    return_balance_loss=True
                )
 
                
                y_aqi = torch.transpose(y_aqi, 1, 2)
                y_aqi_mask = torch.transpose(y_aqi_mask, 1, 2)
                y_meo = torch.transpose(y_meo, 1, 2)
                y_meo_mask = torch.transpose(y_meo_mask, 1, 2)
                
                label_aqi = y_aqi[:, :, :, 0].contiguous()
                label_temp = y_meo[:, :, :, 0].contiguous()
                label_humi = y_meo[:, :, :, 1].contiguous()
                label_wind = y_meo[:, :, :, 2].contiguous()
                
                loss_aqi = loss_mse(out_aqi, label_aqi, y_aqi_mask[:,:,:,0])
                loss_temp = loss_mse(out_temp, label_temp, y_meo_mask[:,:,:,0])
                loss_humi = loss_mse(out_humi, label_humi, y_meo_mask[:,:,:,1])
                loss_wind = loss_mse(out_wind, label_wind, y_meo_mask[:,:,:,2])
                
                loss = loss_aqi + loss_temp + loss_humi + loss_wind
                loss = loss + args.balance_loss_alpha * balance_loss + args.diversity_loss_alpha * diversity_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_metrics['aqi_loss'] += loss_aqi.item()
                train_metrics['temp_loss'] += loss_temp.item()
                train_metrics['humi_loss'] += loss_humi.item()
                train_metrics['wind_loss'] += loss_wind.item()
                
                out_aqi_denorm = data_loader.denormalize(out_aqi.detach().cpu().numpy(),  'aqi', 0)
                label_aqi_denorm = data_loader.denormalize(label_aqi.detach().cpu().numpy(),  'aqi', 0)
                out_temp_denorm = data_loader.denormalize(out_temp.detach().cpu().numpy(),  'meo', 0)
                label_temp_denorm = data_loader.denormalize(label_temp.detach().cpu().numpy(),  'meo', 0)
                out_humi_denorm = data_loader.denormalize(out_humi.detach().cpu().numpy(),  'meo', 1)
                label_humi_denorm = data_loader.denormalize(label_humi.detach().cpu().numpy(),  'meo', 1)
                out_wind_denorm = data_loader.denormalize(out_wind.detach().cpu().numpy(),  'meo', 2)
                label_wind_denorm = data_loader.denormalize(label_wind.detach().cpu().numpy(),  'meo', 2)
                
                train_metrics['aqi_mae'] += mae(out_aqi_denorm, label_aqi_denorm, y_aqi_mask[:,:,:,0].cpu().numpy())
                train_metrics['temp_mae'] += mae(out_temp_denorm, label_temp_denorm, y_meo_mask[:,:,:,0].cpu().numpy())
                train_metrics['humi_mae'] += mae(out_humi_denorm, label_humi_denorm, y_meo_mask[:,:,:,1].cpu().numpy())
                train_metrics['wind_mae'] += mae(out_wind_denorm, label_wind_denorm, y_meo_mask[:,:,:,2].cpu().numpy())
                
                train_metrics['aqi_smape'] += smape(out_aqi_denorm, label_aqi_denorm, y_aqi_mask[:,:,:,0].cpu().numpy())
                train_metrics['temp_smape'] += smape(out_temp_denorm, label_temp_denorm, y_meo_mask[:,:,:,0].cpu().numpy())
                train_metrics['humi_smape'] += smape(out_humi_denorm, label_humi_denorm, y_meo_mask[:,:,:,1].cpu().numpy())
                train_metrics['wind_smape'] += smape(out_wind_denorm, label_wind_denorm, y_meo_mask[:,:,:,2].cpu().numpy())
                
                if i % 5 == 1:
                    print('Epoch: {:04d}'.format(epoch+1),
                        'Step: {:06d}'.format(i+1),
                        'aqi mae: {:.4f}'.format(train_metrics['aqi_mae']/5),
                        'temp mae: {:.4f}'.format(train_metrics['temp_mae']/5),
                        'humi mae: {:.4f}'.format(train_metrics['humi_mae']/5),
                        'wind mae: {:.4f}'.format(train_metrics['wind_mae']/5))
                    for key in train_metrics:
                        train_metrics[key] = 0

            print('Validating...')
            generator.eval()
            
            val_metrics = {
                'loss': 0, 'aqi_mae': 0, 'temp_mae': 0, 'humi_mae': 0, 'wind_mae': 0,
                'aqi_smape': 0, 'temp_smape': 0, 'humi_smape': 0, 'wind_smape': 0
            }
            
            with torch.no_grad():
                num_batches = data_loader.get_num_batches('val')
                for i in range(num_batches):
                    X_batch = data_loader.load_X_batch(
                        data_loader.X_aqi_val, data_loader.X_meo_val,
                        data_loader.X_aqi_ex_val, data_loader.X_meo_ex_val, i
                    )
                    y_batch = data_loader.load_y_batch(
                        data_loader.y_aqi_val, data_loader.y_meo_val,
                        data_loader.y_aqi_mask_val, data_loader.y_meo_mask_val, i
                    )
                    
                    X_aqi, X_meo, X_aqi_ex, X_meo_ex = X_batch
                    y_aqi, y_meo, y_aqi_mask, y_meo_mask = y_batch
                    
                    out_aqi, out_temp, out_humi, out_wind = generator(
                        X_aqi, X_meo, X_aqi_ex, X_meo_ex,
                        data_loader.context_feat, data_loader.adj, data_loader.adj_norm
                    )
                    
                
                    
                    y_aqi = torch.transpose(y_aqi, 1, 2)
                    y_aqi_mask = torch.transpose(y_aqi_mask, 1, 2)
                    y_meo = torch.transpose(y_meo, 1, 2)
                    y_meo_mask = torch.transpose(y_meo_mask, 1, 2)
                    
                    label_aqi = y_aqi[:, :, :, 0].contiguous()
                    label_temp = y_meo[:, :, :, 0].contiguous()
                    label_humi = y_meo[:, :, :, 1].contiguous()
                    label_wind = y_meo[:, :, :, 2].contiguous()
                    
                    loss_aqi = loss_mse(out_aqi, label_aqi, y_aqi_mask[:,:,:,0])
                    loss_temp = loss_mse(out_temp, label_temp, y_meo_mask[:,:,:,0])
                    loss_humi = loss_mse(out_humi, label_humi, y_meo_mask[:,:,:,1])
                    loss_wind = loss_mse(out_wind, label_wind, y_meo_mask[:,:,:,2])
                    
                    val_loss = loss_aqi + loss_temp + loss_humi + loss_wind
                    val_metrics['loss'] += val_loss.item()
                    
                    out_aqi_denorm = data_loader.denormalize(out_aqi.detach().cpu().numpy(),  'aqi', 0)
                    label_aqi_denorm = data_loader.denormalize(label_aqi.detach().cpu().numpy(),  'aqi', 0)
                    out_temp_denorm = data_loader.denormalize(out_temp.detach().cpu().numpy(),  'meo', 0)
                    label_temp_denorm = data_loader.denormalize(label_temp.detach().cpu().numpy(),  'meo', 0)
                    out_humi_denorm = data_loader.denormalize(out_humi.detach().cpu().numpy(),  'meo', 1)
                    label_humi_denorm = data_loader.denormalize(label_humi.detach().cpu().numpy(),  'meo', 1)
                    out_wind_denorm = data_loader.denormalize(out_wind.detach().cpu().numpy(),  'meo', 2)
                    label_wind_denorm = data_loader.denormalize(label_wind.detach().cpu().numpy(),  'meo', 2)
                    
                    val_metrics['aqi_mae'] += mae(out_aqi_denorm, label_aqi_denorm, y_aqi_mask[:,:,:,0].cpu().numpy())
                    val_metrics['temp_mae'] += mae(out_temp_denorm, label_temp_denorm, y_meo_mask[:,:,:,0].cpu().numpy())
                    val_metrics['humi_mae'] += mae(out_humi_denorm, label_humi_denorm, y_meo_mask[:,:,:,1].cpu().numpy())
                    val_metrics['wind_mae'] += mae(out_wind_denorm, label_wind_denorm, y_meo_mask[:,:,:,2].cpu().numpy())
                    
                    val_metrics['aqi_smape'] += smape(out_aqi_denorm, label_aqi_denorm, y_aqi_mask[:,:,:,0].cpu().numpy())
                    val_metrics['temp_smape'] += smape(out_temp_denorm, label_temp_denorm, y_meo_mask[:,:,:,0].cpu().numpy())
                    val_metrics['humi_smape'] += smape(out_humi_denorm, label_humi_denorm, y_meo_mask[:,:,:,1].cpu().numpy())
                    val_metrics['wind_smape'] += smape(out_wind_denorm, label_wind_denorm, y_meo_mask[:,:,:,2].cpu().numpy())
            
            for key in val_metrics:
                val_metrics[key] /= num_batches
            
            print("Epoch: {}".format(epoch+1))
            print("val_loss: {:.4f}".format(val_metrics['loss']))
            print('Validation metrics:', 
                'aqi_mae: {:.4f}'.format(val_metrics['aqi_mae']),
                'temp_mae: {:.4f}'.format(val_metrics['temp_mae']),
                'humi_mae: {:.4f}'.format(val_metrics['humi_mae']),
                'wind_mae: {:.4f}'.format(val_metrics['wind_mae']))
            print('time: {:.4f}s'.format(time.time() - st_time))

            if val_metrics['loss'] < min_val_loss:
                min_val_loss = val_metrics['loss']
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(generator.state_dict(), 'best_generator.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print('Early stopping')
                    break
    
    
    print('Testing with best model...')
    generator.load_state_dict(torch.load('best_generator.pth'))
    generator.eval()
    
    test_metrics = {
        'aqi_mae': 0, 'temp_mae': 0, 'humi_mae': 0, 'wind_mae': 0,
        'aqi_smape': 0, 'temp_smape': 0, 'humi_smape': 0, 'wind_smape': 0
    }
    

    with torch.no_grad():
        num_batches = data_loader.get_num_batches('test')
        for i in range(num_batches):
            X_batch = data_loader.load_X_batch(
                data_loader.X_aqi_test, data_loader.X_meo_test,
                data_loader.X_aqi_ex_test, data_loader.X_meo_ex_test, i
            )
            y_batch = data_loader.load_y_batch(
                data_loader.y_aqi_test, data_loader.y_meo_test,
                data_loader.y_aqi_mask_test, data_loader.y_meo_mask_test, i
            )
            
            X_aqi, X_meo, X_aqi_ex, X_meo_ex = X_batch
            y_aqi, y_meo, y_aqi_mask, y_meo_mask = y_batch
            
            out_aqi, out_temp, out_humi, out_wind = generator(
                X_aqi, X_meo, X_aqi_ex, X_meo_ex,
                data_loader.context_feat.to(device), data_loader.adj.to(device), data_loader.adj_norm.to(device)
            )
            
            
            y_aqi = torch.transpose(y_aqi, 1, 2)
            y_aqi_mask = torch.transpose(y_aqi_mask, 1, 2)
            y_meo = torch.transpose(y_meo, 1, 2)
            y_meo_mask = torch.transpose(y_meo_mask, 1, 2)
            
            label_aqi = y_aqi[:, :, :, 0].contiguous()
            label_temp = y_meo[:, :, :, 0].contiguous()
            label_humi = y_meo[:, :, :, 1].contiguous()
            label_wind = y_meo[:, :, :, 2].contiguous()
            
            out_aqi_denorm = data_loader.denormalize(out_aqi.detach().cpu().numpy(),  'aqi', 0)
            label_aqi_denorm = data_loader.denormalize(label_aqi.detach().cpu().numpy(),  'aqi', 0)
            out_temp_denorm = data_loader.denormalize(out_temp.detach().cpu().numpy(),  'meo', 0)
            label_temp_denorm = data_loader.denormalize(label_temp.detach().cpu().numpy(),  'meo', 0)
            out_humi_denorm = data_loader.denormalize(out_humi.detach().cpu().numpy(),  'meo', 1)
            label_humi_denorm = data_loader.denormalize(label_humi.detach().cpu().numpy(),  'meo', 1)
            out_wind_denorm = data_loader.denormalize(out_wind.detach().cpu().numpy(),  'meo', 2)
            label_wind_denorm = data_loader.denormalize(label_wind.detach().cpu().numpy(),  'meo', 2)
            
            test_metrics['aqi_mae'] += mae(out_aqi_denorm, label_aqi_denorm, y_aqi_mask[:,:,:,0].cpu().numpy())
            test_metrics['temp_mae'] += mae(out_temp_denorm, label_temp_denorm, y_meo_mask[:,:,:,0].cpu().numpy())
            test_metrics['humi_mae'] += mae(out_humi_denorm, label_humi_denorm, y_meo_mask[:,:,:,1].cpu().numpy())
            test_metrics['wind_mae'] += mae(out_wind_denorm, label_wind_denorm, y_meo_mask[:,:,:,2].cpu().numpy())
            
            test_metrics['aqi_smape'] += smape(out_aqi_denorm, label_aqi_denorm, y_aqi_mask[:,:,:,0].cpu().numpy())
            test_metrics['temp_smape'] += smape(out_temp_denorm, label_temp_denorm, y_meo_mask[:,:,:,0].cpu().numpy())
            test_metrics['humi_smape'] += smape(out_humi_denorm, label_humi_denorm, y_meo_mask[:,:,:,1].cpu().numpy())
            test_metrics['wind_smape'] += smape(out_wind_denorm, label_wind_denorm, y_meo_mask[:,:,:,2].cpu().numpy())
    



    for key in test_metrics:
        test_metrics[key] /= num_batches
    
    print('Final Testing Results:')
    print('Testing metrics:', 
        'aqi_mae: {:.4f}'.format(test_metrics['aqi_mae']),
        'temp_mae: {:.4f}'.format(test_metrics['temp_mae']),
        'humi_mae: {:.4f}'.format(test_metrics['humi_mae']),
        'wind_mae: {:.4f}'.format(test_metrics['wind_mae']))
    print('Best Epoch: {}'.format(best_epoch))
    
    results = {
        "seed" : args.seed,
        "hidden_dim": args.hidden_dim,
        "ffn_hidden_dim": args.ffn_hidden_dim,
        "num_FHG_layers": args.num_FHG_layers,
        "num_experts": args.num_experts,
        "num_heads": args.num_heads,
        "d": args.d,
        "topk": args.topk,
        "learning_rate": args.learning_rate,
        "balance_loss_alpha": args.balance_loss_alpha,
        "diversity_loss_alpha": args.diversity_loss_alpha,
        "best_epoch": best_epoch,
        "min_val_loss": float(min_val_loss),
        **test_metrics
    }
    
    with open(args.save_path, "w") as f:
        json.dump(results, f, indent=4) 
