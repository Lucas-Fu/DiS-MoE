import numpy as np
import torch
import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path, batch_size=32):
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_data()
        self.load_mean_std() 
        self._to_tensor()


    def load_data(self):
        self.X_aqi_train = np.load(os.path.join(self.data_path, "X_aqi_train.npy"))
        self.X_meo_train = np.load(os.path.join(self.data_path, "X_meo_train.npy"))
        self.y_aqi_train = np.load(os.path.join(self.data_path, "y_aqi_train.npy"))
        self.y_meo_train = np.load(os.path.join(self.data_path, "y_meo_train.npy"))
        self.y_aqi_mask_train = np.load(os.path.join(self.data_path, "y_aqi_mask_train.npy"))
        self.y_meo_mask_train = np.load(os.path.join(self.data_path, "y_meo_mask_train.npy"))

        self.X_aqi_val = np.load(os.path.join(self.data_path, "X_aqi_val.npy"))
        self.X_meo_val = np.load(os.path.join(self.data_path, "X_meo_val.npy"))
        self.y_aqi_val = np.load(os.path.join(self.data_path, "y_aqi_val.npy"))
        self.y_meo_val = np.load(os.path.join(self.data_path, "y_meo_val.npy"))
        self.y_aqi_mask_val = np.load(os.path.join(self.data_path, "y_aqi_mask_val.npy"))
        self.y_meo_mask_val = np.load(os.path.join(self.data_path, "y_meo_mask_val.npy"))

        self.X_aqi_test = np.load(os.path.join(self.data_path, "X_aqi_test.npy"))
        self.X_meo_test = np.load(os.path.join(self.data_path, "X_meo_test.npy"))
        self.y_aqi_test = np.load(os.path.join(self.data_path, "y_aqi_test.npy"))
        self.y_meo_test = np.load(os.path.join(self.data_path, "y_meo_test.npy"))
        self.y_aqi_mask_test = np.load(os.path.join(self.data_path, "y_aqi_mask_test.npy"))
        self.y_meo_mask_test = np.load(os.path.join(self.data_path, "y_meo_mask_test.npy"))
        
        self.context_feat = np.load(os.path.join(self.data_path, "context_feat.npy"))
        self.adj = np.load(os.path.join(self.data_path, "adj.npy"))
        
    def load_mean_std(self):

        
        path_aqi_mean_std = os.path.join(self.data_path, "aqi_mean_std_train.csv")
        path_meo_mean_std = os.path.join(self.data_path, "meo_mean_std_train.csv")

        if not os.path.exists(path_aqi_mean_std):
            raise FileNotFoundError(f"AQI mean/std file not found: {path_aqi_mean_std}. Ensure it's named 'aqi_mean_std_train.csv' and contains training set statistics.")
        if not os.path.exists(path_meo_mean_std):
            raise FileNotFoundError(f"MEO mean/std file not found: {path_meo_mean_std}. Ensure it's named 'meo_mean_std_train.csv' and contains training set statistics.")

        self.aqi_mean_std = pd.read_csv(path_aqi_mean_std)
        self.meo_mean_std = pd.read_csv(path_meo_mean_std)
        


    def _to_tensor(self):
        self.X_aqi_ex_train = torch.LongTensor(self.X_aqi_train[:, :, :, 6:]).to(self.device)
        self.X_meo_ex_train = torch.LongTensor(self.X_meo_train[:, :, :, 4:]).to(self.device)
        self.X_aqi_train = torch.FloatTensor(self.X_aqi_train[:, :, :, :6]).to(self.device)
        self.X_meo_train = torch.FloatTensor(self.X_meo_train[:, :, :, :4]).to(self.device)
        self.y_aqi_train = torch.FloatTensor(self.y_aqi_train).to(self.device)
        self.y_meo_train = torch.FloatTensor(self.y_meo_train).to(self.device)
        self.y_aqi_mask_train = torch.FloatTensor(self.y_aqi_mask_train).to(self.device)
        self.y_meo_mask_train = torch.FloatTensor(self.y_meo_mask_train).to(self.device)

        self.X_aqi_ex_val = torch.LongTensor(self.X_aqi_val[:, :, :, 6:]).to(self.device)
        self.X_meo_ex_val = torch.LongTensor(self.X_meo_val[:, :, :, 4:]).to(self.device)
        self.X_aqi_val = torch.FloatTensor(self.X_aqi_val[:, :, :, :6]).to(self.device)
        self.X_meo_val = torch.FloatTensor(self.X_meo_val[:, :, :, :4]).to(self.device)
        self.y_aqi_val = torch.FloatTensor(self.y_aqi_val).to(self.device)
        self.y_meo_val = torch.FloatTensor(self.y_meo_val).to(self.device)
        self.y_aqi_mask_val = torch.FloatTensor(self.y_aqi_mask_val).to(self.device)
        self.y_meo_mask_val = torch.FloatTensor(self.y_meo_mask_val).to(self.device)

        self.X_aqi_ex_test = torch.LongTensor(self.X_aqi_test[:, :, :, 6:]).to(self.device)
        self.X_meo_ex_test = torch.LongTensor(self.X_meo_test[:, :, :, 4:]).to(self.device)
        self.X_aqi_test = torch.FloatTensor(self.X_aqi_test[:, :, :, :6]).to(self.device)
        self.X_meo_test = torch.FloatTensor(self.X_meo_test[:, :, :, :4]).to(self.device)
        self.y_aqi_test = torch.FloatTensor(self.y_aqi_test).to(self.device)
        self.y_meo_test = torch.FloatTensor(self.y_meo_test).to(self.device)
        self.y_aqi_mask_test = torch.FloatTensor(self.y_aqi_mask_test).to(self.device)
        self.y_meo_mask_test = torch.FloatTensor(self.y_meo_mask_test).to(self.device)
        
        self.context_feat = torch.FloatTensor(self.context_feat).to(self.device)
        self.adj = torch.FloatTensor(self.adj).to(self.device)
        self.adj_norm = (self.adj / 10000.0).to(self.device) 
        
    def load_X_batch(self, X_aqi, X_meo, X_aqi_ex, X_meo_ex, i):
        X_aqi_batch = X_aqi[i*self.batch_size:i*self.batch_size+self.batch_size]
        X_meo_batch = X_meo[i*self.batch_size:i*self.batch_size+self.batch_size]
        X_aqi_ex_batch = X_aqi_ex[i*self.batch_size:i*self.batch_size+self.batch_size]
        X_meo_ex_batch = X_meo_ex[i*self.batch_size:i*self.batch_size+self.batch_size]
        return X_aqi_batch, X_meo_batch, X_aqi_ex_batch, X_meo_ex_batch
    
    def load_y_batch(self, y_aqi, y_meo, y_aqi_mask, y_meo_mask, i):
        y_aqi_batch = y_aqi[i*self.batch_size:i*self.batch_size+self.batch_size]
        y_meo_batch = y_meo[i*self.batch_size:i*self.batch_size+self.batch_size]
        y_aqi_mask_batch = y_aqi_mask[i*self.batch_size:i*self.batch_size+self.batch_size]
        y_meo_mask_batch = y_meo_mask[i*self.batch_size:i*self.batch_size+self.batch_size]
        return y_aqi_batch, y_meo_batch, y_aqi_mask_batch, y_meo_mask_batch
    
    def denormalize(self, data, feature_type='aqi', feature_index=0, dataset_type='train'): 
        
        mean_std_df = self.aqi_mean_std if feature_type == 'aqi' else self.meo_mean_std
            
        if mean_std_df.empty:
            raise ValueError(f"Mean/Std DataFrame is empty for feature_type '{feature_type}'. Ensure 'aqi_mean_std_train.csv' or 'meo_mean_std_train.csv' is loaded correctly.")

        if not (0 <= feature_index < len(mean_std_df)):
            raise ValueError(
                f"feature_index {feature_index} is out of bounds for {feature_type} "
                f"which has {len(mean_std_df)} features in its mean/std data."
            )
            
        mean_scalar = mean_std_df['mean'].iloc[feature_index]
        std_scalar = mean_std_df['std'].iloc[feature_index]
        
        if std_scalar == 0:
            print(f"Warning: Standard deviation is zero for {feature_type}, feature index {feature_index}. Denormalized value will be the mean: {mean_scalar}")
            return np.full_like(data, mean_scalar) 

        return data * std_scalar + mean_scalar
    
    def get_num_batches(self, dataset_type='train'):
        if dataset_type == 'train':
            return len(self.X_aqi_train) // self.batch_size
        elif dataset_type == 'val':
            return len(self.X_aqi_val) // self.batch_size
        else:
            return len(self.X_aqi_test) // self.batch_size
    
    def shuffle_train_data(self):
        indices = torch.randperm(len(self.X_aqi_train))
        self.X_aqi_train = self.X_aqi_train[indices]
        self.X_meo_train = self.X_meo_train[indices]
        self.X_aqi_ex_train = self.X_aqi_ex_train[indices]
        self.X_meo_ex_train = self.X_meo_ex_train[indices]
        self.y_aqi_train = self.y_aqi_train[indices]
        self.y_meo_train = self.y_meo_train[indices]
        self.y_aqi_mask_train = self.y_aqi_mask_train[indices]
        self.y_meo_mask_train = self.y_meo_mask_train[indices]