import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess_raw_data(file_path):
    data = pd.read_csv(file_path)
    data['utc_time'] = pd.to_datetime(data['utc_time'])
    data = data.sort_values(by=['station_id', 'utc_time']).reset_index(drop=True)
    specific_anomalies = [999999, 999017]
    for anomaly_val in specific_anomalies:
        count = data.isin([anomaly_val]).sum().sum()
        
    data.replace(specific_anomalies, np.nan, inplace=True)
    initial_mask_df = data.isna() 
    return data, initial_mask_df

def align_datasets_temporally(df1, df1_mask, df2, df2_mask, time_col='utc_time'):
    min_time_df1 = df1[time_col].min()
    max_time_df1 = df1[time_col].max()
    min_time_df2 = df2[time_col].min()
    max_time_df2 = df2[time_col].max()
    common_start_time = max(min_time_df1, min_time_df2)
    common_end_time = min(max_time_df1, max_time_df2)

    if common_start_time >= common_end_time:
        raise ValueError("No overlapping time period found.")
    df1_aligned = df1[(df1[time_col] >= common_start_time) & (df1[time_col] <= common_end_time)].copy()
    df1_mask_aligned = df1_mask.loc[df1_aligned.index] 
    df2_aligned = df2[(df2[time_col] >= common_start_time) & (df2[time_col] <= common_end_time)].copy()
    df2_mask_aligned = df2_mask.loc[df2_aligned.index]
    df1_aligned.reset_index(drop=True, inplace=True)
    df1_mask_aligned.reset_index(drop=True, inplace=True)
    df2_aligned.reset_index(drop=True, inplace=True)
    df2_mask_aligned.reset_index(drop=True, inplace=True)

    return df1_aligned, df1_mask_aligned, df2_aligned, df2_mask_aligned


def engineer_features(df, is_aq=True):
    data = df.copy()
    data['station_num'] = data['station_id'].astype('category').cat.codes
    data['month'] = data['utc_time'].dt.month
    data['weekday'] = data['utc_time'].dt.weekday
    data['hour'] = data['utc_time'].dt.hour

    if is_aq:
        numerical_features_to_scale = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
        categorical_like_features = ['station_num', 'month', 'weekday', 'hour']

        all_features_ordered = numerical_features_to_scale + categorical_like_features + ['utc_time', 'station_id'] 
    else:
        if 'wind_direction' not in data.columns:
            raise ValueError("'wind_direction' column not found in MEO data.")
        if data['wind_direction'].isna().any():
            data['wind_direction'].fillna(0, inplace=True)
        data['wind_direction_cat'] = np.floor(data['wind_direction'] / 45.0).astype(int).clip(0, 8)
        
        numerical_features_to_scale = ['temperature', 'humidity', 'wind_speed', 'pressure']
        categorical_like_features = ['wind_direction_cat', 'station_num', 'month', 'weekday', 'hour']
        all_features_ordered = numerical_features_to_scale + categorical_like_features + ['utc_time', 'station_id'] 


    final_columns_to_return = []
    for col_list in [numerical_features_to_scale, categorical_like_features, ['utc_time', 'station_id']]:
        for col in col_list:
            if col in data.columns and col not in final_columns_to_return: 
                 final_columns_to_return.append(col)
            elif col not in data.columns : 
                 raise ValueError(f"Column '{col}' not found in {'AQ' if is_aq else 'MEO'} data during feature engineering. Available columns: {data.columns.tolist()}")

    return data[final_columns_to_return], final_columns_to_return, numerical_features_to_scale

def fill_missing_per_station_feature(df_processed, is_aq=True, station_col_id='station_id'):

    df_filled = df_processed.copy()
    if is_aq:
        cols_to_fill_robustly = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    else:
        cols_to_fill_robustly = ['temperature', 'humidity', 'wind_speed', 'pressure', 'wind_direction'] 
    cols_to_fill_robustly = [col for col in cols_to_fill_robustly if col in df_filled.columns]


    initial_nans_count = df_filled[cols_to_fill_robustly].isna().sum().sum()


    if initial_nans_count > 0:
        for col in cols_to_fill_robustly:
            if df_filled[col].isna().any():
                df_filled[col] = df_filled.groupby(station_col_id, group_keys=False)[col].apply(lambda x: x.ffill().bfill())
                if df_filled[col].isna().any():
                    global_median = df_filled[col].median()
                    if pd.notna(global_median):
                        df_filled[col].fillna(global_median, inplace=True)
                    else:
                        df_filled[col].fillna(0, inplace=True)
    
    final_nans_robust = df_filled[cols_to_fill_robustly].isna().sum().sum()
    return df_filled

def split_data_by_time(data_df, initial_mask_df, train_ratio=0.8, val_ratio=0.1):
    if 'utc_time' not in data_df.columns:
        raise ValueError("'utc_time' column is required in data_df for sorting.")
    
    if not data_df.index.equals(initial_mask_df.index):

        print("Warning: Data and mask indices do not match. Attempting alignment via reset_index and then loc if utc_time exists in mask.")

        data_df_copy = data_df.copy()
        initial_mask_df_copy = initial_mask_df.copy()
        
        data_df_copy.reset_index(drop=True, inplace=True)
        initial_mask_df_copy.reset_index(drop=True, inplace=True)
        

        if not data_df_copy.index.equals(initial_mask_df_copy.index):
             initial_mask_df_copy = initial_mask_df_copy.reindex(data_df_copy.index)


    data_sorted = data_df.sort_values('utc_time') 
    mask_sorted = initial_mask_df.loc[data_sorted.index] 

    n = len(data_sorted)
    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    train_df = data_sorted.iloc[:train_end_idx].copy()
    val_df = data_sorted.iloc[train_end_idx:val_end_idx].copy()
    test_df = data_sorted.iloc[val_end_idx:].copy()

    train_mask_df = mask_sorted.iloc[:train_end_idx].copy()
    val_mask_df = mask_sorted.iloc[train_end_idx:val_end_idx].copy()
    test_mask_df = mask_sorted.iloc[val_end_idx:].copy()
    
    return (train_df, val_df, test_df), (train_mask_df, val_mask_df, test_mask_df)

def create_sequences_grouped_by_station(
    data_split_df, 
    initial_mask_split_df, 
    T_in, T_out,
    all_feature_cols_ordered_with_time_id,
    target_numerical_feature_cols,
    station_id_col='station_num'
):
    
    model_feature_cols = [c for c in all_feature_cols_ordered_with_time_id if c not in ['utc_time', 'station_id']]

    unique_stations = sorted(data_split_df[station_id_col].unique())
    min_samples = float('inf')
    X_per_station, y_per_station, y_mask_per_station = [], [], []

    for st in unique_stations:
        
        df_st = data_split_df[data_split_df[station_id_col] == st]
        mask_st = initial_mask_split_df.loc[df_st.index]
        n = len(df_st)
        if n < T_in + T_out:
            print(f"Station {st} records {n} < T_in+T_out ({T_in+T_out}), skip.")
            continue


        mask_list = []
        for col in model_feature_cols:
            if col in mask_st.columns:
                mask_list.append(mask_st[col].values.astype(int))
            else:
                mask_list.append(np.zeros(n, dtype=int))
        mask_matrix = np.stack(mask_list, axis=1)

        feat_matrix = df_st[model_feature_cols].values

        X_s, y_s, ym_s = [], [], []
        for i in range(n - T_in - T_out + 1):
            X_s.append(feat_matrix[i : i + T_in, :])
            y_s.append(feat_matrix[i + T_in : i + T_in + T_out, :])
            ym_s.append(1 - mask_matrix[i + T_in : i + T_in + T_out, :])

        X_per_station.append(np.array(X_s))
        y_per_station.append(np.array(y_s))
        y_mask_per_station.append(np.array(ym_s))
        min_samples = min(min_samples, len(X_s))


    if not X_per_station:
        S = len(unique_stations)
        F = len(model_feature_cols)
        return (np.empty((0, T_in, S, F)), np.empty((0, T_out, S, F)), np.empty((0, T_out, S, F)))


    if min_samples == float('inf'):
        min_samples = 0
    X_final = np.stack([x[:min_samples] for x in X_per_station], axis=2)
    y_final = np.stack([y[:min_samples] for y in y_per_station], axis=2)
    y_mask_final = np.stack([m[:min_samples] for m in y_mask_per_station], axis=2).astype(np.float32)

    return X_final, y_final, y_mask_final


def save_final_data(X_data, y_data, y_mask_data, 
                    train_mean_std_df, dataset_prefix, data_type_prefix, base_save_path,
                    context_feat=None, adj=None, save_context_adj=False):
    os.makedirs(base_save_path, exist_ok=True)
    np.save(os.path.join(base_save_path, f"X_{data_type_prefix}_{dataset_prefix}.npy"), X_data)
    np.save(os.path.join(base_save_path, f"y_{data_type_prefix}_{dataset_prefix}.npy"), y_data)
    np.save(os.path.join(base_save_path, f"y_{data_type_prefix}_mask_{dataset_prefix}.npy"), y_mask_data)
    train_mean_std_df.to_csv(os.path.join(base_save_path, f"{data_type_prefix}_mean_std_{dataset_prefix}.csv"), index=False)
    if save_context_adj:
        if context_feat is not None: np.save(os.path.join(base_save_path, "context_feat.npy"), context_feat)
        if adj is not None: np.save(os.path.join(base_save_path, "adj.npy"), adj)


class DataPreprocessor:
    def __init__(self, 
                 T_in=72, 
                 T_out=48, 
                 data_dir='./Data/Beijing'):
        self.T_in = T_in
        self.T_out = T_out
        self.aqi_file_path = data_dir + '/beijing_17_18_aq.csv'
        self.meo_file_path = data_dir + '/beijing_17_18_meo.csv'
        self.save_path = data_dir

    def run(self):
        print("--- start data preprocessing ---")

        aq_df_raw, aq_initial_mask_df = load_and_preprocess_raw_data(self.aqi_file_path)
        meo_df_raw, meo_initial_mask_df = load_and_preprocess_raw_data(self.meo_file_path)


        aq_df_aligned, aq_initial_mask_aligned, \
        meo_df_aligned, meo_initial_mask_aligned = align_datasets_temporally(
            aq_df_raw, aq_initial_mask_df, meo_df_raw, meo_initial_mask_df
        )

        aq_df_filled = fill_missing_per_station_feature(aq_df_aligned, is_aq=True, station_col_id='station_id')
        meo_df_filled = fill_missing_per_station_feature(meo_df_aligned, is_aq=False, station_col_id='station_id')


        aq_df_feat_eng, aq_all_feat_cols_time_id, aq_numerical_to_scale = engineer_features(aq_df_filled, is_aq=True)
        meo_df_feat_eng, meo_all_feat_cols_time_id, meo_numerical_to_scale = engineer_features(meo_df_filled, is_aq=False)
        

        (train_aq_df, val_aq_df, test_aq_df), \
        (train_aq_initial_mask, val_aq_initial_mask, test_aq_initial_mask) = \
            split_data_by_time(aq_df_feat_eng, aq_initial_mask_aligned)

        (train_meo_df, val_meo_df, test_meo_df), \
        (train_meo_initial_mask, val_meo_initial_mask, test_meo_initial_mask) = \
            split_data_by_time(meo_df_feat_eng, meo_initial_mask_aligned)


        aq_scaler = StandardScaler()
        norm_train_aq_df = train_aq_df.copy()
        norm_train_aq_df[aq_numerical_to_scale] = aq_scaler.fit_transform(train_aq_df[aq_numerical_to_scale])
        norm_val_aq_df = val_aq_df.copy()
        norm_val_aq_df[aq_numerical_to_scale] = aq_scaler.transform(val_aq_df[aq_numerical_to_scale])
        norm_test_aq_df = test_aq_df.copy()
        norm_test_aq_df[aq_numerical_to_scale] = aq_scaler.transform(test_aq_df[aq_numerical_to_scale])

        meo_scaler = StandardScaler()
        norm_train_meo_df = train_meo_df.copy()
        norm_train_meo_df[meo_numerical_to_scale] = meo_scaler.fit_transform(train_meo_df[meo_numerical_to_scale])
        norm_val_meo_df = val_meo_df.copy()
        norm_val_meo_df[meo_numerical_to_scale] = meo_scaler.transform(val_meo_df[meo_numerical_to_scale])
        norm_test_meo_df = test_meo_df.copy()
        norm_test_meo_df[meo_numerical_to_scale] = meo_scaler.transform(test_meo_df[meo_numerical_to_scale])
        
        aq_train_mean_std_df = pd.DataFrame({'feature': aq_numerical_to_scale, 'mean': aq_scaler.mean_, 'std': aq_scaler.scale_})
        meo_train_mean_std_df = pd.DataFrame({'feature': meo_numerical_to_scale, 'mean': meo_scaler.mean_, 'std': meo_scaler.scale_})

        X_train_aqi_full, y_train_aqi_full, y_mask_train_aqi_full = create_sequences_grouped_by_station(
            norm_train_aq_df, train_aq_initial_mask, self.T_in, self.T_out, aq_all_feat_cols_time_id, aq_numerical_to_scale)
        X_val_aqi_full, y_val_aqi_full, y_mask_val_aqi_full = create_sequences_grouped_by_station(
            norm_val_aq_df, val_aq_initial_mask, self.T_in, self.T_out, aq_all_feat_cols_time_id, aq_numerical_to_scale)
        X_test_aqi_full, y_test_aqi_full, y_mask_test_aqi_full = create_sequences_grouped_by_station(
            norm_test_aq_df, test_aq_initial_mask, self.T_in, self.T_out, aq_all_feat_cols_time_id, aq_numerical_to_scale)

        X_train_meo_full, y_train_meo_full, y_mask_train_meo_full = create_sequences_grouped_by_station(
            norm_train_meo_df, train_meo_initial_mask, self.T_in, self.T_out, meo_all_feat_cols_time_id, meo_numerical_to_scale)
        X_val_meo_full, y_val_meo_full, y_mask_val_meo_full = create_sequences_grouped_by_station(
            norm_val_meo_df, val_meo_initial_mask, self.T_in, self.T_out, meo_all_feat_cols_time_id, meo_numerical_to_scale)
        X_test_meo_full, y_test_meo_full, y_mask_test_meo_full = create_sequences_grouped_by_station(
            norm_test_meo_df, test_meo_initial_mask, self.T_in, self.T_out, meo_all_feat_cols_time_id, meo_numerical_to_scale)

        
        num_train_samples = min(X_train_aqi_full.shape[0], X_train_meo_full.shape[0])
        if num_train_samples == 0 :
            print("Warning: No common training samples can be formed for AQI and MEO. Check sequence generation.")
        X_train_aqi, y_train_aqi, y_mask_train_aqi = X_train_aqi_full[:num_train_samples], y_train_aqi_full[:num_train_samples], y_mask_train_aqi_full[:num_train_samples]
        X_train_meo, y_train_meo, y_mask_train_meo = X_train_meo_full[:num_train_samples], y_train_meo_full[:num_train_samples], y_mask_train_meo_full[:num_train_samples]

     
        num_val_samples = min(X_val_aqi_full.shape[0], X_val_meo_full.shape[0])
        if num_val_samples == 0:
            print("Warning: No common validation samples can be formed for AQI and MEO.")
        X_val_aqi, y_val_aqi, y_mask_val_aqi = X_val_aqi_full[:num_val_samples], y_val_aqi_full[:num_val_samples], y_mask_val_aqi_full[:num_val_samples]
        X_val_meo, y_val_meo, y_mask_val_meo = X_val_meo_full[:num_val_samples], y_val_meo_full[:num_val_samples], y_mask_val_meo_full[:num_val_samples]

 
        num_test_samples = min(X_test_aqi_full.shape[0], X_test_meo_full.shape[0])
        if num_test_samples == 0:
            print("Warning: No common test samples can be formed for AQI and MEO.")
        X_test_aqi, y_test_aqi, y_mask_test_aqi = X_test_aqi_full[:num_test_samples], y_test_aqi_full[:num_test_samples], y_mask_test_aqi_full[:num_test_samples]
        X_test_meo, y_test_meo, y_mask_test_meo = X_test_meo_full[:num_test_samples], y_test_meo_full[:num_test_samples], y_mask_test_meo_full[:num_test_samples]
                
        context_feat_for_saving = None 
        adj_for_saving = None      

  
        print("\n--- Saving processed data ---")
        save_final_data(X_train_aqi, y_train_aqi, y_mask_train_aqi, aq_train_mean_std_df, "train", "aqi", self.save_path, 
                        context_feat=context_feat_for_saving, adj=adj_for_saving, save_context_adj=True)
        save_final_data(X_train_meo, y_train_meo, y_mask_train_meo, meo_train_mean_std_df, "train", "meo", self.save_path)

        save_final_data(X_val_aqi, y_val_aqi, y_mask_val_aqi, aq_train_mean_std_df, "val", "aqi", self.save_path)
        save_final_data(X_val_meo, y_val_meo, y_mask_val_meo, meo_train_mean_std_df, "val", "meo", self.save_path)
        save_final_data(X_test_aqi, y_test_aqi, y_mask_test_aqi, aq_train_mean_std_df, "test", "aqi", self.save_path)
        save_final_data(X_test_meo, y_test_meo, y_mask_test_meo, meo_train_mean_std_df, "test", "meo", self.save_path)

        print(f"\nPreprocessing complete. Data saved to: {self.save_path}")


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run()