# A Spatial-Temporal Distribution Shift Aware Mixture-of-Experts Model for Urban Atmospheric Forecasting

This is the official PyTorch implementation for the ICDM 2026 paper "A Spatial-Temporal Distribution Shift Aware Mixture-of-Experts Model for Urban Atmospheric Forecasting".

This paper proposes a Spatio-Temporal Distribution Shift aware Mixture-of-Experts model (DiS-MoE) to address challenging Spatial-Temporal Distribution shift problems in urban atmospheric forecasting.

## Table of Contents
- [A Spatial-Temporal Distribution Shift Aware Mixture-of-Experts Model for Urban Atmospheric Forecasting](#a-spatial-temporal-distribution-shift-aware-mixture-of-experts-model-for-urban-atmospheric-forecasting)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Dataset](#dataset)
  - [How to Run](#how-to-run)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Training the Model](#2-training-the-model)
    - [3. Testing the Model](#3-testing-the-model)
  - [File Structure](#file-structure)
  - [Key Arguments](#key-arguments)

## Requirements
You can install all the necessary dependencies using the following command:
```bash
pip install -r requirements.txt
```
This project has been tested on **Python 3.9+**. The main dependencies and their versions are listed below:
- `numpy==2.3.1`
- `pandas==1.5.3`
- `scikit_learn==1.2.2`
- `torch==2.2.2`

## Dataset
1.  **Data Source**: This demo uses the dataset from the [KDD Cup 2018](https://www.biendata.xyz/competition/kdd_2018/), which contains public Air Quality (AQI) and Meteorological (MEO) data from the Beijing area. Other datasets used in our paper can be found at [here](http://urban-computing.com/data/Data-1.zip)
2.  **File Preparation**: Please place the dataset files `beijing_17_18_aq.csv` and `beijing_17_18_meo.csv` into the `./Data/Beijing/` directory.

In addition, this project requires an adjacency matrix, **`adj.npy`**, with a shape of `(N, N)`, and an urban context feature matrix, **`context_feat.npy`**, with a shape of `(N, D_context)`. Here, `N` is the total number of stations, and `D_context` is the dimension of the context features. You will need to prepare these two files yourself if you change the dataset.

For the construction of the adjacency matrix, you can refer to the methodology in [MasterGNN](https://arxiv.org/pdf/2012.15037). The urban context features can be constructed using Point of Interest (POI) and road network information surrounding each station. Both files should also be placed in the `./Data/Beijing/` directory.

The final expected file structure is as follows:

    The expected file structure is as follows:
    ```
    .
    ├── Data/
    │   └── Beijing/
    │       ├── beijing_17_18_aq.csv
    │       ├── beijing_17_18_meo.csv
    │       ├── adj.npy                # Adjacency matrix, shape (N, N), provided
    │       └── context_feat.npy       # Urban context features, shape (N, D_context), provided
    ├── DiSMoE.py
    ├── main.py
    └── requirements.txt
    ```

## How to Run

The entire process is divided into three steps: data preprocessing, model training, and model testing.

### 1. Data Preprocessing
The code will automatically preprocess the raw data by calling "preprocessor.run()". This includes cleaning, feature engineering, temporal alignment, and converting the data into the sequence format (`.npy` files) required by the model.

The `data_preprocessor.py` script is automatically called by `main.py`. It will:
- Load the raw AQI and MEO CSV files.
- Align the timestamps of the two datasets.
- Impute missing values.
- Create temporal features (e.g., hour of day, day of week).
- Normalize numerical features and split the data into training, validation, and test sets.
- Generate `X_*.npy` and `y_*.npy` files and save them to the `./Data/Beijing/` directory.

**Note**: The `preprocessor.run()` function is called by default at the beginning of `main.py`. This step is essential on your **first run**, or anytime you **change the input/output sequence lengths** (e.g., `T_in`, `T_out`), as this will regenerate the `.npy` files with the correct dimensions. **To skip preprocessing** for subsequent runs with the same parameters, you can comment out the `preprocessor.run()` line in `main.py` to save time.

### 2. Training the Model
Use the following command to train the model from scratch. During training, the model is validated, and the best-performing model is automatically saved as `best_generator.pth`.

```bash
python main.py --data_path ./Data/Beijing --save_path ./result/demo.json
```

- Training logs will be printed to the console in real-time.
- After training is complete, the final test results and hyperparameter configuration will be saved in JSON format to the file specified by `--save_path` (defaults to `./result/demo.json`).

### 3. Testing the Model
If you already have a trained `best_generator.pth` file, you can use the `--test_only` argument to skip the training phase and directly evaluate the model's performance on the test set.

```bash
python main.py --data_path ./Data/Beijing --save_path ./result/test_results.json --test_only True
```

This command will load `best_generator.pth`, run evaluation on the test set, and save the final MAE and sMAPE metrics to the `test_results.json` file.

## File Structure
```
.
├── Data/
│   └── Beijing/
│       ├── beijing_17_18_aq.csv   # Raw air quality data
│       └── beijing_17_18_meo.csv  # Raw meteorological data
│       ├── adj.npy                # Adjacency matrix, shape (N, N), provided
│       └── context_feat.npy       # Urban context features, shape (N, D_context), provided
│       └── *.npy                  # Data files generated after preprocessing
├── result/
│   └── demo.json                  # JSON file to store training/testing results
├── DiSMoE.py                      # Core implementation of the DiS-MoE model
├── data_preprocessor.py           # Data preprocessing script
├── data_loader.py                 # Data loader for training and evaluation
├── main.py                        # Main script for training and testing
└── requirements.txt               # Project dependencies
```

## Key Arguments

You can adjust the model configuration and training process via command-line arguments. Here are some of the key parameters:

- `--data_path`: Directory where the dataset is located.
- `--save_path`: Path to the JSON file for saving results.
- `--hidden_dim`: The size of hidden dimensions in the model.
- `--num_experts`: The number of experts in the DiS-MoE model.
- `--num_heads`: Number of heads in the router's multi-head attention.
- `--topk`: The frequencies with top-K largest amplitude that used to reconstruct the high-energy non-stationary component.
- `--balance_loss_alpha`: The weight of balance loss.
- `--diversity_loss_alpha`: The weight of diversity loss.
- `--learning_rate`: The learning rate for the optimizer.
- `--batch_size`: The batch size for training and evaluation.
- `--epochs`: The total number of epochs for training.
- `--patience`: The number of patience epochs for Early Stopping.
- `--test_only`: Boolean. If `True`, skips the training phase and runs testing only.

For more parameters and their default values, please refer to the `main.py` file.

## Results Correction

A table transcription error affected four entries for Corrformer on Beijing: the MAE and SMAPE values for humidity and wind speed. These entries duplicated Corrformer's Shenzhen AQI and temperature results. The corrected values below are restored from the earlier experiment-result tables. This correction does not involve rerunning experiments; all other reported values remain unchanged.

### Corrected Entries

All entries below are for Corrformer on Beijing and are reported as mean ± standard deviation.

| Target | Metric | Originally reported | Corrected |
| --- | --- | ---: | ---: |
| Humidity | MAE | 16.010 ± 2.724 | 19.128 ± 2.520 |
| Humidity | SMAPE | 0.610 ± 0.143 | 0.488 ± 0.042 |
| Wind speed | MAE | 4.961 ± 0.938 | 1.166 ± 0.036 |
| Wind speed | SMAPE | 0.204 ± 0.032 | 0.617 ± 0.014 |

### Full Corrected Results

Results are reported as mean ± standard deviation over three runs. Lower values are better for both MAE and SMAPE. **Bold** indicates the lowest mean in each column.

#### Beijing (BJ)

| Model | AQI MAE ↓ | AQI SMAPE ↓ | Temperature MAE ↓ | Temperature SMAPE ↓ | Humidity MAE ↓ | Humidity SMAPE ↓ | Wind speed MAE ↓ | Wind speed SMAPE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DCRNN | 35.591 ± 2.463 | 0.880 ± 0.027 | 18.778 ± 4.931 | 1.880 ± 0.032 | 22.932 ± 4.176 | 0.558 ± 0.065 | 1.194 ± 0.005 | 0.610 ± 0.001 |
| STGCN | 40.905 ± 0.013 | 0.922 ± 0.000 | 20.889 ± 0.010 | 1.901 ± 0.000 | 22.566 ± 0.056 | 0.562 ± 0.001 | 1.143 ± 0.000 | 0.591 ± 0.000 |
| GMAN | 35.183 ± 4.033 | 0.896 ± 0.026 | 4.764 ± 0.215 | 1.291 ± 0.039 | 14.062 ± 0.559 | 0.425 ± 0.052 | 1.194 ± 0.038 | 0.607 ± 0.013 |
| GTS | 40.237 ± 1.804 | 0.929 ± 0.016 | 19.404 ± 1.355 | 1.893 ± 0.010 | 19.658 ± 2.649 | 0.508 ± 0.048 | 1.189 ± 0.037 | 0.608 ± 0.011 |
| HimNet | 34.434 ± 2.899 | 0.858 ± 0.030 | 6.402 ± 0.342 | 1.586 ± 0.108 | 15.031 ± 1.498 | 0.413 ± 0.031 | 1.161 ± 0.054 | 0.592 ± 0.021 |
| ARIMA | 70.921 ± 0.000 | 1.057 ± 0.000 | 6.544 ± 0.000 | 1.554 ± 0.000 | 19.175 ± 0.000 | 0.586 ± 0.000 | 1.226 ± 0.000 | 0.623 ± 0.000 |
| LR | 38.466 ± 1.825 | 0.933 ± 0.003 | 5.304 ± 0.051 | 1.322 ± 0.022 | 16.518 ± 0.063 | 0.457 ± 0.014 | 1.246 ± 0.027 | 0.632 ± 0.005 |
| DLinear | 37.743 ± 0.379 | 0.908 ± 0.006 | 4.600 ± 0.179 | 1.165 ± 0.035 | 13.728 ± 0.204 | 0.390 ± 0.004 | 1.148 ± 0.013 | 0.591 ± 0.005 |
| PatchTST | 36.785 ± 0.475 | 0.873 ± 0.004 | 4.183 ± 0.127 | 1.111 ± 0.019 | 12.999 ± 0.275 | 0.390 ± 0.009 | 1.185 ± 0.013 | 0.609 ± 0.006 |
| AirFormer | 42.738 ± 2.317 | 0.928 ± 0.021 | 11.447 ± 0.870 | 1.812 ± 0.011 | 17.085 ± 0.161 | 0.462 ± 0.001 | **1.113 ± 0.008** | 0.574 ± 0.003 |
| Corrformer | 40.245 ± 5.242 | 0.925 ± 0.051 | 12.580 ± 1.884 | 1.791 ± 0.039 | 19.128 ± 2.520 | 0.488 ± 0.042 | 1.166 ± 0.036 | 0.617 ± 0.014 |
| MasterGNN | 45.402 ± 14.592 | 0.925 ± 0.117 | 16.617 ± 8.744 | 1.713 ± 0.336 | 20.743 ± 9.032 | 0.562 ± 0.125 | 1.119 ± 0.090 | 0.586 ± 0.029 |
| DiS-MoE | **33.909 ± 0.655** | **0.825 ± 0.006** | **2.698 ± 0.183** | **0.792 ± 0.043** | **11.836 ± 0.168** | **0.365 ± 0.009** | 1.119 ± 0.010 | **0.573 ± 0.002** |

#### Shenzhen (SZ)

| Model | AQI MAE ↓ | AQI SMAPE ↓ | Temperature MAE ↓ | Temperature SMAPE ↓ | Humidity MAE ↓ | Humidity SMAPE ↓ | Wind speed MAE ↓ | Wind speed SMAPE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DCRNN | 13.612 ± 2.395 | 0.459 ± 0.059 | 3.238 ± 0.121 | 0.148 ± 0.005 | 11.800 ± 0.302 | 0.165 ± 0.005 | 0.997 ± 0.034 | 0.506 ± 0.011 |
| STGCN | 12.765 ± 0.071 | 0.439 ± 0.002 | 3.117 ± 0.025 | 0.143 ± 0.001 | 11.444 ± 0.050 | 0.161 ± 0.001 | 0.869 ± 0.010 | 0.450 ± 0.005 |
| GMAN | 14.643 ± 0.903 | 0.486 ± 0.021 | 3.372 ± 0.116 | 0.153 ± 0.004 | 11.272 ± 0.395 | 0.159 ± 0.005 | 0.988 ± 0.071 | 0.519 ± 0.053 |
| GTS | 12.135 ± 1.456 | 0.421 ± 0.040 | 3.238 ± 0.112 | 0.148 ± 0.004 | 11.610 ± 0.150 | 0.163 ± 0.002 | 0.971 ± 0.033 | 0.498 ± 0.010 |
| HimNet | 12.178 ± 1.120 | 0.398 ± 0.024 | 3.062 ± 0.141 | 0.143 ± 0.007 | 9.319 ± 0.199 | 0.134 ± 0.004 | 0.858 ± 0.014 | 0.433 ± 0.005 |
| ARIMA | 13.146 ± 0.000 | 0.385 ± 0.000 | 2.173 ± 0.000 | 0.104 ± 0.000 | 10.688 ± 0.000 | 0.153 ± 0.000 | 0.931 ± 0.000 | 0.483 ± 0.000 |
| LR | 11.733 ± 0.187 | 0.403 ± 0.010 | 2.791 ± 0.013 | 0.128 ± 0.001 | 11.050 ± 0.147 | 0.156 ± 0.002 | 0.869 ± 0.001 | 0.450 ± 0.001 |
| DLinear | 12.587 ± 0.124 | 0.437 ± 0.011 | 2.601 ± 0.226 | 0.121 ± 0.010 | 9.250 ± 0.372 | 0.133 ± 0.004 | 0.847 ± 0.022 | 0.438 ± 0.011 |
| PatchTST | 12.228 ± 0.220 | 0.419 ± 0.006 | 2.254 ± 0.230 | 0.105 ± 0.010 | 9.010 ± 0.090 | 0.131 ± 0.001 | 0.830 ± 0.012 | 0.424 ± 0.005 |
| AirFormer | 11.854 ± 0.520 | 0.403 ± 0.018 | 3.085 ± 0.283 | 0.140 ± 0.013 | 10.927 ± 0.716 | 0.155 ± 0.009 | 0.898 ± 0.042 | 0.468 ± 0.022 |
| Corrformer | 16.010 ± 2.724 | 0.610 ± 0.143 | 4.961 ± 0.938 | 0.204 ± 0.032 | 12.417 ± 0.819 | 0.170 ± 0.009 | 0.895 ± 0.028 | 0.476 ± 0.010 |
| MasterGNN | 13.798 ± 2.574 | 0.454 ± 0.060 | 3.207 ± 0.147 | 0.147 ± 0.006 | 11.410 ± 0.290 | 0.161 ± 0.003 | 0.946 ± 0.020 | 0.495 ± 0.015 |
| DiS-MoE | **10.500 ± 0.513** | **0.368 ± 0.019** | **2.145 ± 0.021** | **0.100 ± 0.001** | **8.393 ± 0.087** | **0.122 ± 0.001** | **0.784 ± 0.016** | **0.410 ± 0.007** |
