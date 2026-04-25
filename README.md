# DiS-MoE: Spatial-Temporal Distribution Shift Aware Mixture-of-Experts Model for Urban Atmospheric Forecasting

This is the official PyTorch implementation for the paper "DiS-MoE: Spatial-Temporal Distribution Shift Aware Mixture-of-Experts Model for Urban Atmospheric Forecasting".

This paper proposes a Spatio-Temporal Distribution Shift aware Mixture-of-Experts model (DiS-MoE) to address challenging Spatial-Temporal Distribution shift problems in urban atmospheric forecasting.

## Table of Contents
- [DiS-MoE: Spatial-Temporal Distribution Shift Aware Mixture-of-Experts Model for Urban Atmospheric Forecasting](#dis-moe-spatial-temporal-distribution-shift-aware-mixture-of-experts-model-for-urban-atmospheric-forecasting)
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
