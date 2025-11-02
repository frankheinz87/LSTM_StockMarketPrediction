# Make sure that you have all these libaries available to run the code successfully
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
import sys
import subprocess
from pathlib import Path
import json



def dataprep(df):

    # First calculate the mid prices from the highest and lowest
    #high_prices = df.loc[:,'High'].to_numpy()
    #low_prices = df.loc[:,'Low'].to_numpy()
    mid_prices = (df["High"]+df["Low"])/2.0
    mid_prices = mid_prices.to_numpy()

    train_data = mid_prices[:11000]
    test_data = mid_prices[11000:]

    # Scale the data to be between 0 and 1
    # When scaling remember! You normalize both test and train data with respect to training data
    # Because you are not supposed to have access to test data
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)

    # Train the Scaler with training data and smooth data
    smoothing_window_size = 2500
    for di in range(0,10000,smoothing_window_size):
        scaler.fit(train_data[di:di+smoothing_window_size,:])
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

    # You normalize the last bit of remaining data
    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

    # Reshape both train and test data
    train_data = train_data.reshape(-1)

    # Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)

    # Now perform exponential moving average smoothing
    # So the data will have a smoother curve than the original ragged data
    EMA = 0.0
    gamma = 0.1
    for ti in range(11000):
        EMA = gamma*train_data[ti] + (1-gamma)*EMA
        train_data[ti] = EMA

    # Used for visualization and test purposes
    all_mid_data = np.concatenate([train_data,test_data],axis=0)
    return train_data, test_data, all_mid_data

def save_predictions(preds_by_epoch, predictions_start_idx, best_epoch, folder="saved_preds"):
    import os
    os.makedirs(folder, exist_ok=True)

    # Save best_epoch as JSON
    with open(f"{folder}/best_epoch.json", "w") as f:
        json.dump({"best_epoch": best_epoch}, f)

    # Save predictions_start_idx as CSV
    pd.DataFrame({"start_idx": predictions_start_idx}).to_csv(f"{folder}/predictions_start_idx.csv", index=False)

    # Save preds_by_epoch
    # Flatten to a DataFrame: columns = epoch, window, time_step, prediction
    rows = []
    for ep_idx, epoch_preds in enumerate(preds_by_epoch):
        for win_idx, window_preds in enumerate(epoch_preds):
            for t_idx, pred in enumerate(window_preds):
                rows.append([ep_idx, win_idx, t_idx, pred])
    df_preds = pd.DataFrame(rows, columns=["epoch", "window", "time_step", "prediction"])
    df_preds.to_csv(f"{folder}/preds_by_epoch.csv", index=False)
    print(f"Predictions saved to folder '{folder}'")

def load_predictions(folder="saved_preds"):

    # Load best_epoch
    with open(f"{folder}/best_epoch.json", "r") as f:
        best_epoch = json.load(f)["best_epoch"]

    # Load start indices
    df_start = pd.read_csv(f"{folder}/predictions_start_idx.csv")
    predictions_start_idx = df_start["start_idx"].tolist()

    # Load preds_by_epoch
    df_preds = pd.read_csv(f"{folder}/preds_by_epoch.csv")
    # Reconstruct preds_by_epoch as list of lists of arrays
    preds_by_epoch = []
    for ep_idx in sorted(df_preds["epoch"].unique()):
        epoch_preds = []
        df_ep = df_preds[df_preds["epoch"] == ep_idx]
        for win_idx in sorted(df_ep["window"].unique()):
            window_preds = df_ep[df_ep["window"] == win_idx].sort_values("time_step")["prediction"].values
            epoch_preds.append(np.array(window_preds))
        preds_by_epoch.append(epoch_preds)

    return preds_by_epoch, predictions_start_idx, best_epoch



def folder_has_all_files(folder, required_files):
    if not os.path.exists(folder):
        return False
    for f in required_files:
        if not os.path.isfile(os.path.join(folder, f)):
            print(f"Warning: Required file '{f}' is missing in '{folder}'. Will retrain LSTM.")
            return False
    if not any(os.scandir(folder)):
        print(f"Warning: Folder '{folder}' is empty. Will retrain LSTM.")
        return False
    return True





