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

#local import
from dataacquisition import dataacquisition
from dataexploration import dataexploration
from dataprep import dataprep, save_predictions, load_predictions, folder_has_all_files
from prediction import sa_prediction, ema_prediction
from LSTMmodel import LSTM, plot_predictions
from constants import folder, required_files

def install():
    req = Path("requirements.txt")
    if not req.exists():
        print("No requirements.txt found")
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])

def main():
    #install()
    #data = dataacquisition("kaggle", "Stocks", "hpq.us.txt")
    data = dataacquisition(sys.argv[1], sys.argv[2], sys.argv[3])
    train_data, test_data, mid_data = dataprep(data)

    if len(sys.argv) > 4 and sys.argv[4] == "plot" and folder_has_all_files(folder, required_files):
        preds, preds_start_idx, best_epoch = load_predictions(folder)
        plot_predictions(preds, preds_start_idx, best_epoch, data, mid_data)
    else:
        preds, preds_start_idx, best_epoch = LSTM(train_data, test_data, mid_data)
        save_predictions(preds, preds_start_idx, best_epoch)
        plot_predictions(preds, preds_start_idx, best_epoch, data, mid_data)

if __name__ == "__main__":
    main()