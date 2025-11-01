# Make sure that you have all these libaries available to run the code successfully
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import sys
import subprocess
from pathlib import Path

class DataGeneratorSeq(object):

    def __init__(self,prices,batch_size,num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length //self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):

        batch_data = np.zeros((self._batch_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size),dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length:
                #self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0,(b+1)*self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b]= self._prices[self._cursor[b]+np.random.randint(0,5)]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length

        return batch_data,batch_labels

    def unroll_batches(self):

        unroll_data,unroll_labels = [],[]
        init_data, init_label = None,None
        for ui in range(self._num_unroll):

            data, labels = self.next_batch()    

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0,min((b+1)*self._segments,self._prices_length-1))

def LSTM(train_data, test_data, mid_data):
    # --- Hyperparameters ---
    D = 1
    num_unrollings = 50
    batch_size = 500
    num_nodes = [200, 200, 150]
    dropout = 0.2
    epochs = 30
    n_predict_once = 50
    test_points_seq = np.arange(11000, 12000, 50).tolist()

    # --- Data preparation ---
    print("Preparing training batches...")
    dg = DataGeneratorSeq(train_data, batch_size, num_unrollings)
    u_data, u_labels = dg.unroll_batches()

    # Convert unrolled batches to NumPy arrays suitable for Keras
    X = np.stack(u_data, axis=1).reshape(batch_size, num_unrollings, D)
    y = np.stack(u_labels, axis=1)[:, -1].reshape(batch_size, 1)

    # --- Build LSTM model ---
    model = tf.keras.Sequential([
        layers.LSTM(num_nodes[0], return_sequences=True, input_shape=(num_unrollings, D), dropout=dropout),
        layers.LSTM(num_nodes[1], return_sequences=True, dropout=dropout),
        layers.LSTM(num_nodes[2], dropout=dropout),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # --- Create validation (test) dataset ---
    print("\nPreparing validation data...")
    def create_sequences(series, num_unrollings):
        X_seq, y_seq = [], []
        for i in range(len(series) - num_unrollings):
            X_seq.append(series[i:i+num_unrollings])
            y_seq.append(series[i+num_unrollings])
        return np.array(X_seq).reshape(-1, num_unrollings, 1), np.array(y_seq).reshape(-1, 1)

    X_test, y_test = create_sequences(test_data, num_unrollings)

    # --- Train the model (with green progress bar) ---
    print("\nStarting training...")
    history = model.fit(
        X, y,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1
    )

    # --- Evaluate on test data ---
    print("\nEvaluating model performance on test data...")
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test MSE: {test_loss:.6f}")

    # --- Make predictions over test points ---
    print("\nGenerating multi-step predictions for visualization...")
    predictions_over_time = []

    for w_i in test_points_seq:
        if w_i < num_unrollings or w_i >= len(mid_data):
            continue  # skip invalid indices

        input_seq = mid_data[w_i - num_unrollings:w_i].reshape(1, num_unrollings, D)
        preds = []
        current_input = input_seq

        for _ in range(n_predict_once):
            pred = model.predict(current_input, verbose=0)
            preds.append(pred[0, 0])
            # shift window
            current_input = np.roll(current_input, -1)
            current_input[0, -1, 0] = pred

        predictions_over_time.append(np.array(preds))

    print("\nFinished all predictions successfully ✅")
    print(f"Generated {len(predictions_over_time)} prediction sequences.")

    return predictions_over_time, history, test_loss



