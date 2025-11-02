import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class DataGeneratorSeq:
    def __init__(self, prices, batch_size, num_unroll):
        self._prices = prices
        self._prices_length = len(prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b] + 1 >= self._prices_length:
                self._cursor[b] = np.random.randint(0, (b + 1) * self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0, 5)]

            self._cursor[b] = (self._cursor[b] + 1) % self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self):
        unroll_data, unroll_labels = [], []
        for _ in range(self._num_unroll):
            data, labels = self.next_batch()
            unroll_data.append(data)
            unroll_labels.append(labels)
        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0, min((b + 1) * self._segments, self._prices_length - 1))


def LSTM(train_data, test_data, mid_data):
    D = 1
    num_unrollings = 50
    batch_size = 500
    num_nodes = [200, 200, 150]
    dropout = 0.2
    epochs = 30
    n_predict_once = 50

    # Learning rate schedule
    initial_learning_rate = 0.001
    decay_speed = 1000.0

    # Test points (ensure non-overlapping predictions)
    test_points_seq = np.arange(len(train_data), len(mid_data) - n_predict_once, n_predict_once * 2).tolist()

    # Training batches
    dg = DataGeneratorSeq(train_data, batch_size, num_unrollings)
    u_data, u_labels = dg.unroll_batches()
    X_train = np.stack(u_data, axis=1).reshape(batch_size, num_unrollings, D)
    y_train = np.stack(u_labels, axis=1)[:, -1].reshape(batch_size, 1)

    # Validation batches
    dg_test = DataGeneratorSeq(test_data, batch_size, num_unrollings)
    u_data_t, u_labels_t = dg_test.unroll_batches()
    X_val = np.stack(u_data_t, axis=1).reshape(batch_size, num_unrollings, D)
    y_val = np.stack(u_labels_t, axis=1)[:, -1].reshape(batch_size, 1)

    # Build model
    model = tf.keras.Sequential([
        layers.LSTM(num_nodes[0], return_sequences=True, input_shape=(num_unrollings, D), dropout=dropout),
        layers.LSTM(num_nodes[1], return_sequences=True, dropout=dropout),
        layers.LSTM(num_nodes[2], dropout=dropout),
        layers.Dense(1)
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_speed,
        decay_rate=0.5,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')

    preds_by_epoch = []
    history_val_loss = []

    for ep in range(epochs):
        # Train for one epoch
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=1, batch_size=batch_size, shuffle=True, verbose=1)
        val_loss = hist.history.get('val_loss', [None])[-1]
        history_val_loss.append(val_loss)

        # Predictions
        preds_for_epoch = []
        for w_i in test_points_seq:
            # Warm-up sequence: last known true prices
            input_seq = mid_data[w_i - num_unrollings:w_i].reshape(1, num_unrollings, D)
            current_input = input_seq.copy()
            preds = []
            for _ in range(n_predict_once):
                pred = model.predict(current_input, verbose=0)
                preds.append(pred[0, 0])
                # Shift left and append predicted value
                current_input[:, :-1, 0] = current_input[:, 1:, 0]
                current_input[:, -1, 0] = pred
            preds_for_epoch.append(np.array(preds))
        preds_by_epoch.append(preds_for_epoch)

    best_epoch = int(np.argmin(history_val_loss))
    print(f"Best epoch (lowest val_loss): {best_epoch}, val_loss={history_val_loss[best_epoch]:.6f}")

    return preds_by_epoch, test_points_seq, best_epoch


def plot_predictions(preds_by_epoch, predictions_start_idx, best_epoch, df, all_mid_data):
    epochs = len(preds_by_epoch)
    epoch_slice = range(0, epochs)
    to_plot_epochs = [preds_by_epoch[e] for e in epoch_slice]

    plt.figure(figsize=(18, 18))

    # ---- Top subplot: evolution over epochs ----
    plt.subplot(2, 1, 1)
    plt.plot(range(len(all_mid_data)), all_mid_data, color='b')

    start_alpha = 0.25
    alpha_vals = np.linspace(start_alpha, 1.0, len(to_plot_epochs))

    for e_i, epoch_idx in enumerate(epoch_slice):
        preds_for_epoch = preds_by_epoch[epoch_idx]
        alpha = alpha_vals[e_i]
        for start_idx, preds in zip(predictions_start_idx, preds_for_epoch):
            x_vals = np.arange(start_idx - 1, start_idx - 1 + len(preds) + 1)
            y_vals = np.concatenate([[all_mid_data[start_idx - 1]], preds])
            plt.plot(x_vals, y_vals, color='r', alpha=alpha)

    plt.title('Evolution of Test Predictions Over Time', fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.xlim(len(df) - 1500, len(df))

    # ---- Bottom subplot: best epoch ----
    plt.subplot(2, 1, 2)
    plt.plot(range(len(all_mid_data)), all_mid_data, color='b')

    preds_best = preds_by_epoch[best_epoch]
    for start_idx, preds in zip(predictions_start_idx, preds_best):
        x_vals = np.arange(start_idx - 1, start_idx - 1 + len(preds) + 1)
        y_vals = np.concatenate([[all_mid_data[start_idx - 1]], preds])
        plt.plot(x_vals, y_vals, color='r')

    plt.title('Best Test Predictions Over Time', fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.xlim(len(df) - 1500, len(df))

    plt.show()
