import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json
import math
import tensorflow as tf

def readData(targetFile):
    """Reads data from target file and returns as object"""
    with open(targetFile, "r") as data_file:
        data = json.load(data_file)
    return data


#Pandas dataframe was used for the neural network


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step = False):
    """Splits the data into the desired set"""
    data = []
    labels = []
    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)


def prepare_data(TRAIN_SPLIT, features_considered, past_history, future_target, STEP):
    tf.random.set_seed(13)
    features = df[features_considered]
    features.index = df["Date Time"]

    dataset = features.values

    # Normalise the values
    data_train_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_train_std = dataset[:TRAIN_SPLIT].std(axis=0)

    dataset = (dataset - data_train_mean) / data_train_std
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)

    return [(x_train_multi, y_train_multi), (x_val_multi, y_val_multi)]


def create_time_steps(length):
    return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction, data_train_std, data_train_mean):
    plt.figure()
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    plot_history = np.array(history[:, 0])
    plot_future = np.array(true_future)
    plot_prediction = np.array(prediction)

    plot_history = np.add(np.multiply(plot_history, data_train_std[0]), data_train_mean[0])
    plot_future = np.add(np.multiply(plot_future, data_train_std[0]), data_train_mean[0])
    plot_prediction = np.add(np.multiply(plot_prediction, data_train_std[0]), data_train_mean[0])

    plt.plot(num_in, np.array(plot_history), label="History")
    plt.plot(np.arange(num_out)/STEP, np.array(plot_future), "bo", label="True Future")
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(plot_prediction), "ro", label="Predicted Future")

    ticks = [-96, -84, -72, -60, -48, -36, -24, -12, 0, 12]
    plt.xticks(ticks, np.divide(ticks, 4))
    plt.ylabel("Wind Speeds $(ms^{-1})$")
    plt.xlabel("Forecast Horizon (hours)")
    plt.legend()
    plt.show()


class LSTMNetwork:

    def __init__(self, future_target, training_data, validation_data, epochs, evaluation_interval, lookahead):
        self.model = tf.keras.models.Sequential()
        self.future_target = future_target
        self.training_data = training_data
        self.validation_data = validation_data
        self.epochs = epochs
        self.evaluation_interval = evaluation_interval
        self.lookahead = lookahead


    def add_intermediate_LSTM_layer(self, nodes):
        self.model.add(tf.keras.layers.LSTM(nodes, return_sequences=True,
                                          input_shape=self.training_data[0].shape[-2:],
                                          activation="sigmoid"))


    def add_last_LSTM_layer(self, nodes):
        self.model.add(tf.keras.layers.LSTM(nodes, activation="sigmoid"))
        self.model.add(tf.keras.layers.Dense(self.future_target))


    def compile_model(self):
        self.model.compule(optimizer="Adam", loss="mae")


    def train_model(self):
        multi_step_history = self.model.fit(self.training_data, epochs=self.epochs,
                                            steps_per_epoch=self.evaluation_interval,
                                            validation_data=self.validation_data,
                                            validation_steps=50)


    def save_model(self):
        self.model.save(f"multivariate_multistep_eval{self.evaluation_interval}_lookahead{self.lookahead}.h5")



if __name__ == "__main__":
    df = pd.read_csv("AllDataDecJanFeb.csv")

    TRAIN_SPLIT = 7000

    features_considered = ['Wind Speed ms', 'DSwindSpeed0', "DSwindSpeed1", "DSwindSpeed2", "DSwindSpeed3", "DSwindSpeed4",
                           "DSwindSpeed5", "DSwindSpeed6", "DSwindSpeed7", "DSwindSpeed8", "DSwindSpeed9", "DSwindSpeed10",
                           "DSwindSpeed11", "DSwindSpeed12", "DSwindSpeed13", "DSwindSpeed14", "DSwindSpeed15",
                           "DSwindSpeed16", "DSwindSpeed17", "DSwindSpeed18", "DSwindSpeed19", "DSwindSpeed20",
                           "DSwindSpeed21", "DSwindSpeed22", "DSwindSpeed23", "DSwindSpeed24"]

    past_history = 480
    future_target = 48
    STEP = 1

    lookahead = len(features_considered)-2

    data = prepare_data(TRAIN_SPLIT, features_considered, past_history, future_target, STEP)

    training_data = data[0]
    validation_data = data[1]

    multivariate_rnn = LSTMNetwork(future_target, training_data, validation_data,
                                   epochs=10, evaluation_interval=1000, lookahead=lookahead)
    multivariate_rnn.add_intermediate_LSTM_layer(32)
    multivariate_rnn.add_last_LSTM_layer(16)
    multivariate_rnn.compile_model()
    multivariate_rnn.train_model()
    multivariate_rnn.save_model()


