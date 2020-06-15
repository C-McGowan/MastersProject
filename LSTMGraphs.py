import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json
import math
import tensorflow as tf
import LSTMRNNClass


def create_time_steps(length):
    return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction, mean, std):
    """Plots model predictions vs the true future"""
    plt.figure()
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    plot_history = np.array(history[:, 0])
    plot_future = np.array(true_future)
    plot_prediction = np.array(prediction)

    plot_history = np.add(np.multiply(plot_history, std[0]), mean[0])
    plot_future = np.add(np.multiply(plot_future, std[0]), mean[0])
    plot_prediction = np.add(np.multiply(plot_prediction, std[0]), mean[0])

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


def undo_normalisation(x, mean, std):
    return np.add(np.multiply(x, std[0]), mean[0])


def get_predictions(step, DSstep, model, val_data_multi):
    """Gets the unnormalised predictions from the model as well as the exact data and the DarkSky forecasted value"""
    rnn_prediction = []
    exact_data = []
    forecasts = df[f"DSwindSpeed{DSstep}"]
    index = TRAIN_SPLIT + past_history
    darksky_prediction = forecasts.values[index:-future_target]
    for x, y in val_data_multi.take(5):
        for i in range(len(y)):
            exact = undo_normalisation(y[i])[step]
            exact_data.append(exact)
            rnn = undo_normalisation(model.predict(x)[i][step])
            rnn_prediction.append(rnn)
    return exact_data, rnn_prediction, darksky_prediction


def get_differences(exact, predictions):
    differences = np.subtract(predictions, exact)
    return differences


def get_absolute_error(differences):
    abs_differences = abs(differences)
    return abs_differences.mean()


def plot_differences_histogram(rnn_difference, darksky_difference):
    """Plots histogram of differences in m/s"""
    rnn_std = rnn_difference.std()
    rnn_mean = rnn_difference.mean()
    darksky_std = darksky_difference.std()
    darksky_mean = darksky_difference.mean()

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    #Plot darksky data with mean and std in legend
    ax1.hist(darksky_difference, bins=15, alpha=0.5, color="r",
             label=f"DarkSky Forecasts \n $\mu = {darksky_mean:.2f}$ \n $\sigma = {darksky_std:.2f}$",
             density=True)
    #Plot rnn predictions with mean and std in legend
    ax1.hist(rnn_difference, bins=15, alpha=0.5, color="b",
             label=f"RNN predictions \n $\mu = {rnn_mean:.2f}$ \n $\sigma = {rnn_std:.2f}$",
             density=True)
    ax1.legend()
    ax1.set_ylim(bottom=0, top=0.75)
    ax1.set_ylabel("Number Density")
    ax1.set_xlabel(r"Prediction difference $(ms^{-1})$")
    plt.show()


def plot_scatter(exact_data, rnn_predictions, darksky_predictions):
    """Plots scatter of differences in m/s"""
    rnn_differences = get_differences(exact_data, rnn_predictions)
    darksky_differences = get_differences(exact_data, darksky_predictions)

    abs_rnn = get_absolute_error(rnn_differences)
    abs_darksky = get_absolute_error(darksky_differences)

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.scatter(exact_data, darksky_predictions, color="r",
                label=f"Darksky \n $MAE = {abs_darksky:.2f}$")
    ax1.scatter(exact_data, rnn_predictions, color="b",
                label=f"RNN \n $MAE = {abs_rnn:.2f}$")
    ax1.plot([0,12], [0,12], color="k", linestyle="--")
    ax1.set_ylabel("Wind Speed Predictions $(ms^{-1})$")
    ax1.set_xlabel("Wind Speed Measurements $(ms^{-1})$")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0)
    ax1.legend()
    ax1.set_aspect("equal")
    plt.show()


def create_timesteps_array(future_target):
    """Creates an array with the value of timesteps starting at 15 minutes and then every subsequent hour
    (e.g. (15 mins, 1 hour, 2 hours, ...))"""
    array = [0]
    for i in range(int(future_target / 4)):
        array.append(4 * i - 1 + 4)
    return array


def create_predictions_csv(future_target, model):
    """Creates a csv with all the predictions for a certain model"""
    array = create_timesteps_array(future_target)
    for step in array:
        DSstep = math.floor((step + 1) / 4)
        exact_data, rnn_predictions, darksky_predictions = get_predictions(step, DSstep, model)
        dataframe = pd.DataFrame(exact_data)
        dataframe["darksky_predictions"] = darksky_predictions
        dataframe["rnn_predictions"] = rnn_predictions
        dataframe.to_csv(f"step{step}lookahead{future_target/4}.csv")


def full_MAE_comparison(lookaheads, future_target):
    """Plot of the mean absolute errors for each of the desired lookaheads for the same model"""
    timesteps = create_timesteps_array(future_target)
    minutes = np.multiply(np.add(timesteps, 1), 15)
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    for lookahead in lookaheads:
        dataframe = pd.read_csv(f"std_mae_lookahead{lookahead}.csv")
        minutes_used = minutes[:lookahead + 1]
        rnn_mae = dataframe.values[:, 1]
        rnn_std = dataframe.values[:, 2]
        darksky_mae = dataframe.values[:, 3]
        darksky_std = dataframe.values[:, 4]
        ax1.plot(minutes_used, rnn_std, color="b", label="RNN")
    ax1.plot(minutes_used, darksky_std, color="r", label="Darksky")
    ax1.legend()
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("Standard Deviation $(ms^{-1})$")
    ax1.set_xlabel("Forecast Horizon (minutes)")
    ax1.set_xticks([0, 180, 360, 480, 720, 900, 1080, 1260, 1440])
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("AllDataDecJanFeb.csv")

    TRAIN_SPLIT = 7000

    features_considered = ['Wind Speed ms', 'DSwindSpeed0', "DSwindSpeed1", "DSwindSpeed2", "DSwindSpeed3",
                           "DSwindSpeed4",
                           "DSwindSpeed5", "DSwindSpeed6", "DSwindSpeed7", "DSwindSpeed8", "DSwindSpeed9",
                           "DSwindSpeed10",
                           "DSwindSpeed11", "DSwindSpeed12", "DSwindSpeed13", "DSwindSpeed14", "DSwindSpeed15",
                           "DSwindSpeed16", "DSwindSpeed17", "DSwindSpeed18", "DSwindSpeed19", "DSwindSpeed20",
                           "DSwindSpeed21", "DSwindSpeed22", "DSwindSpeed23", "DSwindSpeed24"]

    past_history = 480
    future_target = 48
    STEP = 1

    lookahead = len(features_considered) - 2

    data = LSTMRNNClass.prepare_data(TRAIN_SPLIT, features_considered, past_history, future_target, STEP)

    multi_step_model = tf.keras.models.load_model("multivariate_multistep_eval1000_lookahead12.h5")

    training_data = data[0]
    validation_data = data[1]
    data_mean = data[2]
    data_std = data[3]
    create_predictions_csv(96, multi_step_model)
