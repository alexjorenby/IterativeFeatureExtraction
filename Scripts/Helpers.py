import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tensorflow import keras
import math



class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    print(hist)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['acc'],
             label='Train Acc')
    plt.plot(hist['epoch'], hist['val_acc'],
             label='Val Acc')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Loss')
    plt.plot(hist['epoch'], hist['val_loss'],
             label='Val Loss')
    plt.legend()

def csv_format(columns):
    s_acc = ""
    for i in range(len(columns)):
        s_acc += str(columns[i])
        if i < len(columns)-1:
            s_acc += ","
        else:
            s_acc += "\n"
    return s_acc


def combinations(start, end, features):
    acc = 0
    for i in range(start, end):
        acc += len(list(itertools.combinations([i for i in range(features)], i)))
    return acc


def loops(num_features, num_starting_set, desired_features):
    start_point = num_features - num_starting_set
    stop_point = start_point - (desired_features - num_starting_set)
    acc = 0
    while start_point > stop_point:
        acc += 35 * (start_point + 1)
        start_point -= 1
    return acc


def LIMSModelAnal(labels, predictions):
    fp = 0
    fn = 0
    ex_neg = 0
    ex_pos = 0

    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    median_pred = np.median(predictions)
    for i in range(len(labels)):
        pred = math.exp(predictions[i])
        actual = math.exp(labels[i])

        if actual > 10:
            ex_neg += 1
        else:
            ex_pos += 1
        if (actual >= 10 and pred < 10):
            fn += 1
        elif (actual < 10 and pred >= 10):
            fp += 1

    return fp, fn, ex_neg, ex_pos, mean_pred, std_pred, median_pred

def binary_model_analysis(labels, predictions):
    fp = 0
    fn = 0
    ex_neg = 0
    ex_pos = 0

    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    median_pred = np.median(predictions)
    for i in range(len(labels)):
        pred = predictions[i]
        actual = labels[i]

        if actual == 1:
            ex_neg += 1
        else:
            ex_pos += 1
        if (actual == 1 and pred < 0.5):
            fn += 1
        elif (actual == 0 and pred >= 0.5):
            fp += 1

    return fp, fn, ex_neg, ex_pos, mean_pred, std_pred, median_pred


def LIMSModelRes(labels, predictions):
    plt.scatter(labels, predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])

    plt.show()

    error = predictions - labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")

    plt.show()


def list_from_file(path):
    f = open(path)
    result = []
    for line in f:
        result.append(str(line).replace('\n',''))
    return result
