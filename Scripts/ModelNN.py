import pandas as pd
import numpy as np
import scipy.stats as ss
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import math
import datetime
import os
from sklearn.model_selection import train_test_split


import Helpers as H


def logistic_binary_crossentropy(inputs, labels, wipe_model=True, seed_directory='', seed_id=''):
    atime = datetime.datetime.now()

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(np.array(inputs), np.array(labels), test_size=0.20)

    x = len(train_inputs[0])
    h1 = int(10000 / (x + 10))
    print(h1)

    model = keras.Sequential([
        keras.layers.Dense(300, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
        keras.layers.Dense(300, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
        keras.layers.Dense(300, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.sigmoid),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    batch_size = 10000 if len(train_inputs) > 10000 else len(train_inputs)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=15)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath='../../models/weights.{epoch:02d}.hdf5', save_weights_only=True, period=5)

    if len(seed_directory) > 1:
        seed_weights = seed_directory + '/init_weights.hdf5'
        if os.path.isfile(seed_weights + '.index'):
            model.load_weights(seed_weights)
            print("Seed Loaded")
        else:
            model.save_weights(filepath=seed_weights)
            print("Seed Created")

    history = model.fit(train_inputs, train_labels, epochs=5000, batch_size=batch_size, validation_split=0.25, callbacks=[early_stop, H.PrintDot(), checkpoint])

    # weights = model.get_weights()
    # print(weights)

    t = datetime.datetime.now() - atime
    e = len(history.epoch)
    time_per_epoch = t / e

    loss, acc = model.evaluate(test_inputs, test_labels)
    test_predictions = model.predict(test_inputs).flatten()

    # H.plot_history(history)
    # plt.show()

    best_model, best_epoch = parse_history_binary_model(model, history, test_inputs, test_labels, batch_size)

    print("BEST MODEL: " + str(best_model))
    model.load_weights(best_model)

    predictions = model.predict(test_inputs, batch_size=batch_size)

    fp, fn, ex_neg, ex_pos, mean_pred, std_pred, median_pred = H.binary_model_analysis(test_labels, predictions)
    # plt.plot(acc_acc)
    # plt.plot(pos_acc_acc)
    # plt.plot(neg_acc_acc)
    # plt.show()

    if wipe_model:
        keras.backend.clear_session()

    folder = '../../models'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    return acc, fn + fp, fn, fp, ex_neg, ex_pos, math.exp(max(test_predictions)), math.exp(min(test_predictions)), math.exp(mean_pred), math.exp(std_pred), math.exp(median_pred), best_epoch, time_per_epoch


def parse_history_binary_model(model, history, test_inputs, test_labels, batch_size):
    ep = len(pd.DataFrame(history.history)) - 1
    acc_acc = []
    pos_acc_acc = []
    neg_acc_acc = []
    best_model = ""
    best_score = -math.inf
    best_epoch = 0

    if ep > 505:
        start_ep = ep - ep % 5 - 500
    else:
        start_ep = 5

    for i in range(start_ep, ep-1, 5):
        if i == 5:
            fi = '05'
        else:
            fi = str(i)
        model_target = '../../models/weights.' + fi + '.hdf5'
        model.load_weights(model_target)

        predictions = model.predict(test_inputs, batch_size=batch_size)

        fp, fn, ex_neg, ex_pos, mean_pred, std_pred, median_pred = H.binary_model_analysis(test_labels, predictions)

        acc = 1 - ((fn + fp) / (ex_neg + ex_pos))
        pos_acc = 1 - (fp / ex_pos)
        neg_acc = 1 - (fn / ex_neg)
        total = ex_pos + ex_neg

        model_score = acc - max(ex_pos/total, ex_neg/total)
        # model_score = acc + (1 - math.fabs(pos_acc - neg_acc))

        if model_score > best_score:
            best_model = model_target
            best_score = model_score
            best_epoch = ep

        acc_acc.append(acc)
        pos_acc_acc.append(pos_acc)
        neg_acc_acc.append(neg_acc)

        # print("baseline negative")
        # print(ex_neg / (ex_pos + ex_neg))
        # print("baseline positive")
        # print(ex_pos / (ex_pos + ex_neg))
        print("baseline acc")
        print(max(ex_pos/total, ex_neg/total))
        print("acc")
        print(acc)
        # print("pos acc")
        # print(pos_acc)
        # print("neg acc")
        # print(neg_acc)

    return best_model, best_epoch



