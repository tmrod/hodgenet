import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# function to make 1D CNN classifier
def cnn_model(layer_depths, kernel_sizes, dropout_rate, learning_rate, num_classes):
    model = keras.Sequential()

    for l, k in zip(layer_depths, kernel_sizes):
        model.add(keras.layers.Conv1D(filters=l, kernel_size=k, strides=1,
                                      padding='same', activation=keras.activations.relu))
        model.add(keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.train.AdamOptimizer(learning_rate),
                  metrics=['accuracy'])

    return model

# callback for model to record accuracy
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []
        
    def on_epoch_end(self, batch, logs={}):
        self.train_acc.append(logs.get('acc'))
        self.train_loss.append(logs.get('loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.val_loss.append(logs.get('val_loss'))

