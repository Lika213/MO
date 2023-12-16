# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:47:08 2023

@author: lika
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds

(train_images, train_labels), (test_images, test_labels) = tfds.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

Y_train = to_categorical(train_labels, num_classes=10)
X_train = train_images
X_test = test_images

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))

model.add(Conv2D(128, kernel_size=4, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Flatten())
model.add(tf.keras.layers.Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


annealer = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

epochs = 45
X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size=0.1)
history = model.fit(X_train2, Y_train2, batch_size=64, epochs=45, validation_data=(X_val2, Y_val2), callbacks=[annealer], verbose=0)

print("Epochs={0:d}, Train accuracy={1:.5f}, Validation accuracy={2:.5f}".format(
    45, max(history.history['accuracy']), max(history.history['val_accuracy'])))
model.save('source_.h5')