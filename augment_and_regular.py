# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:56:59 2023

@author: lika
"""
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# LOAD THE DATA
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# PREPARE DATA FOR NEURAL NETWORK
train_images = train_images / 255.0
test_images = test_images / 255.0

Y_train = to_categorical(train_labels, num_classes=10)
X_train = train_images
X_test = test_images
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)
# PREVIEW AUGMENTED IMAGES

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

X_train3 = X_train[9,].reshape((1,32,32,3))
Y_train3 = Y_train[29,].reshape((1,10))
plt.figure(figsize=(15,4.5))
for i in range(30):  
    plt.subplot(3, 10, i+1)
    X_train2, Y_train2 = datagen.flow(X_train3,Y_train3).next()
    plt.imshow(X_train2[0].reshape((32,32,3)),cmap=plt.cm.binary)
    plt.axis('off')
    if i==9: X_train3 = X_train[762,].reshape((1,32,32,3))
    if i==19: X_train3 = X_train[18,].reshape((1,32,32,3))
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
#plt.show()


# BUILD CONVOLUTIONAL NEURAL NETWORKS

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(32, 32, 3),kernel_regularizer=keras.regularizers.L1L2()))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu',kernel_regularizer=keras.regularizers.L1L2()))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu',kernel_regularizer=keras.regularizers.L1L2()))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, activation='relu',kernel_regularizer=keras.regularizers.L1L2()))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu',kernel_regularizer=keras.regularizers.L1L2()))
model.add(tf.keras.layers.BatchNormalization())
model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu',kernel_regularizer=keras.regularizers.L1L2()))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))

model.add(Conv2D(128, kernel_size=4, activation='relu',kernel_regularizer=keras.regularizers.L1L2()))
model.add(tf.keras.layers.BatchNormalization())
model.add(Flatten())
model.add(tf.keras.layers.Dropout(0.4))
model.add(Dense(10, activation='softmax'))

# COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
##уменьшаем скорость обучения с каждой эпохой
annealer = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# TRAIN NETWORKS
epochs = 45
X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size=0.1)
history = model.fit(X_train2, Y_train2, batch_size=64, epochs=45, validation_data=(X_val2, Y_val2), callbacks=[annealer], verbose=0)

print("Epochs={0:d}, Train accuracy={1:.5f}, Validation accuracy={2:.5f}".format(
    45, max(history.history['accuracy']), max(history.history['val_accuracy'])))
model.save('augment_and_regularizers_.h5')