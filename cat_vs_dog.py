# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:30:29 2023

@author: lika
"""

# import required libraries / dependencies
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import keras
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from tensorflow.keras.optimizers import Adam
# define the paths to the dataset.
training_path = '../input/cat-and-dog/training_set/training_set/'
test_path = '../input/cat-and-dog/test_set/test_set/'

# Create dataset
image_size = (200, 200)
batch_size = 32

training_set = keras.preprocessing.image.image_dataset_from_directory(
    directory=training_path,
    class_names=['cats', 'dogs'],
    image_size=image_size,
    batch_size=batch_size
)
test_set = keras.preprocessing.image.image_dataset_from_directory(
    directory=test_path,
    class_names=['cats', 'dogs'],
    image_size=image_size,
    batch_size=batch_size,
    
)

# define some layers of data augmentation
augmented_data = Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1)
])

# build the model
model = Sequential([
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(200, 200, 3)),
    
    # preprocessing
    layers.CenterCrop(180, 180),
    layers.Rescaling(scale=1./255),
    
    
    # applying image data augmentation
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    
    layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    
    layers.Flatten(),
    
    # output layer
    layers.Dense(1, activation='sigmoid')
])
epochs = 50
# callbacks (save the model at each epoch)
callbacks = [
    keras.callbacks.ModelCheckpoint("checkpoints/model_at_{epoch}.h5"),
]

# compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# fit the model
model.fit(training_set, validation_data=test_set, epochs=epochs, callbacks=callbacks, verbose=2)