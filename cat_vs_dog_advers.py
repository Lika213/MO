# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:18:23 2023

@author: lika
"""
##ДООБУЧЕННАЯ МОДЕЛЬ КОТОВ И ПЕСИКОВ




import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import keras
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from tensorflow.keras.optimizers import Adam
# Load pre-trained model

#Загружаем модельку
loaded_model = keras.models.load_model('cat_vs_dog.h5')
training_path ='C:/Users/lika/Desktop/ML/newwwwwwwwwwwwwww/training_set/training_set/'
test_path = 'C:/Users/lika/Desktop/ML/newwwwwwwwwwwwwww/test_set/test_set/'

# Create dataset
image_size = (200, 200)
batch_size = 16

train_set = keras.preprocessing.image_dataset_from_directory(
    directory=training_path,
    class_names=['cats', 'dogs'],
    image_size=image_size,
    batch_size=batch_size
)
test_set = keras.preprocessing.image_dataset_from_directory(
    directory=test_path,
    class_names=['cats', 'dogs'],
    image_size=image_size,
    batch_size=batch_size,
)

# Function to create adversarial pattern
def create_adversarial_pattern(input_image, input_label, loaded_model, loss_object):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = loaded_model(input_image)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

# Function to apply adversarial training
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
epsilon = 0.1  

def adversarial_training_partial(loaded_model, images, labels, epsilon=0.15, fraction=0.5):
    perturbations = create_adversarial_pattern(images, labels, loaded_model, loss_object)
    adv_images = images + epsilon * perturbations
    adv_images = tf.clip_by_value(adv_images, 0, 1)
    mask = (tf.random.uniform(tf.shape(images)[:-1]) < fraction)[:, :, :, tf.newaxis]

    images = tf.where(mask, adv_images, images)

    return images, labels

# Applying adversarial training to the dataset
train_set_adv = train_set.map(lambda x, y: adversarial_training_partial(loaded_model, x, y, epsilon, fraction=0.5))

# Compiling the model
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model with adversarial training dataset
history = loaded_model.fit(train_set_adv, epochs=10, validation_data=test_set)


# Сохраните модель после обучения
loaded_model.save('adversarial_trained_model.h5')
