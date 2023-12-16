# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:45:54 2023

@author: lika
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

loaded_model = keras.models.load_model('model_name.h5')
loaded_model.summary()

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Преобразование меток в one-hot кодировку
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

# Функция для генерации адверсарных примеров с использованием PGD
def generate_adversarials_pgd(input_image, input_label, epsilon=0.01, alpha=0.001, num_iter=40):
    perturbed_image = tf.expand_dims(input_image, 0)  # Добавление измерения пакета
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(perturbed_image)
            prediction = loaded_model(perturbed_image)
            true_label = tf.one_hot(np.argmax(input_label), 10)  # Преобразование формата метки
            true_label = tf.reshape(true_label, (1, 10))  # Изменение формы метки
            loss = keras.losses.mean_squared_error(true_label, prediction)  # Использование другой функции потерь
        gradient = tape.gradient(loss, perturbed_image)
        signed_grad = tf.sign(gradient)
        perturbed_image = perturbed_image + alpha * signed_grad
        perturbed_image = tf.clip_by_value(perturbed_image, input_image - epsilon, input_image + epsilon)
        perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
    perturbed_image = tf.squeeze(perturbed_image, 0)  
    return perturbed_image

datagen_pgd = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=lambda x: generate_adversarials_pgd(x, None, epsilon=0.01, alpha=0.001, num_iter=40)  # Использование адверсарных примеров в качестве аугментации
)

datagen_pgd.fit(train_images)

epochs = 50

for epoch in range(epochs):
    for x_aug, y_aug in datagen_pgd.flow(train_images, train_labels, batch_size=128):
        loaded_model.fit(x_aug, y_aug, epochs=1, batch_size=256)  
        break 
test_loss_pgd, test_acc_pgd = loaded_model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy (PGD): {test_acc_pgd}')

loaded_model.save('target_pgd.h5')
