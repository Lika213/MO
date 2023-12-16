# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:20:31 2023

@author: lika
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Загрузка модели
loaded_model = keras.models.load_model('augment_and_regularizers_.h5')
loaded_model.summary()

# Загрузка данных
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Предварительная обработка данных
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Преобразование меток в one-hot кодировку
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

# Функция для генерации адверсарных примеров с использованием FGSM
def generate_adversarials(input_image, input_label, epsilon=0.0):
    input_image = tf.convert_to_tensor(input_image)
    input_image = tf.expand_dims(input_image, axis=0)  # Добавление размерности пакета
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = loaded_model(input_image)
        true_label = tf.one_hot(np.argmax(input_label), 10)  # Изменение формата метки
        true_label = tf.reshape(true_label, (1, 10))  # Изменение формы метки
        loss = keras.losses.mean_squared_error(true_label, prediction)  # Использование другой функции потерь
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    perturbed_image = input_image + epsilon * signed_grad
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
    perturbed_image = tf.squeeze(perturbed_image, axis=0)  # Удаление размерности пакета
    return perturbed_image

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=lambda x: generate_adversarials(x, None, epsilon=0.002)  # Использование адверсарных примеров в качестве аугментации
)

datagen.fit(train_images)

epochs = 500

for epoch in range(epochs):
    for x_aug, y_aug in datagen.flow(train_images, train_labels, batch_size=128):
        loaded_model.fit(x_aug, y_aug, epochs=3, batch_size=256)  
        break  
    
test_loss, test_acc = loaded_model.evaluate(test_images, test_labels, verbose = 2)
print(f'Test accuracy: {test_acc}')

loaded_model.save('Имя_модели.h5')
