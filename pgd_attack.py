# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:55:21 2023

@author: lika
"""


import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim
loaded_model = keras.models.load_model('source_.h5')
directory_path = 'Путь к папке с изображениями'
output_directory = 'Путь к папке, в которой сохраняются состязательные изображения'

image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Функция для генерации адверсарных примеров с использованием PGD атаки
def pgd_attack(model, image, epsilon, alpha, num_iter, target_class):
    adv_image = tf.identity(image)  # Копирование исходного изображения
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            loss = tf.keras.losses.categorical_crossentropy(target_class, prediction)
        gradient = tape.gradient(loss, adv_image)
        signed_grad = tf.sign(gradient)
        adv_image = adv_image + alpha * signed_grad
        adv_image = tf.clip_by_value(adv_image, image - epsilon, image + epsilon)
        adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image

# Параметры атаки
epsilon = 0.01
alpha = 0.001
num_iter = 10

result = 0
for image_file in image_files:
    img_path = os.path.join(directory_path, image_file)
    img = Image.open(img_path)
    img_array = image.img_to_array(img.resize((32, 32))) / 255.0
    image_tensor = tf.convert_to_tensor([img_array], dtype=tf.float32)
    target_class = tf.convert_to_tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    adv_image = pgd_attack(loaded_model, image_tensor, epsilon, alpha, num_iter, target_class)
    predictions = loaded_model.predict(adv_image)
    predicted_class = tf.argmax(predictions[0])
    adv_img_path = os.path.join(output_directory, image_file)
    adv_img_pil = image.array_to_img(adv_image[0])
    adv_img_pil.save(adv_img_path)
    if int(predicted_class.numpy()) == 5:
        result +=1
#print(f"Image: {image_file} | Predicted class: {predicted_class.numpy()}")
if result != 0:
    print(result/201 * 100)
else:
    print(0)
##print(f"Predicted class: {predicted_class.numpy()}")
