# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 01:49:14 2023

@author: lika
"""


import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from tensorflow import keras
##loaded_model = keras.models.load_model('Source_model_.h5')
loaded_model = keras.models.load_model('source_.h5')

directory_path = 'Указывается путь к папке с изображениями'

output_directory = 'Путь к папке для сохранения состязательных изображений'  

##img = Image.open('image.png')
image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
label = [5]
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
epsilon = 0.
result = 0

def create_adversarial_pattern(input_image, input_label, loaded_model, loss_object):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = loaded_model(input_image)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

for image_file in image_files:
    img_path = os.path.join(directory_path, image_file)
    #adv_img_path = os.path.join(output_directory, image_file)
    img = Image.open(img_path)

    # Преобразование изображения в массив и нормализация
    img_array = image.img_to_array(img.resize((32, 32)))
    img_array = img_array / 255.0

    # Расширение размерности для создания адверсариального искажения
    expanded_img_array = tf.expand_dims(img_array, 0)

    perturbations = create_adversarial_pattern(expanded_img_array, label, loaded_model, loss_object)
    adv_img = img_array + epsilon * perturbations[0]

    # Обрезка значений изображения, чтобы они оставались в допустимом диапазоне [0,1]
    adv_img = tf.clip_by_value(adv_img, 0, 1)

    # Расширение размерности для совместимости с моделью
    adv_img = tf.expand_dims(adv_img, 0)
    predictions = loaded_model.predict(adv_img)
    predicted_class = tf.argmax(predictions[0])
    adv_img_pil = image.array_to_img(adv_img[0])
    #adv_img_pil.save(adv_img_path)
    if int(predicted_class.numpy()) == 5:
        result +=1
       ## print(result)
    else:
        result = result
       
    print(f"Image: {image_file} | Predicted class: {predicted_class.numpy()}")

if result != 0:
    print(result/201 * 100)
else:
    print(0)