# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 01:43:58 2023

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
loaded_model = keras.models.load_model('Название тестируемой модели.h5')
loaded_model.summary()
directory_path = 'Путь до папки с состязательными изображениями'


image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]


result = 0


for image_file in image_files:
    img_path = os.path.join(directory_path, image_file)
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)  
    img_array = img_array / 255.0  
    img_tensor = tf.expand_dims(img_array, axis=0)

    predictions = loaded_model.predict(img_tensor)
    predicted_class = tf.argmax(predictions[0])

    if int(predicted_class.numpy()) == 5:
        result +=1
     
    print(f"Image: {image_file} | Predicted class: {predicted_class.numpy()}")

if result != 0:
    print(result/201 * 100)


