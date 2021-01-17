import tensorflow as tf
import h5py
import json
import numpy as np
from PIL import Image
from skimage.transform import resize
from flask import jsonify 
import os

import matplotlib.pyplot as plt

model_w = None

def predict(data):  
    global model_w

    if model_w is None:
        txt=os.path.abspath(__file__)
        x = txt.split("/", 3)
        my_path="/"+x[1]+"/"+x[2]+"/my_model"
        my_file="/"+x[1]+"/"+x[2]+"/my_model.h5"
        model_w = tf.keras.models.load_model(my_path)
        model_w.load_weights(my_file) 

    
    IMG_SHAPE = (256,256)
    image = Image.open(data)
    image=np.array(image)
    print(image.shape)
    
    image = image.astype('float32')

    image = resize(image, IMG_SHAPE, anti_aliasing=True)
    image /= 255

    test_images=[image]
    test_images = np.array(test_images)

    Y_pred_test = model_w.predict(test_images) # Predict probability of image belonging to a class, for each class

    fig,ax = plt.subplots(1)
    fig.patch.set_visible(False)
    plt.axis('off')
    fig = plt.figure(figsize=(20, 20))

    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(image)

    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(Y_pred_test[0])

    encoded = fig_to_base64(fig)
    
    return '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
