import base64
import io
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

@tf.function
def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


@tf.function
def dice(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def predict(data):  
    global model_w

    if model_w is None:
        txt=os.path.abspath(__file__)
        x = txt.split("/", 3)
        my_path="/"+x[1]+"/"+x[2]+"/my_model"
        my_file="/"+x[1]+"/"+x[2]+"/my_model.h5"
        model_w = tf.keras.models.load_model(my_path,custom_objects={"iou": iou, "dice":dice})
        model_w.load_weights(my_file) 

    image = Image.open(data)
    image=np.array(image)        
    image = image.astype('float32')

    image = resize(image, (256, 256), anti_aliasing=True)
    image /= 255

    test_images=[image]
    test_images = np.array(test_images)

    loc = model_w.predict(test_images)

    f = plt.figure()
    ax1 = f.add_subplot(1,2, 1) 
    plt.imshow(image)

    ax2 = f.add_subplot(1,2, 2)
    plt.imshow(loc)

    ax1.title.set_text('Original image')
    ax2.title.set_text('Segmented image')

    img = Image.fromarray(f.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    a = format(img_base64.decode('utf-8'))
    obj = { 'image': str(a)}
    return json.dumps(obj)

