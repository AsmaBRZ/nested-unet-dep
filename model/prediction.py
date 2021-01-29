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
import typing
model_w = None


def weighted_loss(original_loss_function: typing.Callable, weights_list: dict) -> typing.Callable:
    def loss_function(true, pred):
        class_selectors = tf.cast(K.argmax(true, axis=-1), tf.int32)
        class_selectors = [K.equal(i, class_selectors) for i in range(len(weights_list))]
        class_selectors = [K.cast(x, K.floatx()) for x in class_selectors]
        weights = [sel * w for sel, w in zip(class_selectors, weights_list)]
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]
        loss = original_loss_function(true, pred)
        loss = loss * weight_multiplier
        return loss
    return loss_function


@tf.function
def loss(y_true, y_pred, smooth=1, cat_weight=1, iou_weight=1, dice_weight=1):
    return cat_weight * K.categorical_crossentropy(y_true, y_pred) \
           + iou_weight * log_iou(y_true, y_pred, smooth) \
           + dice_weight * log_dice(y_true, y_pred, smooth)

@tf.function
def log_iou(y_true, y_pred, smooth=1):
    return - K.log(iou(y_true, y_pred, smooth))


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

weights = {0.0: 1.0, 1.0: 6.063862886734454, 2.0: 72.90944265835127}
def predict(data):  
    print("IN DATA")
    global model_w

    if model_w is None:
        
        txt=os.path.abspath(__file__)
        x = txt.split("/", 3)
        my_path="/"+x[1]+"/"+x[2]+"/my_model"
        my_file="/"+x[1]+"/"+x[2]+"/my_model.h5"
        print("before load model")
        model_w = tf.keras.models.load_model(my_path,custom_objects={"iou": iou, "dice":dice,"loss_function":weighted_loss(loss,weights)})
        print("after load model")
        #model_w.load_weights(my_file) 
    
    print("before open image")
    image = Image.open(data)
    image=np.array(image)        
    image = image.astype('float32')

    image = resize(image, (256, 256), anti_aliasing=True)
    image /= 255

    test_images=[image]
    test_images = np.array(test_images)
    print("before predict")
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

