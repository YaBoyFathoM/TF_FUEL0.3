import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
import cv2
def show(im,time,title=None):
    plt.imshow(im,cmap="inferno")
    plt.title(title)
    plt.pause(time)
    plt.cla()
def reshape_lambda(input_tensor):
    height, width, channels = tf.shape(input_tensor)[1], tf.shape(input_tensor)[2], input_tensor.shape[3]
    return tf.reshape(input_tensor, (-1, height*width, channels))
def box(image):
    o=random.randint(0,1)
    r=random.randint(0,2)
    box=np.zeros_like(image[r:28-r,r:140-r])
    bxo=box.shape[0]-o
    byo=box.shape[1]-o
    box[o:o+1,o:byo]=random.randint(0,255)
    box[bxo:bxo+1,o:byo+1]=random.randint(0,255)
    box[o:bxo,o:o+1]=random.randint(0,255)
    box[o:bxo,byo:byo+1]=random.randint(0,255)
    image[r:28-r,r:140-r]+=box
    return image
def fields_generator():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    datagen = ImageDataGenerator(rotation_range = 10,
    brightness_range=[0.4,1.2],
    width_shift_range = 0.30,
    height_shift_range = 0.30,
    zoom_range=[0.8,1.7],dtype="uint8")
    datagen.fit(X_train.reshape(X_train.shape[0], 28, 28, 1))
    for X, Y in datagen.flow(X_train.reshape(X_train.shape[0], 28, 28, 1),
    Y_train.reshape(Y_train.shape[0], 1),batch_size=8,shuffle=True):
            image=np.zeros((28,140,1))
            ids=np.zeros((7,10))
            order=random.sample(range(7),7)
            bx=random.randint(0,1)
            if bx==1:
                image=box(image)
            image=np.minimum(255,image).astype(np.int32)
            for i in order:
                dig=X[i]
                contours,hierarchy = cv2.findContours(dig,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                cnt = contours[0]
                x,y,w,h = cv2.boundingRect(cnt)
                dig=dig[:,x:x+w].astype(np.int32)
                if dig.size>=100:
                    image[:,18*i:18*i+w,:]+=dig
                    image=(np.minimum(255,image)/255).astype(np.float32)
                    ids[i,:]=tf.one_hot(int(Y[i]),10,dtype="float32")
            image=tf.repeat(tf.expand_dims(image,0),7,axis=0)
            yield (image, ids), ids
