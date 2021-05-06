# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:33:41 2021

@author: vishali jothimuthu
"""

#load lib
import os
os.chdir('C:/Users/vishali jothimuthu/Desktop/')
#MNIST cnn 8april21
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
#checking the shape of the data
train_images.shape
train_labels.shape 
#exploring the first img
first_img=train_images[0]
plt.imshow(first_img,cmap='gray')
train_labels[0]
#shaping and scaling
train_images=train_images.reshape((60000,28,28,1))
train_images.shape 
#convert to float
train_images=train_images.astype('float32')/255
#test img reshaping
tset_images=test_images.reshape((10000,28,28,1))
test_images.shape 
#test convert to float32
test_images=test_images.astype('float32')/255

#train labels
train_labels

train_labels= to_categorical(train_labels)
train_labels.shape

test_labels
test_labels= to_categorical(test_labels)
test_labels.shape

#model
from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

#compile
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
'''
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 3, 64)          36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 576)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                36928     
_________________________________________________________________
dense_3 (Dense)              (None, 10)                650       
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
_________________________________________________________________
'''

#feeding data
history=model.fit(train_images,train_labels,epochs=5,batch_size=100,validation_data=(test_images,test_labels))
'''_______________________________________

history=model.fit(train_images,train_labels,epochs=5,batch_size=100,validation_data=(test_images,test_labels))
Epoch 1/5
600/600 [==============================] - ETA: 0s - loss: 0.4864 - 
accuracy: 0.8415  

'''
model.summary()
'''

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 3, 64)          36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 576)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                36928     
_________________________________________________________________
dense_3 (Dense)              (None, 10)                650       
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
_________________________________________________________________
'''
#Saving the model
model.save('C:/Users/Dr Vinod/Desktop/WD_python/mnist.h5')

#Viewing the data stored in history
history.history.keys()
'''dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])'''

# plot loss during training
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

# plot accuracy during training
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()




















