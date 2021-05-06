# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:39:50 2021

@author: vishali jothimuthu
"""
import os
os.chdir('C:/Users/vishali jothimuthu/Desktop/data/Breast Cancer Prediction/Dataset')
import pandas  as pd 
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_column',None)
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
 
#load data
bc=pd.read_csv('wisconsin_breast_cancer.csv')
bc.info()
bc.shape
type(bc)
bc.isnull().sum()
#removing missing values
bc=bc.dropna()
bc.info()
#predictors and response variables
#Assigning predictors and response variables
x=bc.iloc[:,1:10]
x.info()
x.shape


y = bc['class']
y.shape
y.dtype
#split dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)
#standardizing the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#standardizing the x_train data
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_train
x_train.shape
#standardizing the x_test data
scaler.fit(x_test)
x_test=scaler.transform(x_test)
x_test
x_test.shape
#_________DEEP LEARNING-sequential model
from keras.models import Sequential
from keras.layers import Dense
#processing the data
#y_train[y_train=='malignant']=1
#y_train[y_train=='benign']=0
y_train=np.asarray(y_train).astype('float32')
x_train=np.asarray(y_train).astype('float32')
x_train.shape
y_train.shape
#processing test  data
#y_test[y_test=='malignant']=1
#y_test[y_test=='benign']=0
y_test=np.asarray(y_test).astype('float32')
x_test=np.asarray(y_test).astype('float32')
x_test.shape
y_test.shape
#MODEL
#___Model with hidden layer activation_relu & outplayer_sigmoid
'''
bc_seq=Sequential()
bc_seq.add(Dense(9,activation='relu'))
bc_seq.add(Dense(30,activation='relu'))
bc_seq.add(Dense(20,activation='relu'))
bc_seq.add(Dense(10,activation='relu'))
bc_seq.add(Dense(1,activation='relu'))
''''
bc_seq=Sequential()
bc_seq.add(Dense(9,activation='relu'))
bc_seq.add(Dense(20,activation='relu'))
bc_seq.add(Dense(10,activation='relu'))
bc_seq.add(Dense(15,activation='relu'))
bc_seq.add(Dense(1,activation='relu'))

#compilation
bc_seq.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
bc_seq.fit(x_train,y_train,epochs=10)
bc_seq.summary()

#Evaluating model on train
train_loss,train_acc=bc_seq.evaluate(x_train,y_train)

#evaluating model on test data
test_loss,test_acc=bc_seq.evaluate(x_test,y_test)
print('test_acc:',train_acc)
____________________________

bc_seq=Sequential()
bc_seq.add(Dense(20,activation='relu'))
bc_seq.add(Dense(30,activation='relu'))
bc_seq.add(Dense(10,activation='relu'))
bc_seq.add(Dense(15,activation='relu'))
bc_seq.add(Dense(1,activation='relu'))

#compilation
bc_seq.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
bc_seq.fit(x_train,y_train,epochs=10)
bc_seq.summary()

#Evaluating model on train
train_loss,train_acc=bc_seq.evaluate(x_train,y_train)

#evaluating model on test data
test_loss,test_acc=bc_seq.evaluate(x_test,y_test)
print('test_acc:',train_acc)
'''
[==============================] - 0s 1ms/step - loss: 0.0000e+00 - accuracy: 1.0000

test_loss,test_acc=bc_seq.evaluate(x_test,y_test)
6/6 [==============================] - 0s 2ms/step - loss: 0.0000e+00 - accuracy: 1.0000

print('test_acc:',train_acc)
test_acc: 1.0

'''
_____________________________
bc_seq=Sequential()
bc_seq.add(Dense(9,activation='relu'))
bc_seq.add(Dense(42,activation='relu'))
bc_seq.add(Dense(9,activation='relu'))
bc_seq.add(Dense(65,activation='relu'))
bc_seq.add(Dense(1,activation='relu'))

#compilation
bc_seq.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
bc_seq.fit(x_train,y_train,epochs=10)
bc_seq.summary()

#Evaluating model on train
train_loss,train_acc=bc_seq.evaluate(x_train,y_train)

#evaluating model on test data
test_loss,test_acc=bc_seq.evaluate(x_test,y_test)
print('test_acc:',train_acc)

''' test_acc: 1.0 '''
________________________________________

bc_seq=Sequential()
bc_seq.add(Dense(9,activation='relu'))
bc_seq.add(Dense(30,activation='relu'))
bc_seq.add(Dense(34,activation='relu'))
bc_seq.add(Dense(25,activation='relu'))
bc_seq.add(Dense(1,activation='relu'))

#compilation
bc_seq.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
bc_seq.fit(x_train,y_train,epochs=10)
bc_seq.summary()

#Evaluating model on train
train_loss,train_acc=bc_seq.evaluate(x_train,y_train)

#evaluating model on test data
test_loss,test_acc=bc_seq.evaluate(x_test,y_test)
print('test_acc:',train_acc)

''' test_acc: 0.658203125 '''
















