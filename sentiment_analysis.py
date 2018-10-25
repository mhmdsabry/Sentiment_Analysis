# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:17:33 2018

@author: mkenny
"""

from keras.datasets import imdb
from keras import models,layers,optimizers
import numpy as np
from keras.callbacks import Callback

#prepare the dataset
(training_data,training_targets),(testing_data,testing_targets) = imdb.load_data(num_words = 10000)
data = np.concatenate((training_data,testing_data),axis=0)
targets = np.concatenate((training_targets,testing_targets),axis=0)

def vectorized(sequence,dim = 10000):
    result = np.zeros((len(sequence),dim))
    for i,seq in enumerate(sequence):
        result[i,seq] = 1       
    return result

data = vectorized(data) 
targets = np.array(targets).astype("float32")

test_data = data[:10000]
test_targets = targets[:10000]
train_data = data[10000:]
train_targets = targets[10000:]


model = models.Sequential()

#input_layer
model.add(layers.Dense(50,activation = 'relu',input_shape=(10000, )))

#hidden_layer
model.add(layers.Dropout(0.3,noise_shape =None,seed=None))
model.add(layers.Dense(50,activation ='relu'))

#output_layer
model.add(layers.Dense(1,activation = 'sigmoid'))

model.summary()
#custom callback for switching between sgd and RMSprop
class changeOptimizer(Callback):
    def on_batch_end(self,batch=150,log={}):
        sgd = optimizers.SGD(lr=0.0001, decay = 1e-6, momentum=0.99,nesterov= True)
        model.compile(optimizer = sgd,loss = 'binary_crossentropy',metrics=['accuracy'])
       
#compiling_model
callbacks = [changeOptimizer()]
model.compile(optimizer = 'RMSprop',loss = 'binary_crossentropy',metrics=['accuracy'])
results = model.fit(train_data,train_targets,epochs=2,batch_size=500,validation_data=(test_data,test_targets),callbacks = callbacks)

print('test Accuracy = ',np.mean(results.history["val_acc"]))