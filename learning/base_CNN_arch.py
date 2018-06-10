'''This network will classify Red, Gree and Blue images'''

from __future__ import print_function
import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

#import make_food as yum

#def read_data():
#     train, validate = yum.generate()
#     return(train,validate)
all_data = np.load('all_data.npy')
np.shape(all_data)
x_train = all_data[:110,3:]

x_train = x_train.reshape(np.shape(x_train)[0],225,301,1)

y_train = all_data[:110,:3]
y_test = all_data[110:,:3]

x_test = all_data[400:,3:]
x_test = x_test.reshape(np.shape(x_test)[0],225,301,1)


train = x_train,y_train
validate = x_test,y_test

def network_model():
     model = Sequential()
     model.add(Conv2D(32, kernel_size=(3, 3),
                      activation='relu',
                      input_shape=(225,301,1)))
     model.add(MaxPooling2D(pool_size=(3, 3)))
     model.add(Conv2D(32, (3, 3), activation='relu'))
     model.add(MaxPooling2D(pool_size=(3, 3)))
     model.add(Dropout(0.25))
     model.add(Flatten())
     model.add(Dense(128, activation='relu'))
     model.add(Dropout(0.2))
     model.add(Dense(3, activation='softmax'))

     model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adadelta(),
                   metrics=['accuracy'])

     return(model)

def network_train(model,train,validate):
     history = model.fit(x = train[0], y = train[1],
          steps_per_epoch = 10,
          epochs = 7)

     model.save('rgb_small_1_test.h5')
     return(history)

def network_evaluation(history):
     acc = history.history['acc']
     val_acc = history.history['val_acc']
     loss = history.history['loss']
     val_loss = history.history['val_loss']

     epochs = range(1,len(acc) + 1)

     plt.plot(epochs, acc, 'bo', label = 'Training loss')
     plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
     plt.title('Training and validation accuracy')

     plt.figure()

     plt.plot(epochs, loss, 'bo', label = 'Training loss')
     plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

     plt.title('Training and validation loss')
     plt.legend

     plt.show()

def run():
     #train, validate = read_data()
     model = network_model()
     history = network_train(model,train,validate)
     network_evaluation(history)


run()

import prediction_rgb as pre
pre.predict('\\Users\\cross\\Desktop\\butter.jpg',150,80)
