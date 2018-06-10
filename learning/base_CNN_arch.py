'''Base network architecture for multi classifiers. The data needs to come
in as One Hot encoded already shuffled. 
Andy Cross and Adam Forland Red Rocks Community College, Data_Lab
June 10, 2018'''

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
import time


class centromere_trainer:

    def prep_all_data():
        all_data = np.load('all_data.npy')
        x_train = all_data[:110,3:]
        x_train = x_train.reshape(np.shape(x_train)[0],225,301,1)

        y_train = all_data[:110,:3]
        y_test = all_data[110:,:3]

        x_test = all_data[400:,3:]
        x_test = x_test.reshape(np.shape(x_test)[0],225,301,1)


        train = x_train,y_train
        validate = x_test,y_test

        return(train)
        return(test)

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

        model.save('models/CNN_name_of_feature_+'+time.strftime("%Y-%m-%d") +'.h5')
        with open('/History_name_of_feature' + time.strftime("%Y-%m-%d") , 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

cent = centromere_trainer()
train , test = cent.prep_all_data()
model = cent.network_model()
network_train(model, train, test)

