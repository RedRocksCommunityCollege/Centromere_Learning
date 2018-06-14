'''Base network architecture for multi classifiers. The data needs to come
in as One Hot encoded already shuffled.
Andy Cross and Adam Forland Red Rocks Community College, Data_Lab
June 10, 2018'''

from __future__ import print_function
import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.models import load_model
import time
import h5py
from os import getcwd
import pickle

mypath = getcwd()
print(mypath)

class centromere_trainer:

    def prep_all_data(self):
        
        h5f = h5py.File('all_data.h5','r')
        all_data = h5f['dataset_1'][:]
        
        print(np.shape(all_data))

        x_train = all_data[:350,2:]
        print(np.shape(x_train))
        x_train = x_train.reshape(np.shape(x_train)[0],10,10,3)
        print(np.shape(x_train))
        y_train = all_data[:350,:2]
        y_test = all_data[350:,:2]

        x_test = all_data[350:,2:]
        x_test = x_test.reshape(np.shape(x_test)[0],10,10,3)


        train = x_train,y_train
        test = x_test,y_test

        return(train,test)

    def network_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2, 2),
                         activation='relu',
                         input_shape=(10,10,3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))

        #parallel_model = model.multi_gpu_model(model,gpus = 2)

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        return(model)

    def network_train(self,model,train,validate):
        history = model.fit(x = train[0], y = train[1],
            steps_per_epoch = 10,
            epochs = 7)

        model.save(mypath + '/models/CNN_name_of_feature_' + time.strftime("%Y-%m-%d_%H-%M-%S") +'.h5')
        with open(mypath + '/models/history/History_name_of_feature' + time.strftime("%Y-%m-%d_%H-%M-%S") , 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

cent = centromere_trainer()
train , test = cent.prep_all_data()
model = cent.network_model()
cent.network_train(model, train, test)
