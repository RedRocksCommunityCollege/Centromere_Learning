import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import glob
import os
from os import getcwd
import pickle

mypath = getcwd()
print(mypath)

#location of pretrained model
model = load_model(mypath + '/learning/models/CNN_name_of_feature_2018-06-13_02-36-22.h5')

#folder of images to be predicted
image_path = ('/Users/cross/Desktop/test')

#create folders for where to save sorted images
if not os.path.exists("/Users/cross/Desktop/images_sharp_not"):
            os.makedirs("/Users/cross/Desktop/images_sharp_not")



#create sub folders for sorting images

if not os.path.exists("/Users/cross/Desktop/images_sharp_not/sharp"):
            os.makedirs("/Users/cross/Desktop/images_sharp_not/sharp")



if not os.path.exists("/Users/cross/Desktop/images_sharp_not/not_sharp"):
            os.makedirs("/Users/cross/Desktop/images_sharp_not/not_sharp")



#if not os.path.exists("/Users/andycross/Desktop/animals/fish"):

#            os.makedirs("/Users/andycross/Desktop/animals/fish")

#code for making predictions and sorting predicted images

def predict(loc,x_size,y_size):

    i=0

    for image in glob.glob(image_path+'/*.jp2'):

        img = cv2.imread(image,0) #read in images from glob

        img1 = cv2.resize(img,(100,100)) #give same shape as keras model was trained on

        plt.imshow(img)

        img = img1.reshape(x_size,y_size,3) #reshape, last integer represents color channels

        #plt.imshow(img)

        img = np.array([img])

        #plt.imshow(img)

        #print(img)

        #img = np.reshape((img[0],img[1]))

        probs = model.predict_proba(img)  #prbabilities- how certain

        preds = model.predict_classes(img) #which class

        i = i+1

        #saving the predicted images into correct folders

        if preds == 0:

            print('This is not sharp')

            cv2.imwrite('/Users/cross/Desktop/images_sharp_not/not_sharp'+str(i)+'.jpg',img1)

        elif preds == 1:

            print('This is sharp')

            cv2.imwrite('/Users/cross/Desktop/images_sharp_not/sharp'+str(i)+'.jpg',img1)

        #elif preds == 2:

        #    print('This is a fish')

        #    cv2.imwrite('/Users/andycross/Desktop/animals/fish/image'+str(i)+'.jpg',img1)

        print(preds)

        print(probs)



predict('at',100,100)
