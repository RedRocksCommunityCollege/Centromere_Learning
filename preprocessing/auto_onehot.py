from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy import argmax
from numpy import zeros
from numpy import shape
from numpy import chararray
import numpy as np
from matplotlib.image import imread
import glob
from PIL import Image
import cv2
import os
from os import getcwd
import pickle

mypath = getcwd()
print(mypath)

class Make_One_Hot:

    def __init__(self):
        # This is where the images will be
        self.dirloc = mypath + "/preprocessing/data"

    def Parameters(self):
        parA = []
        for dirnames in os.walk(self.dirloc):
            parA.append(dirnames[0])

            #if dirnames[0] == self.dirloc:
            #    continue
            #else:
                #for images in glob.glob(dirnames[0] + '/*'):
        print(parA)


    # Resizes all the images so that they are all the same size.
    def resize_image(self):
        x_sizes = [] # Catch for sizes
        y_sizes = []
        # Loop over all the images
        for dirnames in os.walk(self.dirloc):
            if dirnames[0] == self.dirloc:
                continue
            else:
                for images in glob.glob(dirnames[0] + '/*'):
                    image_array = imread(images) # Read in as an array
                    x = np.shape(image_array)[0] # Pull x value of the shape
                    y = np.shape(image_array)[1] # Pull the y value of the shape

                    # Append to the array
                    x_sizes = np.append(x,x_sizes)
                    y_sizes = np.append(y,y_sizes)

                    # Find the x min
                    x_min = np.amin(x_sizes)
                    x_min = int(x_min)

                    #Find the y min
                    y_min = np.amin(y_sizes)
                    y_min = int(y_min)

        return(x_min,y_min)

    # This reads in the images and appends them together
    def Make_Data_Matrix(self,location):

        # Gets the size for x and y (Smallest)
        x_size, y_size = MOH.resize_image()

        # Makes a vector of the correct size so that we can append to it.
        vector1 = np.zeros((x_size,y_size,3))
        vector1 = vector1.flatten()
        vector1 = vector1.reshape((np.shape(vector1)[0],1))


        # Loop through all the images.

        i = 1
        for images in glob.glob(location + '/*'):
            print("Image" + str(i))
            image = imread(images) # Read in each image
            image = cv2.resize(image,(int(y_size),int(x_size))) # Size the images so that they are all the same size
            j = image.flatten() # Flatten them all
            B = j.reshape((np.shape(j)[0],1))# Reshape so that the arrays are (n,1)
            print(np.shape(B))
            print(np.shape(vector1))
            vector1 = np.append(vector1,B,axis = 1) # Append them all to the main array
            i = i + 1

        # Tip so that it is a row of images with columns of values
        vector1 = vector1.T
        print("Number of images in" + location  , np.shape(vector1))
        return vector1

    def make_all_data(self,raw_master_array):
        d=[]
        a = []
        a1 = []
        i = 0
        collect = []
        for i in range(np.shape(raw_master_array)[0]):
            a = np.append(np.ones(np.shape(raw_master_array[i])[0]),a1,axis = 0)
            c = i * a
            d = np.append(d,c)
            for j in range(np.shape(raw_master_array[i])[0]):
                collect.append(raw_master_array[i][j])

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(d)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        y_train_base = onehot_encoder.fit_transform(integer_encoded)
        all_data = np.append(y_train_base, collect, axis = 1)
        np.random.shuffle(all_data)
        print("Here is the shape of all_data: " , np.shape(all_data))
        np.save('all_data',all_data)


    def run(self):
        # Initialize the class
        MOH = Make_One_Hot()

        num_of_images = []
        raw_master_array = []


        # Make the collection of data
        for dirnames in os.walk(self.dirloc):
            if dirnames[0] == self.dirloc:
                continue
            else:
                # All the flattened images for all the classes in all the label directories
                raw_master_array.append(MOH.Make_Data_Matrix(dirnames[0]))

        #raw_master_array = np.array(raw_master_array)
        for i in range(np.shape(raw_master_array)[0]):
            #Look through each of the n matrices and find the one that has the least num of images
            num_of_images.append(np.shape(raw_master_array[i])[0])

        MOH.make_all_data(raw_master_array)



MOH = Make_One_Hot()
MOH.Parameters()


MOH.run()
