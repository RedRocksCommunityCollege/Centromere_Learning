from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from matplotlib.image import imread
import glob
import cv2
import os
import sys
import time
import h5py

class Make_One_Hot:

    def __init__(self):
        # This is where the images will be
        self.dirloc = "/home/adam/MountPt/data/Sharp_NotSharp/"
    
    # Got this from here:  https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    def progress(self, count, total, status):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()
    
    # Resizes all the images so that they are all the same size.
    def resize_image(self):
        x_sizes = []  # Catch for sizes
        y_sizes = []
        img_count = 0
        # Loop over all the images
        for dirnames in os.walk(self.dirloc):
            if dirnames[0] == self.dirloc:
                continue
            else:
                for images in glob.glob(dirnames[0] + '/*'):
                    image_array = imread(images)  # Read in as an array
                    x = np.shape(image_array)[0]  # Pull x value of the shape
                    y = np.shape(image_array)[1]  # Pull the y value of the shape
                    # Append to the array
                    x_sizes = np.append(x, x_sizes)
                    y_sizes = np.append(y, y_sizes)

                    # Find the x min
                    x_min = np.amin(x_sizes)
                    x_min = int(x_min)

                    #Find the y min
                    y_min = np.amin(y_sizes)
                    y_min = int(y_min)

                    img_count += 1
        
        x_min = 10 
        y_min = 10
        return(x_min, y_min, img_count)

    # This reads in the images and appends them together
    def Make_Data_Matrix(self, location):

        # Gets the size for x and y (Smallest)
        x_size, y_size, img_count = MOH.resize_image()

        # Makes a vector of the correct size so that we can append to it.
        #vector1 = np.zeros((x_size,y_size,3))
        #vector1 = vector1.flatten()
        #vector1 = vector1.reshape((np.shape(vector1)[0],1))


        # Loop through all the images.
        i = 0
        total = img_count
        vector1 = []
        for images in glob.glob(location + '/*'):
            image = imread(images)  # Read in each image
            image = cv2.resize(image, (int(y_size), int(x_size)))  # Size the images so that they are all the same size
            j = image.flatten()  # Flatten them all
            B = j.reshape((np.shape(j)[0], 1))  # Reshape so that the arrays are (n,1)
            vector1.append(B)  # Append them all to the main array

            MOH.progress(i, total, 'Working on ' + location)
            time.sleep(0.5)  # emulating long-playing job

            i += 1

        # Tip so that it is a row of images with columns of values
        vector1 = np.array(vector1)
        vector1 = vector1.T
        vector1 = np.squeeze(vector1, axis=0)
        vector1 = np.transpose(vector1)
        print("Number of images in" + location, np.shape(vector1))
        return vector1

    def make_all_data(self, raw_master_array):
        i = 0
        collect = []
        for i in range(np.shape(raw_master_array)[0]):
            one_hot = [0] * np.shape(raw_master_array)[0]
            for j in range(np.shape(raw_master_array[i])[0]):
                one_hot[i] = 1
                collect1 = one_hot + raw_master_array[i][j].tolist()
                collect.append(collect1)
                print(i,j)
                sys.stdout.flush()
        
        collect = np.array(collect, dtype=np.float16)


        h5f = h5py.File("all_data.h5", "w")
        h5f.create_dataset('dataset_1', data=collect)
        h5f.close()

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
        #for i in range(np.shape(raw_master_array)[0]):
            #Look through each of the n matrices and find the one that has the least num of images
        #    num_of_images.append(np.shape(raw_master_array[i])[0])

        MOH.make_all_data(raw_master_array)

MOH = Make_One_Hot()
MOH.run()

