
# coding: utf-8

# In[28]:


'''
Code for augmenting images for training. 
'''


import matplotlib.pyplot as plt
import glob
import numpy as np
from skimage.io import imread
from skimage import exposure, color
from skimage.transform import resize
import cv2
import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


# In[29]:


#initialize keras's image augmentation
def imgGen(img, zca=False, rotation=0., w_shift=0., h_shift=0., shear=0., zoom=0., h_flip=False, v_flip=False,  preprocess_fcn=None, batch_size=9):
    datagen = ImageDataGenerator(
            zca_whitening=zca,
            rotation_range=rotation,
            width_shift_range=w_shift,
            height_shift_range=h_shift,
            shear_range=shear,
            zoom_range=zoom,
            fill_mode='nearest',
            horizontal_flip=h_flip,
            vertical_flip=v_flip,
            preprocessing_function=preprocess_fcn,
            data_format=K.image_data_format())
    
    datagen.fit(img)
    #display and save augmented images, currently overwrites existing files when new ones are augmented, needs to be fixed
    i=0
    for img_batch in datagen.flow(img, batch_size=9, shuffle=False):
        
        for img in img_batch:
            #plt.subplot(330 + 1 + i)
            plt.imshow(img)
            print(np.shape(img))
            i=i+1  
            np.save("/Users/andycross/Desktop/transformed/image"+str(i),img)
            plt.savefig('/Users/andycross/Desktop/transformed/image'+str(i)+'.jpg')
        if i >= batch_size:
            break
    plt.show()


# In[31]:


#displays augmented images from npy files, maintaines all color channels.
g= np.load('/Users/andycross/Desktop/transformed/image1.npy')
plt.imshow(g)
print(np.shape(g))
b,t,r = cv2.split(g)
print(np.shape(t))
plt.imshow(g)
b  = g.flatten()
print(np.shape([b]))
d = b.reshape(681,968,3)
plt.imshow(d)


# In[12]:



# visualize the image# visua 
for img in glob.glob('/Users/andycross/Desktop/test/*.jpeg'):
    img = imread(img)
    plt.imshow(img)
    plt.show()
    print(np.shape([img]))

# reshape it to prepare for data generator
    img = img.astype('float32')
    img /= 255
    h_dim = np.shape(img)[0]
    print(h_dim)
    w_dim = np.shape(img)[1]
    print(w_dim)
    num_channel = np.shape(img)[2]
    img = img.reshape(1, h_dim, w_dim, num_channel)
    print(img.shape)

# generate images using function imgGen
    imgGen(img, shear=10,rotation = 10, h_shift=0.1)
   
    
    

