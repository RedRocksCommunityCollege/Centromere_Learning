
# coding: utf-8

# In[1]:


'''
Code for assessing and sorting predictions made by keras model. Andy Cross and Adam Forland, Red Rocks Community
College Data Lab, summer 2018
'''


import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import glob
import os


# In[2]:


#location of pretrained model
model = load_model('/Users/andycross/Desktop/rgb_net_v4/rgb_small_1_test.h5')
#folder of images to be predicted
image_path = ('/Users/andycross/Desktop/test')


# In[3]:


#create folders for where to save sorted images
if not os.path.exists("/Users/andycross/Desktop/animals"):
            os.makedirs("/Users/andycross/Desktop/animals")
        
#create sub folders for sorting images        
if not os.path.exists("/Users/andycross/Desktop/animals/cats"):
            os.makedirs("/Users/andycross/Desktop/animals/cats")
        
if not os.path.exists("/Users/andycross/Desktop/animals/dogs"):
            os.makedirs("/Users/andycross/Desktop/animals/dogs")
        
if not os.path.exists("/Users/andycross/Desktop/animals/fish"):
            os.makedirs("/Users/andycross/Desktop/animals/fish")
        


# In[5]:


#code for making predictions and sorting predicted images
def predict(loc,x_size,y_size):
    i=0 
    for image in glob.glob(image_path+'/*.jpeg'):
        img = cv2.imread(image,0) #read in images from glob
        img1 = cv2.resize(img,(150,113)) #give same shape as keras model was trained on 
        plt.imshow(img)
        img = img1.reshape(x_size,y_size,1) #reshape, last integer represents color channels
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
            print('This is a Cat')
            cv2.imwrite('/Users/andycross/Desktop/animals/cats/image'+str(i)+'.jpg',img1)
        elif preds == 1:
            print('This is a Dog')
            cv2.imwrite('/Users/andycross/Desktop/animals/dogs/image'+str(i)+'.jpg',img1)
        elif preds == 2:
            print('This is a fish')
            cv2.imwrite('/Users/andycross/Desktop/animals/fish/image'+str(i)+'.jpg',img1)
        print(preds)
        print(probs)

predict('at',113,150)

