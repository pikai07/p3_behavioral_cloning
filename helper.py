import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_data():
    """ Loads the data with information from udacity provided 'drive_log.csv'
        file. Add 2 additional columns or left and right image steering angle
        
        left image steer angle = center image steer angle + correction
        right image steer angle = center image steer angle - correction
        
        correction: the angle required to steer the car back to the center """
    path = 'data/'
    column_names=['center', 'left', 'right', 'Steering Angle', 'throttle', 'break', 'speed']
                    
    data = pd.read_csv('data/driving_log.csv',index_col = False)
    data.columns = column_names

    data['Center Image'] = path + data['center'].apply(str.strip)
    data['Left Image'] = path + data['left'].apply(str.strip)
    data['Right Image'] = path + data['right'].apply(str.strip)

    # Remove data with no throttle
    ind = data['throttle']>0    
    data = data[ind].reset_index(drop=True)
  
    
    # Add left and right image steering angles to the angle correction factor
    correction = 0.22
    data['Left Steering Angle'] = data['Steering Angle'] + correction
    data['Right Steering Angle'] = data['Steering Angle'] - correction 
    return data


def preprocess_image(image, new_row_size, new_col_size):
    """ Takes in a numpy array and converts it HSV color space 
        only keeping the S channel. Resizes the image to 64x64 size
        and crop image to remove the horizon and the hood part of the car"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1]
    image = image[40:138]  
    image = cv2.resize(image,(new_row_size, new_col_size), interpolation=cv2.INTER_AREA)
    return image 


def load_training_data(data, new_row_size=64, new_col_size=64):
    """ Takes in a dataframe containing training data information and
        returns preprocessed image data and labels in the form of numpy arrays. Images are resized
        to 64x64"""
    num_samples = len(data)
    label_data = np.zeros(num_samples*3, dtype=np.float32)
    image_data = np.zeros((num_samples*3, new_row_size, new_col_size), dtype=np.float32)
    
    # Load center image data
    for index in range(0, num_samples):
        image_c = cv2.imread(data['Center Image'][index])
        label_data[index] = data['Steering Angle'][index]
        image_data[index] = preprocess_image(image_c, new_row_size, new_col_size)
    
    # Load left image data
    for read_index, write_index in zip(range(num_samples), range(num_samples, num_samples*2)):
        image_l = cv2.imread(data['Left Image'][read_index])
        label_data[write_index] = data['Left Steering Angle'][read_index]
        image_data[write_index] = preprocess_image(image_l, new_row_size, new_col_size)
    
    # Load right image data
    for read_index, write_index in zip(range(num_samples), range(num_samples*2, num_samples*3)):
        image_r = cv2.imread(data['Right Image'][read_index])
        label_data[write_index] = data['Right Steering Angle'][read_index]
        image_data[write_index] = preprocess_image(image_r, new_row_size, new_col_size)
    
    # Add additional dimension to the image - convert (None,64,64) to (None,64,64,1)
    image_data = np.reshape(image_data, (image_data.shape[0],new_row_size,
                                         new_col_size,-1))
    return image_data, label_data