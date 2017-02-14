Behavioral Cloning Project

The goals / steps of this project are the following:

Use the simulator to collect data of good driving behavior
Build, a convolution neural network in Keras that predicts steering angles from images
Train and validate the model with a training and validation set
Test that the model successfully drives around track one without leaving the road
Summarize the results with a written report

Files Submitted & Code Quality

1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

model_lenet.py containing the script to create and train the model
drive.py for driving the car in autonomous mode
model.h5 containing a trained convolution neural network
writeup_report.md summarizing the results

2. Submssion includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.json

Model Architecture and Training Strategy

1. The Neural network used by NVIDIA and comma ai are dense architectures which tend to overfit.

For this exercise I used LeNet architecture. 
LeNet has just 2 convolutional layers and 2 fully connected layers and surprisingly it did well without much of image augmentation. 
Few subtle changes were made as listed below
1) Fed input image of size 64x64 to neural network.
2) Normalized the image using Keras lambda layer
3) Changed the size of the convolution kernels to 3x3 in order extract finer features and that made a big difference.
4) Changed the activation layer from Relu to ELU(Exponential Linear Units)
5) Changed the size of the pooling kernel from 2x2 to 4x4 - That is because the input image I am using is the size of 64x64 
instead of the standard 32x32 that was used for LeNet.
6) Using droput made the loss worse and the car would keep falling off the road, so got rid of it.


2. Preprocessing steps to reduce overfitting
a) Normalize the image
b) Shuffle the data before every epoch
c) Convert the image to converts it to HSV color space keeping only the S channel
d) Crop the image to eliminate the horizon and the hood part of the car.

Used the udacity provided dataset to train the model.

3. Model parameter tuning

The model used an adam optimizer, so the learning rate did not have to be tuned manually.




