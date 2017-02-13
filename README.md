Behavrioal Cloning Project

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
writeup_report.md or writeup_report.pdf summarizing the results

2. Submssion includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.json

Model Architecture and Training Strategy

1. The Neural network used by NVIDIA and comma ai use hyperparamters which tend to overfit.

For this exercise I used a smaller network, the LeNet architecture. 
LeNet has 2 convolutional layers and 2 fully connected layers. 
Made few changes as listed below
1) Use image size of 64x64.
2) Normalize the image using Keras lambda layer
3)Changed the size of the convolution kernels to 3x3 in order extract finer features and that made a big difference.
4)Change the activation layer from Relu to ELU(Exponential Linear Units)
5) Changed the size of the pooling kernel from 2x2 to 4x4 - That is because the input image I am using is the size of 64x64 
instead of the standard 32x32 that was used for LeNet.
My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)
6) Dropout of 


2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.



