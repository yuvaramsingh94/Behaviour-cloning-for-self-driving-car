**Behavioral Cloning Project**

steps of this project are the following:
* use the simulator to collect the image and steering angle as labels (best practice is to drive around the track twice and also in the opposite direction )
* import the image and steering angle 
* construct neural network to calculate the weights to find and adjust the steering angle to drive the car inside the track 
* train a model based on the training data collected from the step 1 and save the model 
* after the training step , open the simulator in  "Autonomous mode"
* run the drive.py file by passing the location path of the model as an argument in the terminal window
          (python drive.py /path/of/themodel/)
* once the model is initialized , it should start driving the car . based on your model and the system you are using , the accuracy of the driving might differ i used (Alienware i5 NVIDIA® GeForce® GTX 965M   my friends lap :) ) to train and test my model



[//]: # (Image References)

[image1]: ./img/conv.jpg "Convolution layer"
[image2]: ./img/fullyConnected.jpg "Fully connected"
[image3]: ./img/fullModel.jpg "full model"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Note : i developed my model by using Keras 2.0.1 version . please use this version to test this program
* model.py this has the python program to create , train ,and save a Convolutional neural network model to predict the steering angle (I used keras version 2.0.1)
* drive.py for driving the car in autonomous mode
* modelNvidiaMax_2 is the saved model which has the neural network graph and the corresponding weights which can be used to recreate this model for future use
* writeup_report.md  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py /path/of/themodel/modelNviidaMax_2.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Nvidia deep neural network has been implemented to predict my steering angle in my model (with few changes )

My model consist 5 convolutional layers with maxpool layers . the layer structure is described below

![alt text][image1]

following these conv layers . a flatten layer and _ fully connected layer is fitted . finally a single node is used to find the steering angle (Regression not classification )


![alt text][image2]

#### 2. Attempts to reduce overfitting in the model
 
My model is not using Dropout or relu to generalize  . i have used maxpool layers in between the convolution layers 

#### 3. Model parameter tuning

After playing with a lot of optimizer and learning rate , i ended up using adem optimiser and mean square error . these yielded me a good accuracy over the validation dataset and also performed good during the driving test  

#### 4. Appropriate training data

I collected the training data by driving around the track two times by maintaining my car in the center lane , additionally i drived in the opposite direction for a single lap , aditional images where take in the trick spots where the road and the mud mixup 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a model which i am familiar with and work through other models by increasing the depth , adding more conv layers , changing the number of nodes in the fully connected layers 

My first step was to use a convolutional neural network model similar to the Lenet  . the lenet has a powerful image classification properties , this model works good on straight roads but not on curves . 

After going through a lot of neural network papers , i found Nvidia's self driving cars paper provided by udacity . after going through the paper , i modified the conv layers to fit the image shape 160X320X3 provided by the simulator 

my model consist of 5 conv layers fitted with max pool and 4 fully connected layers followed by a single node . it  takes 160X320X3 as input and spits out 1 steering angle 



after training this model , i was able to drive around the given track autonomously without leaving the track 

#### 2. Final Model Architecture

this is my final model 
![alt text][image3]

#### 3. Creation of the Training Set & Training Process
To create a more generalized model to drive the car , i started collecting the training data from the muddy track (easiest one) . i drove two laps around the track . one lap in the opposite side , one lap on the other track (the hard one)
After the collection process, I had 23000 (approx) number of data points. I then increased  this data by flipping the images , using the right , left images to serve as a recovery image (if my model sees those image  , it steers hard towards the center) 


I shuffled the training data and used 30% for validation data 

i used these data to train and test my model , i run my model for 10 epochs and saved it under the name modelNvidiaMax_2.h5 . using this model, i was able to drive the car within the road as shown in the video 


