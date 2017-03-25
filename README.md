#**Behavioral Cloning** 



**Behavioral Cloning Project**

steps of this project are the following:
* use the simulator to collect the image and strearing angle as labels (best practice is to drive around the track twice and also in the opposite direction )
* import the image and stearing angle 
* construct neural network to calculate the weights to find and adjust the stearing angle to drive the car inside the track 
* train a model based on the training dat collected from the step 1 and save the model 
* after the training step , open the simulator in  "Autonomous mode"
* run the drive.py file by passing the location path of the model as an arguement in the terminal window
          (python drive.py /path/of/themodel/)
* once the model is initialized , it should start driving the car . based on your model and the system you are using , the accuracy of the driving might differ i used (Alienware i5 NVIDIA® GeForce® GTX 965M   my friends lap :) ) to train and test my model



[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Note : i developed my modelby using Keras 2.0.1 version . please use this version t test this program
* model.py this has the python program to create , train ,and save a Convolutional neural network model to predice hte stearing angle (I used keras version 2.0.1)
* drive.py for driving the car in autonomous mode
* modelNvidiaMax_2 is the saved model which has the neural network graph and the corresponding weights which can be used to recreate this model for futhure use
* writeup_report.md  summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py /path/of/themodel/modelNviidaMax_2.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. Nvidia deep neural network has been implemented to predict my stearing ange in my model (wiht few changes )

My model consist 5 convolution layers with maxpooling layers . the layer structure is discribed below
![alt text][image1]
following these conv layers . a flatten layer and _ fully connected layer is fitted . finaly a single node is used to find the stearing angle (Regressin not classification )
![alt text][image1]

####2. Attempts to reduce overfitting in the model
 
My model is not using Dropout or relu to generalize  . i have used maxpool layers inbetween the convolution layers 

####3. Model parameter tuning

After playing with a lot of optimizer nad learning rate , i ended up using adem optimiser and mean square error . these yielded me a good accuracy over the validation dataset and also performed good during the driving test  

####4. Appropriate training data

I collected the training data by driving around the track two times by maintainig my car in the center lane , additionaly i drived in the opposite direction for a single lap , aditional images where take in the trick spots where the rode and the mud mixup 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
