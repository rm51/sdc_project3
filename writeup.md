#**Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


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
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 24 and 64 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

I also tried with an ELU layer but found that the model performed worse with ELU so I went back to RELU.

####2. Attempts to reduce overfitting in the model

I added two dropout layers and found that the model performed the best with those two layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and watching that it would keep running on the track without crashing or hitting any barriers or going off the track until I stopped the simulator. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data


I first drove around the track twice and then recorded recovery data from the side of the track as well as
more corner data where the car was falling off the track such as after the bridge and on the next corner where
it was supposed to go right but didn't. But even with that data I wasn't able to get a full lap until I started
using the three cameras. With the three cameras I recorded  new data by driving around the road closewise and 
one time counterclockwise.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I first used a convolutional neural network similar to the LeNet model and then later in the lessons learned about the Nvidia model so I switched to that.

The Nvidia model was pretty good without any finetuning because it got the car to the bridge. But then after the bridge it would go on the dirt path.

I orignally used just the center camera but wasn't able to complete an entire lap so I switched to adding a parameter for the left and right camera angle. 

I also resized the images to make the model quicker to train so that I could more easily tune the parameters since I was training on my laptop without a gpu. 

After resizing images I found that each epoch took less than a minute down from ten minutes so that allowed me to quickly finetune the model.

At first I wasn't using dropout but I found that I had a low mean squared error on the training set but a high mean squared error on the validation set implying overfitting. 

So I added dropout first at a rate of 0.5 and saw that the model performed worse so I tried a rate of 0.3 and the model did worse so I changed it to 0.7 and then the model did better and made it over the bridge but crashed into the barrier. 

I then decided to add an additional Dropout layer  and again experimented with the values to find the best one and again found that 0.7 was the best choice. I also moved the Dropout to inb etween various layers to see where the best results were and left the Dropout where it performed the best. And the car was able to make it consistently around the track and would continue going around until I stopped the simulator. 

I originally tried five epochs but saw that the loss was still decreasing so I increased it to ten to see if that would give a better model and I did so I kept it there. I had also tried two 
and three epochs but the model was worse with that.

To augment the data set I flipped the images so that I would have more data. I also filtered 70% of the images that were almost straight to combat the tendency to drive straight.

I didn't train on track two because I found it difficult to stay in the lane. I plan to train on that track and also add brightness to my images to further improve my model because currently my model goes to the left pretty quickly on track 2. 

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![sample center image][https://github.com/rm51/sdc_project3/blob/master/center_2017_08_31_22_49_07_970.jpg]


To augment the data sat, I also flipped images and angles thinking that this would give me more data while also having the benefit on not having the model memorize the track.


After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. I plotted the mse vs loss and saw that at five loss was still decreasing. Also I looked at other graphs as to when to increase the number of epochs and saw that mine was the case where I should use more epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I was planning to use a generator if I ran out of memory but I didn't need to as I didn't run out of memory and one epoch took less than a minute.
