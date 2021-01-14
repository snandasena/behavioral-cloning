[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Behavioral Cloning Project
---

### Introduction
The goal of the project is to build a ML model to simulate a car to run in a autonomus mode that is provided by Udacity. The simulator can be found [here](https://github.com/udacity/self-driving-car-sim). The deep neural network will be used to build this model primalary Convolutional Neuaral Network(CNN) will be implemented by using [NVIDIA End-to-End Deep Learning for Self-Driving Cars architecture](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/). There are two provided lane tracks to generate training data for this project and this project requirement is to build a model for first track. To continue this project successfuly following steps were proccessed.

* Data Preprocessing
* Image Data Aumentation
* Traning Data Preparation
* Building the Model Architecture
* Training the Model
* Testing with Simulator

### Data Preprocessing
To undestand about these simulator generated data I used a 'Jupyter Notebook](image_utils.ipynb) and did some preliminary data precessing steps.  
**Note: This Jupyter file is not covering all the steps I was followed and it was used to start image preprocessing.**

Following is the simulator generated driving logs CSV's Pandas dataframe head.

![](resources/data-look-a-like.png)

#### The Data
This simulator is generating a CSV file with **7** columns and there are **3** of columns contain images related details namely **center, left, and right**. These are the our input lables that we want to use to build a this **Regression** model. The output of the model is **streering** the streering angle for three images. In real case these three images are taken from three different cameras at the same time.  Following is the high level view of the data collection system that is used by NVIDIA.

![](resources/data-collection-system-624x411.png)

Image source: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/



