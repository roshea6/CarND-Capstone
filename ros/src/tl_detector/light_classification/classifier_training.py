import tensorflow as tf 
import cv2, glob
import numpy as np 

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

# Class for training and loading a convolutional neural net classifier on the data traffic light data generated from the simulator
class tlConvNet(object):
    def __init__(self):
        pass
    
    def train_model(self):
        pass
    
    def load_model(self):
        pass