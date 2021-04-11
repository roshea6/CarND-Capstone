# from styx_msgs.msg import TrafficLight

# import tensorflow as tf 
import cv2, glob
import numpy as np 

from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.layers import Convolution2D, Dense, Flatten, Lambda
from keras.layers import MaxPooling2D, Cropping2D, ELU, Dropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical


# from skimage import io
# from sklearn.model_selection import train_test_split

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # self.save_path = "/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/data/"
        self.save_path = "./data/"
        self.green_path = self.save_path + 'green/'
        self.yellow_path = self.save_path + 'yellow/'
        self.red_path = self.save_path + 'red/'
        
        self.img_shape = None
        
        self.save_file = "/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/models/test.h5"
    
    def loadData(self):
        training_imgs = []
        training_labels = []
        
        # Loop through the different data folders and append the img and label to the lists
        for img_name in glob.glob(self.green_path + '*.jpg'):
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            if not img.shape == (600, 800, 3):
                print "Bad"
            training_imgs.append(img)
            training_labels.append(0)
            
        for img_name in glob.glob(self.yellow_path + '*.jpg'):
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            if not img.shape == (600, 800, 3):
                print "Bad"
            training_imgs.append(img)
            training_labels.append(1)
            
        for img_name in glob.glob(self.red_path + '*.jpg'):
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            if not img.shape == (600, 800, 3):
                print "Bad"
            training_imgs.append(img)
            training_labels.append(2)
         
        if len(training_imgs) > 0:
            self.img_shape = training_imgs[0].shape
            print(self.img_shape)
            #print(len(training_imgs))
            #print(training_imgs[0])
            
        return np.asarray(training_imgs), np.asarray(training_labels)
    
    def trainModel(self):
        # Get the training images and labels
        imgs, labels = self.loadData()
        
        # Reshape the data into the proper shape
        #imgs = imgs.reshape(self.img_shape[0], self.img_shape[1], self.img_shape[2], 1)
        
        print imgs.shape
        
        labels = to_categorical(labels, num_classes=3)
        
        # TODO: Define our neural net, optimizer, split into training and testing data, and train the network
        # Define our CNN
        model = Sequential()

        # Takes in a 600x800x3 image
        model.add(Convolution2D(32, 3, 3, subsample=(2,2), input_shape = self.img_shape, activation = 'relu'))
        # model.add(Convolution2D(32, 3, 3, subsample=(2,2), activation = 'relu'))

        model.add(MaxPooling2D((2,2)))

        model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation = 'relu'))
        # model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation = 'relu'))
        #model.add(Conv2D(64, (3,3), activation = 'relu'))

        model.add(MaxPooling2D((2,2)))

        # model.add(Convolution2D(128, 3, 3, subsample=(2,2), activation = 'relu'))
        # model.add(Convolution2D(128, 3, 3, subsample=(2,2), activation = 'relu'))
        #model.add(Conv2D(128, (3,3), activation = 'relu'))

        # model.add(MaxPooling2D((2,2)))

        model.add(Flatten())

        model.add(Dropout(0.40))
        model.add(Dense(256, activation = 'relu'))
        # model.add(Dropout(0.40))

        model.add(Dense(3, activation = 'softmax'))

        # Convolution layer 1 with 24 5x5 filters
        # model.add(Convolution2D(24, 5, 5, subsample=(2,2), input_shape = self.img_shape, activation='relu'))
        # # model.add(Dropout(.2))

        # # Convolution layer 2 with 36 5x5 filters
        # model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))

        # # Convolution layer 3 with 48 5x5 filters
        # model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))

        # # Convolution layer 4 with 64 3x3 filters
        # model.add(Convolution2D(64, 3, 3, activation='relu'))

        # # Convolution layer 5 with 64 3x3 filters
        # model.add(Convolution2D(64, 3, 3, activation='relu'))

        # # Flatten the output of the last convolution layer so it can be connected with a fully connected layer
        # model.add(Flatten())

        # # Fully Connected layer 1
        # model.add(Dense(100))
        # model.add(Dropout(.50))

        # # Fully Connected layer 2
        # model.add(Dense(50))
        # model.add(Dropout(.50))

        # # Fully Connected layer 2
        # model.add(Dense(10))

        # # Output single node (The steering angle)
        # model.add(Dense(3, activation = 'softmax'))
        
        # Split into training and validation data
        # img_train, img_test, label_train, label_test = train_test_split(imgs, labels, test_size = 0.20, random_state = 7, shuffle = True)

        # Compile the model with the optimizer, loss function, and desired metric
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Callback checkpoint to save the model's weights whenever it improves over the best
        checkpoint = ModelCheckpoint("test.h5", monitor='accuracy', verbose=1, save_best_only=True, mode='auto', period=1)

        # Train the model with our data 
        model.fit(x = imgs, y = labels, epochs = 10, validation_split=.2, callbacks=[checkpoint])

        model.save("test.h5")

        # self.evaluateModel(model, img_train, label_train)

    # def get_classification(self, image):
    #     """Determines the color of the traffic light in the image

    #     Args:
    #         image (cv::Mat): image containing the traffic light

    #     Returns:
    #         int: ID of traffic light color (specified in styx_msgs/TrafficLight)

    #     """
    #     #TODO implement light color prediction
    #     return TrafficLight.UNKNOWN
    
# Use main function to test data loading and training
if __name__ == '__main__':
    classifier = TLClassifier()
    
#     imgs, labels = classifier.loadData()
    
#     print imgs.shape
    
    classifier.trainModel()
