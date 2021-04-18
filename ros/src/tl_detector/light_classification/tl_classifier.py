from styx_msgs.msg import TrafficLight

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

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
graph = tf.get_default_graph()


class TLClassifier(object):
    def __init__(self):
        # self.save_path = "/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/data/"
        self.save_path = "./data/"
        self.green_path = self.save_path + 'green/'
        self.yellow_path = self.save_path + 'yellow/'
        self.red_path = self.save_path + 'red/'
        
        self.img_shape = (600, 800, 3)
        
        self.save_file = "./models/newest_model.h5"
        
        self.current_best_model = "/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/models/best_model.h5"
        
        # Load the model once so it can be used by the get_classification function
        self.model = self.loadModel()
    
    def loadData(self):
        training_imgs = []
        training_labels = []
        
        # Loop through the different data folders and append the img and label to the lists
        for img_name in glob.glob(self.green_path + '*.jpg'):
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            training_imgs.append(img)
            training_labels.append(0)
            
            # Flip the image vertically and horizantally to augment the image
            v_flip = cv2.flip(img, 0)
            training_imgs.append(v_flip)
            training_labels.append(0)
            
            h_flip = cv2.flip(img, 1)
            training_imgs.append(h_flip)
            training_labels.append(0)
            
        for img_name in glob.glob(self.yellow_path + '*.jpg'):
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            training_imgs.append(img)
            training_labels.append(1)
            
            # Flip the image vertically and horizantally to augment the image
            v_flip = cv2.flip(img, 0)
            training_imgs.append(v_flip)
            training_labels.append(1)
            
            h_flip = cv2.flip(img, 1)
            training_imgs.append(h_flip)
            training_labels.append(1)
            
        for img_name in glob.glob(self.red_path + '*.jpg'):
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            training_imgs.append(img)
            training_labels.append(2)
            
            # Flip the image vertically and horizantally to augment the image
            v_flip = cv2.flip(img, 0)
            training_imgs.append(v_flip)
            training_labels.append(2)
            
            h_flip = cv2.flip(img, 1)
            training_imgs.append(h_flip)
            training_labels.append(2)
         
        if len(training_imgs) > 0:
            self.img_shape = training_imgs[0].shape
            
        return np.array(training_imgs), np.array(training_labels)
    
    def trainModel(self):
        print "Loading training data"
        # Get the training images and labels
        imgs, labels = self.loadData()
        
        print "Training on " + str(len(imgs)) + " images"
        
        # Convert the labels array to a categorical array. Will basically make each entry have a corresponding value for each of the classes
        # For example the entry for a green labeled image will be [1, 0, 0]
        labels = to_categorical(labels, num_classes=3)
        
        # Define our neural net
        model = Sequential()
        
        # Takes in a 600x800x3 image and normalizes it
        model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape = self.img_shape))

        model.add(Convolution2D(32, 3, 3, subsample=(2,2), activation = 'relu'))

        model.add(MaxPooling2D((2,2)))

        model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation = 'relu'))

        model.add(MaxPooling2D((2,2)))

        model.add(Flatten())

        model.add(Dropout(0.40))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dropout(0.40))

        model.add(Dense(3, activation = 'softmax'))

        # Compile the model with the optimizer, loss function, and desired metric
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model with our data 
        model.fit(x = imgs, y = labels, epochs = 10, shuffle=True, validation_split=.2)

        # Save the model when it's done training
        model.save(self.save_file)

        
    # Loads and returns the current best model
    def loadModel(self):
        model = load_model(self.current_best_model)

        return model
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        # Resize the image to the size the model takes
        image = cv2.resize(image, (self.img_shape[1], self.img_shape[0]))
        
        # Convert to a numpy array
        arr_img = np.asarray(image)
        
        # Add an extra dimension so the network will accept it. Essentially creates a batch of size 1 to feed into the network
        arr_img = arr_img.reshape(1, arr_img.shape[0], arr_img.shape[1], arr_img.shape[2])
        
        # Need to use this for some odd reason to get the predict function to run without errors. Possibly a conflict with ROS or a quirk or running as a ROS node
        global graph
        with graph.as_default():
            # Get the output of the neural net
            pred = self.model.predict(arr_img)
        
        # Order the output from highest to lowest
        pred_class = pred.argmax(axis=-1)
        
        # Get the actual classification
        best_class = pred_class[0]
        
        # Return the state of the traffic light based on the classification returned
        if best_class == 0:
            print "Green Light"
            return TrafficLight.GREEN
        elif best_class == 1:
            print "Yellow Light"
            return TrafficLight.YELLOW
        elif best_class == 2:
            print "Red Light"
            return TrafficLight.RED
        
        return TrafficLight.UNKNOWN
    
# Use main function to test data loading and training
if __name__ == '__main__':
    classifier = TLClassifier()
    
    classifier.trainModel()

#     img = cv2.imread("./data/red/0.jpg")
    
#     print classifier.get_classification(img)
    
    
