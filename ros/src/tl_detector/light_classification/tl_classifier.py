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
        
        self.img_shape = (600, 800, 3)
        
        self.save_file = "./models/newest_model.h5"
        
        self.current_best_model = "./models/newest_model.h5"
    
    def loadData(self):
        training_imgs = []
        training_labels = []
        
        # Loop through the different data folders and append the img and label to the lists
        for img_name in glob.glob(self.green_path + '*.jpg'):
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            training_imgs.append(img)
            training_labels.append(0)
            
        for img_name in glob.glob(self.yellow_path + '*.jpg'):
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            training_imgs.append(img)
            training_labels.append(1)
            
        for img_name in glob.glob(self.red_path + '*.jpg'):
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            training_imgs.append(img)
            training_labels.append(2)
         
        if len(training_imgs) > 0:
            self.img_shape = training_imgs[0].shape
            
        return np.array(training_imgs), np.array(training_labels)
    
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
        
        # Takes in a 600x800x3 image and normalizes it
        model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape = self.img_shape))

        model.add(Convolution2D(32, 3, 3, subsample=(2,2), activation = 'relu'))

        model.add(MaxPooling2D((2,2)))

        model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation = 'relu'))

        model.add(MaxPooling2D((2,2)))

        model.add(Flatten())

        model.add(Dropout(0.40))
        model.add(Dense(256, activation = 'relu'))
        # model.add(Dropout(0.40))

        model.add(Dense(3, activation = 'softmax'))

        # Compile the model with the optimizer, loss function, and desired metric
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Callback checkpoint to save the model's weights whenever it improves over the best
        checkpoint = ModelCheckpoint(self.save_file, monitor='accuracy', verbose=1, save_best_only=True, mode='auto', period=1)

        # Train the model with our data 
        model.fit(x = imgs, y = labels, epochs = 4, validation_split=.2, callbacks=[checkpoint])

        model.save(self.save_file)

        # self.evaluateModel(model, img_train, label_train)
        
    # Load and returns the current best model
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
        #TODO implement light color prediction
        # Load the model
        model = self.loadModel()
        
        # Resize the image to the size the model takes
        image = cv2.resize(image, (self.img_shape[1], self.img_shape[0]))
        
        # Convert to a numpy array
        arr_img = np.asarray(image)
        
        # Add an extra dimension so the network will accept it. Essentially creates a batch of size 1 to feed into the network
        arr_img = arr_img.reshape(1, arr_img.shape[0], arr_img.shape[1], arr_img.shape[2])
        
        # Get the output of the neural net
        pred = model.predict(arr_img)
        
        # Order the output from highest to lowest
        pred_class = pred.argmax(axis=-1)
        
        # Get the actual classification
        best_class = pred_class[0]
        
        print best_class
        
        # return TrafficLight.UNKNOWN
    
# Use main function to test data loading and training
if __name__ == '__main__':
    classifier = TLClassifier()
    
#     classifier.trainModel()
    
#     imgs, labels = classifier.loadData()
    
#     print imgs.shape

    img = cv2.imread("./data/red/0.jpg")
    
    classifier.get_classification(img)
    
    
