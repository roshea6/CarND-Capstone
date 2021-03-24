#!/usr/bin/env python
import rospy
from sensor_msg.msg import Image
from geometry_msgs.msg import PoseStamped, Pose
import cv2
from cv_bridge import CvBridge

from styx_msgs.msg import TrafficLightArray, TrafficLight

# TODO: Make this node executable and add it launch file or see if it can be run from the terminal in the sim
# Class for subsribing to the relevant ROS topics and capturing images when near a trafic light
class DataCapture(object):
    def __init__(self):
        rospy.init_node("data_capture")
        
        rospy.Subscriber('/image_callback', Image, self.cameraCallback)
        rospy.Subscriber('/current_pose', PoseStamped, self.carPoseCallback)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.tlCallback)
    
    # Saves images from 
    def cameraCallback(self, img_msg):
        pass
    
    # Updates the stored current pose of the car
    def carPoseCallback(self, msg):
        pass
    
    # Stores the position of all traffic lights as well as their states
    # Uses a KD Tree to store the position once to increase search eficiency against the car's pose
    def tlCallback(self, msg):
        pass

if __name__ == '__main__':
    try:
        DataCapture()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start data capture node.')