#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.xy_waypoints = None
        self.kdt_waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        
        # If the 2d waypoints and kd tree waypoints haven't been created yet then create them
        if not self.xy_waypoints:
            # Grab just the x and y values from the pose of the waypoints in the list
            self.xy_waypoints = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            
            # Use the x,y coordinates to create a KDTree that can be efficiently searched in nearest neighbor calculations
            self.kdt_waypoints = KDTree(self.xy_waypoints)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        
        # Gets the index closest light waypoint and it's current state
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        # Query the KD Tree of waypoints to get the index of the one closest to the car
        closest_wp_idx = self.kdt_waypoints.query([x, y], 1)[1] # Index 1 grabs the index of the result instead of the result point
        return closest_wp_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        return light.state
#         if(not self.has_image):
#             self.prev_light_loc = None
#             return False

#         cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

#         #Get classification
#         return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        # This will all be different once the images are actually checked intead of the raw data from the traffic light topic
        # The location of the traffic light and the stop line for it are different so we need to store both
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            # TODO: Check if this matches with the car wp in the waypoint updater code
            car_nearest_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            
            # Find the closest traffic light waypoint to the car's wp
            # Do this by finding the two waypoints with the fewest number of waypoints between them
            diff = len(self.waypoints.waypoints)
            
            # Loop through the lights to find the closest one
            for i, light in enumerate(self.lights):
                # Get the stop line position
                stop_line = stop_line_positions[i]
                
                # Find the waypoint closest to the stop line by querying the KD Tree
                closest_wp_idx = self.get_closest_waypoint(stop_line[0], stop_line[1])
                
                # Get the number of waypoints between the car and the stop line
                dist = closest_wp_idx - car_nearest_wp_idx
                
                # If the stop line is in front of the car and there are fewer waypoints in between than the current best then update
                if dist >= 0 and dist < diff:
                    diff = dist
                    closest_light = light
                    line_wp_idx = closest_wp_idx

        #TODO find the closest visible traffic light (if one exists)

        # This will be changed to return the classifier output instead
        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state
        #self.waypoints = None
        
        # If no traffic light is detected or we can't determine its state then return -1
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
