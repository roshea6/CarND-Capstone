#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
import numpy as np
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        # Apparently only published once so we need to store this
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        
        # Create a blank lane message to store the list of waypoints in when it's first published
        self.all_waypoints = Lane()
        
        # Create a placeholder for the x,y waypoints so we can easily reference them
        self.xy_waypoints = None
        
        # Create a placeholder for the x,y waypoints stored in a KDTree
        self.kdt_waypoints = None

        rospy.spin()

    # Looks through all the waypoints to find the first n waypoints in front of the car's current pose
    def pose_cb(self, msg):
        # TODO: Implement
        # Blank lane message to be populated with waypoints and then published to the final waypoints publisher
        final_waypoints = Lane()
        
        # Used to keep track of how many waypoints we've added to our message
        num_waypoints = 0
        
        # Grab the x and y values from the car pose 
        x = msg.pose.position.x
        y = msg.pose.position.y
        
        # Get the index of the closest waypoint to the car from the KD Tree
        # This is basically just performing a vry efficient nearest neighbor search
        if not self.kdt_waypoints:
            return
        else:
            closest_idx = self.kdt_waypoints.query([x, y], 1)[1] # Index 1 grabs the index of the result instead of the result point
        
        # Get the closest coordinate and the one directly behind it in the list of waypoints
        closest_coord = self.xy_waypoints[closest_idx]
        prev_coord = self.xy_waypoints[closest_idx - 1] # If these ends up being negative it should just grab the last point in the array
        
        # Create vectors from the 3 relevant points so we can determine which direction they're facing in
        closest_coord_vec = np.array(closest_coord)
        prev_coord_vec = np.array(prev_coord)
        car_pos_vec = np.array([x, y])
        
        # Determine if the vector between the car and closest point and the vector between the closest point and and previous point are pointing in the same direction using the dot product
        
        # Scenario 1
        # If the dot product between these two vectors is less than zero then the vectors are pointing in opposite directions
        # This means that the closest point is in front of the car and the point before that is behind the car   *(prev_coord)<---vec1--- car ----vec2--->*(closest_coord)
        
        # Scenario 2 
        # If the dot product between these two vectors is greater than zero then the vectors are pointing in the same direction
        # This means that the closest point is behind the car and the point before that is also behind the car  *(prev_coord)<---vec1--- *(closest_coord)<---vec2--- car
        prod = np.dot(closest_coord_vec - car_pos_vec, car_pos_vec - closest_coord_vec)
        
        # If the result is positive then take the point next in the list after the closest point
        if prod > 0:
            closest_idx = (closest_idx+1)%len(self.xy_waypoints) # Mod by the number of waypoints to loop around instead of going out of bounds
        
        
        # Populate the final waypoints with the n many waypoints after the closest coordinate index
        final_waypoints.header = self.all_waypoints.header
        final_waypoints.waypoints = self.all_waypoints.waypoints[closest_idx: closest_idx + LOOKAHEAD_WPS]                    
        
        # Publish the waypoints
        self.final_waypoints_pub.publish(final_waypoints)
                
        

    # Stores the list of all waypoints in a local Lane message which will be used to determine which points are directly ahead of the car
    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.all_waypoints = waypoints
        
        # If the 2d waypoints and kd tree waypoints haven't been created yet then create them
        if not self.xy_waypoints:
            # Grab just the x and y values from the pose of the waypoints in the list
            self.xy_waypoints = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            
            # Use the x,y coordinates to create a KDTree that can be efficiently searched in nearest neighbor calculations
            self.kdt_waypoints = KDTree(self.xy_waypoints)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
