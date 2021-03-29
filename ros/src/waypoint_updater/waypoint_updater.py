#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
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

# TODO: Car seems to break after 1 full lap? Waypoints just stopped updating for a while and car swered off track

# Having this number at 200 broke the code. I think it was taking too long to generate that many waypoints so the path would lag in the simulator
LOOKAHEAD_WPS = 75 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        # Apparently only published once so we need to store this
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        
       

        # TODO: Add other member variables you need below
        # Store the pose of the car from the simulator
        self.car_pose = None
        
        self.car_vel = None
        
        # Create a blank lane message to store the list of waypoints in when it's first published
        self.all_waypoints = None
        
        # Create a placeholder for the x,y waypoints so we can easily reference them
        self.xy_waypoints = None
        
        # Create a placeholder for the x,y waypoints stored in a KDTree
        self.kdt_waypoints = None
        
         # Use -1 to represent no lights currently being red
        self.light_idx = -1
        
        # Call the main waypoint generation function which will loop at 50hz
        self.generate_waypoints()

    # Looks through all the waypoints to find the first n waypoints in front of the car's current pose
    def pose_cb(self, msg):
        # TODO: Implement
        self.car_pose = msg
        
    # Generate waypoints at a 50 hz rate
    def generate_waypoints(self):
        rate = rospy.Rate(50)
        # Loop until roscore is shut down
        while not rospy.is_shutdown():
            # Make sure that a car pose and the waypoints are available
            if self.car_pose and self.all_waypoints:
                # Grab the x and y values from the car pose 
                x = self.car_pose.pose.position.x
                y = self.car_pose.pose.position.y

                # Get the index of the closest waypoint to the car from the KD Tree
                # This is basically just performing a very efficient nearest neighbor search
                if not self.kdt_waypoints:
                    # If the KD tree hasn't been made yet then just skip to the next iteration in the loop
                    continue
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
                prod = np.dot(closest_coord_vec - prev_coord_vec, car_pos_vec - closest_coord_vec)

                # If the result is positive then take the point next in the list after the closest point
                if prod > 0:
                    closest_idx = (closest_idx+1)%len(self.xy_waypoints) # Mod by the number of waypoints to loop around instead of going out of bounds
                    
                    
                # Blank lane message to be populated with waypoints and then published to the final waypoints publisher
                final_waypoints = Lane()
                    
                # Populate the final waypoints with the n many waypoints after the closest coordinate index
                final_waypoints.header = self.all_waypoints.header
                final_waypoints.waypoints = self.all_waypoints.waypoints[closest_idx: closest_idx + LOOKAHEAD_WPS]    
                
                
                # TODO: Determine if the closest traffic light waypoint is within a certain distance threshold
                if not self.light_idx == -1:
                    print("Car: {}".format(closest_idx))
                    print("Light: {}".format(self.light_idx))
                    
                    # TODO: Light idx also keeps being behind the closest idx
                    dist = self.distance(self.all_waypoints.waypoints, closest_idx, self.light_idx)
                    
                    print("Distance {}".format(dist))
                    
                    # TODO: Fix car not going again after coming to a stop. Car also doesn't seems to decrease speed for stop lights again now. Cool
                    
                    # If within the threshold then begin decelerating the car
                    if dist <= 100 and closest_idx < self.light_idx:
                        current_vel = self.get_waypoint_velocity(self.all_waypoints.waypoints[closest_idx])
                        speed_decrement = float(self.light_idx - closest_idx)/current_vel
                        
                        print("Decrement {}".format(speed_decrement))
                        
                        num_wp_between = closest_idx - self.light_idx
                        
                        for i in range(num_wp_between):
                            temp_current_vel = self.get_waypoint_velocity(final_waypoints.waypoints[i])
                            new_vel = temp_current_vel - speed_decrement*(i+1)
                            
                            self.set_waypoint_velocity(final_waypoints.waypoints, i, new_vel)
                            
                            # Check to make sure we don't go out of bounds
                            if i >= len(final_waypoints.waypoints) - 1:
                                break

                                

                # Publish the waypoints
                self.final_waypoints_pub.publish(final_waypoints)
            # Sleep to hit the desired frequency
            rate.sleep()
                
        

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
        # Grab the index of the closest red light to the car
        self.light_idx = msg.data

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
