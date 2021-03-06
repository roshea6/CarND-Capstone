
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

from yaw_controller import YawController
from pid import PID
import rospy

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        # Create the controllers we'll use to generate the throttle, brake, and steer commands
        
        self.total_mass = vehicle_mass + fuel_capacity*GAS_DENSITY
        self.wheel_radius = wheel_radius
        
        self.pid = PID(0.3, 0.1, 0.0, decel_limit, accel_limit)
        
        self.yaw_control = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        
        # Use rospy to keep track of the time between steps
        self.prev_time = rospy.get_time()
        

    def control(self, proposed_lin_vel, proposed_ang_vel, current_vel, dbw_status):
        # Calculate the current velocity error
        vel_error = proposed_lin_vel - current_vel
        
        # Get the time between steps
        current_time = rospy.get_time()
        sample_time = current_time - self.prev_time
        self.prev_time = current_time
        
        # If drive by wire isn't active then return nothing before doing any calculations to prevent unnecessary error buildup in PID
        if dbw_status == False:
            # Reset the PID as well 
            self.pid.reset()
            return 0.0, 0.0, 0.0
        
        # Get the result from the PID controller based on current error and time passed
        res = self.pid.step(vel_error, sample_time)
        
        throttle = 0
        brake = 0
        
        # If the result of the PID is positive then we want to accelerate and need to have throttle and no brake
        if res > 0:
            throttle = res
        # If the output is negative then we want to decelerate and need to apply the brake
        else:
            # Calculate the break output in terms of newton meters
            brake = self.total_mass * abs(res) * self.wheel_radius
        
        # Use the yaw controller to calculate the steering angle
        steer = self.yaw_control.get_steering(proposed_lin_vel, proposed_ang_vel, current_vel)

        return throttle, brake, steer
