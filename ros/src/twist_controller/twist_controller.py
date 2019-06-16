from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, 
	accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        
	self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

	self.vel_lpf = LowPassFilter(tau=0.5, ts=.02)

	self.vehicle_mass = vehicle_mass
	self.fuel_capacity = fuel_capacity
	self.brake_deadband = brake_deadband
	self.decel_limit = decel_limit
	self.accel_limit = accel_limit
	self.wheel_radius = wheel_radius
	self.wheel_base = wheel_base
	self.steer_ratio = steer_ratio
	self.max_lat_accel = max_lat_accel
	self.max_steer_angle = max_steer_angle
	
	self.last_time = rospy.get_time()
	min_speed = 0

	self.linear_pid_controller = PID(kp=0.8, ki=0, kd=0.05, mn=self.decel_limit, mx=0.5 * self.accel_limit)
        self.yaw_controller = YawController(self.wheel_base, self.steer_ratio, min_speed, self.max_lat_accel, self.max_steer_angle)
        self.steering_pid_controller = PID(kp=0.15, ki=0.001, kd=0.1, mn=-self.max_steer_angle, mx=self.max_steer_angle)

    def reset(self):
	self.linear_pid_controller.reset()
	self.steering_pid_controller.reset()
	

    def control(self, current_vel, linear_vel, angular_vel, cross_track_error, duration_in_seconds):
	#current_vel = self.vel_lpf.filt(current_vel)
	linear_velocity_error = linear_vel - current_vel
	velocity_correction = self.linear_pid_controller.step(linear_velocity_error, duration_in_seconds)
	brake = 0
	throttle = velocity_correction
	rospy.loginfo("throttle is: %f", throttle)
	if (throttle < 0):
	    deceleration = abs(throttle)
	    brake = (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY) * self.wheel_radius * deceleration if deceleration > self.brake_deadband else 0.
            throttle = 0

	predictive_steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        corrective_steering = self.steering_pid_controller.step(cross_track_error, duration_in_seconds)
        steering = predictive_steering + corrective_steering

        return throttle, brake, steering


