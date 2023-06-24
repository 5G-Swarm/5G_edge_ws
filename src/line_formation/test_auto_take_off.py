#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from mavros_msgs.msg import PositionTarget, State, HomePosition
from sensor_msgs.msg import Image, Imu, NavSatFix
import numpy as np
import time 

NAME_SPACE = "/iris_0"
mavros_state = State()
setModeServer = rospy.ServiceProxy(NAME_SPACE+'/mavros/set_mode', SetMode)
mavros_altitude0 = 0
mavros_altitude1 = 0
mavros_altitude2 = 0

########
set_mavros_altitude0 = 0
set_mavros_altitude1 = 0
set_mavros_altitude1 = 0

kp_z = 1#1.5
expected_height = 6

def mavros_state_callback(msg):
	global mavros_state
	mavros_state = msg

def mavros_altitude_callback0(ros_gps_pos):
	global mavros_altitude0
	mavros_altitude0 = ros_gps_pos.altitude
    #     _ = ros_gps_pos.latitude
    #     _ = ros_gps_pos.longtitude
def mavros_altitude_callback1(ros_gps_pos):
	global mavros_altitude1
	mavros_altitude1 = ros_gps_pos.altitude
 
def mavros_altitude_callback2(ros_gps_pos):
	global mavros_altitude2
	mavros_altitude2 = ros_gps_pos.altitude


def drone_test_publisher():
	
    rospy.init_node('drone_test', anonymous=True)
    global target_pos_pub
    rospy.Subscriber(NAME_SPACE+"/mavros/global_position/global", NavSatFix, mavros_altitude_callback0)
    rospy.Subscriber("/iris_1/mavros/global_position/global", NavSatFix, mavros_altitude_callback1)
    rospy.Subscriber("/iris_2/mavros/global_position/global", NavSatFix, mavros_altitude_callback2)      
    drone_vel_pub0 = rospy.Publisher(NAME_SPACE+'/mavros/setpoint_velocity/cmd_vel_unstamped',Twist, queue_size=10)
    drone_vel_pub1 = rospy.Publisher('/iris_1/mavros/setpoint_velocity/cmd_vel_unstamped',Twist, queue_size=10)
    drone_vel_pub2 = rospy.Publisher('/iris_2/mavros/setpoint_velocity/cmd_vel_unstamped',Twist, queue_size=10)
    # rospy.Subscriber(NAME_SPACE+"/mavros/state", State, mavros_state_callback)
    
    rate = rospy.Rate(10) 
    drone_vel0 = Twist()
    drone_vel1 = Twist()
    drone_vel2 = Twist()
    wait_time = 5
    run_time = 1
    pause_time = 1.0
    flight_time = time.time()#rospy.Time.now()

    finish_take_off_flag = 0
    init_mavros_altitude0 = mavros_altitude0
    init_mavros_altitude1 = mavros_altitude1
    init_mavros_altitude2 = mavros_altitude2
    while not rospy.is_shutdown():

        if (time.time()-flight_time)<=wait_time:#rospy.Duration(wait_time):#mavros_state.mode != "OFFBOARD":
            drone_vel0.linear.x = 0
            drone_vel0.linear.y = 0
            drone_vel0.linear.z = 0#[0.05,0.05,0.05]
            drone_vel0.angular.z = 0
            drone_vel1.linear.x = 0
            drone_vel1.linear.y = 0
            drone_vel1.linear.z = 0#[0.05,0.05,0.05]
            drone_vel1.angular.z = 0
            drone_vel2.linear.x = 0
            drone_vel2.linear.y = 0
            drone_vel2.linear.z = 0#[0.05,0.05,0.05]
            drone_vel2.angular.z = 0
            print("wait for take off",time.time()-flight_time)
            init_mavros_altitude0 = mavros_altitude0
            init_mavros_altitude1 = mavros_altitude1
            init_mavros_altitude2 = mavros_altitude2
            
        else:

            # if finish_take_off_flag ==0:
            set_mavros_altitude0 = init_mavros_altitude0+expected_height
            set_mavros_altitude1 = init_mavros_altitude1+expected_height
            set_mavros_altitude2 = init_mavros_altitude2+expected_height
            err_z0 = set_mavros_altitude0 - mavros_altitude0
            err_z1 = set_mavros_altitude1 - mavros_altitude1
            err_z2 = set_mavros_altitude2 - mavros_altitude2
            if np.abs(err_z0)<0.2 and np.abs(err_z1)<0.2 and np.abs(err_z2)<0.2:
                finish_take_off_flag = 1
                print("finish_take_off")
                vz0 = 0
                vz1 = 0
                vz2 = 0
                # continue
            else:
            # if mavros_altitude0:
                vz0 = err_z0*kp_z
                vz1 = err_z1*kp_z
                vz2 = err_z2*kp_z
            # else:
            #     vz0 = 0
            #     vz1 = 0
            #     vz2 = 0 
            vz0 = np.clip(vz0,-1,1)
            vz1 = np.clip(vz1,-1,1)
            vz2 = np.clip(vz2,-1,1)
            drone_vel0.linear.x = 0
            drone_vel0.linear.y = 0
            drone_vel0.linear.z = vz0#[0.05,0.05,0.05]
            drone_vel0.angular.z = 0
            drone_vel1.linear.x = 0
            drone_vel1.linear.y = 0
            drone_vel1.linear.z = vz1#[0.05,0.05,0.05]
            drone_vel1.angular.z = 0
            drone_vel2.linear.x = 0
            drone_vel2.linear.y = 0
            drone_vel2.linear.z = vz2#[0.05,0.05,0.05]
            drone_vel2.angular.z = 0
            # print("z_pid0",vz0)
            # print("z_pid1",vz1)
            # print("z_pid2",vz2)

        drone_vel_pub0.publish(drone_vel0)
        drone_vel_pub1.publish(drone_vel1)
        drone_vel_pub2.publish(drone_vel2)
        # print("flight_time:",time.time()-flight_time)
       
        rate.sleep()

    

    # if mavros_state.mode != "AUTO.LAND":
    #     setModeServer(custom_mode="AUTO.LAND")
    #     # drone_vel_pub.publish(drone_vel)
    #     print("landing",time.time()-flight_time)

if __name__ == '__main__':
    try:
        drone_test_publisher()
    except rospy.ROSInterruptException:
        pass

