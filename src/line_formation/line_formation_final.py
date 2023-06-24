#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist,PoseStamped
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from mavros_msgs.msg import PositionTarget, State, HomePosition
from sensor_msgs.msg import Image, Imu, NavSatFix
import numpy as np
import time 

NAME_SPACE = "/iris_0"
mavros_state = State()
setModeServer = rospy.ServiceProxy(NAME_SPACE+'/mavros/set_mode', SetMode)
mavros_altitude = 0
pos0 = PoseStamped()
pos1 = PoseStamped()
pos2 = PoseStamped()
########
set_mavros_altitude = 0

kp_x = 1#1.5

      

def pose_callback0(data):
    global pos0
    pos0 = data
    #print("received",pos.pose.position.x)

def pose_callback1(data):
    global pos1
    pos1 = data
    
def pose_callback2(data):
    global pos2
    pos2 = data

   

def line_formation():

    rospy.init_node('line_formation', anonymous=True)
    rospy.Subscriber(NAME_SPACE+"/mavros/local_position/pose", PoseStamped, pose_callback0)
    rospy.Subscriber("/iris_1/mavros/local_position/pose", PoseStamped, pose_callback1)
    rospy.Subscriber("/iris_2/mavros/local_position/pose", PoseStamped, pose_callback2)
    drone_vel_pub0 = rospy.Publisher(NAME_SPACE+'/mavros/setpoint_velocity/cmd_vel_unstamped',Twist, queue_size=10)
    drone_vel_pub1 = rospy.Publisher('/iris_1/mavros/setpoint_velocity/cmd_vel_unstamped',Twist, queue_size=10)
    drone_vel_pub2 = rospy.Publisher('/iris_2/mavros/setpoint_velocity/cmd_vel_unstamped',Twist, queue_size=10)
    
    rate = rospy.Rate(10) 
    drone_vel0 = Twist()
    drone_vel1 = Twist()
    drone_vel2 = Twist()
    wait_time = 0.5
    run_time = 1
    pause_time = 1.0
    flight_time = time.time()#rospy.Time.now()

    finish_take_off_flag = 0
    init_pos0 = pos0
    init_pos1 = pos1
    init_pos2 = pos2
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
            drone_vel2.linear.z = 0#[2.05,0.05,0.05]
            drone_vel2.angular.z = 0
            print("wait for take off",time.time()-flight_time)
            init_pos0 = pos0.pose.position.x
            init_pos1 = pos1.pose.position.x
            init_pos2 = pos2.pose.position.x
        else:
            set_pos0 = PoseStamped()
            set_pos1 = PoseStamped()
            set_pos2 = PoseStamped()
            set_pos0.pose.position.x = init_pos0 + 10
            set_pos1.pose.position.x = init_pos1 + 10
            set_pos2.pose.position.x = init_pos2 + 10
            err_x0 = set_pos0.pose.position.x - pos0.pose.position.x
            err_x1 = set_pos1.pose.position.x - pos1.pose.position.x
            err_x2 = set_pos2.pose.position.x - pos2.pose.position.x
            # if np.abs(err_x0)<0.2 & np.abs(err_x1)<0.2 & np.abs(err_x2)<0.2:
            #     finish_take_off_flag = 1
            #     print("finish_take_off")
            #     # continue
            #     vx0 = 0
            #     vx1 = 0
            #     vx2 = 0
            # else:
            vx0 = err_x0*kp_x 
            vx1 = err_x1*kp_x
            vx2 = err_x2*kp_x
            vx0 = np.clip(vx0,-2,2)
            vx1 = np.clip(vx1,-2,2)
            vx2 = np.clip(vx2,-2,2)
            drone_vel0.linear.x = vx0
            drone_vel0.linear.y = 0
            drone_vel0.linear.z = 0#[0.05,0.05,0.05]
            drone_vel0.angular.z = 0
            drone_vel1.linear.x = vx1
            drone_vel1.linear.y = 0
            drone_vel1.linear.z = 0#[0.05,0.05,0.05]
            drone_vel1.angular.z = 0
            drone_vel2.linear.x = vx2
            drone_vel2.linear.y = 0
            drone_vel2.linear.z = 0#[0.05,0.05,0.05]
            drone_vel2.angular.z = 0
            print("x0_pid",vx0)
            print("x1_pid",vx1)
            print("x2_pid",vx2)

        drone_vel_pub0.publish(drone_vel0)
        drone_vel_pub1.publish(drone_vel1)
        drone_vel_pub2.publish(drone_vel2)
        print("flight_time:",time.time()-flight_time)
      
        rate.sleep()


    

    # if mavros_state.mode != "AUTO.LAND":
    #     setModeServer(custom_mode="AUTO.LAND")
    #     # drone_vel_pub.publish(drone_vel)
    #     print("landing",time.time()-flight_time)

if __name__ == '__main__':
    try:
        line_formation()
    except rospy.ROSInterruptException:
        pass



