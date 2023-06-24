#!/usr/bin/env python2

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist,PoseStamped
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from mavros_msgs.msg import PositionTarget, State, HomePosition
from sensor_msgs.msg import Image, Imu, NavSatFix
from tf.transformations import euler_from_quaternion
import numpy as np
import time 
from autoware_msgs.msg import DroneSyn
NAME_SPACE = "/iris_0"
mavros_state = State()
setModeServer = rospy.ServiceProxy(NAME_SPACE+'/mavros/set_mode', SetMode)
mavros_altitude = 0
pos0 = PoseStamped()
pos1 = PoseStamped()
pos2 = PoseStamped()
########
set_mavros_altitude = 0
kp_x = 0.5#1.5
kp_y = 0.5
kp_z = 0.5
kp_ang = 0.5

def pose_callback0(data):
    global pos0
    pos0 = data.gps
    global pos0_yaw
    pos0_yaw = euler_from_quaternion(data.imu)[2]
    #print("received",pos.pose.position.x)

def pose_callback1(data):
    global pos1
    global pos1_yaw
    pos1 = data.gps
    pos1_yaw = euler_from_quaternion(data.imu)[2]
    
def pose_callback2(data):
    global pos2
    global pos2_yaw
    pos2 = data.gps
    pos2_yaw = euler_from_quaternion(data.imu)[2]
   

def line_formation():

    rospy.init_node('line_formation', anonymous=True)
    rospy.Subscriber("/drone_state_0", DroneSyn, pose_callback0)
    rospy.Subscriber("/drone_state_1", DroneSyn, pose_callback1)
    rospy.Subscriber("/drone_state_2", DroneSyn, pose_callback2)
    drone_vel_pub0 = rospy.Publisher('/teamrl_controller_vel',Twist, queue_size=10)
    # drone_vel_pub1 = rospy.Publisher('/iris_1/mavros/setpoint_velocity/cmd_vel_unstamped',Twist, queue_size=10)
    # drone_vel_pub2 = rospy.Publisher('/iris_2/mavros/setpoint_velocity/cmd_vel_unstamped',Twist, queue_size=10)
    
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
        else:
            set_pos0 = PoseStamped()
            set_pos1 = PoseStamped()
            set_pos2 = PoseStamped()
            
            # set_pos0.pose.position.x = init_pos0.pose.position.x + 10
            set_pos0.pose.position.x = init_pos0[0] + 3
            set_pos0.pose.position.y = init_pos0[1]
            set_pos0.pose.position.z = init_pos0[2]
            set_pos0_yaw = np.pi/2
            
            
            # set_pos1.pose.position.x = init_pos1.pose.position.x + 10
            set_pos1.pose.position.x = init_pos1[0] + 3
            set_pos1.pose.position.y = init_pos1[1]
            set_pos1.pose.position.z = init_pos1[2]
            set_pos1_yaw = np.pi/2
            
            # set_pos2.pose.position.x = init_pos2.pose.position.x + 10
            set_pos2.pose.position.x = init_pos2[0] + 3
            set_pos2.pose.position.y = init_pos2[1]
            set_pos2.pose.position.z = init_pos2[2]
            set_pos2_yaw = np.pi/2
            
            err_x0 = set_pos0.pose.position.x - pos0[0]
            err_y0 = set_pos0.pose.position.y - pos0[1]
            err_z0 = set_pos0.pose.position.z - pos0[2]
            err_posx0 = pos1[0]- pos0[0] - 1
            err_posy0 = pos1[1]- pos0[1]
            
            err_x1 = set_pos1.pose.position.x - pos1[0]
            err_y1 = set_pos1.pose.position.y - pos1[1]
            err_z1 = set_pos1.pose.position.z - pos1[2]
            
            err_x2 = set_pos2.pose.position.x - pos2[0]
            err_y2 = set_pos2.pose.position.y - pos2[1]
            err_z2 = set_pos2.pose.position.z - pos2[2]
            err_posx2 = pos1.pose.position.x - pos2[0] + 1
            err_posy2 = pos1.pose.position.y - pos2[1] 
            
            err_ang0 = set_pos0_yaw - pos0_yaw
            err_ang1 = set_pos1_yaw - pos1_yaw
            err_ang2 = set_pos2_yaw - pos2_yaw
            # if np.abs(err_x0)<0.2 & np.abs(err_x1)<0.2 & np.abs(err_x2)<0.2:
            #     finish_take_off_flag = 1
            #     print("finish_take_off")
            #     # continue
            #     vx0 = 0
            #     vx1 = 0
            #     vx2 = 0
            # else:
            vx0 = np.clip(err_x0*kp_x,-2,2)
            vy0 = np.clip(err_y0*kp_y,-2,2) 
            vz0 = np.clip(err_z0*kp_z,-2,2) 
            
            vx1 = np.clip(err_x1*kp_x,-2,2)
            vy1 = np.clip(err_y1*kp_y,-2,2) 
            vz1 = np.clip(err_z1*kp_z,-2,2) 
            
            vx2 = np.clip(err_x2*kp_x,-2,2)
            vy2 = np.clip(err_y2*kp_y,-2,2) 
            vz2 = np.clip(err_z2*kp_z,-2,2) 
            
            ang0 = np.clip(err_ang0 * kp_ang,-2,2)
            ang1 = np.clip(err_ang1 * kp_ang,-2,2)
            ang2 = np.clip(err_ang2 * kp_ang,-2,2)
             
            
            
            drone_vel0.linear.x = vx0 + err_posx0*kp_x
            drone_vel0.linear.y = vy0 + err_posy0*kp_y
            drone_vel0.linear.z = vz0#[0.05,0.05,0.05]
            drone_vel0.angular.z = ang0
            drone_vel1.linear.x = vx1
            drone_vel1.linear.y = vy1
            drone_vel1.linear.z = vz1#[0.05,0.05,0.05]
            drone_vel1.angular.z = ang1
            drone_vel2.linear.x = vx2 + err_posx2*kp_x
            drone_vel2.linear.y = vy2 + err_posy2*kp_y
            drone_vel2.linear.z = vz2#[0.05,0.05,0.05]
            drone_vel2.angular.z = ang2
            # print("err_posx0",err_posx0)
            # print("err_posy0",err_posy0)
            # print("err_posx2",err_posx2)
            # print("err_posy2",err_posy2)
            print('pos2',pos2)
            

        drone_vel_pub0.publish(drone_vel0)
        
        # drone_vel_pub1.publish(drone_vel1)
        # drone_vel_pub2.publish(drone_vel2)
        print("flight_time:",time.time()-flight_time)
      
        rate.sleep()


if __name__ == '__main__':
    try:
        line_formation()
    except rospy.ROSInterruptException:
        pass



