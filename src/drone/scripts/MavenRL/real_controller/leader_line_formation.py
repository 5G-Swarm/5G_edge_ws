#!/usr/bin/env python2

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray,Header
from geometry_msgs.msg import Twist,PoseStamped,TwistStamped
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from mavros_msgs.msg import PositionTarget, State, HomePosition
from sensor_msgs.msg import Image, Imu, NavSatFix
from tf.transformations import euler_from_quaternion
import numpy as np
import time 
from autoware_msgs.msg import DroneSyn

drone_pub_list = []
NUM = 3
kp_x = 1
kp_y = 1
kp_z = 1
k_error = 1
kp_ang = 0.5
sx0=0
sy0=0
sx1=0
sy1=1.5
sx2=0
sy2=3
gx=2
gy=0
global flag
pos0=[]
pos1=[]
pos2=[]

flag=0#

def pose_callback0(data):
    global pos0 
    global pos0_yaw
    pos0 = data.gps
    pos0_yaw = euler_from_quaternion(data.imu)[2]
    print('receivedpos0',pos0,pos0_yaw)

def pose_callback1(data):
    global pos1
    global pos1_yaw
    pos1 = data.gps
    pos1_yaw = euler_from_quaternion(data.imu)[2]
    print('receivedpos1',pos1,pos1_yaw)
    
def pose_callback2(data):
    global pos2
    global pos2_yaw
    pos2 = data.gps
    pos2_yaw = euler_from_quaternion(data.imu)[2]
    print('receivedpos2',pos2,pos2_yaw)
   

def line_formation():
    rospy.init_node('line_formation', anonymous=True)
    rospy.Subscriber("/drone_state_0", DroneSyn, pose_callback0)
    rospy.Subscriber("/drone_state_1", DroneSyn, pose_callback1)
    rospy.Subscriber("/drone_state_2", DroneSyn, pose_callback2)
    # for i in range(NUM):    
    #     drone_pub_list.append(rospy.Publisher('/teamrl_controller_vel',TwistStamped, queue_size=10))
    drone_pub_0=rospy.Publisher('/teamrl_controller_vel',TwistStamped, queue_size=10)
    rate = rospy.Rate(10) 
    drone_vel_0 = TwistStamped()
    drone_vel_1 = TwistStamped()
    drone_vel_2 = TwistStamped()
    wait_time = 1
    finish_take_off_flag = 0
    
    init_pos1 = pos1
    init_pos2 = pos2
    err_0 = []
    err_1 = []
    err_2 = []
    set_pos0 = []
    set_pos1 = []
    set_pos2 = []
    flight_time = time.time()
    while not rospy.is_shutdown():
        # if (flag%2) :
        #     for i in range(NUM):
        #         f'set_pos{i}'[0],f'set_pos{i}'[1],f'set_pos{i}'[2]=[f'init_pos{i}[0]' + gx,f'init_pos{i}[1]' + gy,f'init_pos{i}[2]',np.pi/2]
        #         f'err_{i}'[0],f'err_{i}'[1],f'err_{i}'[2]=[f'set_pos{i}[0]'-f'pos{i}[0]',f'set_pos{i}[1]'-f'pos{i}[1]',f'set_pos{i}[2]'-f'pos{i}[2]',f'set_pos{i}[3]'-f'pos{i}_yaw']
        if pos0==[]:
            continue
        if (time.time()-flight_time)<=wait_time:#rospy.Duration(wait_time):#mavros_state.mode != "OFFBOARD":
            init_pos0 = pos0
            print("wait for take off",time.time()-flight_time)
            continue
        set_pos0=[init_pos0[0]+gx,init_pos0[1]+gy,init_pos0[2],np.pi/2]
        err_0=[set_pos0[0]-pos0[0],set_pos0[1]-pos0[1],set_pos0[2]-pos0[2],set_pos0[3]-pos0_yaw]

        # else:
        #     for i in range(NUM):
        #         f'set_pos{i}'[0],f'set_pos{i}'[1],f'set_pos{i}'[2],=[f'init_pos{i}[0]',f'init_pos{i}[1]',f'init_pos{i}[2]',np.pi/2]
        #         f'err_{i}'[0],f'err_{i}'[1],f'err_{i}'[2]=[f'set_pos{i}[0]'-f'pos{i}[0]',f'set_pos{i}[1]'-f'pos{i}[1]',f'set_pos{i}[2]'-f'pos{i}[2]',f'set_pos{i}[3]'-f'pos{i}_yaw']

        # err_pos0=[pos1[0]-pos0[0]-(sx1-sx0),pos1[1]-pos0[1]-(sy0-sy1)]
        # err_pos1=[0,0] 
        # err_pos2=[pos1[0]-pos2[0]-(sx1-sx2),pos1[1]-pos2[1]-(sy1-sy2)] 

        # for i in range(NUM):
        robot_id = 0
        drone_vel_0.header = Header()
        drone_vel_0.header.stamp = rospy.Time.now()
        drone_vel_0.header.frame_id = str(robot_id)
        drone_vel_0.twist.linear.x = kp_x * np.clip(err_0[0],-1,1) #+ k_error * np.clip(f'err_pos{i}[0]',-1,1)
        drone_vel_0.twist.linear.y = kp_y * np.clip(err_0[1],-1,1) #+ k_error * np.clip(f'err_pos{i}[1]',-1,1)
        drone_vel_0.twist.linear.z = kp_z * np.clip(err_0[2],-1,1)
        drone_vel_0.twist.angular.x = 0
        drone_vel_0.twist.angular.y = 0
        drone_vel_0.twist.angular.z = np.clip(err_0[3],-0.5,0.5)#0.2#float(wz)
        drone_pub_0.publish(drone_vel_0)
        print("pos0",pos0)
        print("init_pos0",init_pos0)
        print("angle",pos0_yaw)
        print("vx,z",drone_vel_0.twist.linear.x,drone_vel_0.twist.angular.z)
    
        if  np.abs(err_0[0]) <=0.2 and np.abs(err_0[1]) <=0.2 and np.abs(err_0[2]) <=0.2:
            flight_time = time.time()#rospy.Time.now()
            if (time.time()-flight_time)<=5:
                for i in range(NUM):
                    drone_pub_0.publish(TwistStamped())
            flag += 1

    rate.sleep()
    


if __name__ == '__main__':
    try:
        line_formation()
    except rospy.ROSInterruptException:
        pass



