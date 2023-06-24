#!/usr/bin/env python2

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
from geometry_msgs.msg import Twist,PoseStamped,TwistStamped
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from mavros_msgs.msg import PositionTarget, State, HomePosition
from sensor_msgs.msg import Image, Imu, NavSatFix
from tf.transformations import euler_from_quaternion
import numpy as np
from autoware_msgs.msg import DroneSyn
import time 

gx=0
gy=4
kp_x = 0.5
kp_y = 0.5
kp_z = 0.5
kp_ang = 0.5
pos0=[1,1,1,1]
pos1=[1,1,1,1]
pos2=[1,1,1,1]
pos_list=[pos0,pos1,pos2]
pos0_yaw=[0]
wait_time=5

def pose_callback0(data):
    global pos0
    pos0_yaw = euler_from_quaternion(data.imu)[2]
    pos0=[data.gps[0], data.gps[1], data.gps[2], pos0_yaw]

class drone:
    def __init__(self,pos_array):
        self.pos=pos_array
        self.initpos=[0,0,0,0]
        self.setpos=[self.initpos[0]+gx,self.initpos[1]+gy,self.initpos[2],np.pi/2]
        self.err=[0,0,0,0]
        self.v=[0,0,0,0]

     

    def fly(self,flag):
        if(flag%2):
            self.setpos=[self.initpos[0]+gx,self.initpos[1]+gy,self.initpos[2],np.pi/2]
        else:
            self.setpos=[self.initpos[0],self.initpos[1],self.initpos[2],np.pi/2]
        self.err=np.array(self.setpos)-np.array(self.pos)
        print('self.flag',flag)
        self.v=np.clip(self.err,[-1,-1,-1,-0.5],[1,1,1,0.5])     
        return self.v
    
   

def line_formation():

    rospy.init_node('line_formation')
    rospy.Subscriber("/drone_state_0", DroneSyn, pose_callback0)
    drone_vel_pub0 = rospy.Publisher('/teamrl_controller_vel',TwistStamped, queue_size=10) 
    global flag,pos0
    init_pos0 = pos0
    # print('init_pos0',init_pos0)
    flag=1
    rate = rospy.Rate(10) 
    flight_time = time.time()
    while not rospy.is_shutdown():
        if (time.time()-flight_time)<=wait_time:
            v0=[0,0,0,0]
            init_pos0 = pos0
         
        else:
            drone0=drone(pos0)
            drone0.initpos=init_pos0
            print('drone.initpos',drone0.initpos)
            print('drone.pos',drone0.pos)
            v0=drone0.fly(flag)
            
            if(abs(v0[0])<0.5 and abs(v0[1])<0.5 and abs(v0[2])<0.5 and abs(v0[3])<0.2):
                flag=flag+1
                stop_time=time.time()
                while (time.time()-stop_time)<= wait_time:
                    drone_vel0 = TwistStamped()
                    robot_id = 0
                    drone_vel0.header = Header()
                    drone_vel0.header.stamp = rospy.Time.now()
                    drone_vel0.header.frame_id = str(robot_id)
                    drone_vel0.twist.linear.x = 0
                    drone_vel0.twist.linear.y = 0
                    drone_vel0.twist.linear.z = 0
                    drone_vel0.twist.angular.x = 0
                    drone_vel0.twist.angular.y = 0
                    drone_vel0.twist.angular.z = 0
                    drone_vel_pub0.publish(drone_vel0)
                # print('flag',flag)
                
            
        drone_vel0 = TwistStamped()
        robot_id = 0
        drone_vel0.header = Header()
        drone_vel0.header.stamp = rospy.Time.now()
        drone_vel0.header.frame_id = str(robot_id)
        drone_vel0.twist.linear.x = kp_x *v0[0]
        drone_vel0.twist.linear.y = kp_y *v0[1]
        drone_vel0.twist.linear.z = kp_z *v0[2]
        drone_vel0.twist.angular.x = 0
        drone_vel0.twist.angular.y = 0
        drone_vel0.twist.angular.z = kp_ang *v0[3]
        print("v0",v0)
        drone_vel_pub0.publish(drone_vel0)    
        # print("v0",v0)     
        rate.sleep()


if __name__ == '__main__':
    try:
        line_formation()
    except rospy.ROSInterruptException:
        pass



