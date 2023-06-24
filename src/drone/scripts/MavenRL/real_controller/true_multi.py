#!/usr/bin/env python2

import rospy
import math
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

gx=5.5
gy=-2.1
gz=2

NUM=[2,7,12]
formation_x=0
formation_y=1
kp_x = 0.5
kp_y = 0.5
kp_z = 0.5
kp_ang = 0.5
wait_takeoff_time=5
err0=[formation_x,formation_y,0,0]
err1=[0,0,0,0]
err2=[0,0,0,0]
err_f=[err0,err1,err2]
pos0=[1,1,1,1]
pos1=[1,1,1,1]
pos2=[1,1,1,1]

pos_list=[pos0,pos1,pos2]
pos0_yaw=[0]
wait_time=2

def pose_callback0(data):
    global pos0
    pos0_yaw = euler_from_quaternion(data.imu)[2]
    pos0=[data.gps[0], data.gps[1], data.gps[2], pos0_yaw]

def pose_callback1(data):
    global pos1
    pos1_yaw = euler_from_quaternion(data.imu)[2]
    pos1=[data.gps[0], data.gps[1], data.gps[2], pos1_yaw]

def pose_callback2(data):
    global pos2
    pos2_yaw = euler_from_quaternion(data.imu)[2]
    pos2=[data.gps[0], data.gps[1], data.gps[2], pos2_yaw]

class drone:
    def __init__(self,pos_array):
        self.pos=pos_array
        self.initpos=[0,0,0,0]
        self.setpos=[self.initpos[0]+gx,self.initpos[1]+gy,self.initpos[2],np.pi/2]
        self.err=[0,0,0,0]
        self.v=[0,0,0,0]

     

    def fly(self,flag):
        global gx
        global gy
        if(flag%2):
            self.setpos=[self.initpos[0]+gx,self.initpos[1]+gy,self.initpos[2],np.pi/2]
        else:
            self.setpos=[self.initpos[0],self.initpos[1],self.initpos[2],np.pi/2]
        self.err=np.array(self.setpos)-np.array(self.pos)
        print('self.flag',flag)
        if abs(gx)>abs(gy):
            k=abs(gx)/abs(gy)
            self.v=np.clip(self.err,[-1,-1/k,-1,-0.5],[1,1/k,1,0.5])    
        else:
            k=abs(gy)/abs(gx)
            self.v=np.clip(self.err,[-1/k,-1,-1,-0.5],[1/k,1,1,0.5])        
        return self.v
    
   

def line_formation():

    rospy.init_node('line_formation')
    rospy.Subscriber("drone_state_2", DroneSyn, pose_callback0)
    rospy.Subscriber("drone_state_7", DroneSyn, pose_callback1)
    rospy.Subscriber("drone_state_12", DroneSyn, pose_callback2)
    drone_vel_pub = rospy.Publisher('/teamrl_controller_vel',TwistStamped, queue_size=10) 
    global flag,pos0,pos1,pos2,err0,err1,err2,err_f,NUM,gx,gy
    init_pos0 = pos0
    init_pos1 = pos1
    init_pos2 = pos2
    flag=1
    flag_land=0
    flag_z=1
    flag_x=1
    rate = rospy.Rate(10) 
    flight_time = time.time()
    while not rospy.is_shutdown():
        if (time.time()-flight_time)<=wait_time:
            v0=[0,0,0,0]
            v1=[0,0,0,0]
            v2=[0,0,0,0]
            v=[v0,v1,v2]
            init_pos0 = pos0
            init_pos1 = pos1
            init_pos2 = pos2
            setpos0_z=pos0[2]+gz
            setpos1_z=pos1[2]+gz
            setpos2_z=pos2[2]+gz
        else:
            if (flag_z%2 or flag_x%2):
                v0=[0,0,np.clip(setpos0_z-pos0[2],0,1),0]
                v1=[0,0,np.clip(setpos1_z-pos1[2],0,1),0]
                # v1=[0,0,np.clip(setpos1_z-pos1[2],0,1),0]
                v2=[0,0,np.clip(setpos2_z-pos2[2],0,1),0]
                v=[v0,v1,v2]
                if (abs(v0[2])<0.2 and abs(v1[2])<0.2):#and abs(v1[2]<0.2) and abs(v2[2]<0.2)):
                    init_pos0[2] = setpos0_z
                    init_pos1[2] = setpos1_z 
                    init_pos2[2] = setpos2_z
                    flag_z=2
                    print('flag_z',flag_z)
                    if (abs(init_pos0[0]-pos0[0])>0.2 or abs(init_pos0[1]-pos0[1])>0.2 or abs(init_pos1[0]-pos1[0])>0.2 or abs(init_pos1[1]-pos1[1])>0.2):
                        v0=np.clip(np.array(init_pos0)-np.array(pos0),[-1,-1,-1,-0.5],[1,1,1,0.5])
                        v1=np.clip(np.array(init_pos1)-np.array(pos1),[-1,-1,-1,-0.5],[1,1,1,0.5])
                        v2=np.clip(np.array(init_pos2)-np.array(pos2),[-1,-1,-1,-0.5],[1,1,1,0.5])
                        v=[v0,v1,v2]
                    else:
                        flag_x=2
                        
                    
                
            else:
                drone0=drone(pos0)
                drone0.initpos=init_pos0
                drone1=drone(pos1)
                drone1.initpos=init_pos1
                drone2=drone(pos2)
                drone2.initpos=init_pos2
                v0=drone0.fly(flag)
                v1=drone1.fly(flag)
                v2=drone2.fly(flag)
                v=[v0,v1,v2]
                err1=np.array(pos1)-np.array(pos0)
                err2=np.array(pos2)-np.array(pos1)
                err_f=[err0,err1,err2]
                print('err_f',err_f)

                if (abs(v0[0])<0.2 and abs(v0[1])<0.2 and abs(v0[2])<0.2 and abs(v1[0])<0.2 and abs(v1[1])<0.2 and abs(v1[2])<0.2 ):
                # if abs(v0[0])<0.05 and abs(v0[1])<0.05 and abs(v0[2])<0.05 and abs(v0[3])<0.05:
                    flag=flag+1
                    flag_land=0
                    if flag==6:
                        flag_land=1
                    stop_time=time.time()
                    while (time.time()-stop_time)<= wait_time:  
                        for i in range(3):
                            drone_vel0 = TwistStamped()
                            robot_id = NUM[i]
                            drone_vel0.header = Header()
                            drone_vel0.header.stamp = rospy.Time.now()
                            drone_vel0.header.frame_id = str(robot_id)
                            drone_vel0.twist.linear.x = 0
                            drone_vel0.twist.linear.y = 0
                            drone_vel0.twist.linear.z = 0
                            drone_vel0.twist.angular.x = flag_land
                            drone_vel0.twist.angular.y = 0
                            drone_vel0.twist.angular.z = 0
                            drone_vel_pub.publish(drone_vel0)
                    print('flag',flag)
    
            for i in range(3):       
                drone_vel0 = TwistStamped()
                robot_id = NUM[i]
                drone_vel0.header = Header()
                drone_vel0.header.stamp = rospy.Time.now()
                drone_vel0.header.frame_id = str(robot_id)
                drone_vel0.twist.linear.x = kp_x *v[i][0]
                drone_vel0.twist.linear.y = kp_y *v[i][1]
                drone_vel0.twist.linear.z = kp_z *v[i][2]
                drone_vel0.twist.angular.x = flag_land
                drone_vel0.twist.angular.y = 0
                drone_vel0.twist.angular.z = kp_ang *v[i][3] 
                drone_vel_pub.publish(drone_vel0)    
                print("v0",v0)    
                print("v1",v1)   
                print("v2",v2)   
            rate.sleep()


if __name__ == '__main__':
    try:
        line_formation()
    except rospy.ROSInterruptException:
        pass




