#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import cv2
from time import sleep
import numpy as np
import threading

from informer import Informer

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header
from autoware_msgs.msg import DroneSyn

from proto.python_out import drone_state_msgs_pb2, drone_cmd_msgs_pb2

cv_bridge = CvBridge()

robot_num = 10
ifm_r2e_dict = {}
ifm_e2c_dict = {}

def gps2xy(longtitude, latitude):
    L = 6381372*math.pi*2
    W = L
    H = L/2
    mill = 2.3
    x = longtitude*math.pi/180
    y = latitude*math.pi/180
    y = 1.25*math.log(math.tan(0.25*math.pi+0.4*y))
    x = (W/2)+(W/(2*math.pi))*x
    y = (H/2)-(H/(2*mill))*y
    return x, y

def xy2gps(x, y):
    L = 6381372 * math.pi*2
    W = L
    H = L/2
    mill = 2.3
    latitude = ((H/2-y)*2*mill)/(1.25*H)
    latitude = ((math.atan(math.exp(latitude))-0.25*math.pi)*180)/(0.4*math.pi)
    longtitude = (x-W/2)*360/W
    return round(latitude,7), round(longtitude,7)

def parse_img(message, robot_id):
    global img_pub_list
    # print('get img!')
    # relay_img(message, robot_id)
    # print(message[:18].decode())

    ts_secs = int(message[:10].decode())
    ts_nsecs = int(message[10:19].decode())
    
    img_data = message[19:]
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr,  cv2.IMREAD_COLOR)

    ros_img = cv_bridge.cv2_to_imgmsg(img)
    ros_img.header = Header()
    # ros_img.header.stamp = rospy.Time.now()
    ros_img.header.stamp.secs = ts_secs
    ros_img.header.stamp.nsecs = ts_nsecs
    ros_img.header.frame_id = str(robot_id)
    # print(ros_img)

    try:
        img_pub_list[robot_id].publish(ros_img)
    except:
        pass
    # cv2.imshow('Image',img)
    # cv2.waitKey(2)

# def relay_img(message, robot_id):
#     global ifm_e2c_dict
#     if robot_id in ifm_e2c_dict.keys():
#         ifm_e2c_dict[robot_id].send_img(message)

def parse_state(message, robot_id):
    global state_pub_list

    ts_secs = int(message[:10].decode())
    ts_nsecs = int(message[10:19].decode())
    message_data = message[19:]

    state = drone_state_msgs_pb2.DroneState()
    state.ParseFromString(message_data)
    
    x, y = gps2xy(state.gps.lon_x, state.gps.lat_y)

    ros_state = DroneSyn()
    ros_state.header = Header()
    # ros_state.header.stamp = rospy.Time.now()
    ros_state.header.stamp.secs = ts_secs 
    ros_state.header.stamp.nsecs = ts_nsecs
    ros_state.header.frame_id = str(robot_id)
    #33426109.17371788, 7649906.485160576#33426121.959677193, 7649910.113136261
    # ros_state.gps[0] = state.gps.lon_x33426121.001844004, 7649888.462029896,,33426124.120370667, 7649907.702826089,
    ros_state.gps[0] = x-33426121.959677193#-33426125.122754235#-33426109.17371788#x-33426121.001844004#-33426124.120370667#-33425779.16677153#x-33425787.976609346#x-33425791.173#-33425802.4777#x-33425788.8125#x - 33425788.8125#center of top#x-33425781.539yard
    # ros_state.gps[1] = state.gps.lat_y
    ros_state.gps[1] = -(y-7649910.113136261)#-(y-7649903.858005151)#-(y-7649906.485160576)#-(y-7649888.462029896)#-7649907.702826089)#-7650170.090460286)#-(y-7650150.041062016)#-(y-7650161.17498)#-(y-7650161.05)#-(y - 7650161.05)#center of top#-(y-7650162.7679)yard
    ros_state.gps[2] = state.gps.alt_z
    ros_state.gps[3] = state.gps.vx
    ros_state.gps[4] = state.gps.vy
    ros_state.gps[5] = state.gps.vz
    ros_state.imu[0] = state.imu.quan_x
    ros_state.imu[1] = state.imu.quan_y
    ros_state.imu[2] = state.imu.quan_z
    ros_state.imu[3] = state.imu.quan_w
    ros_state.imu[4] = state.imu.w_x
    ros_state.imu[5] = state.imu.w_y
    ros_state.imu[6] = state.imu.w_z

    state_pub_list[robot_id].publish(ros_state)
    # print('11111\n\n\n\n state', robot_id)
    # relay_state(message, robot_id)
    pass

# def relay_state(message, robot_id):
#     global ifm_e2c_dict
#     if robot_id in ifm_e2c_dict.keys():
#         ifm_e2c_dict[robot_id].send_state(message)

def parse_cmd(message, robot_id):
    # relay_cmd(message, robot_id)
    pass

# def relay_cmd(message, robot_id):
#     global ifm_e2c_dict
#     if robot_id in ifm_e2c_dict.keys():
#         ifm_e2c_dict[robot_id].send_cmd(message)

class ServerR2E(Informer):
    def img_recv(self):
        self.recv('img', parse_img)

    def state_recv(self):
        self.recv('state', parse_state)

    def send_cmd(self, message):
        self.send(message, 'cmd')

#############################################

def parse_path(message, robot_id):
    relay_path(message, robot_id)

def relay_path(message, robot_id):
    global ifm_r2e_dict
    if robot_id in ifm_r2e_dict.keys():
        ifm_r2e_dict[robot_id].send_path(message)

def parse_ctrl(message, robot_id):
    relay_ctrl(message, robot_id)

def relay_ctrl(message, robot_id):
    global ifm_r2e_dict
    if robot_id in  ifm_r2e_dict.keys():
        ifm_r2e_dict[robot_id].send_ctrl(message)

# class ServerE2C(Informer):
#     def send_state(self, message):
#         self.send(message, 'state')

#     def send_cmd(self, message):
#         self.send(message, 'cmd')

#     def send_img(self, message):
#         self.send(message, 'img')

def start_r2e():
    global ifm_r2e_dict
    for i in range(robot_num):
        ifm_r2e_dict[i] = ServerR2E(config = 'config_drone.yaml', robot_id = i)

# def start_e2c():
#     global ifm_e2c_dict
#     for i in range(1, robot_num+1):
#         ifm_e2c_dict[i] = ServerE2C(config = 'config_e2c.yaml', robot_id = i)

#########callback
def send_cmd_callback(ros_cmd_msg):
    robot_id = int(ros_cmd_msg.header.frame_id)
    cmd = drone_cmd_msgs_pb2.DroneCmd()
    cmd.vx = ros_cmd_msg.twist.linear.x#1.0
    cmd.vy = ros_cmd_msg.twist.linear.y#1.0
    cmd.vz = ros_cmd_msg.twist.linear.z#1.0
    cmd.wz = ros_cmd_msg.twist.angular.z#1.0
    cmd.flag = drone_cmd_msgs_pb2.CmdType.Value("TAKE_OFF")
    print("shahao",drone_cmd_msgs_pb2.CmdType.Name(cmd.flag))
    # import pdb;pdb.set_trace()
    cmd_data = cmd.SerializeToString()
    ifm_r2e_dict[robot_id].send_cmd(cmd_data)


if __name__ == '__main__':
    rospy.init_node('drone_edge_transfer', anonymous=True)
    # img_pub = rospy.Publisher('/drone_image', Image, queue_size=0)
    img_pub_list = [rospy.Publisher('/drone_image_'+str(i), Image, queue_size=0) for i in range(robot_num)]
    state_pub_list = [rospy.Publisher('/drone_state_'+str(i), DroneSyn, queue_size=0) for i in range(robot_num)]
    rospy.Subscriber("/teamrl_controller_vel", TwistStamped, send_cmd_callback)

    start_r2e_thread = threading.Thread(
        target = start_r2e, args=()
    )
    # start_e2c_thread = threading.Thread(
    #     target = start_e2c, args=()
    # )
    start_r2e_thread.start()
    # start_e2c_thread.start()
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        rate.sleep()