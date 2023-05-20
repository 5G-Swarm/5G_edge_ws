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
from std_msgs.msg import Header
from autoware_msgs.msg import DroneSyn

from proto.python_out import drone_state_msgs_pb2

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
    nparr = np.frombuffer(message, np.uint8)
    img = cv2.imdecode(nparr,  cv2.IMREAD_COLOR)

    ros_img = cv_bridge.cv2_to_imgmsg(img)
    ros_img.header = Header()
    ros_img.header.stamp = rospy.Time.now()
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
    state = drone_state_msgs_pb2.DroneState()
    state.ParseFromString(message)
    
    x, y = gps2xy(state.gps.lon_x, state.gps.lat_y)

    ros_state = DroneSyn()
    ros_state.header = Header()
    ros_state.header.stamp = rospy.Time.now()
    ros_state.header.frame_id = str(robot_id)

    # ros_state.gps[0] = state.gps.lon_x
    ros_state.gps[0] = x - 33425780
    # ros_state.gps[1] = state.gps.lat_y
    ros_state.gps[1] = 7650172 - y #y - 7650172
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

    def cmd_recv(self):
        self.recv('cmd', parse_cmd)

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

if __name__ == '__main__':
    rospy.init_node('drone_edge_transfer', anonymous=True)
    # img_pub = rospy.Publisher('/drone_image', Image, queue_size=0)
    img_pub_list = [rospy.Publisher('/drone_image_'+str(i), Image, queue_size=0) for i in range(robot_num)]
    state_pub_list = [rospy.Publisher('/drone_state_'+str(i), DroneSyn, queue_size=0) for i in range(robot_num)]


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