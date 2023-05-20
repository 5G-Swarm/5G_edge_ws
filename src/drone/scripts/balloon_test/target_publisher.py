#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################
####          Copyright 2020 GuYueHome (www.guyuehome.com).          ###
########################################################################

# 该例程将发布/person_info话题，自定义消息类型learning_topic::Person

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from autoware_msgs.msg import DroneSyn
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from Triangulation import triangulation
from detect_balloon import detect
# from learning_topic.msg import Person



def targetpos_Callback(ros_drone_1,ros_drone_2):
     
    global target_pos_pub
    
    global drone_1, drone_2
    drone_1.update(ros_drone_1)
    drone_2.update(ros_drone_2)
    
    config = USB_Camera(drone_1,drone_2)
    
    target = triangulation(config.Roi_points_left,config.Roi_points_right,config.R,config.T,config.cam_matrix_left,config.cam_matrix_right,1)
    target_msg = Float32MultiArray(data=target[0])
    target_pos_pub.publish(target_msg)
    print("success")

def target_publisher():
	# ROS节点初始化
    rospy.init_node('target_location', anonymous=True)
    global target_pos_pub
    target_pos_pub = rospy.Publisher('/target_pos',Float32MultiArray, queue_size=10)
    

	# 创建一个Publisher，发布名为/person_info的topic，消息类型为learning_topic::Person，队列长度10
	# rospy.Subscriber("/img_1", Image, targetpos_Callback)
    sub_drone_1 = message_filters.Subscriber("/drone_1",DroneSyn)
    sub_drone_2 = message_filters.Subscriber("/drone_2",DroneSyn)

    sync = message_filters.ApproximateTimeSynchronizer([sub_drone_1,sub_drone_2],10,1,allow_headerless = True)
    sync.registerCallback(targetpos_Callback)
    global drone_1,drone_2 
    drone_1 = Drone_camera()
    drone_2 = Drone_camera()
    #设置循环的频率
    rate = rospy.Rate(10) 

    while not rospy.is_shutdown():
        # 初始化learning_topic::Person类型的消息

        # 按照循环频率延时
        rate.sleep()


class Drone_camera(object):
    def __init__(self,extrinsic_mat =np.eye(4),intrinsic_mat=np.eye(3),distortion=np.zeros((1,4))):
        self.transform_world = np.eye(4)
        # self.transform_world[:3,:3] = Rotation.from_quat([imu_ori.x,imu_ori.y,imu_ori.z,imu_ori.w]).as_matrix()
        # self.transform_world[:3,3] = np.array([gps_pos.longtitude,gps_pos.latitude,gps_pos.altitude])
        self.extrinsic_mat = extrinsic_mat
        self.intrinsic_mat = intrinsic_mat
        self.distortion = distortion
        # self.pix_detect = np.array([0,0]) 
    
    def update(self,drone_msg):
        temp_bridge = CvBridge()
        img = temp_bridge.imgmsg_to_cv2(drone_msg.img)
        self.pix_detect = np.array([[100,100]])#detect(img)
        self.transform_world[:3,:3] = Rotation.from_quat(drone_msg.imu[:4]).as_matrix()
        self.transform_world[:3,3] = np.array(drone_msg.gps[:3])
 


class USB_Camera(object):   
    def __init__(self,left_drone,right_drone):
        # 左相机内参 及 畸变系数
        self.cam_matrix_left = left_drone.intrinsic_mat#np.array([[1271.71,-0.1385,376.5282],[0.,1270.87,258.1373],[0.,0.,1.]])
        self.distortion_l = left_drone.distortion#np.array([[-0.5688,5.9214,-0.00038018,-0.00052731,-61.7538]])
        # 右相机内参 及 畸变系数
        self.cam_matrix_right = right_drone.intrinsic_mat#np.array([[1269.2524,-2.098026,367.9874],[0.,1267.1973,246.2712],[0.,0.,1.]])
        self.distortion_r = right_drone.distortion#np.array([[-0.5176,2.4704,-0.0011,0.0012,-16.1715]])
        # 右边相机相对于左边相机的 旋转矩阵 R ， 平移矩阵 T
        self.tran_l_to_r = np.multiply(np.linalg.inv(left_drone.transform_world),right_drone.transform_world)
        self.R = self.tran_l_to_r[:3,:3]#np.array([[1.0000,-0.0088,-0.0043],[0.0088,0.9999,0.0072],[0.0042,-0.0072,1.0000]])
        self.T = self.tran_l_to_r[:3,3].reshape(3,1)#np.array([[-68.5321],[-0.5832],[-4.0933]])
        # 焦距
        self.focal_length = 6.00
        # 左右相机之间的距离 取 T 向量的第一维数值 单位 mm
        self.baseline = np.abs(self.T[0])
        self.Roi_points_left = left_drone.pix_detect
        self.Roi_points_right = right_drone.pix_detect

if __name__ == '__main__':
    try:
        target_publisher()
    except rospy.ROSInterruptException:
        pass
