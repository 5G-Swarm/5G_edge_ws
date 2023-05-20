#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import rospy
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, Vector3, PoseStamped
from autoware_msgs.msg import TrackingObjectMarker, TrackingObjectMarkerArray
from sensor_msgs.msg import Image, PointCloud2, Imu, Image, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

import time
from informer import Informer
from proto.python_out import marker_msgs_pb2, geometry_msgs_pb2, path_msgs_pb2, cmd_msgs_pb2, ctrl_msgs_pb2

class Client(Informer):
    def pcd_recv(self):
        self.recv('pcd', parse_pcd)

    def img_recv(self):
        self.recv('img', parse_img)


def parse_img(message, robot_id):
    # print('img', len(message))
    img = Image()
    img.data = message
    img.height = 768
    img.width = 2048
    img.encoding = "bgr8"
    img.is_bigendian = 0
    img.step = 6144

    img_pub.publish(img)


def parse_pcd(message, robot_id):
    # print('pcd', len(message))
    pcd = PointCloud2()
    pcd.header = Header()
    # pcd.header.stamp = rospy.Time.now()
    pcd.header.frame_id = 'lidar_center'
    pcd.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='ring', offset=16, datatype=PointField.UINT16, count=1),
    ]
    pcd.data = message
    pcd.point_step = 18
    pcd.width = len(pcd.data)//pcd.point_step
    pcd.height = 1
    pcd.row_step = 0
    pcd.is_bigendian = False
    pcd.is_dense = True

    pcd_pub.publish(pcd)

    # print('point_num:', ros_pcd.width)
    # pc = pc2.read_points(new_pcd, skip_nans=True, field_names=("x", "y", "z", "intensity", "ring"))
    # pc_list = []
    # for p in pc:
    #     pc_list.append( [p[0],p[1],p[2]] )
    # pc_list = np.array(pc_list)
    # print(pc_list.shape)



if __name__ == '__main__':
    rospy.init_node('cloud_5g_transfer', anonymous=True)
    ifm = Client(config = 'config-cloud.yaml')
    pcd_pub = rospy.Publisher('/lidar_center/velodyne_points2', PointCloud2, queue_size=0)
    img_pub = rospy.Publisher('/stereo_color/right/image_color2', Image, queue_size=0)
    # rospy.spin()
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        rate.sleep()
