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
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2

import time
from informer import Informer
from proto.python_out import marker_msgs_pb2, geometry_msgs_pb2, path_msgs_pb2, cmd_msgs_pb2, ctrl_msgs_pb2


class Client(Informer):
    def send_msg(self, message):
        self.send(message, 'msg')
    
    def send_odm(self, message):
        self.send(message, 'odm')

    def send_pcd(self, message):
        self.send(message, 'pcd')

    def send_img(self, message):
        self.send(message, 'img')

def ros_marker2pb(ros_marker : TrackingObjectMarker):
    marker = marker_msgs_pb2.Marker()
    # type: TrackingObjectMarker
    marker.time_stamp = ros_marker.header.stamp.secs
    marker.id = ros_marker.track_id
    marker.pose.position.x = ros_marker.marker.pose.position.x
    marker.pose.position.y = ros_marker.marker.pose.position.y
    marker.pose.position.z = ros_marker.marker.pose.position.z
    marker.pose.orientation.x = ros_marker.marker.pose.orientation.x
    marker.pose.orientation.y = ros_marker.marker.pose.orientation.y
    marker.pose.orientation.z = ros_marker.marker.pose.orientation.z
    marker.pose.orientation.w = ros_marker.marker.pose.orientation.w
    marker.scale.x = ros_marker.marker.scale.x
    marker.scale.y = ros_marker.marker.scale.y
    marker.scale.z = ros_marker.marker.scale.z
    marker.color.r = ros_marker.marker.color.r
    marker.color.g = ros_marker.marker.color.g
    marker.color.b = ros_marker.marker.color.b
    return marker

def parse_ros_marker_list(ros_marker_array: TrackingObjectMarkerArray):
    marker_list = marker_msgs_pb2.MarkerList()
    for ros_mark in ros_marker_array.markers:
        mark = ros_marker2pb(ros_mark)
        marker_list.marker_list.append(mark)
    return marker_list

def callback_mark_array(ros_marker_array: TrackingObjectMarkerArray):
    global ifm
    marker_list = parse_ros_marker_list(ros_marker_array)
    sent_data = marker_list.SerializeToString()
    # print('send', len(sent_data))
    ifm.send_msg(sent_data)

def callback_pcd(ros_pcd : PointCloud2):
    global ifm
    ifm.send_pcd(ros_pcd.data)

def callback_imu(ros_imu_data : Imu):
    pass

def callback_img(ros_img : Image):
    global ifm
    ifm.send_img(ros_img.data)
    # img = np.ndarray(shape=(768, 2048, 3), dtype=np.dtype("uint8"), buffer=ros_img.data)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('img', img)
    # cv2.waitKey(1)
    # _, jpeg = cv2.imencode('.jpg', img)
    # data = jpeg.tobytes()
    # ifm.send_img(data)


if __name__ == '__main__':
    rospy.init_node('vehicle_5g_transfer', anonymous=True)
    ifm = Client(config = 'config-vehicle.yaml')
    # path_pub = rospy.Publisher('/global_path_gps', Path, queue_size=0)
    # ctrl_pub = rospy.Publisher('/manual_ctrl', Vector3, queue_size=0)
    # rospy.Subscriber('/detection/lidar_detector/objects_markers_withID', TrackingObjectMarkerArray, callback_mark_array)
    # rospy.Subscriber('/base2gps', Odometry, callback_odometry)
    # rospy.Subscriber('/camera/color/image_raw', Image, callback_img)
    # rospy.Subscriber('/cmd_vel', Twist, callback_cmd)
    # rospy.Subscriber('/license_plate_recognition', String, callback_recg)

    rospy.Subscriber('/lidar_center/velodyne_points', PointCloud2, callback_pcd)
    rospy.Subscriber('/stereo_color/right/image_color', Image, callback_img)
    rospy.Subscriber('/xsens/imu_data', Imu, callback_imu)

    # rospy.spin()
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        rate.sleep()
