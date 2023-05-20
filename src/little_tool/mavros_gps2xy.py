#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import pyrealsense2 as rs
# import cv2
import numpy as np
# import time
# import message_filters
import rospy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
import utm
# from geometry_msgs.msg import TwistStamped,Twist
# from nav_msgs.msg import Odometry
origin_gps = [30.2628442,120.1167266, 92.983952]

def gps2xy_new(lat,lon):
    return utm.from_latlon(lat,lon)[:2]


def callback_gps(ros_gps_pos):
    global xy_pub,origin_xy
    # global drone_syn_pub
    print("begin")
    lon_x = ros_gps_pos.longitude
    lat_y = ros_gps_pos.latitude
    alt_z = ros_gps_pos.altitude
    temp_x,temp_y = gps2xy_new(lat_y,lon_x)
    
    xy_msg = PoseStamped()
    xy_msg.header.stamp = rospy.Time.now()
    # xy_msg.header.frame_id = "123"#
    xy_msg.pose.position.x = temp_x-origin_xy[0]
    xy_msg.pose.position.y = temp_y-origin_xy[1]
    xy_msg.pose.position.z = alt_z - origin_gps[2]
    
    xy_pub.publish(xy_msg)



if __name__ == '__main__':

    # import pdb;pdb.set_trace()
    rospy.init_node('drone_5g_transfer', anonymous=True)
    rospy.Subscriber('/mavros/global_position/global', NavSatFix, callback_gps)    
    xy_pub = rospy.Publisher('/gps2xy',PoseStamped, queue_size=10)

    origin_xy = list(gps2xy_new(origin_gps[0], origin_gps[1]))
    rospy.spin()
    rate = rospy.Rate(1000)
    #try:
    while not rospy.is_shutdown():
        rate.sleep()