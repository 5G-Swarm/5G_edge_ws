from time import sleep
import cv2
import numpy as np

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, Vector3 ,PoseStamped
from sensor_msgs.msg import Image,BatteryState,Imu
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

def callback(ros_msg):
    global msg_pub
    msg = PoseStamped()
    msg.header = Header()
    msg.header.frame_id = 'map'
    msg.pose.orientation.x = ros_msg.orientation.x
    msg.pose.orientation.y = ros_msg.orientation.y
    msg.pose.orientation.z = ros_msg.orientation.z
    msg.pose.orientation.w = ros_msg.orientation.w
    msg.pose.position.y = 0
    msg.pose.position.z = 0
    msg_pub.publish(msg)
    print("finish")



if __name__ == '__main__':
    rospy.init_node('5g-transfer', anonymous=True)
    global msg_pub 
    msg_pub = rospy.Publisher('/target_state', Odometry, queue_size=0)
    # rospy.Subscriber('/camera/color/image_raw', Image, callback_img)
    # rospy.Subscriber('/mavros/altitude', Altitude, callback_altitude)
    # rospy.Subscriber('/mavros/battery', BatteryState, callback_battery)
    # rospy.Subscriber('/mavros/global_position/raw/gps_vel', TwistStamped, callback_gps_vel)
    # rospy.Subscriber('/mavros/imu/data', Imu, callback)

    msg = Odometry()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        msg.pose.pose.position.x = 0.1
        msg.pose.pose.position.y = 0.1
        msg.pose.pose.position.z = 0.1
        msg.twist.twist.linear.x = 0
        msg.twist.twist.linear.y = 0
        msg.twist.twist.linear.z = 0
        msg_pub.publish(msg)

        rate.sleep()

