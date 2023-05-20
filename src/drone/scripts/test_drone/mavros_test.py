#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty
import rospy
from mavros_msgs.msg import PositionTarget, State, HomePosition
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32, String
import time
import math


msg = """
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%%%%%%%%%%%%%%%%%%%%%%%
offboard_cotrol
%%%%%%%%%%%%%%%%%%%%%%%
---------------------------
CTRL-C to quit

"""
cur_Position_Target = PositionTarget()
mavros_state = State()
armServer = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
setModeServer = rospy.ServiceProxy('/mavros/set_mode', SetMode)
local_target_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
def __init__():
	rospy.init_node('PX4_AuotFLy' ,anonymous=True)
	rospy.Subscriber("/mavros/state", State, mavros_state_callback)
        #rospy.Subscriber("/mavros/local_position/pose",HomePosition, mavros_pos_callback)
	print("Initialized")

def mavros_state_callback(msg):
	global mavros_state
	mavros_state = msg
#	if mavros_state.armed == 0 :
#		print("disarm")

def mavros_pos_callback(msg):
        global mavros_pos
        mavros_pos = msg

def Intarget_local():
	set_target_local = PositionTarget()
	set_target_local.type_mask = 0b100111111000  
        set_target_local.coordinate_frame = 1
        set_target_local.position.x = 0
        set_target_local.position.y = 0
        set_target_local.position.z = 2
        set_target_local.yaw = 0
	return set_target_local

def run_state_update():
	if mavros_state.mode != 'OFFBOARD':
                setModeServer(custom_mode='OFFBOARD')
                local_target_pub.publish(cur_Position_Target)
        	print("wait offboard")
        else: 
        	local_target_pub.publish(cur_Position_Target)
		print("local_target_pub.publish")

global cur_Position_Target
cur_Position_Target = Intarget_local()

if __name__=="__main__":
	settings = termios.tcgetattr(sys.stdin)
	print (msg)
	__init__()
	if armServer(True) :
			print("Vehicle arming succeed!")
        if setModeServer(custom_mode='OFFBOARD'):
			print("Vehicle offboard succeed!")
	else:
			print("Vehicle offboard failed!")
	while(1):
                run_state_update()
                time.sleep(0.2)