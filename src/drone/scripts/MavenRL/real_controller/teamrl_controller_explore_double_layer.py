

from difflib import restore
import gym
import gym_flock
import numpy as np
import dgl
import torch.nn as nn
import torch.nn.functional as F
import torch as t

import torch.optim as optim
import dgl
import dgl.function as fn
import math
import pdb
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable
from torch.distributions import Categorical
import torch
import networkx as nx
import pdb
# import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')
print(mpl.get_backend())
import matplotlib.pyplot as plt
from utils import *

##### ##############
import rllib
import sys
import os
sys.path.append(os.path.abspath(".."))
from rl_code.td3_dec import TD3_DEC as Method 
from rl_code.args import EnvParams,generate_args
from rl_code.data_graph import Data_Graph
from utils.real_env_double_layer import Real_env
# from utils.real_env_numdec import Real_env
from utils.basic_controller import Pid_controller,Visual_servo
from utils.function_generate import FunctorFactory
#####ros#####
import rospy
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import TwistStamped,PoseStamped,Pose
from std_msgs.msg import Header
from autoware_msgs.msg import DroneSyn


model_name = "test"
restore = False
test_flag = True#False#True#False# 
model_num = -1#774600#971800#227900
steps_per_episode = 500
load_path = "../rl_code/results/test_real/first_4n/saved_models"#first_4n#"../rl_code/results/test_real/forth/saved_models"#"../rl_code/results/test_ten/2o_ten_vel/saved_models"
start_episode = 0

##########

n_agent = 15
high_layer_id_list = [14,13,7,8,9,10]#[7,8,9,12,13]
low_layer_id_list = [12,1,2,3,4]#[11,2,3,4,5]
id_list_double_layer = low_layer_id_list + high_layer_id_list
real_env = Real_env(n_agents = n_agent)
real_env.radius = 10
real_env.radius_high = 12
real_env.radius_low = 8#12
real_env.expected_high_layer_id_list = high_layer_id_list
real_env.expected_low_layer_id_list = low_layer_id_list  
CONTROL_FREQ = 10
flag_estimate = 0
temp_vel_sum = 0
temp_count = 0
##########
def call_back_playground_xy(ros_playground):
	for i in range(4):
		real_env.pg_boundary[i] = np.array([ros_playground.poses[i].pose.position.x,ros_playground.poses[i].pose.position.y])
		

def call_back_target_state(ros_target_odom):
	# real_env.target_center[0] = ros_target_odom.pose.pose.position.x
	# real_env.target_center[1] = ros_target_odom.pose.pose.position.y
	# real_env.target_center[2] = ros_target_odom.pose.pose.position.z
	# real_env.target_center[3] = ros_target_odom.twist.twist.linear.x
	# real_env.target_center[4] = ros_target_odom.twist.twist.linear.y
	# real_env.target_center[5] = ros_target_odom.twist.twist.linear.z
	global flag_estimate 
	flag_estimate = 1
	real_env.estimate_target_center[0] = ros_target_odom.pose.pose.position.x
	real_env.estimate_target_center[1] = ros_target_odom.pose.pose.position.y
	real_env.estimate_target_center[2] = ros_target_odom.pose.pose.position.z
	real_env.estimate_target_center[3] = ros_target_odom.twist.twist.linear.x
	real_env.estimate_target_center[4] = ros_target_odom.twist.twist.linear.y
	real_env.estimate_target_center[5] = ros_target_odom.twist.twist.linear.z
	# print("real_env.estimate_target_center:",real_env.estimate_target_center)
	
def call_back_target_groundtruth(drone_state):
	# temp_sec = drone_state.header.stamp.secs
	# print("temp_sec",temp_sec)
	# global temp_vel_sum,temp_count
	# if temp_sec>(1656231780+360) and temp_sec<(1656231780+380):
	# 	temp_vel_sum += drone_state.gps[4]
	# 	temp_count +=1
	# 	print("shahao_vel",temp_count,drone_state.gps[4])
	# if temp_sec > (1656231780+380):
	# 	print("finish vel",temp_vel_sum/temp_count)
	real_env.target_center[0] = drone_state.gps[0]
	real_env.target_center[1] = drone_state.gps[1]
	real_env.target_center[2] = drone_state.gps[2]
	real_env.target_center[3] = drone_state.gps[3]
	real_env.target_center[4] = drone_state.gps[4]
	real_env.target_center[5] = drone_state.gps[5]

	temp_vel = np.linalg.norm(np.array([drone_state.gps[3],drone_state.gps[4]]))
	
	temp_vel_xy = Pose()
	temp_vel_xy.position.x = temp_vel 
	global temp_vel_publisher 
	temp_vel_publisher.publish(temp_vel_xy)
	



class call_back_template_bbx(object):
	def __init__(self, name):
		self._name = name
		# self._id = int(name[-1])
		self._id = int(name.split("_")[-1])

	def __call__(self,ros_target_Pose):
		# if real_env.state_drones_dict.has_key(self._id):
		target_center = [(ros_target_Pose.pose.orientation.x+ros_target_Pose.pose.orientation.z)/2,(ros_target_Pose.pose.orientation.y+ros_target_Pose.pose.orientation.w)/2]
		real_env.in_vision_bbx_dict[self._id] = np.array(target_center)

class FunctorFactory_bbx(object):
    def create(self,name):
        globals()[name] = call_back_template_bbx(name)



class call_back_template_state(object):
	def __init__(self, name):
		self._name = name
		self._id = int(name.split("_")[-1])

	def __call__(self,drone_state):
		# if real_env.state_drones_dict.has_key(self._id):
		temp = np.array(drone_state.gps+drone_state.imu)
		# temp[0] = temp[0]-33425788.8125
		# temp[1] = -(temp[1]-7650161.05)
		# if self._id == 0:
		# 	print("robot0_stamp:",drone_state.header.stamp)
		real_env.state_drones_dict[self._id] = temp
		if (self._id in high_layer_id_list) and (self._id not in real_env.real_high_layer_id_list):
			real_env.real_high_layer_id_list.append(self._id)
		elif (self._id in low_layer_id_list) and (self._id not in real_env.real_low_layer_id_list):
			real_env.real_low_layer_id_list.append(self._id)
			

class FunctorFactory_state(object):
    def create(self,name):
        globals()[name] = call_back_template_state(name)

def main(episodes):

	###global configuration###
	config = rllib.basic.YamlConfig()
	args = generate_args()
	config.update(args)
	config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
	config.set("model_dir",load_path)
	config.set("model_num",model_num)
	config.set("restore",restore)
	config.set("dim_state",real_env.observation_space.shape[1])
	config.set("dim_action",real_env.action_space.shape[1]) 
	config.set("n_agents",n_agent)
	config.set("test_flag",test_flag)
	config.env = "Two_Order_DecentralFormationFlying-v0"#"DecentralFormationFlying-v0"
	
	###ros configuration###
	rospy.init_node('teamrl_controller' ,anonymous=True)
	# import pdb;pdb.set_trace()
	factory_state = FunctorFactory_state()
	factory_bbx = FunctorFactory_bbx()
	
	callback_list_state = []
	callback_list_bbx = []
	for i in range(n_agent):
		name = "drone_state_callback_"+str(i)
		callback_list_state.append(name)
		factory_state.create(name)

		name = "target_bbx_callback_"+str(i)
		callback_list_bbx.append(name)
		factory_bbx.create(name)


	for i in range(n_agent):
		rospy.Subscriber("/drone_state_"+str(i), DroneSyn, globals()[callback_list_state[i]])

		rospy.Subscriber("/target_bbx_"+str(i), PoseStamped, globals()[callback_list_bbx[i]])
		# controller_pub_list.append(rospy.Publisher('/teamrl_controller_vel', Odometry, queue_size=10))
	controller_pub = rospy.Publisher('/teamrl_controller_vel', TwistStamped, queue_size=10)
	init_target_xy_publisher = rospy.Publisher('/init_target_xy', Pose, queue_size=10)
	global temp_vel_publisher
	temp_vel_publisher = rospy.Publisher('/compare_target_xy', Pose, queue_size=10)
	rospy.Subscriber("/balloon_estimation/odom", Odometry, call_back_target_state)
	#####new added 
	rospy.Subscriber("/drone_state_10", DroneSyn, call_back_target_groundtruth)
	# rospy.Subscriber("/target_bbx", PoseStamped, call_back_target_bbx)
	
	rospy.Subscriber("/place_xy", Path, call_back_playground_xy)
	


	rate = rospy.Rate(CONTROL_FREQ)
	drone_vel_msg = TwistStamped()


	##Controller configuration###
	writer = rllib.basic.create_dir(config, model_name)
	xy_controller = Method(config, writer)
	xy_controller.models_to_load = [xy_controller.actor,xy_controller.critic]
	xy_controller._load_model()
	z_controller = Pid_controller(1/CONTROL_FREQ)
	wz_controller = Visual_servo(1/CONTROL_FREQ)
	center_speed_x = 0#1#-1.2#float(np.random.uniform(-1.4,1.4,1))
	center_speed_y = 0#0.5#0.5#float(np.random.uniform(-1.4,1.4,1))
	
	controlvxy_dict_high = np.zeros((len(real_env.expected_high_layer_id_list),2))
	controlvxy_dict_low = np.zeros((len(real_env.expected_low_layer_id_list),2))

	###fake drones
	# real_env.state_drones_dict[11] = np.zeros(13)
	# real_env.state_drones_dict[11][0]=55
	# real_env.id_list.append(11)
	# real_env.real_high_layer_id_list.append(11) 
	
	# real_env.state_drones_dict[12] = np.zeros(13)
	# real_env.state_drones_dict[12][0]=55
	# real_env.id_list.append(12)
	# real_env.real_high_layer_id_list.append(12)

	while not rospy.is_shutdown():

		# real_env.target_center[2] = 39.3+2

		# real_env.target_center[3] = center_speed_x*math.sin(real_env.counter/60)
		# real_env.target_center[4] = center_speed_y*math.cos(real_env.counter/60)
		# real_env.target_center[:2] = real_env.target_center[:2] + real_env.target_center[3:5]*0.1
		# real_env.target_center = real_env.estimate_target_center.copy()
		global flag_estimate 
		if flag_estimate==0:
			real_env.estimate_target_center = real_env.target_center.copy()
			real_env.estimate_target_center[0] += 3.5
			msg_target_xy = Pose()
			msg_target_xy.position.x = real_env.estimate_target_center[0]
			msg_target_xy.position.y = real_env.estimate_target_center[1]
			init_target_xy_publisher.publish(msg_target_xy) 

		###observator###
		real_env.id_list = list(real_env.state_drones_dict.keys())
		obsxy_dict_low , obsxy_dict_high = real_env._get_obs_xy_double_layer()
		print("state:",real_env.id_list)
		
		if len(real_env.real_low_layer_id_list)<len(real_env.expected_low_layer_id_list):
			print("low_layer number less than",len(real_env.expected_low_layer_id_list))
		else:		
			###xy_controler###
			axay_dict = xy_controller.actor(torch.from_numpy(obsxy_dict_low).float().to(config.device))
			axay_dict = axay_dict.cpu().detach().numpy()*1.3
			controlvxy_dict_low = real_env._preprocessAction(axay_dict,real_env.feat_xy_low,kp=30)#30#5#15###90
			print("axay_dict_low:",real_env.expected_low_layer_id_list,axay_dict)
		
		if len(real_env.real_high_layer_id_list)<len(real_env.expected_high_layer_id_list):
			print("high_layer number less than",len(real_env.expected_high_layer_id_list))
		else:		
			###xy_controler###
			axay_dict = xy_controller.actor(torch.from_numpy(obsxy_dict_high).float().to(config.device))
			axay_dict = axay_dict.cpu().detach().numpy()*1.3
			controlvxy_dict_high = real_env._preprocessAction(axay_dict,real_env.feat_xy_high,kp=30)#30#5#15###90
			print("axay_dict_high:",real_env.expected_high_layer_id_list,axay_dict)
		
		controlvxy_dict = np.vstack((controlvxy_dict_low,controlvxy_dict_high))		
		print("controlvxy_dict:",id_list_double_layer,controlvxy_dict)	
		###z_controler###	
		# obsz_dict = real_env._get_obs_z()
		# temp_id = real_env.id_list.index(2)
		# print("error_z:",obsz_dict[temp_id])			
		
		# controlvz_dict = z_controller.err_control(obsz_dict,kp = 0.5,ki = 0.0, kd = 0.0,OUTPUT_BOUND=2)

		# print("output_z:",controlvz_dict[temp_id])
		
		###wz_controler###
		real_env.in_vision_id_list = list(real_env.in_vision_bbx_dict.keys())
		print("in_vision:",real_env.in_vision_id_list)
		if len(real_env.in_vision_id_list):
			obswz_dict,obslvz_dict = real_env._get_obs_wz()			
			controlwz_dict,control_lvz_dict = wz_controller.control(obswz_dict,obslvz_dict,kp = 0.005,ki = 0.0, kd = 0.0, OUTPUT_BOUND=0.5)
			# print("error_wz:",obswz_dict,obslvz_dict)	
			# print("output_wz:",controlwz_dict,control_lvz_dict)
		for i in range(len(id_list_double_layer)):#一次性循环发十个可能会有问题，需要那边callback的quene_size>10
			robot_id = id_list_double_layer[i]
			drone_vel_msg.header = Header()
			drone_vel_msg.header.stamp = rospy.Time.now()
			drone_vel_msg.header.frame_id = str(robot_id)
			drone_vel_msg.twist.linear.x = controlvxy_dict[i,0]
			drone_vel_msg.twist.linear.y = controlvxy_dict[i,1]
			if robot_id in real_env.real_low_layer_id_list:
				if robot_id in real_env.in_vision_id_list:
					temp_id = real_env.in_vision_id_list.index(robot_id) 
					wz = controlwz_dict[temp_id]
					local_vz = control_lvz_dict[temp_id]
					# drone_vel_msg.twist.linear.z = 0#local_vz
					# import pdb;pdb.set_trace()
				else:
					# if robot_id in real_env.pre_in_vision_id_list:
					# 	temp_id = real_env.pre_in_vision_id_list.index(robot_id) 
					# 	wz = 0.4*np.sign(pre_controlwz_dict[temp_id])
					# else:
					wz = 0.3#0
					local_vz = 0
			else:
					wz = 0
					local_vz = 0
			print("wz,local_vz",wz,local_vz)
			drone_vel_msg.twist.linear.z = local_vz#controlvz_dict[i]
			drone_vel_msg.twist.angular.x = 0
			drone_vel_msg.twist.angular.y = 0
			drone_vel_msg.twist.angular.z = wz#0.2#float(wz)
			controller_pub.publish(drone_vel_msg)
		#reset
		real_env.render()
		real_env.in_vision_bbx_dict = {}
		real_env.pre_in_vision_id_list = real_env.in_vision_id_list.copy() 
		if len(real_env.in_vision_id_list):
			pre_controlwz_dict = controlwz_dict.copy()
		# real_env.state_drones_dict = {}
		# real_env.id_list = []

		rate.sleep()


# def drone_state_callback_0(drone_state):
# 	real_env.state_drones[0] = np.array(drone_state.gps+drone_state.imu)

# def drone_state_callback_0(drone_state):
# 	real_env.state_drones[0] = np.array(drone_state.gps+drone_state.imu)



if __name__ == '__main__':
	episodes = 50000
	main(episodes)











#pdb.set_trace()
