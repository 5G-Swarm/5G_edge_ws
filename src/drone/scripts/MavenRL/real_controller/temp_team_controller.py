

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
from utils.real_env import Real_env
# from utils.real_env_numdec import Real_env
from utils.basic_controller import Pid_controller,Visual_servo
from utils.function_generate import FunctorFactory
#####ros#####
import rospy
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import TwistStamped,PoseStamped
from std_msgs.msg import Header
from autoware_msgs.msg import DroneSyn


model_name = "test"
restore = False
test_flag = True#False#True#False# 
model_num = -1#774600#971800#227900
steps_per_episode = 500
load_path = "../rl_code/results/test_real/first_4n/saved_models"#"../rl_code/results/test_real/third/saved_models"#"../rl_code/results/test_ten/2o_ten_vel/saved_models"
start_episode = 0

##########
n_agent = 10
real_env = Real_env(n_agents = n_agent)
CONTROL_FREQ = 10

##########

def call_back_target_state(ros_target_odom):
	real_env.target_center[0] = ros_target_odom.pose.pose.position.x
	real_env.target_center[1] = ros_target_odom.pose.pose.position.y
	real_env.target_center[2] = ros_target_odom.pose.pose.position.z
	real_env.target_center[3] = ros_target_odom.twist.twist.linear.x
	real_env.target_center[4] = ros_target_odom.twist.twist.linear.y
	real_env.target_center[5] = ros_target_odom.twist.twist.linear.z
	# real_env.estimate_target_center[0] = ros_target_odom.pose.pose.position.x
	# real_env.estimate_target_center[1] = ros_target_odom.pose.pose.position.y
	# real_env.estimate_target_center[2] = ros_target_odom.pose.pose.position.z
	# real_env.estimate_target_center[3] = ros_target_odom.twist.twist.linear.x
	# real_env.estimate_target_center[4] = ros_target_odom.twist.twist.linear.y
	# real_env.estimate_target_center[5] = ros_target_odom.twist.twist.linear.z
	# print("real_env.estimate_target_center:",real_env.estimate_target_center)
	
def call_back_target_groundtruth(drone_state):
	real_env.estimate_target_center[0] = drone_state.gps[0]
	real_env.estimate_target_center[1] = drone_state.gps[1]
	real_env.estimate_target_center[2] = drone_state.gps[2]
	real_env.estimate_target_center[3] = drone_state.gps[3]
	real_env.estimate_target_center[4] = drone_state.gps[4]
	real_env.estimate_target_center[5] = drone_state.gps[5]



class call_back_template_bbx(object):
	def __init__(self, name):
		self._name = name
		self._id = int(name[-1])

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
		self._id = int(name[-1])

	def __call__(self,drone_state):
		# if real_env.state_drones_dict.has_key(self._id):
		temp = np.array(drone_state.gps+drone_state.imu)
		# temp[0] = temp[0]-33425788.8125
		# temp[1] = -(temp[1]-7650161.05)
		real_env.state_drones_dict[self._id] = temp

class FunctorFactory_state(object):
    def create(self,name):
        globals()[name] = call_back_template_state(name)

def main(episodes):	
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

	controller_pub = rospy.Publisher('/teamrl_controller_vel', TwistStamped, queue_size=10)

	


	rate = rospy.Rate(CONTROL_FREQ)
	drone_vel_msg = TwistStamped()


	##Controller configuration###
	# xy_controller = Method(config, writer)
	# xy_controller.models_to_load = [xy_controller.actor,xy_controller.critic]
	# xy_controller._load_model()

	z_controller = Pid_controller(1/CONTROL_FREQ)
	wz_controller = Visual_servo(1/CONTROL_FREQ)
	center_speed_x = 0#1#-1.2#float(np.random.uniform(-1.4,1.4,1))
	center_speed_y = 0#0.5#0.5#float(np.random.uniform(-1.4,1.4,1))
	
	while not rospy.is_shutdown():

		robot_id = 0#real_env.id_list[i]
		drone_vel_msg.header = Header()
		drone_vel_msg.header.stamp = rospy.Time.now()
		drone_vel_msg.header.frame_id = str(robot_id)
		drone_vel_msg.twist.linear.x = 0#controlvxy_dict[i,0]
		drone_vel_msg.twist.linear.y = 0#controlvxy_dict[i,1]
		

		drone_vel_msg.twist.linear.z = 1#controlvz_dict[i]
		drone_vel_msg.twist.angular.x = 0
		drone_vel_msg.twist.angular.y = 0
		drone_vel_msg.twist.angular.z = 1#0.2#float(wz)
		controller_pub.publish(drone_vel_msg)
		print("sending ")

		# real_env.target_center[2] = 39.3+2

		# real_env.target_center[3] = center_speed_x*math.sin(real_env.counter/60)
		# real_env.target_center[4] = center_speed_y*math.cos(real_env.counter/60)
		# real_env.target_center[:2] = real_env.target_center[:2] + real_env.target_center[3:5]*0.1
		# real_env.target_center = real_env.estimate_target_center.copy()
		###observator###
		# real_env.id_list = list(real_env.state_drones_dict.keys())
		# print("state:",real_env.id_list)
		# # if len(real_env.id_list)<3:
		# # 	continue

		# obsxy_dict = real_env._get_obs_xy()

		# ###xy_controler###

		# axay_dict = xy_controller.actor(torch.from_numpy(obsxy_dict).float().to(config.device))
		# axay_dict = axay_dict.cpu().detach().numpy()
		# controlvxy_dict = real_env._preprocessAction(axay_dict,kp=30)#30#5#15
		# print("axay_dict:",axay_dict)
		# print("axay_dict:",controlvxy_dict)	
		# ###z_controler###	
		# obsz_dict = real_env._get_obs_z()
		# # temp_id = real_env.id_list.index(2)
		# # print("error_z:",obsz_dict[temp_id])			
		# controlvz_dict = z_controller.err_control(obsz_dict,kp = 0.5,ki = 0.0, kd = 0.0,OUTPUT_BOUND=2)
		# # print("output_z:",controlvz_dict[temp_id])
		
		# ###wz_controler###
		# real_env.in_vision_id_list = list(real_env.in_vision_bbx_dict.keys())
		# print("in_vision:",real_env.in_vision_id_list)
		# if len(real_env.in_vision_id_list):
		# 	obswz_dict,obslvz_dict = real_env._get_obs_wz()			
		# 	controlwz_dict,control_lvz_dict = wz_controller.control(obswz_dict,obslvz_dict,kp = 0.005,ki = 0.0, kd = 0.0, OUTPUT_BOUND=0.5)
		# 	# print("error_wz:",obswz_dict,obslvz_dict)	
		# 	# print("output_wz:",controlwz_dict,control_lvz_dict)
		# for i in range(len(real_env.id_list)):#一次性循环发十个可能会有问题，需要那边callback的quene_size>10
		# 	robot_id = real_env.id_list[i]
		# 	drone_vel_msg.header = Header()
		# 	drone_vel_msg.header.stamp = rospy.Time.now()
		# 	drone_vel_msg.header.frame_id = str(robot_id)
		# 	drone_vel_msg.twist.linear.x = controlvxy_dict[i,0]
		# 	drone_vel_msg.twist.linear.y = controlvxy_dict[i,1]
			
		# 	if robot_id in real_env.in_vision_id_list:
		# 		temp_id = real_env.in_vision_id_list.index(robot_id) 
		# 		wz = controlwz_dict[temp_id]
		# 		local_vz = control_lvz_dict[temp_id]
		# 		# drone_vel_msg.twist.linear.z = 0#local_vz
		# 		# import pdb;pdb.set_trace()
		# 	else:
		# 		wz = 0
		# 		local_vz = 0

		# 	drone_vel_msg.twist.linear.z = local_vz#controlvz_dict[i]
		# 	drone_vel_msg.twist.angular.x = 0
		# 	drone_vel_msg.twist.angular.y = 0
		# 	drone_vel_msg.twist.angular.z = wz#0.2#float(wz)
		# 	controller_pub.publish(drone_vel_msg)
		# #reset
		# real_env.render()
		# real_env.in_vision_bbx_dict = {}
		# # real_env.state_drones_dict = {}
		# # real_env.id_list = []

		rate.sleep()


# def drone_state_callback_0(drone_state):
# 	real_env.state_drones[0] = np.array(drone_state.gps+drone_state.imu)

# def drone_state_callback_0(drone_state):
# 	real_env.state_drones[0] = np.array(drone_state.gps+drone_state.imu)



if __name__ == '__main__':
	episodes = 50000
	main(episodes)











#pdb.set_trace()
