from bdb import set_trace
from difflib import restore
import gym
import gym_flock
import numpy as np
import pdb
import dgl
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from make_g import build_graph
import torch.optim as optim
import dgl
import dgl.function as fn
import math
import pdb

from torch.autograd import Variable
from torch.distributions import Categorical
import torch
import networkx as nx
import pdb
import matplotlib.pyplot as plt
from policy import Net
#from linear_policy import Net
from make_g import build_graph
from utils import *

import os
import datetime
from torch.utils.tensorboard import SummaryWriter
import rllib
from td3_dec import TD3_DEC as Method 
from args import EnvParams,generate_args
from data_graph import Data_Graph



model_name = "test"
restore = False
test_flag = True#False#True#False# 

model_num = -1#774600#971800#227900
steps_per_episode = 500
load_path = "./results/test_real/third/saved_models"#"./results/single/2022-04-24-16:01:38----td3_gnn/saved_models"#"./results/tri/2022-04-24-17:49:06----td3_gnn/saved_models"
start_episode = 0


def main(episodes):
	plotting_rew = []
	config = rllib.basic.YamlConfig()
	args = generate_args()
	config.update(args)
	config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
	config.set("model_dir",load_path)
	config.set("model_num",model_num)
	config.set("restore",restore)
	# config.set("num_agent",num_agent)
	config.set("test_flag",test_flag)
	config.env = "Two_Order_DecentralFormationFlying-v0"#"DecentralFormationFlying-v0"
	writer = rllib.basic.create_dir(config, model_name)
	
	# env = gym.make('FormationFlying-v2')
	env = gym.make(config.env)
	# env.n_agents = num_agent
	# import pdb;pdb.set_trace()
	config.update(EnvParams(env))
	config.n_agents = env.n_agents
	method = Method(config, writer)
	if test_flag:
		method.models_to_load = [method.actor,method.critic]
		method._load_model()

	for episode in range(start_episode+1,episodes):
		obs_dict = env.reset() # Reset environment and record the starting state
		
		# g = build_graph(env)
		running_reward = 0
		avg_length = 0
		# if test_flag==False:
		# 	if episode%100:
		# 		env.render()

		for time in range(steps_per_episode):
			# g = build_graph(env,env.n_agents,env.x[:,:2])
			# import pdb;pdb.set_trace()
			if test_flag==False:
				action_dict = method.select_action(torch.from_numpy(obs_dict).float())
			else:
				action_dict = method.actor(torch.from_numpy(obs_dict).float().to(config.device))
				action_dict = action_dict.cpu().detach().numpy()
				action_dict = np.clip(action_dict,-env.max_accel,env.max_accel) 

				
			next_obs_dict, reward_dict, done_dict, _ = env.step(action_dict)
				# import pdb;pdb.set_trace()
			# if time ==steps_per_episode-1:
			# 	done_dict['__all__'] = True
				

			if test_flag==False:
				for agent_id in range(env.n_agents):
					experience = rllib.template.Experience(
						state=torch.from_numpy(obs_dict[agent_id]).float().unsqueeze(0),
						next_state=torch.from_numpy(next_obs_dict[agent_id]).float().unsqueeze(0),
						action=action_dict[agent_id].unsqueeze(0), 
						reward=reward_dict[agent_id], 
						done=done_dict[agent_id])
					
					method.store(experience)
				method.update_parameters()
			else:	
				env.render()
			# Save reward
		
			# print("episode:",episode,reward)
			obs_dict = next_obs_dict

			

			running_reward += reward_dict.sum()*0.99
			avg_length += 1
			if config.render: env.render()
			if 1 in done_dict:
				break

		

		#pdb.set_trace()
		### logging
		print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, running_reward))
		writer.add_scalar('env/reward', running_reward, episode)
		writer.add_scalar('env/avg_length', avg_length, episode)



if __name__ == '__main__':
	episodes = 50000
	main(episodes)











#pdb.set_trace()
