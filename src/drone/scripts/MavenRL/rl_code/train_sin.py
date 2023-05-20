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
from rllib.td3 import TD3 as Method 
from args import EnvParams,generate_args
from data_graph import Data_Graph



# policy = Net()
# optimizer = optim.Adam(policy.parameters(), lr=1e-3)
# env = gym.make('FormationFlying-v2')

# if restore:
# 	start_episode = 13600
# 	filename = "./logs/all10:30:50.560231_5agents_fixed_fcnpolicy"
# 	policy.load_state_dict(torch.load(filename+"/"+str(start_episode)+".pt"))
# else:
# 	start_episode = 0
# 	annotation = "all"
# 	filename = "./logs/"+annotation+str(datetime.datetime.now())[-15:]+str('_%dagents_fixed_fcnpolicy'%env.n_agents)


# if not os.path.exists('./logs'):
# 	os.makedirs('./logs')

# filename = filename+str('.pt')
# torch.save(policy.state_dict(),'./logs/%s'%filename)
# savedir = str('./logs/%s'%filename)
# writer = SummaryWriter(filename)
model_name = "test"
restore = False
test_flag = True#False#True#False# 

model_num = -1
steps_per_episode = 100
load_path = "./results/tri/2022-04-27-13:37:43----td3_gnn/saved_models"#"./results/single/2022-04-24-16:01:38----td3_gnn/saved_models"#"./results/tri/2022-04-24-17:49:06----td3_gnn/saved_models"
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
	config.env = "SingleFormationFlying-v0"
	writer = rllib.basic.create_dir(config, model_name)
	
	# env = gym.make('FormationFlying-v2')
	env = gym.make(config.env)
	# env.n_agents = num_agent
	# import pdb;pdb.set_trace()
	config.update(EnvParams(env))
	method = Method(config, writer)
	if test_flag:
		method.models_to_load = [method.actor,method.critic]
		method._load_model()

	for episode in range(start_episode+1,episodes):
		reward_over_eps = []
		state = env.reset() # Reset environment and record the starting state
		# g = build_graph(env)
		running_reward = 0
		avg_length = 0
		done = False
		# if test_flag==False:
		# 	if episode%100:
		# 		env.render()

		for time in range(steps_per_episode):
			# g = build_graph(env,env.n_agents,env.x[:,:2])
			#if episode%50==0:
			# 
			if test_flag==False:
				action = method.select_action( torch.from_numpy(state).float())
			else:
				action = method.actor(torch.from_numpy(state).float().to(config.device))
			# action = select_action(state,g,policy)

			action = action.cpu().detach().numpy()
			# action = np.reshape(action,[-1])

			# Step through environment using chosen action
			action = np.clip(action,-env.max_accel,env.max_accel) 
			
			next_state, reward, done, _ = env.step(action)
			# import pdb;pdb.set_trace()
			if time ==steps_per_episode-1:
				done = True

			if test_flag==False:
				experience = rllib.template.Experience(
					state=torch.from_numpy(state).float(),
					next_state=torch.from_numpy(next_state).float(),
					action=action.squeeze(), reward=reward, done=done)
				
				method.store(experience)
				method.update_parameters()
			else:	
				env.render()

			reward_over_eps.append(reward)
			# Save reward
		
			# print("episode:",episode,reward)
			state = next_state

			

			running_reward += reward*0.99
			avg_length += 1
			if config.render: env.render()
			if done: break

		

		#pdb.set_trace()
		### logging
		print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, running_reward))
		writer.add_scalar('env/reward', running_reward, episode)
		writer.add_scalar('env/avg_length', avg_length, episode)

if __name__ == '__main__':
	episodes = 20000
	main(episodes)











#pdb.set_trace()
