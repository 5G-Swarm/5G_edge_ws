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

restore = False


policy = Net()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
env = gym.make('FormationFlying-v2')

if restore:
	start_episode = 13600
	filename = "./logs/all10:30:50.560231_5agents_fixed_fcnpolicy"
	policy.load_state_dict(torch.load(filename+"/"+str(start_episode)+".pt"))
else:
	start_episode = 0
	annotation = "all"
	filename = "./logs/"+annotation+str(datetime.datetime.now())[-15:]+str('_%dagents_fixed_fcnpolicy'%env.n_agents)


if not os.path.exists('./logs'):
	os.makedirs('./logs')

# filename = filename+str('.pt')
# torch.save(policy.state_dict(),'./logs/%s'%filename)
# savedir = str('./logs/%s'%filename)
writer = SummaryWriter(filename)


def main(episodes):
	running_reward = 10
	plotting_rew = []

	for episode in range(start_episode+1,episodes):
		reward_over_eps = []
		state = env.reset() # Reset environment and record the starting state
		# g = build_graph(env)
		done = False

		for time in range(2000):
			g = build_graph(env)
			#if episode%50==0:
			#	env.render()
			#g = build_graph(env)
			action = select_action(state,g,policy)

			action = action.numpy()
			action = np.reshape(action,[-1])

			# Step through environment using chosen action
			action = np.clip(action,-env.max_accel,env.max_accel)

			state, reward, done, _ = env.step(action)

			reward_over_eps.append(reward)
			# Save reward
			policy.reward_episode.append(reward)
			print("episode:",episode)
			if done:
				break

		# Used to determine when the environment is solved.
		running_reward = (running_reward * 0.99) + (time * 0.01)

		update_policy(policy,optimizer)

		if episode % 50 == 0:
			print('Episode {}\tLast length: {:5d}\tAverage running reward: {:.2f}\tAverage reward over episode: {:.2f}'.format(episode, time, running_reward, np.mean(reward_over_eps)))

		if episode % 100 == 0 :
			torch.save(policy.state_dict(),filename+'/%s'%(str(episode)+".pt"))
		
		
		writer.add_scalar('rewards', reward, episode)
		writer.add_scalar('loss', policy.loss_history[-1], episode)
		# writer.add_scalar('lr', policy.lr_history[-1],episode)

		plotting_rew.append(np.mean(reward_over_eps))
	#pdb.set_trace()
	np.savetxt('Relative_Goal_Reaching_for_%d_agents_rs_rg.txt' %(env.n_agents), plotting_rew)
	fig = plt.figure()
	x = np.linspace(0,len(plotting_rew),len(plotting_rew))
	plt.plot(x,plotting_rew)
	plt.savefig('Relative_Goal_Reaching_for_%d_agents_rs_rg.png' %(env.n_agents),filename)
	plt.show()

	#pdb.set_trace()

episodes = 2000000
main(episodes)











#pdb.set_trace()
