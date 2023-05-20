
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import dgl
import dgl.function as fn

from replay_buffer_graph import ReplayBuffer_Graph
from rllib.utils import init_weights, soft_update
from rllib.template import MethodSingleAgent, Model
# from rllib.template.model import FeatureExtractor, FeatureMapper


class TD3_GNN(MethodSingleAgent):
    gamma = 0.99
    
    lr_critic = 0.0003
    lr_actor = 0.0003

    tau = 0.005

    buffer_size = 439310#1000000
    batch_size = 256

    policy_freq = 2
    explore_noise = 0.1
    policy_noise = 0.2
    noise_clip = 0.5

    start_timesteps = 30000

    save_model_interval = 200

    def __init__(self, config, writer):
        super(TD3_GNN, self).__init__(config, writer)
        config.set("batch_size",self.batch_size)
        self.critic = config.get('net_critic', Critic)(config).to(self.device)
        self.actor = config.get('net_actor', Actor)(config).to(self.device)
        
        if config.restore or config.test_flag:
            self.actor.model_num = config.model_num
            self.critic.model_num = config.model_num
            self.models_to_load = [self.critic, self.actor]
            self._load_model()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.models = [self.critic, self.actor, self.critic_target, self.actor_target]
        self.models_to_save = [self.critic, self.actor]

        self.critic_optimizer= Adam(self.critic.parameters(), lr=self.lr_critic)#weight_decay=1e-4,RMPProp,
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_loss = nn.MSELoss()

        self.buffer: ReplayBuffer_Graph = config.get('buffer', ReplayBuffer_Graph)(config, self.buffer_size, self.batch_size, self.device)


    def update_parameters(self):
        if len(self.buffer) < self.start_timesteps:
            return
        self.update_parameters_start()
        self.writer.add_scalar('method/buffer_size', len(self.buffer), self.step_update)

        '''load data batch'''
        experience = self.buffer.sample()
        state = experience.state
        action = experience.action
        next_state = experience.next_state
        reward = experience.reward
        done = experience.done
        graph = experience.graph
        

        '''critic'''
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(graph,next_state) + noise).clamp(-1,1)#, step_update=self.step_update

            target_q1, target_q2 = self.critic_target(graph,next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * (1-done) * target_q

        current_q1, current_q2 = self.critic(graph,state, action)
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.writer.add_scalar('method/loss_critic', critic_loss.detach().item(), self.step_update)

        '''actor'''
        if self.step_update % self.policy_freq == 0:
            actor_loss = -self.critic.q1(graph,state, self.actor(graph,state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._update_model()

            self.writer.add_scalar('method/loss_actor', actor_loss.detach().item(), self.step_update)

        if self.step_update % self.save_model_interval == 0:
            self._save_model()

        self.update_callback(locals())
        return


    @torch.no_grad()
    def select_action(self,g,state):
        self.select_action_start()

        if self.step_select < self.start_timesteps:
            action = torch.Tensor(self.dim_action[0],self.dim_action[1]).uniform_(-1,1)
        else:
            noise = torch.normal(0, self.explore_noise, size=self.dim_action)
            # import pdb;pdb.set_trace()
            action = self.actor(g.to(self.device),state.to(self.device))
            action = (action.cpu() + noise).clamp(-1,1)
            
        return action

    def _update_model(self):
        # print('[update_parameters] soft update')
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class GCN(nn.Module):
	def __init__(self, in_feats, out_feats, activation):
		super(GCN, self).__init__()
		self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

	def forward(self, g, feature):
		g.ndata['h'] = feature
		g.update_all(gcn_msg, gcn_reduce)
		g.apply_nodes(func=self.apply_mod)
		return g.ndata.pop('h')



class Actor(Model):
    def __init__(self, config, model_id=0):
        super(Actor, self).__init__(config, model_id)

        self.gcn1 = GCN(2, 16, torch.tanh)
	# 	#self.gcn11 = GCN(16, 16, t.tanh)
	# 	#self.gcn111 = GCN(64, 32, F.relu)
        self.gcn2 = GCN(16, 2, torch.tanh)
        # self.gcn2_ = GCN(16,2,torch.tanh)

        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.gamma = 0.99

    def forward(self, g, features):
        
        x = self.gcn1(g, features)
        return self.gcn2(g, x)


class Feature_extractor_GCN(nn.Module):
    def __init__(self):
        super(Feature_extractor_GCN, self).__init__()        
        self.gcn1 = GCN(4, 16, nn.ReLU())
        self.gcn2 = GCN(16, 4, nn.ReLU())

        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.gamma = 0.99


    def forward(self, g, features):
        x = self.gcn1(g, features)
        return self.gcn2(g, x)


class Critic(Model):
    def __init__(self, config, model_id=0):
        super(Critic, self).__init__(config, model_id)
        self.batch_size = config.batch_size
        self.num_agent = config.num_agent
        # self.fe = config.get('net_critic_fe', FeatureExtractor)(config, model_id)
        self.fe1 = Feature_extractor_GCN()
        self.fcn1 = torch.nn.Linear(4,1)
        # self.fm1 = config.get('net_critic_fm', FeatureMapper)(config, model_id, self.fe.dim_feature+config.dim_action, 1)
        self.fe2 = copy.deepcopy(self.fe1)
        self.fcn2 = copy.deepcopy(self.fcn1)

        self.apply(init_weights)

    def forward(self,g, state, action):
        # x = self.fe(state)
        x = torch.cat([state, action], 1)
        
        output1 = self.fe1(g,x)
        output1 = output1.view(self.batch_size,self.num_agent,4)
        output1 = output1.mean(dim=1)
        output1 = self.fcn1(output1)
        # output1 = self.act1(output1)

        output2 = self.fe2(g,x)
        output2 = output2.view(self.batch_size,self.num_agent,4)
        output2 = output2.mean(dim=1)
        output2 = self.fcn2(output2)
        # output2 = self.act2(output2)

        # import pdb;pdb.set_trace()
        return output1, output2
    
    def q1(self, g, state, action):
        
        x = torch.cat([state, action], 1)
        return self.fe1(g,x)

class NodeApplyModule(nn.Module):
	def __init__(self, in_feats, out_feats, activation):
		super(NodeApplyModule, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)
		self.activation = activation

	def forward(self, node):
		h = self.linear(node.data['h'])
		h = self.activation(h)
		return {'h' : h}

###############################################################################
# We then proceed to define the GCN module. A GCN layer essentially performs
# message passing on all the nodes then applies the `NodeApplyModule`. Note
# that we omitted the dropout in the paper for simplicity.



