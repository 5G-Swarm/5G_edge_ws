
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv

from rl_code.replay_buffer_graph import ReplayBuffer_Graph
from rllib.utils import init_weights, soft_update
from rllib.template import MethodSingleAgent, Model
# from run_scripts.algos.CNN import CNN_Encoder,DEC_CNN_Encoder
# from run_scripts.algos.GNN import GNN_Encoder_img, GNN_Encoder
# from rllib.template.model import FeatureExtractor, FeatureMapper
import time
torch.autograd.set_detect_anomaly(True)




class MATD3_DEC(MethodSingleAgent):
    gamma = 0.99
    lambda_bc = 4
    lambda_q = 2
    

    lr_critic = 2e-4#2e-4#5e-5#0.0003
    lr_actor = 5e-5#1e-4#0.0003 
    max_grad_norm = 20

    tau = 0.005

    # buffer_size = 500000
    buffer_size = 20000#10000#50000

    batch_size = 16#16#256#16#256#16

    policy_freq = 4
    explore_noise_ub = 0.1#0.2
    explore_noise_lb = 0.05#0.7
    explore_noise_decay_rate = 0.9999

    policy_noise = 0.05#0.05#0.05#0.2
    noise_clip = 0.5

    start_timesteps = 0#60000#30000#10000#30000#10000#30000
    start_buffersize = 0#20000
    start_actor_update = 20000#300000

    save_model_interval = 4000#4000#4000#800

    def __init__(self, config, writer):
        super(MATD3_DEC, self).__init__(config, writer)
        
        # self.n_agents = config.n_agents
        self.critic = config.get('net_critic', Critic)(config).to(self.device)
        self.actor = config.get('net_actor', Actor)(config).to(self.device)
        
        if config.restore or config.test_flag:
            self.actor.model_num = config.model_num
            self.critic.model_num = config.model_num
        
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.models = [self.critic, self.actor, self.critic_target, self.actor_target]
        self.models_to_save = [self.critic, self.actor]

        self.critic_optimizer= Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_loss = nn.MSELoss()
        # self.policies = 
        self.buffer: ReplayBuffer_Graph = config.get('buffer', ReplayBuffer_Graph)(config, self.buffer_size, self.batch_size, self.device)

        # self.OU_noise = OrnsteinUhlenbeckActionNoise(mu=torch.zeros((n_agent,self.dim_action)), sigma= )

    def update_parameters(self):
        # if len(self.buffer) < self.start_timesteps:
        #     return
        self.update_parameters_start()
        self.writer.add_scalar('method/buffer_size', len(self.buffer), self.step_update)
        if len(self.buffer) < self.start_buffersize:
            return
        # import pdb;pdb.set_trace()
        # time1 = time.time()
        '''load data batch'''
        experience,num_agent_record = self.buffer.sample()
        # time_i = time.time()
                
        cen_state_graph = experience.cen_state_graph
        dec_state_graph = experience.dec_state_graph
        cen_state_img = experience.cen_state_img
        dec_state_img = experience.dec_state_img
        dec_state = (dec_state_graph, dec_state_img, num_agent_record)
        cen_state = (cen_state_graph, cen_state_img, num_agent_record)

        action_graph = experience.action_graph

        next_cen_state_graph = experience.next_cen_state_graph
        next_dec_state_graph = experience.next_dec_state_graph
        next_cen_state_img = experience.next_cen_state_img
        next_dec_state_img = experience.next_dec_state_img
        next_dec_state = (next_dec_state_graph, next_dec_state_img, num_agent_record)
        next_cen_state = (next_cen_state_graph, next_cen_state_img, num_agent_record)


        reward = experience.reward
        done = experience.done
        # time2 = time.time()
        '''critic'''
        with torch.no_grad():
            noise = (torch.randn_like(action_graph.x) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # next_action = (self.actor_target(next_state, step_update=self.step_update) + noise).clamp(-1,1)
            
            next_action = (self.actor_target(next_dec_state) + noise).clamp(-1,1)
            
            target_q1, target_q2 = self.critic_target(next_cen_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * (1-done) * target_q

        # time3 = time.time()
        current_q1, current_q2 = self.critic(cen_state, action_graph)
        critic_loss = self.critic_loss(current_q1, target_q.detach()) + self.critic_loss(current_q2, target_q.detach())
        # time4 = time.time()
        self.critic_optimizer.zero_grad()
        # try:
        #     critic_loss.backward()
        # except:
        #     import pdb;pdb.set_trace()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        
        self.critic_optimizer.step()
        self.writer.add_scalar('method/loss_critic', critic_loss.detach().item(), self.step_update)
        self.writer.add_scalar('method/critic_grad_norm', critic_grad_norm.detach().item(), self.step_update)

        # time5 = time.time()
        '''actor'''
        if self.step_select>self.start_actor_update and self.step_update % self.policy_freq == 0:
            # num_update_agents = 4
            # mask_temp = []
            # mask_temp.append(np.zeros(sum_act_dim, dtype=np.float32))

            # masks = []
            # for i in range(num_update_agents):
            #     curr_mask_temp = copy.deepcopy(mask_temp)
            

            onpolicy_action = self.actor(dec_state)

            # onpolicy_mask = torch.zeros(onpolicy_action.shape)
            # fill_index_list = [0]
            # base = 0
            # for i in range(len(num_agent_record)-1):
            #     base += num_agent_record[i]
            #     fill_index_list.append(base)
            
            # onpolicy_mask[fill_index_list,:] = torch.ones(onpolicy_mask.shape[1]) 
            # onpolicy_mask = onpolicy_mask.cuda()
            # offpolicy_mask = 1 - onpolicy_mask

            # # offpolicy_action = self.actor_target(dec_state)#action_graph.x
            # offpolicy_action = action_graph.x#action_graph.x
            
            # update_action =  onpolicy_mask*onpolicy_action + offpolicy_mask*offpolicy_action


            # import pdb;pdb.set_trace()
            actor_loss = -self.critic.q1(cen_state, onpolicy_action).mean()
            # Q_scale = self.critic.q1(state, self.actor(state)).mean().detach()
            # BC_scale = 24             
            # actor_loss = (-self.critic.q1(state, self.actor(state))/Q_scale*self.lambda_q + torch.linalg.norm((self.actor(state)-action),ord=2,dim=1)**2/BC_scale*self.lambda_bc).mean() 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self._update_model()

            self.writer.add_scalar('method/loss_actor', actor_loss.detach().item(), self.step_update)
            self.writer.add_scalar('method/actor_grad_norm', actor_grad_norm.detach().item(), self.step_update)

        
			# import pdb;pdb.set_trace()
            
        if self.step_update % self.save_model_interval == 0:
            self._save_model()
        # time6 = time.time()
        self.update_callback(locals())
        # time7 = time.time()
        # print('timeepisode',time_i-time1,time2-time1,time3-time2,time4-time3,time5-time4,time6-time5,time7-time6,time7-time1)
        # import pdb;pdb.set_trace()
        return

    def expert_policy(self, state_array):
        kp = 1#5
        kv = 1#5
        act_vel = state_array[:,:2]
        ref_vel = (state_array[:,4:6] - state_array[:,2:4])/0.1
        err_pos = state_array[:,2:4]
        try:
            err_vel = ref_vel - act_vel
        except:
            import pdb;pdb.set_trace() 
        output = kp*err_pos + kv*err_vel

        return output.clamp(-1,1)
        


    @torch.no_grad()
    def select_action(self, state, n_agent, random_flag): 
        self.select_action_start()
        flag_expert = False
        if self.step_select < self.start_timesteps:
            flag_expert = True
            # if torch.Tensor(1).uniform_(0,1)<0.5:
            if random_flag:
                action = torch.Tensor(n_agent,self.dim_action).uniform_(-1,1)#.cuda() 
            else:
                state_graph = state[0]
                id_list = [i+i*n_agent for i in range(n_agent)]
                # action = ((state_graph.x[id_list, 2:4] - state_graph.x[id_list, :2])*10).clamp(-1,1)
                # action = (state_graph.x[id_list, :2]*10).clamp(-1,1)
                # action = (state_graph.x[id_list, 2:4]*3).clamp(-1,1)
                action = (0.5*state_graph.x[id_list, 2:4]*3 + 0.5*(state_graph.x[id_list, 4:6] - state_graph.x[id_list, 2:4])/0.1  ).clamp(-1,1)
                
                # action = self.expert_policy(state_graph.x[id_list, :])
        else:
            flag_expert = False
            self.explore_noise = max(self.explore_noise_ub * np.power(self.explore_noise_decay_rate,self.step_select/50),self.explore_noise_lb)
            noise = torch.normal(0, self.explore_noise, size=(n_agent,self.dim_action)).cuda()
            self.writer.add_scalar('method/explore_noise', self.explore_noise, self.step_update)
            
            self.actor.eval()
            action = self.actor(state)
            action = (action + noise.squeeze()).clamp(-1,1)
            # action = action.cpu()
        
        return action, flag_expert

    def _update_model(self):
        # print('[update_parameters] soft update')
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)



# class OrnsteinUhlenbeckActionNoise(object):
#     def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
#         self.theta = theta
#         self.mu = mu
#         self.sigma = sigma
#         self.dt = dt
#         self.x0 = x0
#         self.reset()

#     def __call__(self):
#         x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
#         self.x_prev = x
#         return x

#     def reset(self):
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)





# ######################不考虑障碍物
class Lite_Actor(Model):
    def __init__(self, config, model_id=0):
        super(Lite_Actor, self).__init__(config, model_id)
        # self.fe_img = DEC_CNN_Encoder(channel_input=1,dim_output = 256)#config.get('net_actor_fe', FeatureExtractor)(config, model_id)
        self.fe_graph = DEC_GNN_Encoder(node_feature_dim=config.dim_state, edge_dim = 2, hidden_feature_dim=config.dim_action, heads=4, use_GAT=True)
        # self.fe_state1 = nn.Sequential(nn.Linear(16, 256), nn.LeakyReLU())
        # self.fe_affparam = nn.Sequential(nn.Linear(7, 256), nn.LeakyReLU())
        # self.fm = config.get('net_actor_fm', Lite_Actor_FeatureMapper)(config, model_id, self.fe_graph.hidden_feature_dim, config.dim_action)
        self.no = nn.Tanh()  ## normalize output
        self.apply(init_weights)
    
    def forward(self, state):
        state_graph = state[0]
        num_agent_record = state[2]
        # state_img = state[1]
        # import pdb;pdb.set_trace()
        # hidden_img = self.fe_img(state_img)
        # hidden_graph = self.fe_graph(state_graph)
        z = self.fe_graph(state_graph, num_agent_record)

        # z = self.fm(hidden_graph)
        
        return self.no(z)


class Lite_Critic(Model):
    def __init__(self, config, model_id=0):
        super(Lite_Critic, self).__init__(config, model_id)
        
        # self.fe_img = CEN_CNN_Encoder(channel_input=1,dim_output = 256)#config.get('net_critic_fe', FeatureExtractor)(config, model_id)
        
        self.fe_graph1 = Lite_CEN_GNN_Encoder(node_feature_dim=config.dim_state+config.dim_action, edge_dim = 2, hidden_feature_dim=256, heads=4, use_GAT=True)
        
        # self.fe_graph1 = Lite_CEN_GNN_Encoder(node_feature_dim=config.dim_state+config.dim_action, hidden_feature_dim=256, use_GAT=True)
        
        # self.fe_graph2 = copy.deepcopy(self.fe_graph1)
        # self.fe_affparam = nn.Sequential(nn.Linear(7+config.dim_action, 256), nn.LeakyReLU(),nn.Linear(256, 512), nn.LeakyReLU())
        self.fm1 = config.get('net_critic_fm', Lite_Critic_FeatureMapper)(config, model_id, self.fe_graph1.hidden_feature_dim, 1)
        self.fm2 = copy.deepcopy(self.fm1)
        self.apply(init_weights)

    def forward(self, state, action):

        state_graph = state[0]

        z = self.fe_graph1(state_graph, action)
        # z2 = self.fe_graph2(state_graph, action)
        # z = torch.cat([hidden_graph,hidden_affparam], 1)
        # import pdb;pdb.set_trace()
        return self.fm1(z), self.fm2(z)
    
    def q1(self, state, action):
        state_graph = state[0]
        z1 = self.fe_graph1(state_graph, action)
        # z = torch.cat([hidden_graph,hidden_affparam], 1)
        return self.fm1(z1)



# # ###################### 考虑障碍物
class Actor(Model):
    def __init__(self, config, model_id=0):
        super(Actor, self).__init__(config, model_id)
        self.fe_img = DEC_CNN_Encoder(channel_input=1,dim_output = 256)#config.get('net_actor_fe', FeatureExtractor)(config, model_id)
        self.fe_graph = DEC_GNN_Encoder(node_feature_dim=config.dim_state, edge_dim = 3, hidden_feature_dim=256, heads=4, use_GAT=True)
        # self.fe_state1 = nn.Sequential(nn.Linear(16, 256), nn.LeakyReLU())
        # self.fe_affparam = nn.Sequential(nn.Linear(7, 256), nn.LeakyReLU())
        self.fm = config.get('net_actor_fm', Actor_FeatureMapper)(config, model_id, self.fe_graph.hidden_feature_dim+self.fe_img.dim_output, config.dim_action)
        self.no = nn.Tanh()  ## normalize output
        self.apply(init_weights)
    
    def forward(self, state):

        state_graph = state[0]
        state_img = state[1]
        num_agent_record = state[2]
        # import pdb;pdb.set_trace()
        hidden_img = self.fe_img(state_img)
        hidden_graph = self.fe_graph(state_graph, num_agent_record)

        z = self.fm(torch.cat([hidden_img,hidden_graph], 1))
        
        has_nan = np.isnan(z.cpu().detach().numpy()).any()
        if has_nan:
            import pdb;pdb.set_trace()
        
        return self.no(z)


class Critic(Model):
    def __init__(self, config, model_id=0):
        super(Critic, self).__init__(config, model_id)
        
        self.fe_img = DEC_CNN_Encoder(channel_input=1,dim_output = 256)#config.get('net_critic_fe', FeatureExtractor)(config, model_id)
        self.fe_graph = CEN_GNN_Encoder(node_feature_dim=config.dim_state+config.dim_action+self.fe_img.dim_output, edge_dim = 3, hidden_feature_dim=256, heads=4, use_GAT=True)
        # self.fe_affparam = nn.Sequential(nn.Linear(7+config.dim_action, 256), nn.LeakyReLU(),nn.Linear(256, 512), nn.LeakyReLU())
        self.fm1 = config.get('net_critic_fm', Critic_FeatureMapper)(config, model_id, self.fe_graph.hidden_feature_dim, 1)
        self.fm2 = copy.deepcopy(self.fm1)
        self.apply(init_weights)

    def forward(self, state, action):

        state_graph = state[0]
        state_img = state[1]
        num_agent_record = state[2]
        
        hidden_img = self.fe_img(state_img)
        # temp_img = hidden_img[0].repeat(num_agent_record[0],1)
        # for i in range(1,len(num_agent_record)):
        #     temp_img = torch.concat((temp_img,hidden_img[i].repeat(num_agent_record[i],1)),dim=0)
        # import pdb;pdb.set_trace()
        # state_graph.x = torch.concat((state_graph.x,temp_img),dim=1)

        # hidden_graph = self.fe_graph(state_graph,temp_img)
        z = self.fe_graph(state_graph, action, hidden_img)
        # z = torch.cat([hidden_graph,hidden_affparam], 1)
        # import pdb;pdb.set_trace()
        return self.fm1(z), self.fm2(z)
    
    def q1(self, state, action):
        state_graph = state[0]
        state_img = state[1]
        num_agent_record = state[2]
            
        hidden_img = self.fe_img(state_img)
        # temp_img = hidden_img[0].repeat(num_agent_record[0],1)
        # for i in range(1,len(num_agent_record)):
        #     temp_img = torch.concat((temp_img,hidden_img[i].repeat(num_agent_record[i],1)),dim=0)

        # try:
        z = self.fe_graph(state_graph, action, hidden_img)
        # except:
        #     import pdb;pdb.set_trace()
        # z = torch.cat([hidden_graph,hidden_affparam], 1)
        return self.fm1(z)



# ######################
# class Actor(Model):
#     def __init__(self, config, model_id=0):
#         super(Actor, self).__init__(config, model_id)

#         self.fe_img = CNN_Encoder(channel_input=1,dim_output = 256)#config.get('net_actor_fe', FeatureExtractor)(config, model_id)
#         self.fe_graph = GNN_Encoder(node_feature_dim=4, hidden_feature_dim=256)
#         # self.fe_state1 = nn.Sequential(nn.Linear(16, 256), nn.LeakyReLU())
#         self.fe_affparam = nn.Sequential(nn.Linear(7, 256), nn.LeakyReLU())
#         self.fm = config.get('net_actor_fm', FeatureMapper)(config, model_id,  self.fe_img.dim_output+self.fe_graph.hidden_feature_dim+256, config.dim_action)
#         self.no = nn.Tanh()  ## normalize output
#         self.apply(init_weights)
    
#     def forward(self, state):
#         state_vec = state[0]
#         state_graph = state[1]
#         state_img = state[2]
#         hidden_img = self.fe_img(state_img)

#         hidden_graph = self.fe_graph(state_graph)
#         hidden_affparam = self.fe_affparam(state_vec)
#         z = self.fm(torch.cat([hidden_img,hidden_graph,hidden_affparam], 1))
#         return self.no(z)


# class Critic(Model):
#     def __init__(self, config, model_id=0):
#         super(Critic, self).__init__(config, model_id)
        
#         self.fe_img = CNN_Encoder(channel_input=1,dim_output = 256)#config.get('net_critic_fe', FeatureExtractor)(config, model_id)
#         self.fe_graph = GNN_Encoder(node_feature_dim=4, hidden_feature_dim=256)
#         self.fe_affparam = nn.Sequential(nn.Linear(7+config.dim_action, 256), nn.LeakyReLU())

#         self.fm1 = config.get('net_critic_fm', FeatureMapper)(config, model_id, self.fe_img.dim_output+self.fe_graph.hidden_feature_dim+256, 1)
#         self.fm2 = copy.deepcopy(self.fm1)
#         self.apply(init_weights)

#     def forward(self, state, action):
#         state_vec = state[0]
#         state_graph = state[1]
#         state_img = state[2]

#         hidden_img = self.fe_img(state_img)

#         hidden_graph = self.fe_graph(state_graph)
#         hidden_affparam = self.fe_affparam(torch.cat([state_vec, action], 1))
#         z = torch.cat([hidden_img,hidden_graph,hidden_affparam], 1)
#         # import pdb;pdb.set_trace()
#         return self.fm1(z), self.fm2(z)
    
#     def q1(self, state, action):
#         state_vec = state[0]
#         state_graph = state[1]
#         state_img = state[2]
#         hidden_img = self.fe_img(state_img)
#         hidden_graph = self.fe_graph(state_graph)
#         hidden_affparam = self.fe_affparam(torch.cat([state_vec, action], 1))
#         z = torch.cat([hidden_img,hidden_graph,hidden_affparam], 1)
#         return self.fm1(z)


class Lite_Actor_FeatureMapper(Model):
    def __init__(self, config, model_id, dim_input, dim_output):
        super().__init__(config, model_id)
        
        self.fm = nn.Sequential(
            nn.Linear(dim_input, 512), nn.LeakyReLU(),
            # nn.Linear(1024, 256), nn.LeakyReLU(),
            # nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(512, dim_output),
        )
    
    def forward(self, x):
        return self.fm(x)


class Actor_FeatureMapper(Model):
    def __init__(self, config, model_id, dim_input, dim_output):
        super().__init__(config, model_id)

        self.fm = nn.Sequential(
            nn.Linear(dim_input, 1024), nn.LeakyReLU(),
            # nn.Linear(1024, 256), nn.LeakyReLU(),
            # nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(1024, dim_output),
        )

        # self.fm = nn.Sequential(
        #     nn.Linear(dim_input, 1024), nn.LeakyReLU(),
        #     nn.Linear(1024, 256), nn.LeakyReLU(),
        #     # nn.Linear(512, 256), nn.LeakyReLU(),
        #     nn.Linear(256, dim_output),
        # )
    
    def forward(self, x):
        return self.fm(x)

class Lite_Critic_FeatureMapper(Model):
    def __init__(self, config, model_id, dim_input, dim_output):
        super().__init__(config, model_id)
        
        self.fm = nn.Sequential(
            nn.Linear(dim_input, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 256), nn.LeakyReLU(),
            nn.Linear(256, 32), nn.LeakyReLU(),
            nn.Linear(32, dim_output),
        )
    
    def forward(self, x):
        return self.fm(x)

class Critic_FeatureMapper(Model):
    def __init__(self, config, model_id, dim_input, dim_output):
        super().__init__(config, model_id)
        
        self.fm = nn.Sequential(
            nn.Linear(dim_input, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 256), nn.LeakyReLU(),
            nn.Linear(256, 32), nn.LeakyReLU(),
            nn.Linear(32, dim_output),
        )
    
    def forward(self, x):
        return self.fm(x)


class Lite_CEN_GNN_Encoder(torch.nn.Module):
    def __init__(self,node_feature_dim, hidden_feature_dim, edge_dim=None, heads=4, use_GAT=False):
        super().__init__()
        self.hidden_feature_dim = hidden_feature_dim
        # self.fc_input = torch.nn.Linear(node_feature_dim, 512)
        self.use_GAT = use_GAT
        if self.use_GAT:
            self.conv1 = GATConv(node_feature_dim, 256, edge_dim=edge_dim, heads=heads, concat=True)
            self.conv2 = GATConv(256*heads, 256*heads, edge_dim=edge_dim, heads=heads, concat=True)
        else:
            self.conv1 = GCNConv(512, 1024)
            self.conv2 = GCNConv(1024, 1024)

        self.fc_output = torch.nn.Linear(256*heads*heads, self.hidden_feature_dim)        

    ###data_graph当训练时是Batch类型,是已经拼接好的对角块矩阵；当仅前向推理时,是Data类型
    def forward(self, data_graph, action, prob=0.5):
        if isinstance(action,Batch):
            x = torch.concat((data_graph.x,action.x),dim=1)
        else:
            x = torch.concat((data_graph.x,action),dim=1)

        edge_index, edge_attr = data_graph.edge_index, data_graph.edge_attr
        
        # x = F.leaky_relu(self.fc(x))
        # x = F.leaky_relu(self.fc_input(x))
        if self.use_GAT:
            x = F.leaky_relu(self.conv1(x, edge_index, edge_attr=edge_attr))#, edge_weight=edge_weight
            x = F.leaky_relu(self.conv2(x, edge_index, edge_attr=edge_attr))#, edge_weight=edge_weight
        else:
            x = F.leaky_relu(self.conv1(x, edge_index))#, edge_weight=edge_weight
            x = F.leaky_relu(self.conv2(x, edge_index))#, edge_weight=edge_weight
        # x = F.dropout(x, p=prob)
        x = F.leaky_relu(self.fc_output(x))
        x = global_mean_pool(x, data_graph.batch)

        return x



class CEN_GNN_Encoder(torch.nn.Module):
    def __init__(self,node_feature_dim, hidden_feature_dim, edge_dim=None, heads=4, use_GAT=False):
        super().__init__()
        self.hidden_feature_dim = hidden_feature_dim
        self.use_GAT = use_GAT
        if self.use_GAT:
            self.conv1 = GATConv(node_feature_dim, 512, edge_dim=edge_dim, heads=heads, concat=True)
            self.conv2 = GATConv(512*heads, 512*heads, edge_dim=edge_dim, heads=heads, concat=True)
        else:
            self.conv1 = GCNConv(512, 1024)
            self.conv2 = GCNConv(1024, 1024)

        self.fc_output = torch.nn.Linear(512*heads*heads, self.hidden_feature_dim)       

    ###data_graph当训练时是Batch类型,是已经拼接好的对角块矩阵；当仅前向推理时,是Data类型
    def forward(self, data_graph, action, temp_img, prob=0.5):
        if isinstance(action,Batch):
            x = torch.concat((data_graph.x,action.x,temp_img),dim=1)
        else:
            x = torch.concat((data_graph.x,action,temp_img),dim=1)

        edge_index, edge_attr = data_graph.edge_index, data_graph.edge_attr
        
        # x = F.leaky_relu(self.fc(x))
        # x = F.leaky_relu(self.fc_input(x))
        if self.use_GAT:
            x = F.leaky_relu(self.conv1(x, edge_index, edge_attr=edge_attr))#, edge_weight=edge_weight
            x = F.leaky_relu(self.conv2(x, edge_index, edge_attr=edge_attr))#, edge_weight=edge_weight
        else:
            x = F.leaky_relu(self.conv1(x, edge_index))#, edge_weight=edge_weight
            x = F.leaky_relu(self.conv2(x, edge_index))#, edge_weight=edge_weight
        # x = F.dropout(x, p=prob)
        x = F.leaky_relu(self.fc_output(x))
        x = global_mean_pool(x, data_graph.batch)

        
        return x






class DEC_GNN_Encoder(torch.nn.Module):
    def __init__(self,node_feature_dim, hidden_feature_dim, edge_dim=None, heads=4, use_GAT=False):
        super().__init__()
        self.hidden_feature_dim = hidden_feature_dim

        # self.fc_input = torch.nn.Linear(node_feature_dim, 256)
        self.use_GAT = use_GAT
        if self.use_GAT:
            self.conv1 = GATConv(node_feature_dim, 128, edge_dim=edge_dim, heads=heads, concat=True)
            # self.conv2 = GATConv(128*heads, 128*heads, edge_dim=edge_dim, heads=heads, concat=True)
        else:
            self.conv1 = GCNConv(256, 512)
            # self.conv2 = GCNConv(512, 512)
        self.fc_output = torch.nn.Linear(128*heads, self.hidden_feature_dim)     
        

    ###data_graph当训练时是Batch类型,是已经拼接好的对角块矩阵；当仅前向推理时,是Data类型
    def forward(self, data_graph, num_agent_record, prob=0.5):
        
        x, edge_index, edge_attr = data_graph.x, data_graph.edge_index, data_graph.edge_attr
        # x = F.leaky_relu(self.fc_input(x))
        if self.use_GAT:
            x = F.leaky_relu(self.conv1(x, edge_index, edge_attr=edge_attr))#, edge_weight=edge_weight
            # x = F.leaky_relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        else:
            x = F.leaky_relu(self.conv1(x, edge_index))#, edge_weight=edge_weight

        # x = F.leaky_relu(self.conv2(x, edge_index))#, edge_weight=edge_weight
        # x = F.dropout(x,p=prob)
        output_id = []
        sum = 0
        for i in range(len(num_agent_record)):
            for j in range(num_agent_record[i]):
                id = sum + num_agent_record[i]*j + j
                output_id.append(id)
            sum += num_agent_record[i]**2

        x = F.leaky_relu(self.fc_output(x)[output_id])
        # import pdb;pdb.set_trace()
        # x = global_mean_pool(x, data_graph.batch)

        return x





class CEN_CNN_Encoder(nn.Module):
    def __init__(self,channel_input,dim_output):
        super().__init__()
        self.dim_output = dim_output
        
        self.group1 = nn.Sequential(
            nn.Conv2d(channel_input, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        
        self.group2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        self.group3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))) # 16 * 1 * 5 * 10

        
        self.l_1 = nn.Linear(625*16, 625)
        self.l_2 = nn.Linear(625, dim_output)

    def forward(self, input_image):
        """
        input:   disentangled lidar image sequence   [batch_size, C, H, W]
        output:  latent image info                   [batch_size, 256]
        """
        # import pdb;pdb.set_trace()
        state = self.group1(input_image)#F.leaky_relu(self.group1(input_image))
        state = self.group2(state)#F.leaky_relu(self.group2(state))
        state = self.group3(state)#F.leaky_relu(self.group3(state))

        state = state.view(-1, 625*16)

        state = F.leaky_relu(self.l_1(state))
        state = F.leaky_relu(self.l_2(state))

        return state


class DEC_CNN_Encoder(nn.Module):
    def __init__(self,channel_input,dim_output):
        super().__init__()
        self.dim_output = dim_output
        
        self.group1 = nn.Sequential(
            nn.Conv2d(channel_input, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        
        self.group2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        self.group3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))) # 16 * 1 * 5 * 10

        
        self.l_1 = nn.Linear(225*16, 625)
        self.l_2 = nn.Linear(625, dim_output)

    def forward(self, input_image):
        """
        input:   disentangled lidar image sequence   [batch_size, C, H, W]
        output:  latent image info                   [batch_size, 256]
        """
        # import pdb;pdb.set_trace()
        state = self.group1(input_image)#F.leaky_relu(self.group1(input_image))
        has_nan = np.isnan(state.cpu().detach().numpy()).any()
        if has_nan:
            import pdb;pdb.set_trace()
        state = self.group2(state)#F.leaky_relu(self.group2(state))
        has_nan = np.isnan(state.cpu().detach().numpy()).any()
        if has_nan:
            import pdb;pdb.set_trace()
        state = self.group3(state)#F.leaky_relu(self.group3(state))
        has_nan = np.isnan(state.cpu().detach().numpy()).any()
        if has_nan:
            import pdb;pdb.set_trace()
        state = state.view(-1, 225*16)

        state = F.leaky_relu(self.l_1(state))
        state = F.leaky_relu(self.l_2(state))

        return state