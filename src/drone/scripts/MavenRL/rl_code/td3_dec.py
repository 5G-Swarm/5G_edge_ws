
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from rllib.buffer import ReplayBuffer
from rllib.utils import init_weights, soft_update
from rllib.template import MethodSingleAgent, Model
# from rllib.template.model import FeatureExtractor, FeatureMapper


class TD3_DEC(MethodSingleAgent):
    gamma = 0.99
    
    lr_critic = 5e-4#0.0003
    lr_actor = 1e-4#0.0003 

    tau = 0.005

    buffer_size = 500000
    batch_size = 256

    policy_freq = 2
    explore_noise = 0.3
    policy_noise = 0.05#0.2
    noise_clip = 0.5

    start_timesteps = 30000

    save_model_interval = 200

    def __init__(self, config, writer):
        super(TD3_DEC, self).__init__(config, writer)
        
        self.n_agents = config.n_agents
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

        self.buffer: ReplayBuffer = config.get('buffer', ReplayBuffer)(config, self.buffer_size, self.batch_size, self.device)


    def update_parameters(self):
        if len(self.buffer) < self.start_timesteps:
            return
        self.update_parameters_start()
        self.writer.add_scalar('method/buffer_size', len(self.buffer), self.step_update)
        # import pdb;pdb.set_trace()
        '''load data batch'''
        experience = self.buffer.sample()
        state = experience.state
        action = experience.action
        next_state = experience.next_state
        reward = experience.reward
        done = experience.done

        '''critic'''
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, step_update=self.step_update) + noise).clamp(-1,1)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * (1-done) * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.writer.add_scalar('method/loss_critic', critic_loss.detach().item(), self.step_update)

        '''actor'''
        if self.step_update % self.policy_freq == 0:
            # import pdb;pdb.set_trace()
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
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
    def select_action(self, state):
        self.select_action_start()

        if self.step_select < self.start_timesteps:
            action = torch.Tensor(self.n_agents,self.dim_action).uniform_(-1,1)
        else:
            noise = torch.normal(0, self.explore_noise, size=(self.n_agents,self.dim_action))
            action = self.actor(state.to(self.device))
            action = (action.cpu() + noise).clamp(-1,1)
        return action

    def _update_model(self):
        # print('[update_parameters] soft update')
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)



class Actor(Model):
    def __init__(self, config, model_id=0):
        super(Actor, self).__init__(config, model_id)

        self.fe = config.get('net_actor_fe', FeatureExtractor)(config, model_id)
        self.fm = config.get('net_actor_fm', FeatureMapper)(config, model_id, self.fe.dim_feature, config.dim_action)
        self.no = nn.Tanh()  ## normalize output
        self.apply(init_weights)
    
    def forward(self, state, **kwargs):
        x = self.fe(state, **kwargs)
        return self.no(self.fm(x))


class Critic(Model):
    def __init__(self, config, model_id=0):
        super(Critic, self).__init__(config, model_id)
        
        self.fe = config.get('net_critic_fe', FeatureExtractor)(config, model_id)
        self.fm1 = config.get('net_critic_fm', FeatureMapper)(config, model_id, self.fe.dim_feature+config.dim_action, 1)
        self.fm2 = copy.deepcopy(self.fm1)
        self.apply(init_weights)

    def forward(self, state, action):
        x = self.fe(state)
        x = torch.cat([x, action], 1)
        return self.fm1(x), self.fm2(x)
    
    def q1(self, state, action):
        x = self.fe(state)
        x = torch.cat([x, action], 1)
        return self.fm1(x)


class FeatureExtractor(object):
    def __init__(self, config, model_id):
        self.dim_feature = config.dim_state
    def __call__(self, x, **kwargs):
        return x


# =============================================================================
# -- feature mapping ----------------------------------------------------------
# =============================================================================

# class FeatureMapper(Model):
#     def __init__(self, config, model_id, dim_input, dim_output):
#         super().__init__(config, model_id)
        
#         self.fm = nn.Sequential(
#             nn.Linear(dim_input, 512), nn.ReLU(),
#             nn.Linear(512, 1024), nn.ReLU(),
#             nn.Linear(1024, 512), nn.ReLU(),
#             nn.Linear(512, dim_output),
#         )
    
#     def forward(self, x):
#         return self.fm(x)

class FeatureMapper(Model):
    def __init__(self, config, model_id, dim_input, dim_output):
        super().__init__(config, model_id)
        
        self.fm = nn.Sequential(
            nn.Linear(dim_input, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, dim_output),
        )
    
    def forward(self, x):
        return self.fm(x)