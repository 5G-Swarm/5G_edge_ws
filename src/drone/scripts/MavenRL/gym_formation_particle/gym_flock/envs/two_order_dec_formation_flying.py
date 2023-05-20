import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from matplotlib.patches import Ellipse, Circle
from sklearn.neighbors import NearestNeighbors
import itertools 
import random
import pdb
import math
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}




class two_order_decentralized_FormationFlyingEnv(gym.Env):

    def __init__(self):

        config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['flock']

        self.dynamic = True # if the agents are moving or not
        self.mean_pooling = False # normalize the adjacency matrix by the number of neighbors or not
        #self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
        self.degree = 2#5
        # number states per agent
        self.nx_system = 6
        # numer of features per agent
        self.n_features = 4#2

        # number of actions per agent
        self.nu = 2 

        # problem parameters from file
        self.init_n_agents = 10
        self.n_agents = self.init_n_agents
        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius


        # intitialize agent state 
        self.center = np.array([0.0,0.0,0.0,0.0])
        self.counter = 0   
        self.a_net = np.zeros((self.n_agents, self.n_agents)) 
        self.x = np.zeros((self.n_agents, self.nx_system))
        self.u = np.zeros((self.n_agents, self.nu))
        self.obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))
        self.copy_obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))
        self.pre_obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))
        
        self.radius = 5
        self.adj_distance_matrix = np.zeros((self.n_agents,2))
        self.ag_threshold = 2*self.radius*math.sin(math.pi/self.n_agents)
        self.collision_threshold = 1.5#.5#0.8
        self.on_circle_threshold = 0.5#0.8

        # TODO : what should the action space be? is [-1,1] OK?
        self.max_accel = 1 
        self.gain = 1.0 # TODO - adjust if necessary - may help the NN performance
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(self.n_agents,self.nu),dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents,(1+self.degree)*self.n_features*2),
                                            dtype=np.float32)#(1+self.degree)*self.n_features*2

        #render init     
        self.r_max = 20#float(config['max_rad_init'])
        self.fig = None
        self.line1 = None
        self.line2 = None
        self.line3 = None

        self.seed()

        
        
        ###initialize obstacle set:
        self.n_obstacle = 0
        self.obstacle_state = np.zeros((self.n_obstacle,self.n_features))

        ######limitation######
        self.VEL_BOUND = 1
        self.ACCEL_BOUND = 1




    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # import pdb;pdb.set_trace()
        self.counter = 0
        self.n_agents = self.init_n_agents
        self.x = np.zeros((self.n_agents, self.nx_system)) #keep this to track position
        team_center = [np.random.uniform(-3,3),np.random.uniform(-3,3)]
        for i in range(self.n_agents):
            self.x[i,0] = np.random.uniform(-15,15)+team_center[0]
            self.x[i,1] = np.random.uniform(-15,15)+team_center[1]
        

        self.obstacle_state = np.zeros((self.n_obstacle,self.n_features))
        for i in range(self.n_obstacle):
            self.obstacle_state[i,:2] = np.random.uniform(-1,1,2)+self.center[:2]
        self.obstacle_speed_amplitude = np.random.uniform(-3,3,(self.n_obstacle,2))
        
        
        self.center= np.array([np.random.uniform(-3,3),np.random.uniform(-3,3),0.0,0.0])#np.array([0.0,0.0,0.0,0.0])#  
        self.center_speed_x = 0#float(np.random.uniform(-1.4,1.4,1))
        self.center_speed_y = 0#float(np.random.uniform(-1.4,1.4,1))
        


        self.u = np.zeros((self.n_agents, self.nu))
        self.obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))
        self.pre_obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))
        #init reward:
        self.dist_list = np.zeros(self.n_agents)
        self.pre_dist_list = np.zeros(self.n_agents)
        self.dist_agn2tar_list = np.zeros(self.n_agents)
        self.energy_list  = np.zeros(self.n_agents)
        self.on_circle_time = np.zeros(self.n_agents)
        #done:
        self.done_dict = np.zeros(self.n_agents)        
        return self._get_obs()


    def _get_obs(self):

        self.pre_obs_dict[:] = self.obs_dict[:]
        self.total_state = np.vstack((self.x[:,:self.n_features],self.obstacle_state))
        adj_matrix = self.get_connectivity(self.total_state)

        # import pdb;pdb.set_trace()
        for i in range(self.n_agents):
            temp = list(self.center[:self.n_features] - self.x[i,:self.n_features])

            for j in list(self.neighbor_id_matrix[i]):
                temp_list = list((self.total_state[j,:self.n_features] - self.x[i,:self.n_features]).squeeze())#/3
                temp = temp + temp_list 

            self.obs_dict[i] = np.array(temp)
        self.copy_obs_dict[:] = self.obs_dict[:] 
        self.obs_dict = self.add_noise_to_obs(self.obs_dict,0.5,0.5)#0.03
        #return (state_values, state_network)
        return np.hstack((self.obs_dict,self.pre_obs_dict))#/self.radius*5#self.feats.reshape(1,self.n_agents*self.n_features)


    def add_noise_to_obs(self,obs_dict,pos_noise_scale,vel_noise_scale):
        noise_pos = np.random.normal(loc=0.0, scale=pos_noise_scale, size=(obs_dict.shape[0],int(obs_dict.shape[1]/2)))
        noise_vel = np.random.normal(loc=0.0, scale=vel_noise_scale, size=(obs_dict.shape[0],int(obs_dict.shape[1]/2)))
        temp =  np.hstack((noise_pos[:,:2],noise_vel[:,:2],noise_pos[:,2:4],noise_vel[:,2:4],noise_pos[:,4:6],noise_vel[:,4:6],noise_pos[:,6:8],noise_vel[:,6:8],noise_pos[:,8:10],noise_vel[:,8:10],noise_pos[:,10:12],noise_vel[:,10:12]))
        # for i in range(obs_dict.shape[1]/2):
        #     temp = np.hstack((,))
        noisy_obs_dict = obs_dict + temp
        return noisy_obs_dict
# def _get_obs(self):

#         self.pre_obs_dict[:] = self.obs_dict[:]
#         self.total_state = np.vstack((self.x[:,:self.n_features],self.obstacle_state))
#         adj_matrix = self.get_connectivity(self.total_state)

#         # import pdb;pdb.set_trace()
#         for i in range(self.n_agents):
#             temp = list(self.center[:self.n_features] - self.x[i,:self.n_features])

#             a = list(np.argwhere(adj_matrix[i]==1.0).squeeze())
            
#             list0 = list((self.total_state[a[0],:self.n_features] - self.x[i,:self.n_features]).squeeze())
#             list1 = list((self.total_state[a[1],:self.n_features] - self.x[i,:self.n_features]).squeeze())
            
#             dist0 = np.linalg.norm((self.total_state[a[0],:2] - self.x[i,:2]).squeeze())
#             dist1 = np.linalg.norm((self.total_state[a[1],:2] - self.x[i,:2]).squeeze())
            
#             if dist0<=dist1: 
#                 temp = temp + list0
#                 temp = temp + list1
#             else:
#                 temp = temp + list1
#                 temp = temp + list0

#             self.obs_dict[i] = np.array(temp)
        
#         #return (state_values, state_network)
#         return np.hstack((self.obs_dict,self.pre_obs_dict))#/self.radius*5#self.feats.reshape(1,self.n_agents*self.n_features)

    def step(self, action_dict):
        
        info_dict = {}
        self.counter = self.counter+1

        # temp = np.reshape(action,(self.n_agents, self.nu))
        # for i in range(self.n_agents):
        #     agent_id = 'agent-' + str(i) 
        #     self.u[i] = action_dict[agent_id]
        self.u = action_dict*5
        #     self.u[i,:] = temp[i,:]+self.center[-2:] 
        # import pdb;pdb.set_trace()

        self.center[2] = self.center_speed_x*math.sin(self.counter/30)
        self.center[3] = self.center_speed_y*math.cos(self.counter/30)
        # self.center[:2] = self.center[:2] + self.center[2:]*0.1
        self.center[:2] = self.center[:2] + self.center[2:4]*0.1
        # self.center[1] = self.center[1] + self.center[3]*0.1
        
        self.obstacle_state[:,2] =  self.obstacle_speed_amplitude[:,0]*math.sin(self.counter/20)
        self.obstacle_state[:,3] =  self.obstacle_speed_amplitude[:,1]*math.cos(self.counter/20)
        self.obstacle_state[:,:2] = self.obstacle_state[:,:2] + self.obstacle_state[:,2:4]*0.1

        print("self.center",self.center)
        self.x[:, 4:] = self.u[:, :2]
        self.x[:, 4:] = np.clip(self.x[:, 4:],-self.ACCEL_BOUND,self.ACCEL_BOUND)
        self.x[:, 2:4] = self.x[:, 2:4] + self.x[:, 4:]*0.1
        self.x[:, 2:4] = np.clip(self.x[:, 2:4],-self.VEL_BOUND,self.VEL_BOUND)
        self.x[:, :2] = self.x[:, :2] + self.x[:, 2:4]*0.1

        print("self.x",self.x)

        # done = False 
        # if self.counter > 4000000 :
        #     done = True 
        # if np.all(diffs_x) <0.2 and np.all(diffs_y)<0.2:
            # done = True 
        self.pre_n_agents = self.n_agents
        # if self.counter==50:
        #     self.n_agents = self.init_n_agents +2
        #     self.n_agents_change()
        
        # if self.counter==150:
        #     self.n_agents = self.init_n_agents -2
        #     self.n_agents_change()
        
        # if self.counter==250:
        #     self.n_agents = self.init_n_agents +3
        #     self.n_agents_change()


        return self._get_obs(), self.instant_reward(), self.done_dict, info_dict

    def n_agents_change(self):
        if self.n_agents < self.pre_n_agents:
            self.x = self.x[:self.n_agents,:]
            self.u = self.u[:self.n_agents,:]
            self.obs_dict = self.obs_dict[:self.n_agents,:]
            self.pre_obs_dict = self.pre_obs_dict[:self.n_agents,:]
            #reward:
            self.dist_list = self.dist_list[:self.n_agents]
            self.pre_dist_list = self.pre_dist_list[:self.n_agents]
            self.dist_agn2tar_list = self.dist_agn2tar_list[:self.n_agents]
            self.energy_list  = self.energy_list[:self.n_agents]
            self.on_circle_time = self.on_circle_time[:self.n_agents]
            #done:
            self.done_dict = self.done_dict[:self.n_agents]
        
        elif self.n_agents > self.pre_n_agents:
            temp_pos = np.random.uniform(-14,14,(self.n_agents - self.pre_n_agents,2))
            temp_zero = np.zeros((self.n_agents - self.pre_n_agents,self.x.shape[1]-2))
            temp = np.hstack((temp_pos,temp_zero))
            self.x = np.vstack((self.x,temp))

            temp_zero = np.zeros((self.n_agents - self.pre_n_agents,self.u.shape[1]))
            self.u = np.vstack((self.u,temp_zero))

            temp_zero = np.zeros((self.n_agents - self.pre_n_agents,self.obs_dict.shape[1]))
            self.obs_dict = np.vstack((self.obs_dict,temp_zero))
            self.pre_obs_dict = np.vstack((self.pre_obs_dict,temp_zero))
            #reward:
            temp_zero = np.zeros(self.n_agents - self.pre_n_agents)
            self.dist_list = np.hstack((self.dist_list,temp_zero))
            self.pre_dist_list = np.hstack((self.pre_dist_list,temp_zero))
            self.dist_agn2tar_list = np.hstack((self.dist_agn2tar_list,temp_zero))
            self.energy_list  = np.hstack((self.energy_list,temp_zero))
            self.on_circle_time = np.hstack((self.on_circle_time,temp_zero))
            #done:
            self.done_dict = np.hstack((self.done_dict,temp_zero))
            

   

    def instant_reward(self):  # sum of differences in velocities
        reward_dict = np.zeros(self.n_agents)
        cost_dist = np.zeros(self.n_agents)
        cost_dist_agn2tar = np.zeros(self.n_agents)
        cost_energy = np.zeros(self.n_agents)
        cost_distribution = np.zeros(self.n_agents)
        cost_on_circle = np.zeros(self.n_agents)

        for i in range(self.n_agents):
            # temp_vector = self.x[i,:2] - self.center[:2]
            self.dist_list[i] = math.fabs(np.linalg.norm(self.copy_obs_dict[i,:2])-self.radius)#
            if self.counter==1: 
                cost_dist[i] = 0
            else: 
                # import pdb;pdb.set_trace()
                cost_dist[i] = (self.dist_list[i]-self.pre_dist_list[i])*4.5#12

            if self.dist_list[i]<self.on_circle_threshold:#0.25
                self.on_circle_time[i] = -5#-5
            else:
                self.on_circle_time[i] = 0
            
            cost_on_circle[i] = self.on_circle_time[i]*0.1
            temp_dist_1 = np.linalg.norm(self.copy_obs_dict[i,:2])

            if temp_dist_1>(self.radius-self.on_circle_threshold):#self.ag_threshold*2:#*1.5
                k = 0
            else:
                k = 0.5#3#2.5
            if temp_dist_1>self.radius/2:
                cost_dist_agn2tar[i] = k/(temp_dist_1+0.01) 
            else: 
                # import pdb;pdb.set_trace() 
                cost_dist_agn2tar[i] = 50 
                self.done_dict[i] = 1
            #[1/(dist+0.01) if dist<self.ag_threshold else 0 for dist in self.adj_distance_matrix.reshape(self.n_agents*self.degree)]
            cost_energy[i] = np.linalg.norm(self.u[i])*0.3#0.6
            temp_dist_list = []

            for temp_dist_2 in list(self.adj_distance_matrix[i]):
                if temp_dist_2>self.radius*2:#self.ag_threshold*2:#*1.5
                    k = 0
                else:
                    k = 0.5#3#2.5
                if temp_dist_2>self.collision_threshold:#0.5
                    temp_dist_list.append(k/(temp_dist_2-self.collision_threshold+0.05)) 
                else:
                    # import pdb;pdb.set_trace() 
                    temp_dist_list.append(100)
                    self.done_dict[i] = 1
                    # break

            cost_distribution[i] = np.sum(temp_dist_list)/self.degree  
            
                        # if (temp_vector[0])==0:
            #     temp = -90#math.pi/2 
            # else:
            #     temp = math.atan(temp_vector[1]/temp_vector[0])/math.pi*180
            # angle =  temp if temp_vector[1]<0 else temp+180
            # angle_list.append(angle)

            reward_dict[i] = -(cost_dist[i] + cost_dist_agn2tar[i]+cost_distribution[i] + cost_energy[i]+cost_on_circle[i])



        #cost_accumulate_dist = 0#np.sum(self.dist_list)*0.3/self.n_agents
        print("adj_distance_matrix",self.adj_distance_matrix)


        # angle_list.sort()
        # next_angle_list = angle_list[1:] + [angle_list[0]+360]
        # self.distribution_std = np.std(np.array(next_angle_list)-np.array(angle_list))*0.1
        # if  self.counter>1:#np.sum(self.dist_list) <(0.25*self.n_agents) and 
        #     cost_distribution = (self.pre_distribution_std-self.distribution_std)*10
        # else:


        print("diff_list",self.dist_list)
        print("dist_agn2tar_list",self.dist_agn2tar_list)
        print("reward_dict",reward_dict)
        print("cost_dist",-cost_dist)
        print("cost_distribution",-cost_distribution)
        print("cost_dist_agn2tar",-cost_dist_agn2tar)
        print("cost_energy_list",-cost_energy)
        print("cost_on_circle",-cost_on_circle)
        
        ##更新距离，注意要进行内容层面的深拷贝，不要直接把句柄复制
        self.pre_dist_list[:] = self.dist_list[:]
        # self.pre_distribution_std = self.distribution_std 

        return reward_dict


    

    def dist2_mat(self, x):

        x_loc = np.reshape(x[:, 0:2], (self.n_agents,2,1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net


    def get_connectivity(self, x):

        # a_net = self.adj_distance_matrix = self.dist2_mat(x)
        # import pdb;pdb.set_trace()
        if self.degree == 0: #or self.n_agents < self.n_features:
            a_net = self.dist2_mat(x)
            a_net = (a_net < self.comm_radius2).astype(float)
        else:
            neigh = NearestNeighbors(n_neighbors=self.degree)
            neigh.fit(x[:,:2])
            # neigh.fit(x[:,2:4])
            # import pdb;pdb.set_trace()

            self.adj_distance_matrix,self.neighbor_id_matrix = neigh.kneighbors()
            a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())
            
        # if self.mean_pooling:
        #     # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        #     n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents,1)) # TODO or axis=0? Is the mean in the correct direction?
        #     n_neighbors[n_neighbors == 0] = 1
        #     a_net = a_net / n_neighbors 

        return a_net


    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """
        temp_list = [0,1,2,3,4,5,6,7,8,9]
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            global ax,annotator
            ax = fig.add_subplot(111)
            self.line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')  # Returns a tuple of line objects, thus the comma
            self.line2, = ax.plot(self.center[0], self.center[1], 'rx')
            self.line3, = ax.plot(self.obstacle_state[:,0],self.obstacle_state[:,1], 'gs')
            self.circle, = ax.plot([], [], '-', color='r', lw=1)
            #ax.plot([0], [0], 'kx')
            # ax.plot(self.start_xpoints, self.start_ypoints, 'kx')
            
            # print(self.x[:,2],self.x[:,3])
            # cir1 = Circle(xy = (self.center[0],self.center[1]), radius=self.radius, alpha=0.2)
            # ax.add_patch(cir1)
            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            a = gca()
            # a.set_xticklabels(a.get_xticks(), font)
            # a.set_yticklabels(a.get_yticks(), font)
            annotator = []
            for i in range(self.n_agents):
                annotator.append(ax.annotate(temp_list[i],(self.x[i,0],self.x[i,1])))
            plt.title('MavenRL Simulator')
            self.fig = fig
        
        theta = np.linspace(0, 2*np.pi, 100) # 参数t的数组
        x_out = [self.center[0]+self.radius*np.cos(theta[i]) for i in range(len(theta))]
        y_out = [self.center[1]+self.radius*np.sin(theta[i]) for i in range(len(theta))]
        
        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])

        for i in range(self.n_agents):
            annotator[i].remove()
            annotator[i] = ax.annotate(temp_list[i],(self.x[i,0],self.x[i,1]))
        self.line2.set_xdata(self.center[0])
        self.line2.set_ydata(self.center[1])

        self.line3.set_xdata(self.obstacle_state[:,0])
        self.line3.set_ydata(self.obstacle_state[:,1])
        
        self.circle.set_xdata(x_out)
        self.circle.set_ydata(y_out)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass
 


