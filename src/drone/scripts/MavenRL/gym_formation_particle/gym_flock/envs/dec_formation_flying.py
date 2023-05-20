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


class decentralized_FormationFlyingEnv(gym.Env):

    def __init__(self):

        config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['flock']

        self.dynamic = True # if the agents are moving or not
        self.mean_pooling = False # normalize the adjacency matrix by the number of neighbors or not
        #self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
        self.degree = 2
        # number states per agent
        self.nx_system = 4
        # numer of features per agent
        self.n_features = 2

        # number of actions per agent
        self.nu = 2 

        # problem parameters from file
        self.n_agents = 12
        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max 
        self.r_max = 20#float(config['max_rad_init'])
        self.std_dev = float(config['std_dev']) * self.dt

        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))
        self.u = np.zeros((self.n_agents, self.nu))
        self.a_net = np.zeros((self.n_agents, self.n_agents))
        self.center = np.array([0.0,0.0,0.0,0.0])
        self.counter = 0    
        self.obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))
        self.pre_obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))

        # TODO : what should the action space be? is [-1,1] OK?
        self.max_accel = 1 
        self.gain = 1.0 # TODO - adjust if necessary - may help the NN performance
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(self.n_agents,self.nu),dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents,(1+self.degree)*self.n_features*2),
                                            dtype=np.float32)

        #render init     
        self.fig = None
        self.line1 = None
        self.line2 = None

        self.seed()

        self.radius = 8
        self.adj_distance_matrix = np.zeros((self.n_agents,2))
        self.ag_threshold = 2*self.radius*math.sin(math.pi/self.n_agents)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # import pdb;pdb.set_trace()
        self.x = np.zeros((self.n_agents, self.nx_system)) #keep this to track position
        self.feats = np.zeros((self.n_agents,self.n_features)) #this is the feature we return to the agent
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.01  # 0.25 

        self.counter = 0
        self.agent_xg = []
        self.agent_yg = []


        self.agent_startx = []
        self.agent_starty = []
        
        self.center_speed_x = float(np.random.uniform(-0.7,0.7,1))
        self.center_speed_y = float(np.random.uniform(-0.7,0.7,1))
        
        #scheme : 
        #space all agents in a frontier that looks like -> .'.
        #this means everyone has x goal position separated by two units. 
        #and y goal position varies  
        # let the number of agents be odd. for kosher reasons. 

        #hack, say n agents = 3. Then placer_x is = -2 
        team_center = [np.random.uniform(-4,4),np.random.uniform(-4,4)]
        for i in range(self.n_agents):
            self.x[i,0] = np.random.uniform(-5,5)+team_center[0]
            self.x[i,1] = np.random.uniform(-5,5)+team_center[1]
        
        self.center= np.array([np.random.uniform(-3,3),np.random.uniform(-3,3),0.0,0.0])#np.array([0.0,0.0,0.0,0.0])#  
        # self.placer_x = (self.n_agents/2)*2*(-1)        
        
        #init reward:
        self.dist_list = np.zeros(self.n_agents)
        self.pre_dist_list = np.zeros(self.n_agents)
        self.dist_agn2tar_list = np.zeros(self.n_agents)
        self.energy_list  = np.zeros(self.n_agents)
        self.done_dict = np.zeros(self.n_agents)
        self.on_circle_time = np.zeros(self.n_agents)
        
        return self._get_obs()


    def _get_obs(self):
        
        # self.feats[:,0] = self.x[:,0]-self.center[0]# - self.x[:,2]
        # self.feats[:,1] = self.x[:,1]-self.center[1] #- self.x[:,3]
          
        # self.feats[:,2] = self.x[:,2]-self.center[2]
        # self.feats[:,3] = self.x[:,3]-self.center[3]
        # for i in range(self.n_features):
        #     self.feats[:,i] =  self.x[:,i]-self.center[i]
        self.pre_obs_dict[:] = self.obs_dict[:]
        if self.dynamic:
            adj_matrix = self.get_connectivity(self.x)
        else:
            adj_matrix = self.a_net
        # import pdb;pdb.set_trace()
        for i in range(self.n_agents):
            temp = list(self.center[:self.n_features] - self.x[i,:self.n_features])

            a = list(np.argwhere(adj_matrix[i]==1.0).squeeze())
            
            list0 = list((self.x[a[0],:self.n_features] - self.x[i,:self.n_features]).squeeze())
            list1 = list((self.x[a[1],:self.n_features] - self.x[i,:self.n_features]).squeeze())
            
            dist0 = np.linalg.norm((self.x[a[0],:self.n_features] - self.x[i,:self.n_features]).squeeze())
            dist1 = np.linalg.norm((self.x[a[1],:self.n_features] - self.x[i,:self.n_features]).squeeze())
            
            if dist0<=dist1: 
                temp = temp + list0
                temp = temp + list1
            else:
                temp = temp + list1
                temp = temp + list0

            self.obs_dict[i] = np.array(temp)
        
        #return (state_values, state_network)
        return np.hstack((self.obs_dict,self.pre_obs_dict))/self.radius*5#self.feats.reshape(1,self.n_agents*self.n_features)


    def step(self, action_dict):
        
        
        info_dict = {}
        self.counter = self.counter+1
        # temp = np.reshape(action,(self.n_agents, self.nu))
        # for i in range(self.n_agents):
        #     agent_id = 'agent-' + str(i) 
        #     self.u[i] = action_dict[agent_id]
        self.u = action_dict
        #     self.u[i,:] = temp[i,:]+self.center[-2:] 
        # import pdb;pdb.set_trace()

        self.center[2] = self.center_speed_x*math.sin(self.counter/30)
        self.center[3] = self.center_speed_y*math.cos(self.counter/30)
        # self.center[:2] = self.center[:2] + self.center[2:]*0.1
        self.center[0] = self.center[0] + self.center[2]*0.1
        self.center[1] = self.center[1] + self.center[3]*0.1

        print("self.center",self.center)
        self.x[:, 2:] = self.u[:, :2]
        self.x[:, :2] = self.x[:, :2] + self.x[:, 2:]*0.1

        print("self.x",self.x)

        # done = False 
        # if self.counter > 4000000 :
        #     done = True 
        # if np.all(diffs_x) <0.2 and np.all(diffs_y)<0.2:
            # done = True 

        return self._get_obs(), self.instant_reward(), self.done_dict, info_dict

   

    def instant_reward(self):  # sum of differences in velocities
        reward_dict = np.zeros(self.n_agents)
        cost_dist = np.zeros(self.n_agents)
        cost_dist_agn2tar = np.zeros(self.n_agents)
        cost_energy = np.zeros(self.n_agents)
        cost_distribution = np.zeros(self.n_agents)
        cost_on_circle = np.zeros(self.n_agents)

        for i in range(self.n_agents):
            # temp_vector = self.x[i,:2] - self.center[:2]
            self.dist_list[i] = math.fabs(np.linalg.norm(self.obs_dict[i,:2])-self.radius)#
            if self.counter==1: 
                cost_dist[i] = 0
            else: 
                # import pdb;pdb.set_trace()
                cost_dist[i] = (self.dist_list[i]-self.pre_dist_list[i])*15

            if self.dist_list[i]<0.25:
                self.on_circle_time[i] -= 1
            else:
                self.on_circle_time[i] = 0
            
            cost_on_circle[i] = self.on_circle_time[i]*0.1
            temp_dist_1 = np.linalg.norm(self.obs_dict[i,:2])
            if temp_dist_1>self.radius/3:
                cost_dist_agn2tar[i] = 0.07/(temp_dist_1+0.01) 
            else: 
                # import pdb;pdb.set_trace() 
                cost_dist_agn2tar[i] = 50 
                self.done_dict[i] = 1
            #[1/(dist+0.01) if dist<self.ag_threshold else 0 for dist in self.adj_distance_matrix.reshape(self.n_agents*self.degree)]
            cost_energy[i] = np.linalg.norm(self.u[i])*0.6
            temp_dist_list = []

            for temp_dist_2 in list(self.adj_distance_matrix[i]):
                if temp_dist_2>self.radius*2:#self.ag_threshold*2:#*1.5
                    k = 0
                else:
                    k = 3#2.5
                if temp_dist_2>0.5:
                    temp_dist_list.append(k/(temp_dist_2*temp_dist_2+0.01)) 
                else:
                    # import pdb;pdb.set_trace() 
                    temp_dist_list.append(200)
                    self.done_dict[i] = 1
                    break

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
        print("cost_energy_list",-cost_on_circle)
        
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
            self.adj_distance_matrix,_ = neigh.kneighbors()
            a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())
            # import pdb;pdb.set_trace()

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

        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')  # Returns a tuple of line objects, thus the comma
            line2, = ax.plot(self.center[0], self.center[1], 'rx')
            circle, = ax.plot([], [], '-', color='r', lw=1)
            #ax.plot([0], [0], 'kx')
            # ax.plot(self.start_xpoints, self.start_ypoints, 'kx')
            
            print(self.x[:,2],self.x[:,3])
            # cir1 = Circle(xy = (self.center[0],self.center[1]), radius=self.radius, alpha=0.2)
            # ax.add_patch(cir1)
            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            a = gca()
            # a.set_xticklabels(a.get_xticks(), font)
            # a.set_yticklabels(a.get_yticks(), font)

            plt.title('GNN Controller')
            self.fig = fig
            self.line1 = line1
            self.line2 = line2
            self.circle = circle
            
        theta = np.linspace(0, 2*np.pi, 100) # 参数t的数组
        x_out = [self.center[0]+self.radius*np.cos(theta[i]) for i in range(len(theta))]
        y_out = [self.center[1]+self.radius*np.sin(theta[i]) for i in range(len(theta))]
        
        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])

        self.line2.set_xdata(self.center[0])
        self.line2.set_ydata(self.center[1])
        self.circle.set_xdata(x_out)
        self.circle.set_ydata(y_out)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass
 





# class decentralized_FormationFlyingEnv(gym.Env):

#     def __init__(self):

#         config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
#         config = configparser.ConfigParser()
#         config.read(config_file)
#         config = config['flock']

#         self.dynamic = True # if the agents are moving or not
#         self.mean_pooling = False # normalize the adjacency matrix by the number of neighbors or not
#         #self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
#         self.degree = 2
#         # number states per agent
#         self.nx_system = 4
#         # numer of features per agent
#         self.n_features = 2

#         # number of actions per agent
#         self.nu = 2 

#         # problem parameters from file
#         self.n_agents = 3
#         self.comm_radius = float(config['comm_radius'])
#         self.comm_radius2 = self.comm_radius * self.comm_radius
#         self.dt = float(config['system_dt'])
#         self.v_max = float(config['max_vel_init'])
#         self.v_bias = self.v_max 
#         self.r_max = float(config['max_rad_init'])
#         self.std_dev = float(config['std_dev']) * self.dt

#         # intitialize state matrices
#         self.x = np.zeros((self.n_agents, self.nx_system))
#         self.u = np.zeros((self.n_agents, self.nu))
#         self.a_net = np.zeros((self.n_agents, self.n_agents))
#         self.center=np.array([0.0,0.0,0.0,0.0])
#         self.counter = 0    

#         # TODO : what should the action space be? is [-1,1] OK?
#         self.max_accel = 1 
#         self.gain = 1.0 # TODO - adjust if necessary - may help the NN performance
#         self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(self.n_agents,self.nu),dtype=np.float32)

#         self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents,1+self.degree*self.n_features),
#                                             dtype=np.float32)

#         #render init     
#         self.fig = None
#         self.line1 = None
#         self.line2 = None

#         self.seed()

#         self.radius = 1
#         self.adj_distance_matrix = np.zeros((self.n_agents,2))
#         self.ag_threshold = 2*self.radius*math.sin(math.pi/self.n_agents)

#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def reset(self):
#         # import pdb;pdb.set_trace()
#         self.x = np.zeros((self.n_agents, self.nx_system)) #keep this to track position
#         self.feats = np.zeros((self.n_agents,self.n_features)) #this is the feature we return to the agent
#         degree = 0
#         min_dist = 0
#         min_dist_thresh = 0.01  # 0.25 

#         self.counter = 0
#         self.agent_xg = []
#         self.agent_yg = []

#         self.agent_startx = []
#         self.agent_starty = []
        
        
#         #scheme : 
#         #space all agents in a frontier that looks like -> .'.
#         #this means everyone has x goal position separated by two units. 
#         #and y goal position varies  
#         # let the number of agents be odd. for kosher reasons. 

#         #hack, say n agents = 3. Then placer_x is = -2 
#         team_center = [np.random.uniform(-4,4),np.random.uniform(-4,4)]
#         for i in range(self.n_agents):
#             self.x[i,0] = np.random.uniform(-1,1)+team_center[0]
#             self.x[i,1] = np.random.uniform(-1,1)+team_center[1]
        
#         self.center= np.array([np.random.uniform(-3,3),np.random.uniform(-3,3),0.0,0.0])#np.array([0.0,0.0,0.0,0.0])#  
#         # self.placer_x = (self.n_agents/2)*2*(-1)        
        
#         #init reward:
#         self.dist_list = np.zeros(self.n_agents)
#         self.pre_dist_list = np.zeros(self.n_agents)
#         self.dist_agn2tar_list = np.zeros(self.n_agents)
#         self.energy_list  = np.zeros(self.n_agents)
        
#         return self._get_obs()


#     def _get_obs(self):
        

#         # self.feats[:,0] = self.x[:,0]-self.center[0]# - self.x[:,2]
#         # self.feats[:,1] = self.x[:,1]-self.center[1] #- self.x[:,3]
          
#         # self.feats[:,2] = self.x[:,2]-self.center[2]
#         # self.feats[:,3] = self.x[:,3]-self.center[3]
#         # for i in range(self.n_features):
#         #     self.feats[:,i] =  self.x[:,i]-self.center[i]
#         if self.dynamic:
#             adj_matrix = self.get_connectivity(self.x)
#         else:
#             adj_matrix = self.a_net

#         self.obs_dict = {}
#         for i in range(self.n_agents):
#             agent_id = 'agent-' + str(i) 
#             self.obs_dict[agent_id] = []
#             self.obs_dict[agent_id].append( self.center[:self.n_features] - self.x[i,:self.n_features])
#             for j in list(np.argwhere(adj_matrix[i]==1)):
#                 self.obs_dict[agent_id].append( self.x[j,:self.n_features] - self.x[i,:self.n_features])

#             self.obs_dict[agent_id] = np.array(self.obs_dict[agent_id]).flatten()
        
#         #return (state_values, state_network)
#         return self.obs_dict#self.feats.reshape(1,self.n_agents*self.n_features)


#     def step(self, action_dict):
        
#         done_dict = {'__all__':False}
#         info_dict = {}
#         self.counter = self.counter+1

#         # temp = np.reshape(action,(self.n_agents, self.nu))
#         for i in range(self.n_agents):
#             agent_id = 'agent-' + str(i) 
#             self.u[i] = action_dict[agent_id]
#         #     self.u[i,:] = temp[i,:]+self.center[-2:] 
#         # import pdb;pdb.set_trace()
#         self.center[2] = 0#0.3*math.sin(self.counter/50)
#         self.center[3] = 0#0.3*math.cos(self.counter/50)
#         # self.center[:2] = self.center[:2] + self.center[2:]*0.1
#         self.center[0] = self.center[0] + self.center[2]*0.1
#         self.center[1] = self.center[1] + self.center[3]*0.1

#         print("self.center",self.center)
#         self.x[:, 2:] = self.u[:, :2]
#         self.x[:, :2] = self.x[:, :2] + self.x[:, 2:]*0.1

#         print("self.x",self.x)

#         # done = False 
#         # if self.counter > 4000000 :
#         #     done = True 
#         # if np.all(diffs_x) <0.2 and np.all(diffs_y)<0.2:
#             # done = True 

#         return self._get_obs(), self.instant_reward(), done_dict, info_dict

   

#     def instant_reward(self):  # sum of differences in velocities
#         reward_dict = {}

#         angle_list = []
#         for i in range(self.n_agents):
#             temp_vector = self.x[i,:2] - self.center[:2]
#             self.dist_list[i] = math.fabs(np.linalg.norm(temp_vector))#-self.radius
#             self.dist_agn2tar_list[i] = np.linalg.norm(self.feats[i,:2])
#             self.energy_list = np.linalg.norm(self.u[i,:]) 
#             if (temp_vector[0])==0:
#                 temp = -90#math.pi/2 
#             else:
#                 temp = math.atan(temp_vector[1]/temp_vector[0])/math.pi*180
#             angle =  temp if temp_vector[1]<0 else temp+180
#             angle_list.append(angle)        


#         if self.counter==1: 
#             cost_dist2goal=0
#         else: 
#             cost_dist2goal = np.sum(self.dist_list-self.pre_dist_list)*30/self.n_agents
#         #cost_accumulate_dist = 0#np.sum(self.dist_list)*0.3/self.n_agents
#         print("adj_distance_matrix",self.adj_distance_matrix)
#         dist_distribution = [1/(dist+0.01) if dist<self.ag_threshold else 0 for dist in self.adj_distance_matrix.reshape(self.n_agents*self.degree)]#np.std(np.array(next_angle_list)-np.array(angle_list))*0.1
#         cost_distribution = np.sum(dist_distribution)/self.n_agents*1.5

#         # angle_list.sort()
#         # next_angle_list = angle_list[1:] + [angle_list[0]+360]
#         # self.distribution_std = np.std(np.array(next_angle_list)-np.array(angle_list))*0.1
#         # if  self.counter>1:#np.sum(self.dist_list) <(0.25*self.n_agents) and 
#         #     cost_distribution = (self.pre_distribution_std-self.distribution_std)*10
#         # else:
#         #     cost_distribution = 0 
        
#         cost_dist_var = 0#np.std(self.dist_list)**2
#         dist_agn2tar_list = [1/(dist+0.01) if dist<self.radius else 0 for dist in self.dist_agn2tar_list]
#         cost_dist_agn2tar = np.sum(dist_agn2tar_list)/self.n_agents
#         cost_energy = 0#np.sum(self.energy_list)*0.3/self.n_agents

#         print("diff_list",self.dist_list)
#         print("dist_agn2tar_list",self.dist_agn2tar_list)
#         instant_reward = -(cost_dist2goal +cost_distribution+ cost_dist_var+cost_dist_agn2tar+cost_energy)
#         print("cost",instant_reward,-cost_dist2goal,-cost_distribution,-cost_dist_agn2tar,-cost_energy)
        
#         ##更新距离，注意要进行内容层面的深拷贝，不要直接把句柄复制
#         self.pre_dist_list[:] = self.dist_list[:]
#         # self.pre_distribution_std = self.distribution_std 

#         return instant_reward


    

#     def dist2_mat(self, x):

#         x_loc = np.reshape(x[:, 0:2], (self.n_agents,2,1))
#         a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
#         np.fill_diagonal(a_net, np.Inf)
#         return a_net


#     def get_connectivity(self, x):

#         # a_net = self.adj_distance_matrix = self.dist2_mat(x)
#         # import pdb;pdb.set_trace()
#         if self.degree == 0: #or self.n_agents < self.n_features:
#             a_net = self.dist2_mat(x)
#             a_net = (a_net < self.comm_radius2).astype(float)
#         else:
#             neigh = NearestNeighbors(n_neighbors=self.degree)
#             neigh.fit(x[:,:2])
#             # neigh.fit(x[:,2:4])
#             self.adj_distance_matrix,_ = neigh.kneighbors()
#             a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())
#             # import pdb;pdb.set_trace()

#         # if self.mean_pooling:
#         #     # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
#         #     n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents,1)) # TODO or axis=0? Is the mean in the correct direction?
#         #     n_neighbors[n_neighbors == 0] = 1
#         #     a_net = a_net / n_neighbors 

#         return a_net


#     def render(self, mode='human'):
#         """
#         Render the environment with agents as points in 2D space
#         """

#         if self.fig is None:
#             plt.ion()
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#             line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')  # Returns a tuple of line objects, thus the comma
#             line2, = ax.plot(self.center[0], self.center[1], 'rx')
#             circle, = ax.plot([], [], '-', color='r', lw=1)
#             #ax.plot([0], [0], 'kx')
#             # ax.plot(self.start_xpoints, self.start_ypoints, 'kx')
            
#             print(self.x[:,2],self.x[:,3])
#             # cir1 = Circle(xy = (self.center[0],self.center[1]), radius=self.radius, alpha=0.2)
#             # ax.add_patch(cir1)
#             plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
#             plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
#             a = gca()
#             # a.set_xticklabels(a.get_xticks(), font)
#             # a.set_yticklabels(a.get_yticks(), font)

#             plt.title('GNN Controller')
#             self.fig = fig
#             self.line1 = line1
#             self.line2 = line2
#             self.circle = circle
            
#         theta = np.linspace(0, 2*np.pi, 100) # 参数t的数组
#         x_out = [self.center[0]+self.radius*np.cos(theta[i]) for i in range(len(theta))]
#         y_out = [self.center[1]+self.radius*np.sin(theta[i]) for i in range(len(theta))]
        
#         self.line1.set_xdata(self.x[:, 0])
#         self.line1.set_ydata(self.x[:, 1])

#         self.line2.set_xdata(self.center[0])
#         self.line2.set_ydata(self.center[1])
#         self.circle.set_xdata(x_out)
#         self.circle.set_ydata(y_out)
        
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()

#     def close(self):
#         pass
 














# class decentralized_FormationFlyingEnv(gym.Env):

#     def __init__(self):

#         config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
#         config = configparser.ConfigParser()
#         config.read(config_file)
#         config = config['flock']

#         self.dynamic = True # if the agents are moving or not
#         self.mean_pooling = False # normalize the adjacency matrix by the number of neighbors or not
#         #self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
#         self.degree = 2
#         # number states per agent
#         self.nx_system = 4
#         # numer of features per agent
#         self.n_features = 2

#         # number of actions per agent
#         self.nu = 2 

#         # problem parameters from file
#         self.n_agents = 3
#         self.comm_radius = float(config['comm_radius'])
#         self.comm_radius2 = self.comm_radius * self.comm_radius
#         self.dt = float(config['system_dt'])
#         self.v_max = float(config['max_vel_init'])
#         self.v_bias = self.v_max 
#         self.r_max = float(config['max_rad_init'])
#         self.std_dev = float(config['std_dev']) * self.dt

#         # intitialize state matrices
#         self.x = np.zeros((self.n_agents, self.nx_system))
#         self.u = np.zeros((self.n_agents, self.nu))
#         self.a_net = np.zeros((self.n_agents, self.n_agents))
#         self.center=np.array([0.0,0.0,0.0,0.0])
#         self.counter = 0    
#         self.obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))

#         # TODO : what should the action space be? is [-1,1] OK?
#         self.max_accel = 1 
#         self.gain = 1.0 # TODO - adjust if necessary - may help the NN performance
#         self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(self.n_agents,self.nu),dtype=np.float32)

#         self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents,(1+self.degree)*self.n_features),
#                                             dtype=np.float32)

#         #render init     
#         self.fig = None
#         self.line1 = None
#         self.line2 = None

#         self.seed()

#         self.radius = 3#1
#         self.adj_distance_matrix = np.zeros((self.n_agents,2))
#         self.ag_threshold = 2*self.radius*math.sin(math.pi/self.n_agents)

#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def reset(self):
#         # import pdb;pdb.set_trace()
#         self.x = np.zeros((self.n_agents, self.nx_system)) #keep this to track position
#         self.feats = np.zeros((self.n_agents,self.n_features)) #this is the feature we return to the agent
#         degree = 0
#         min_dist = 0
#         min_dist_thresh = 0.01  # 0.25 

#         self.counter = 0
#         self.agent_xg = []
#         self.agent_yg = []

#         self.agent_startx = []
#         self.agent_starty = []
        
#         self.center_speed_x = float(np.random.uniform(-0.7,0.7,1))
#         self.center_speed_y = float(np.random.uniform(-0.7,0.7,1))
        
#         #scheme : 
#         #space all agents in a frontier that looks like -> .'.
#         #this means everyone has x goal position separated by two units. 
#         #and y goal position varies  
#         # let the number of agents be odd. for kosher reasons. 

#         #hack, say n agents = 3. Then placer_x is = -2 
#         team_center = [np.random.uniform(-4,4),np.random.uniform(-4,4)]
#         for i in range(self.n_agents):
#             self.x[i,0] = np.random.uniform(-3,3)+team_center[0]
#             self.x[i,1] = np.random.uniform(-3,3)+team_center[1]
        
#         self.center= np.array([np.random.uniform(-3,3),np.random.uniform(-3,3),0.0,0.0])#np.array([0.0,0.0,0.0,0.0])#  
        
#         # self.placer_x = (self.n_agents/2)*2*(-1)        
        
#         #init reward:
#         self.dist_list = np.zeros(self.n_agents)
#         self.pre_dist_list = np.zeros(self.n_agents)
#         self.dist_agn2tar_list = np.zeros(self.n_agents)
#         self.energy_list  = np.zeros(self.n_agents)
#         self.done_dict = np.zeros(self.n_agents)
#         self.on_circle_time = np.zeros(self.n_agents)

#         return self._get_obs()


#     def _get_obs(self):
        
#         # self.feats[:,0] = self.x[:,0]-self.center[0]# - self.x[:,2]
#         # self.feats[:,1] = self.x[:,1]-self.center[1] #- self.x[:,3]
          
#         # self.feats[:,2] = self.x[:,2]-self.center[2]
#         # self.feats[:,3] = self.x[:,3]-self.center[3]
#         # for i in range(self.n_features):
#         #     self.feats[:,i] =  self.x[:,i]-self.center[i]
#         if self.dynamic:
#             adj_matrix = self.get_connectivity(self.x)
#         else:
#             adj_matrix = self.a_net
#         # import pdb;pdb.set_trace()
#         for i in range(self.n_agents):
#             temp = list(self.center[:self.n_features] - self.x[i,:self.n_features])

#             a = list(np.argwhere(adj_matrix[i]==1.0).squeeze())
            
#             list0 = list((self.x[a[0],:self.n_features] - self.x[i,:self.n_features]).squeeze())
#             list1 = list((self.x[a[1],:self.n_features] - self.x[i,:self.n_features]).squeeze())
            
#             dist0 = np.linalg.norm((self.x[a[0],:self.n_features] - self.x[i,:self.n_features]).squeeze())
#             dist1 = np.linalg.norm((self.x[a[1],:self.n_features] - self.x[i,:self.n_features]).squeeze())
            
#             if dist0<=dist1: 
#                 temp = temp + list0
#                 temp = temp + list1
#             else:
#                 temp = temp + list1
#                 temp = temp + list0

#             self.obs_dict[i] = np.array(temp)
        
#         #return (state_values, state_network)
#         return self.obs_dict/self.radius#self.feats.reshape(1,self.n_agents*self.n_features)


#     def step(self, action_dict):
        
        
#         info_dict = {}
#         self.counter = self.counter+1
#         # temp = np.reshape(action,(self.n_agents, self.nu))
#         # for i in range(self.n_agents):
#         #     agent_id = 'agent-' + str(i) 
#         #     self.u[i] = action_dict[agent_id]
#         self.u = action_dict
#         #     self.u[i,:] = temp[i,:]+self.center[-2:] 
#         # import pdb;pdb.set_trace()

#         self.center[2] = self.center_speed_x*math.sin(-self.counter/30) 
#         self.center[3] = self.center_speed_y*math.cos(-self.counter/30)
#         # self.center[:2] = self.center[:2] + self.center[2:]*0.1
#         self.center[0] = self.center[0] + self.center[2]*0.1
#         self.center[1] = self.center[1] + self.center[3]*0.1

#         print("self.center",self.center)
#         self.x[:, 2:] = self.u[:, :2]
#         self.x[:, :2] = self.x[:, :2] + self.x[:, 2:]*0.1

#         print("self.x",self.x)

#         # done = False 
#         # if self.counter > 4000000 :
#         #     done = True 
#         # if np.all(diffs_x) <0.2 and np.all(diffs_y)<0.2:
#             # done = True 

#         return self._get_obs(), self.instant_reward(), self.done_dict, info_dict

   

#     def instant_reward(self):  # sum of differences in velocities
#         reward_dict = np.zeros(self.n_agents)

#         angle_list = []
#         cost_dist = np.zeros(self.n_agents)
#         cost_dist_agn2tar = np.zeros(self.n_agents)
#         cost_energy = np.zeros(self.n_agents)
#         cost_distribution = np.zeros(self.n_agents)
#         cost_on_circle = np.zeros(self.n_agents)

#         for i in range(self.n_agents):
#             # temp_vector = self.x[i,:2] - self.center[:2]
#             self.dist_list[i] = math.fabs(np.linalg.norm(self.obs_dict[i,:2])-self.radius)#
#             if self.counter==1: 
#                 cost_dist[i] = 0
#             else: 
#                 # import pdb;pdb.set_trace()
#                 cost_dist[i] = (self.dist_list[i]-self.pre_dist_list[i])*15
            
#             if self.dist_list[i]<0.1:
#                 self.on_circle_time[i] -= 1
#             else:
#                 self.on_circle_time[i] = 0
            
#             cost_on_circle[i] = self.on_circle_time[i]*0.1


#             temp_dist_1 = np.linalg.norm(self.obs_dict[i,:2])
#             if temp_dist_1>self.radius/3:
#                 cost_dist_agn2tar[i] = 0.07/(temp_dist_1+0.01) 
#             else: 
#                 # import pdb;pdb.set_trace() 
#                 cost_dist_agn2tar[i] = 50 
#                 self.done_dict[i] = 1
#             #[1/(dist+0.01) if dist<self.ag_threshold else 0 for dist in self.adj_distance_matrix.reshape(self.n_agents*self.degree)]
#             cost_energy[i] = np.linalg.norm(self.u[i])*0.35

#             temp_dist_list = []

#             for temp_dist_2 in list(self.adj_distance_matrix[i]):
#                 if temp_dist_2>self.ag_threshold:
#                     k = 0
#                 else:
#                     k = 2.5
#                 if temp_dist_2>0.5:
#                     temp_dist_list.append(k/(temp_dist_2*temp_dist_2+0.01)) 
#                 else:
#                     # import pdb;pdb.set_trace() 
#                     temp_dist_list.append(100)
#                     self.done_dict[i] = 1
#                     break

#             cost_distribution[i] = np.sum(temp_dist_list)/self.degree  
            
#                         # if (temp_vector[0])==0:
#             #     temp = -90#math.pi/2 
#             # else:
#             #     temp = math.atan(temp_vector[1]/temp_vector[0])/math.pi*180
#             # angle =  temp if temp_vector[1]<0 else temp+180
#             # angle_list.append(angle)

#             reward_dict[i] = -(cost_dist[i] + cost_dist_agn2tar[i]+ cost_distribution[i] + cost_energy[i]+cost_on_circle[i])



#         #cost_accumulate_dist = 0#np.sum(self.dist_list)*0.3/self.n_agents
#         print("adj_distance_matrix",self.adj_distance_matrix)


#         # angle_list.sort()
#         # next_angle_list = angle_list[1:] + [angle_list[0]+360]
#         # self.distribution_std = np.std(np.array(next_angle_list)-np.array(angle_list))*0.1
#         # if  self.counter>1:#np.sum(self.dist_list) <(0.25*self.n_agents) and 
#         #     cost_distribution = (self.pre_distribution_std-self.distribution_std)*10
#         # else:


#         print("diff_list",self.dist_list)
#         print("dist_agn2tar_list",self.dist_agn2tar_list)
#         print("reward_dict",reward_dict)
#         print("cost_dist",-cost_dist)
#         print("cost_distribution",-cost_distribution)
#         print("cost_dist_agn2tar",-cost_dist_agn2tar)
#         print("cost_energy_list",-cost_energy)
#         print("cost_energy_list",-cost_on_circle)
#         ##更新距离，注意要进行内容层面的深拷贝，不要直接把句柄复制
#         self.pre_dist_list[:] = self.dist_list[:]
#         # self.pre_distribution_std = self.distribution_std 

#         return reward_dict


    

#     def dist2_mat(self, x):

#         x_loc = np.reshape(x[:, 0:2], (self.n_agents,2,1))
#         a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
#         np.fill_diagonal(a_net, np.Inf)
#         return a_net


#     def get_connectivity(self, x):

#         # a_net = self.adj_distance_matrix = self.dist2_mat(x)
#         # import pdb;pdb.set_trace()
#         if self.degree == 0: #or self.n_agents < self.n_features:
#             a_net = self.dist2_mat(x)
#             a_net = (a_net < self.comm_radius2).astype(float)
#         else:
#             neigh = NearestNeighbors(n_neighbors=self.degree)
#             neigh.fit(x[:,:2])
#             # neigh.fit(x[:,2:4])
#             self.adj_distance_matrix,_ = neigh.kneighbors()
#             a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())
#             # import pdb;pdb.set_trace()

#         # if self.mean_pooling:
#         #     # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
#         #     n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents,1)) # TODO or axis=0? Is the mean in the correct direction?
#         #     n_neighbors[n_neighbors == 0] = 1
#         #     a_net = a_net / n_neighbors 

#         return a_net


#     def render(self, mode='human'):
#         """
#         Render the environment with agents as points in 2D space
#         """

#         if self.fig is None:
#             plt.ion()
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#             line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')  # Returns a tuple of line objects, thus the comma
#             line2, = ax.plot(self.center[0], self.center[1], 'rx')
#             circle, = ax.plot([], [], '-', color='r', lw=1)
#             #ax.plot([0], [0], 'kx')
#             # ax.plot(self.start_xpoints, self.start_ypoints, 'kx')
            
#             print(self.x[:,2],self.x[:,3])
#             # cir1 = Circle(xy = (self.center[0],self.center[1]), radius=self.radius, alpha=0.2)
#             # ax.add_patch(cir1)
#             plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
#             plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
#             a = gca()
#             # a.set_xticklabels(a.get_xticks(), font)
#             # a.set_yticklabels(a.get_yticks(), font)

#             plt.title('GNN Controller')
#             self.fig = fig
#             self.line1 = line1
#             self.line2 = line2
#             self.circle = circle
            
#         theta = np.linspace(0, 2*np.pi, 100) # 参数t的数组
#         x_out = [self.center[0]+self.radius*np.cos(theta[i]) for i in range(len(theta))]
#         y_out = [self.center[1]+self.radius*np.sin(theta[i]) for i in range(len(theta))]
        
#         self.line1.set_xdata(self.x[:, 0])
#         self.line1.set_ydata(self.x[:, 1])

#         self.line2.set_xdata(self.center[0])
#         self.line2.set_ydata(self.center[1])
#         self.circle.set_xdata(x_out)
#         self.circle.set_ydata(y_out)
        
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()

#     def close(self):
#         pass
 