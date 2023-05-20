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


class FormationFlyingEnv2(gym.Env):

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
        self.n_agents = 5
        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max 
        self.r_max = float(config['max_rad_init'])
        self.std_dev = float(config['std_dev']) * self.dt

        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))
        
        self.a_net = np.zeros((self.n_agents, self.n_agents))

        # TODO : what should the action space be? is [-1,1] OK?
        self.max_accel = 1 
        self.gain = 1.0 # TODO - adjust if necessary - may help the NN performance
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(self.n_agents,2),dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        self.center=[0,0]        
        self.fig = None
        self.line1 = None
        self.line2 = None
        self.counter = 0 
        self.seed()
        self.center = [0,0]
        self.radius = 1
        self.adj_distance_matrix = np.zeros((self.n_agents,2))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        self.counter = self.counter+1 
        self.u = np.reshape(action,(self.n_agents, self.nu))
        # self.center = [2*math.sin(self.counter/50),2*math.cos(self.counter/50)]
        self.center = [0,0]
        # import pdb;pdb.set_trace()
        # self.x[0,2] = self.center[0]
        # self.x[0,3] = self.center[1]-2

        # self.x[1,2] = self.center[0]-2
        # self.x[1,3] = self.center[1]-1
        
        # self.x[2,2] = self.center[0]-1
        # self.x[2,3] = self.center[1]+2
        
        # self.x[3,2] = self.center[0]+1
        # self.x[3,3] = self.center[1]+2
        
        # self.x[4,2] = self.center[0]+2
        # self.x[4,3] = self.center[1]-1
        
        # self.goal_xpoints = self.x[:,2]
        # self.goal_ypoints = self.x[:,3]

        self.x[:, 0] = self.x[:, 0] + self.u[:, 0]*0.1
        # update  y position 
        self.x[:, 1] = self.x[:, 1] + self.u[:, 1]*0.1

        done = False 

        # diffs_x = np.abs(self.x[:,0] - self.goal_xpoints)
        # diffs_y = np.abs(self.x[:,1] - self.goal_ypoints)
        
        if self.counter > 4000000 :
            done = True 
        # if np.all(diffs_x) <0.2 and np.all(diffs_y)<0.2:
            # done = True 

        return self._get_obs(), self.instant_cost(), done, {}

    # def _step(self, action, time):
             
    #     self.u = np.reshape(action,(self.n_agents, self.nu))
    #     center = [10+4*math.sin(time/50),10+4*math.cos(time/50)]
    #     self.x[0,2] = center[0]
    #     self.x[0,3] = center[1]-2

    #     self.x[1,2] = center[0]-2
    #     self.x[1,3] = center[1]-1
        
    #     self.x[2,2] = center[0]-1
    #     self.x[2,3] = center[1]+2
        
    #     self.x[3,2] = center[0]+1
    #     self.x[3,3] = center[1]+2
        
    #     self.x[4,2] = center[0]+2
    #     self.x[4,3] = center[1]-1
        
    #     self.x[:, 0] = self.x[:, 0] + self.u[:, 0]*0.1
    #     # update  y position 
    #     self.x[:, 1] = self.x[:, 1] + self.u[:, 1]*0.1

    #     done = False 

    #     diffs_x = np.abs(self.x[:,0] - self.goal_xpoints)
    #     diffs_y = np.abs(self.x[:,1] - self.goal_ypoints)
        
    #     if self.counter > 400 :
    #         done = True 
    #     if np.all(diffs_x) <0.2 and np.all(diffs_y)<0.2:
    #         done = True 

    #     return self._get_obs(), self.instant_cost(), done, {}

    # def instant_cost(self):  # sum of differences in velocities
    #     diff_list = []
    #     angle_list = []
    #     for i in range(self.n_agents):
    #         temp_vector = np.array([self.x[i,0],self.x[i,1]])-np.array(self.center)
    #         diff_list.append(math.fabs(np.linalg.norm(temp_vector)-self.radius))
            
    #         if (temp_vector[0])==0:
    #             temp = -90#math.pi/2 
    #         else:
    #             temp = math.atan(temp_vector[1]/temp_vector[0])/math.pi*180
    #         angle =  temp if temp_vector[1]<0 else temp+180
    #         angle_list.append(angle)        
    #     angle_list.sort()
    #     next_angle_list = angle_list[1:] + [angle_list[0]+360]
    #     diff_list = np.array(diff_list)*5
    #     cost_dist2goal = np.sum(diff_list)
    #     cost_distribution = np.std(np.array(next_angle_list)-np.array(angle_list))*3
    #     cost_dist_var = np.std(diff_list)**2*3
    #     # robot_xs = self.x[:,0]
    #     # robot_ys = self.x[:,1]

    #     # robot_goalxs = self.x[:,2]
    #     # robot_goalys = self.x[:,3]


    #     # diff = ((robot_xs - robot_goalxs)**2 + (robot_ys - robot_goalys)**2)**0.5
    #     # return -np.sum(diff)
    #     print("diff_list",diff_list)
    #     print("angle_list",angle_list)
    #     print("next_angle_list",next_angle_list)
    #     instant_reward = -(cost_dist2goal + cost_distribution+ cost_dist_var)
    #     print("cost",instant_reward,-cost_dist2goal,-cost_distribution,-cost_dist_var)
        
    #     return instant_reward

    def instant_cost(self):  # sum of differences in velocities
        diff_list = []
        angle_list = []
        dist_agn2tar_list = []
        for i in range(self.n_agents):
            temp_vector = np.array([self.x[i,0],self.x[i,1]])-np.array(self.center)
            diff_list.append(math.fabs(np.linalg.norm(temp_vector)-self.radius))
            dist_agn2tar_list.append(np.linalg.norm(self.feats[i]))
        #     if (temp_vector[0])==0:
        #         temp = -90#math.pi/2 
        #     else:
        #         temp = math.atan(temp_vector[1]/temp_vector[0])/math.pi*180
        #     angle =  temp if temp_vector[1]<0 else temp+180
        #     angle_list.append(angle)        
        # angle_list.sort()
        # next_angle_list = angle_list[1:] + [angle_list[0]+360]
        diff_list = np.array(diff_list)
        cost_dist2goal = np.sum(diff_list)/self.n_agents
        cost_distribution = np.sum(1./(self.adj_distance_matrix+0.01))/self.n_agents
        cost_dist_var = np.std(diff_list)**2
        dist_agn2tar_list = [1/(dist+0.01) if dist<self.radius else 0 for dist in dist_agn2tar_list]
        cost_dist_agn2tar = np.sum(dist_agn2tar_list)/self.n_agents
        # robot_xs = self.x[:,0]
        # robot_ys = self.x[:,1]

        # robot_goalxs = self.x[:,2]
        # robot_goalys = self.x[:,3]


        # diff = ((robot_xs - robot_goalxs)**2 + (robot_ys - robot_goalys)**2)**0.5
        # return -np.sum(diff)
        print("diff_list",diff_list)
        print("dist_agn2tar_list",dist_agn2tar_list)
        instant_reward = -(cost_dist2goal + cost_distribution+ cost_dist_var+cost_dist_agn2tar)
        print("cost",instant_reward,-cost_dist2goal,-cost_distribution,-cost_dist_var,-cost_dist_agn2tar)
        
        return instant_reward


    def reset(self):
        # import pdb;pdb.set_trace()
        x = np.zeros((self.n_agents, self.n_features+2)) #keep this to track position
        self.feats = np.zeros((self.n_agents,self.n_features)) #this is the feature we return to the agent
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.01  # 0.25 

        self.counter = 0
        self.agent_xg = []
        self.agent_yg = []

        self.agent_startx = []
        self.agent_starty = []
        #scheme : 
        #space all agents in a frontier that looks like -> .'.
        #this means everyone has x goal position separated by two units. 
        #and y goal position varies  
        # let the number of agents be odd. for kosher reasons. 

        #hack, say n agents = 3. Then placer_x is = -2 
        
        self.placer_x = (self.n_agents/2)*2*(-1)

        ########declare goals#######################
        for i in range(0,self.n_agents):
            self.agent_xg.append(self.placer_x) 
            self.placer_x += 2
        
        for i in range(0,self.n_agents):
            self.agent_yg.append(np.random.uniform(2,3)) 
        
        #reset self.placer_x
        self.placer_x = (self.n_agents/2)*2*(-1)        

        ##########declare start positions############
        for i in range(0,self.n_agents):
            self.agent_startx.append(self.placer_x) 
            self.placer_x += 0.3
        
        for i in range(0,self.n_agents):
            self.agent_starty.append(np.random.uniform(-0.5,0.5))


        xpoints = np.array(self.agent_startx)
        ypoints = np.array(self.agent_starty)

        self.start_xpoints = xpoints
        self.start_ypoints = ypoints


        self.goal_xpoints = np.array(self.agent_xg)
        self.goal_ypoints = np.array(self.agent_yg)
        
        x[:,0] = xpoints #- self.goal_xpoints
        x[:,1] = ypoints #- self.goal_ypoints

        
        x[:,2] = self.goal_xpoints
        x[:,3] = self.goal_ypoints
        # compute distances between agents
        a_net = self.dist2_mat(x)

        # compute minimum distance between agents and degree of network to check if good initial configuration
        min_dist = np.sqrt(np.min(np.min(a_net)))
        a_net = a_net < self.comm_radius2
        degree = np.min(np.sum(a_net.astype(int), axis=1))

        self.x = x

        self.a_net = self.get_connectivity(self.x)

        
        return self._get_obs()

    def _get_obs(self):
        
        self.feats[:,0] = self.x[:,0]-self.center[0]# - self.x[:,2]
        self.feats[:,1] = self.x[:,1]-self.center[1] #- self.x[:,3]
          
        if self.dynamic:
            state_network = self.get_connectivity(self.x)
        else:
            state_network = self.a_net

        #return (state_values, state_network)
        return self.feats

    def dist2_mat(self, x):

        x_loc = np.reshape(x[:, 0:2], (self.n_agents,2,1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net


    def get_connectivity(self, x):


        if self.degree == 0:
            a_net = self.dist2_mat(x)
            a_net = (a_net < self.comm_radius2).astype(float)
        else:
            neigh = NearestNeighbors(n_neighbors=self.degree)
            neigh.fit(x[:,:2])
            # neigh.fit(x[:,2:4])
            self.adj_distance_matrix,_ = neigh.kneighbors()
            a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())
            # import pdb;pdb.set_trace()

        if self.mean_pooling:
            # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
            n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents,1)) # TODO or axis=0? Is the mean in the correct direction?
            n_neighbors[n_neighbors == 0] = 1
            a_net = a_net / n_neighbors 

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
 










#####################################
# class FormationFlyingEnv2(gym.Env):

#     def __init__(self):

#         config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
#         config = configparser.ConfigParser()
#         config.read(config_file)
#         config = config['flock']

#         self.dynamic = True # if the agents are moving or not
#         self.mean_pooling = False # normalize the adjacency matrix by the number of neighbors or not
#         #self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
#         self.degree = 1 
#         # number states per agent
#         self.nx_system = 4
#         # numer of features per agent
#         self.n_features = 2
#         # number of actions per agent
#         self.nu = 2 

#         # problem parameters from file
#         self.n_agents = 5
#         self.comm_radius = float(config['comm_radius'])
#         self.comm_radius2 = self.comm_radius * self.comm_radius
#         self.dt = float(config['system_dt'])
#         self.v_max = float(config['max_vel_init'])
#         self.v_bias = self.v_max 
#         self.r_max = float(config['max_rad_init'])
#         self.std_dev = float(config['std_dev']) * self.dt

#         # intitialize state matrices
#         self.x = np.zeros((self.n_agents, self.nx_system))
        
#         self.a_net = np.zeros((self.n_agents, self.n_agents))

#         # TODO : what should the action space be? is [-1,1] OK?
#         self.max_accel = 1 
#         self.gain = 1.0 # TODO - adjust if necessary - may help the NN performance
#         self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 *self.n_agents,),dtype=np.float32)

#         self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
#                                             dtype=np.float32)

        
#         self.fig = None
#         self.line1 = None
#         self.line2 = None
#         self.counter = 0 
#         self.seed()

#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]


#     def step(self, action):
#         self.counter = self.counter+1 
#         self.u = np.reshape(action,(self.n_agents, self.nu))
#         center = [2*math.sin(self.counter/50),2*math.cos(self.counter/50)]
#         center = [0,0]
#         # import pdb;pdb.set_trace()
#         self.x[0,2] = center[0]
#         self.x[0,3] = center[1]-2

#         self.x[1,2] = center[0]-2
#         self.x[1,3] = center[1]-1
        
#         self.x[2,2] = center[0]-1
#         self.x[2,3] = center[1]+2
        
#         self.x[3,2] = center[0]+1
#         self.x[3,3] = center[1]+2
        
#         self.x[4,2] = center[0]+2
#         self.x[4,3] = center[1]-1
        
#         self.goal_xpoints = self.x[:,2]
#         self.goal_ypoints = self.x[:,3]

#         self.x[:, 0] = self.x[:, 0] + self.u[:, 0]*0.1
#         # update  y position 
#         self.x[:, 1] = self.x[:, 1] + self.u[:, 1]*0.1

#         done = False 

#         diffs_x = np.abs(self.x[:,0] - self.goal_xpoints)
#         diffs_y = np.abs(self.x[:,1] - self.goal_ypoints)
        
#         if self.counter > 400000 :
#             done = True 
#         if np.all(diffs_x) <0.2 and np.all(diffs_y)<0.2:
#             done = True 

#         return self._get_obs(), self.instant_cost(), done, {}

#     # def _step(self, action, time):
             
#     #     self.u = np.reshape(action,(self.n_agents, self.nu))
#     #     center = [10+4*math.sin(time/50),10+4*math.cos(time/50)]
#     #     self.x[0,2] = center[0]
#     #     self.x[0,3] = center[1]-2

#     #     self.x[1,2] = center[0]-2
#     #     self.x[1,3] = center[1]-1
        
#     #     self.x[2,2] = center[0]-1
#     #     self.x[2,3] = center[1]+2
        
#     #     self.x[3,2] = center[0]+1
#     #     self.x[3,3] = center[1]+2
        
#     #     self.x[4,2] = center[0]+2
#     #     self.x[4,3] = center[1]-1
        
#     #     self.x[:, 0] = self.x[:, 0] + self.u[:, 0]*0.1
#     #     # update  y position 
#     #     self.x[:, 1] = self.x[:, 1] + self.u[:, 1]*0.1

#     #     done = False 

#     #     diffs_x = np.abs(self.x[:,0] - self.goal_xpoints)
#     #     diffs_y = np.abs(self.x[:,1] - self.goal_ypoints)
        
#     #     if self.counter > 400 :
#     #         done = True 
#     #     if np.all(diffs_x) <0.2 and np.all(diffs_y)<0.2:
#     #         done = True 

#     #     return self._get_obs(), self.instant_cost(), done, {}

#     def instant_cost(self):  # sum of differences in velocities
       
#         robot_xs = self.x[:,0]
#         robot_ys = self.x[:,1]

#         robot_goalxs = self.x[:,2]
#         robot_goalys = self.x[:,3]
        

#         diff = ((robot_xs - robot_goalxs)**2 + (robot_ys - robot_goalys)**2)**0.5
#         print(-np.sum(diff))
#         return -np.sum(diff)


#     def reset(self):
#         # import pdb;pdb.set_trace()
#         x = np.zeros((self.n_agents, self.n_features+2)) #keep this to track position
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
        
#         self.placer_x = (self.n_agents/2)*2*(-1)

#         ########declare goals#######################
#         for i in range(0,self.n_agents):
#             self.agent_xg.append(self.placer_x) 
#             self.placer_x += 2
        
#         for i in range(0,self.n_agents):
#             self.agent_yg.append(np.random.uniform(2,3)) 
        
#         #reset self.placer_x
#         self.placer_x = (self.n_agents/2)*2*(-1)        

#         ##########declare start positions############
#         for i in range(0,self.n_agents):
#             self.agent_startx.append(self.placer_x) 
#             self.placer_x += 2
        
#         for i in range(0,self.n_agents):
#             self.agent_starty.append(np.random.uniform(-1,0))


#         xpoints = np.array(self.agent_startx)
#         ypoints = np.array(self.agent_starty)

#         self.start_xpoints = xpoints
#         self.start_ypoints = ypoints


#         self.goal_xpoints = np.array(self.agent_xg)
#         self.goal_ypoints = np.array(self.agent_yg)
        
#         x[:,0] = xpoints - self.goal_xpoints
#         x[:,1] = ypoints - self.goal_ypoints

        
#         x[:,2] = self.goal_xpoints
#         x[:,3] = self.goal_ypoints
#         # compute distances between agents
#         a_net = self.dist2_mat(x)

#         # compute minimum distance between agents and degree of network to check if good initial configuration
#         min_dist = np.sqrt(np.min(np.min(a_net)))
#         a_net = a_net < self.comm_radius2
#         degree = np.min(np.sum(a_net.astype(int), axis=1))

#         self.x = x

#         self.a_net = self.get_connectivity(self.x)

        
#         return self._get_obs()

#     def _get_obs(self):
        
#         self.feats[:,0] = self.x[:,0] - self.x[:,2]
#         self.feats[:,1] = self.x[:,1] - self.x[:,3]
          
#         if self.dynamic:
#             state_network = self.get_connectivity(self.x)
#         else:
#             state_network = self.a_net

#         #return (state_values, state_network)
#         return self.feats

#     def dist2_mat(self, x):

#         x_loc = np.reshape(x[:, 0:2], (self.n_agents,2,1))
#         a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
#         np.fill_diagonal(a_net, np.Inf)
#         return a_net


#     def get_connectivity(self, x):


#         if self.degree == 0:
#             a_net = self.dist2_mat(x)
#             a_net = (a_net < self.comm_radius2).astype(float)
#         else:
#             neigh = NearestNeighbors(n_neighbors=self.degree)
#             neigh.fit(x[:,2:4])
#             a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())

#         if self.mean_pooling:
#             # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
#             n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents,1)) # TODO or axis=0? Is the mean in the correct direction?
#             n_neighbors[n_neighbors == 0] = 1
#             a_net = a_net / n_neighbors 

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
#             line2, = ax.plot(self.x[:,2], self.x[:,3], 'rx')
#             #ax.plot([0], [0], 'kx')
#             # ax.plot(self.start_xpoints, self.start_ypoints, 'kx')
            
#             print(self.x[:,2],self.x[:,3])

#             plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
#             plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
#             a = gca()
#             # a.set_xticklabels(a.get_xticks(), font)
#             # a.set_yticklabels(a.get_yticks(), font)
#             plt.title('GNN Controller')
#             self.fig = fig
#             self.line1 = line1
#             self.line2 = line2
            

#         self.line1.set_xdata(self.x[:, 0])
#         self.line1.set_ydata(self.x[:, 1])

#         self.line2.set_xdata(self.x[:, 2])
#         self.line2.set_ydata(self.x[:, 3])

        
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()

#     def close(self):
#         pass