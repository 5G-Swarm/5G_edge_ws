import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
print(matplotlib.get_backend())
from matplotlib.patches import Ellipse, Circle
from os import path
import gym
from gym import spaces, error, utils
from gym.utils import seeding

class Real_env(gym.Env): 
    def __init__(self,n_agents = 10):
        self.counter = 0
        self.degree = 4#5
        # number states per agent
        self.nx_system = 13
        # numer of features per agent
        self.n_features = 4#2

        # number of actions per agent
        self.nu = 2 

        # problem parameters from file
        self.init_n_agents = n_agents
        self.n_agents = 0
        self.pre_n_agents = 0
        self.id_list = []
        self.in_vision_bbx_dict = {}
        self.n_in_vision = 0
        self.in_vision_id_list = []

        # self.pre_in_vision_bbx_dict = {}
        # self.n_in_vision = 0
        self.pre_in_vision_id_list = []

        self.target_center = np.zeros(6) 
        self.estimate_target_center = np.zeros(6) 
        self.state_drones_dict = {}
        self.obstacle_index_list = []


        # self.state_drones = np.zeros((self.n_agents, self.nx_system))
        # self.feat_xy = np.zeros((self.n_agents, self.n_features))
        # self.feat_z = np.zeros((self.n_agents, 2))
        # self.feat_wz = np.zeros((self.n_agents, 1))

        # self.obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))
        # self.pre_obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))
        self.ACCEL_BOUND = 1
        self.VELXY_BOUND = 2#1
        self.radius = 5.0
        self.target_err_threshold = 4#error tolerrance to balloon estimation 
        # self.adj_distance_matrix = np.zeros((self.n_agents,2))
        # self.ag_threshold = 2*self.radius*math.sin(math.pi/self.n_agents) 
        self.action_space = spaces.Box(low=-self.ACCEL_BOUND, high=self.ACCEL_BOUND, shape=(self.n_agents,self.nu),dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents,(1+self.degree)*self.n_features*2),
                                            dtype=np.float32)
        ######render########
        self.r_max = 70#float(config['max_rad_init'])
        self.fig = None
        self.line1 = None
        self.line2 = None
        self.line3 = None
        self.line4 = None
        self.line5 = None
        self.pg_boundary = np.zeros((4,2))

    def get_connectivity(self, x):
        # import pdb;pdb.set_trace()
        if self.degree == 0: #or self.n_agents < self.n_features:
            a_net = self.dist2_mat(x)
            a_net = (a_net < self.comm_radius2).astype(float)
        else:
            neigh = NearestNeighbors(n_neighbors=self.degree)
            neigh.fit(x[:,:2])
            self.adj_distance_matrix,self.neighbor_id_matrix = neigh.kneighbors()
            a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())
            
        # if self.mean_pooling:
        #     # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        #     n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents,1)) # TODO or axis=0? Is the mean in the correct direction?
        #     n_neighbors[n_neighbors == 0] = 1
        #     a_net = a_net / n_neighbors 

        return a_net


    def _get_obs_xy(self):
        self.counter = self.counter+1 
        self.n_agents = len(self.id_list)#len(self.state_drones_dict)
        self.target_err = np.linalg.norm(self.target_center[:2] - self.estimate_target_center[:2])
        if self.target_err <= self.target_err_threshold:
            self.feat_target_xy = np.hstack((self.estimate_target_center[0:2],self.estimate_target_center[3:5]))
        else:
            self.feat_target_xy = np.hstack((self.target_center[0:2],self.target_center[3:5]))
        # self.feat_target_z = np.hstack((self.target_center[2],self.target_center[-1]))

        self.state_drones = np.zeros((self.n_agents, self.nx_system))
        for i in range(self.n_agents):
            robot_id = self.id_list[i]
            self.state_drones[i] = self.state_drones_dict[robot_id]

        # print("state_drones",self.state_drones[:,:5])
        # for i in range(self.n_agents):
            # obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
        self.feat_xy = np.hstack((self.state_drones[:,:2],self.state_drones[:,3:5]))#只取出
        # self.feat_z = self.feat_target_z[0] - self.state_drones[:,2]#np.hstack((self.state_drones[:,2],self.state_drones[:,5]))
        # self.feat_wz = self.state_drones[:,-1]#
        # import pdb;pdb.set_trace()

        adj_matrix = self.get_connectivity(self.state_drones)
        obs_list = []
        for i in range(self.n_agents):
            temp = list((self.feat_target_xy[:self.n_features] - self.feat_xy[i,:self.n_features]))#/self.radius*5

            for j in list(self.neighbor_id_matrix[i]):
                temp_list = list((self.feat_xy[j,:self.n_features] - self.feat_xy[i,:self.n_features]).squeeze())#/3
                temp = temp + temp_list 
            #  = np.array(temp)
            obs_list.append(temp)
        print("adj_distance_matrix",self.adj_distance_matrix)

        self.obs_dict = np.array(obs_list) 
        
        if self.pre_n_agents:
            if self.pre_n_agents<self.n_agents:
                self.pre_obs_dict = np.vstack((self.pre_obs_dict,self.obs_dict[self.pre_n_agents:self.n_agents,:]))
        else:
            self.pre_obs_dict = self.obs_dict.copy()
            
        self.pre_n_agents = self.n_agents 
        return_obs_xy = np.hstack((self.obs_dict,self.pre_obs_dict))
        self.pre_obs_dict = self.obs_dict.copy()
        # print("return_obs_xy",return_obs_xy)
        return return_obs_xy/self.radius*5#/self.radius*5#

    def _preprocessAction(self,action_dict,kp):
        action_dict = np.clip(action_dict,-self.ACCEL_BOUND,self.ACCEL_BOUND) 
        vel_output = self.feat_xy[:,2:4] + action_dict*0.1*kp
        vel_output = np.clip(vel_output,-self.VELXY_BOUND,self.VELXY_BOUND)
        return vel_output


    def _get_obs_z(self):
        self.feat_target_z = np.hstack((self.target_center[2],self.target_center[-1]))
        self.feat_z = self.feat_target_z[0] - self.state_drones[:,2]
        return self.feat_z

    def _get_obs_wz(self):
        self.n_in_vision = len(self.in_vision_id_list) 
        self.in_vision_bbx = np.zeros((self.n_in_vision,2))
        for i in range(self.n_in_vision):
            robot_id = self.in_vision_id_list[i]
            self.in_vision_bbx[i] = self.in_vision_bbx_dict[robot_id]
        
        self.feat_wz = np.ones((self.n_in_vision))*320.0-self.in_vision_bbx[:,0]
        self.feat_local_vz = np.ones((self.n_in_vision))*240.0 - self.in_vision_bbx[:,1] 
        return self.feat_wz, self.feat_local_vz
    # def _get_obs_wz(self):
    #     self.n_in_vision = len(self.in_vision_id_list) 
    #     self.in_vision_bbx = np.zeros((self.n_in_vision,2))
    #     for i in range(self.n_in_vision):
    #         robot_id = self.in_vision_id_list[i]
    #         self.in_vision_bbx[i] = self.in_vision_bbx_dict[robot_id]
        
    #     self.feat_wz = np.ones((self.n_in_vision))*320.0-self.in_vision_bbx[:,0]
    #     return self.feat_wz
    
    def _get_reward(self):


        return 0

    def render(self, mode='human'):
        temp_list = [0,1,2,3,4,5,6,7,8,9,10]
        self.obstacle_state = np.zeros((len(self.obstacle_index_list),2))
        for i in self.obstacle_index_list:
            # robot_id = self.id_list.index(i)
            self.obstacle_state[self.obstacle_index_list.index(i)] = self.state_drones_dict[i][:2]
        
        # import pdb;pdb.set_trace()
        
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            global ax,annotator
            ax = fig.add_subplot(111)
            self.line1, = ax.plot(self.state_drones[:, 0], self.state_drones[:, 1], 'bo')  # Returns a tuple of line objects, thus the comma
            self.line2, = ax.plot(self.target_center[0], self.target_center[1], 'rx')
            self.line4, = ax.plot(self.estimate_target_center[0], self.estimate_target_center[1], 'gx')
            self.line3, = ax.plot(self.obstacle_state[:,0],self.obstacle_state[:,1], 'gs')
            self.line5, = ax.plot(self.pg_boundary[:,0],self.pg_boundary[:,1], 'bx')
            
            self.circle, = ax.plot([], [], '-', color='r', lw=1)
            self.circle_3m, = ax.plot([], [], '-', color='y', lw=1)
            #ax.plot([0], [0], 'kx')
            # ax.plot(self.start_xpoints, self.start_ypoints, 'kx')
            
            # print(self.x[:,2],self.x[:,3])
            # cir1 = Circle(xy = (self.center[0],self.center[1]), radius=self.radius, alpha=0.2)
            # ax.add_patch(cir1)
            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            # a.set_xticklabels(a.get_xticks(), font)
            # a.set_yticklabels(a.get_yticks(), font)
            annotator = []
            for i in range(self.n_agents):
                # import pdb;pdb.set_trace()
                annotator.append(ax.annotate(self.id_list[i],(self.state_drones[i,0],self.state_drones[i,1])))
            plt.title('MavenRL Simulator')
            self.fig = fig
        
        theta = np.linspace(0, 2*np.pi, 100) # 参数t的数组
        if self.target_err < self.target_err_threshold:
            x_out = [self.estimate_target_center[0]+self.radius*np.cos(theta[i]) for i in range(len(theta))]
            y_out = [self.estimate_target_center[1]+self.radius*np.sin(theta[i]) for i in range(len(theta))]
        else:
            x_out = [self.target_center[0]+self.radius*np.cos(theta[i]) for i in range(len(theta))]
            y_out = [self.target_center[1]+self.radius*np.sin(theta[i]) for i in range(len(theta))]
        
        x_out_3m = [self.target_center[0]+3*np.cos(theta[i]) for i in range(len(theta))]
        y_out_3m = [self.target_center[1]+3*np.sin(theta[i]) for i in range(len(theta))]

        self.line1.set_xdata(self.state_drones[:, 0])
        self.line1.set_ydata(self.state_drones[:, 1])

        for i in range(self.n_agents):
            if i < len(annotator):
                annotator[i].remove()
            annotator[i] = ax.annotate(self.id_list[i],(self.state_drones[i,0],self.state_drones[i,1]))
        self.line2.set_xdata(self.target_center[0])
        self.line2.set_ydata(self.target_center[1])

        self.line4.set_xdata(self.estimate_target_center[0])
        self.line4.set_ydata(self.estimate_target_center[1])

        self.line3.set_xdata(self.obstacle_state[:,0])
        self.line3.set_ydata(self.obstacle_state[:,1])
        
        self.line5.set_xdata(self.pg_boundary[:,0])
        self.line5.set_ydata(self.pg_boundary[:,1])

        self.circle.set_xdata(x_out)
        self.circle.set_ydata(y_out)

        self.circle_3m.set_xdata(x_out_3m)
        self.circle_3m.set_ydata(y_out_3m)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()