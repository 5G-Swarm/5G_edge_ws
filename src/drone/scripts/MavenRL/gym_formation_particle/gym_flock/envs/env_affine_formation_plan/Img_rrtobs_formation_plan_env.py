import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from matplotlib.pyplot import gca
from matplotlib.patches import Ellipse, Circle
from sklearn.neighbors import NearestNeighbors
import itertools 
import random
import pdb
import math
import cv2
import time 
###################rrt related########### 
# import sys
# sys.path.append("../../../..")
from rtree import index
import uuid
from gym_formation_particle.gym_flock.envs.env_affine_formation_plan.env_utils import SearchSpace



font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class Img_rrtobs_formationPlanEnv(gym.Env):

    def __init__(self,test_flag=True,image_flag=False):

        config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['flock']

        self.level = 0
        self.test_flag = test_flag
        self.image_flag = image_flag
        
        

        self.CBF_flag = False
        self.dynamic = True # if the agents are moving or not
        self.mean_pooling = False # normalize the adjacency matrix by the number of neighbors or not
        #self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
        self.degree = 1#3#5
        self.neighbor_solver = NearestNeighbors(n_neighbors=self.degree)
        # number states per agent
        self.nx_system = 6
        # numer of features per agent
        self.n_features = 4#2

        # number of actions per agent
        self.dim_action = 6

        # problem parameters from file
        self.max_agents = 4
        self.init_n_agents = 4
        self.n_agents = self.init_n_agents
        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius



        self.target = np.array([0.0,0.0,0.0,0.0])
        self.center = np.array([0.0,0.0,0.0,0.0])
        self.counter = 0   

        
        self.radius = 10#5
        self.adj_distance_matrix = np.zeros((self.n_agents,2))

        self.collision_threshold = 1.5#1.5#1.5#1.5#.5#0.8
        self.inter_agent_collision_threshold = 0.3

        
        ####误差阈值
        # self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3),0.5,0.5,0.5,2])#np.array([1-np.cos(np.pi/3),0.5,0.5,0.5,2])#np.array([np.cos(np.pi/3),0.5,0.5,0.5,2]) #np.array([np.cos(np.pi/3)-0.2,0.3,0.3,0.3,2]) 
        # self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3/2),0.3,0.3,0.3,1])#np.array([1-np.cos(np.pi/3/2),0.3,0.3,0.3,1])
        # self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3),0.3,0.3,0.3,1])
        # self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3/2),0.2,0.2,0.2,0.5])#np.array([1-np.cos(np.pi/3/2),0.3,0.3,0.3,1])
        
        
        self.collision_free_threshold = 5#10

        # TODO : what should the action space be? is [-1,1] OK?
        self.max_accel = 1 


        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(1,self.dim_action),dtype=np.float32)
        # self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(1,(1+self.degree)*self.n_features+8*1+4),#+16

        #                                     dtype=np.float32)#(1+self.degree)*self.n_features*2
        # ###任意初始模板
        # self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(1,8+4*2+(1+self.degree)*(self.n_features-2)),#+16
        #                             dtype=np.float32)#(1+self.degree)*self.n_features*2

        ###实际队形+任意初始模板
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(1,8*2+4+(1+self.degree)*(self.n_features-2)),#+16
                                    dtype=np.float32)#(1+self.degree)*self.n_features*2

        #render init     
        self.r_max = 40#50#float(config['max_rad_init'])
        self.fig = None
        self.line1 = None
        self.line2 = None
        self.line3 = None
 
        self.seed(1)

        
        
        ###initialize obstacle set:
        self.n_obstacle = 15#15#5#15
        self.obstacle_state = np.zeros((self.n_obstacle,self.n_features))

        ######Constant######
        self.VEL_BOUND = 2
        self.ACCEL_BOUND = 3
        self.T_step = 0.1
        self.horizon_N = 10#20
        self.lqr_kp = 1#10#10#10

        ##FOV
        self.FOV_radius = 10
        self.FOV_real_size = self.FOV_radius*2
        self.FOV_resolution = 0.1#100×100
        self.FOV_pixel_size = int(self.FOV_real_size/self.FOV_resolution) 
        
        ##limitations
        
        self.max_theta = np.inf
        self.min_theta = -np.inf
        
        self.max_shear_x = 2
        self.min_shear_x = -2
        self.max_scale_x = self.max_scale_y = np.log(3)#5最大为放大5倍
        self.min_scale_x = self.min_scale_y = -np.log(3)#最小为缩小5倍
        
        
        
        self.max_det_theta = math.pi/6
        self.max_det_shear_x = 0.5#0.2
        self.max_det_scale_x = np.log(1.5)#np.log(1.5)#np.log(1.1)
        self.max_det_scale_y = np.log(1.5)#np.log(1.5)#np.log(1.1)

        self.max_det_transition_x = 1.5#1.5
        self.max_det_transition_y = 1.5#1.5
        self.max_transition_x = np.inf
        self.min_transition_x = -np.inf
        self.max_transition_y = np.inf
        self.min_transition_y = -np.inf


        self.max_action = np.array([self.max_det_theta,self.max_det_shear_x,self.max_det_scale_x,self.max_det_scale_y,self.max_det_transition_x,self.max_det_transition_y])
        self.max_state = np.array([self.max_theta,self.max_shear_x,self.max_scale_x,self.max_scale_y,self.max_transition_x,self.max_transition_y])
        self.min_state = np.array([self.min_theta,self.min_shear_x,self.min_scale_x,self.min_scale_y,self.min_transition_x,self.min_transition_y])
        ######search_space###########
        self.X_dimensions = np.array([[self.min_theta, self.max_theta],[self.min_shear_x,self.max_shear_x],[self.min_scale_x,self.max_scale_x],[self.min_scale_y,self.max_scale_y], [self.min_transition_x, self.max_transition_x],[self.min_transition_y, self.max_transition_y]])  # dimensions of Search Space
        self.search_space_resolution = 0.05#0.1

    def rt_formation_array_calculation(self):
        self.rt_Rotation_matrix = np.array([[np.cos(self.rt_theta),np.sin(self.rt_theta)],[-np.sin(self.rt_theta),np.cos(self.rt_theta)]]) 
        self.rt_Shearing_matrix = np.array([[1,self.rt_shear_x],[0,1]])
        self.rt_Scaling_matrix = np.array([[np.exp(self.rt_scale_x),0],[0,np.exp(self.rt_scale_y)]])
        self.rt_A_matrix = self.rt_Rotation_matrix.dot(self.rt_Shearing_matrix.dot(self.rt_Scaling_matrix))
        self.rt_B_matrix = np.array([[self.rt_transition_x],[self.rt_transition_y]])
        return self.rt_A_matrix.dot(self.init_formation_array) + self.rt_B_matrix 


    @staticmethod
    def relative_rt_formation_array_calculation(current_formation_array,affine_param_array):
        relative_theta = affine_param_array[0]
        relative_shear_x = affine_param_array[1]
        relative_scale_x = affine_param_array[2]
        relative_scale_y = affine_param_array[3]
        relative_transition_x = affine_param_array[4]
        relative_transition_y = affine_param_array[5]
        # import pdb;pdb.set_trace()
        relative_Rotation_matrix = np.array([[np.cos(relative_theta),-np.sin(relative_theta)],[np.sin(relative_theta),np.cos(relative_theta)]]) 
        relative_Shearing_matrix = np.array([[1,relative_shear_x],[0,1]])
        relative_Scaling_matrix = np.array([[np.exp(relative_scale_x),0],[0,np.exp(relative_scale_y)]])
        relative_A_matrix = relative_Rotation_matrix.dot(relative_Shearing_matrix.dot(relative_Scaling_matrix))
        relative_B_matrix = np.array([[relative_transition_x],[relative_transition_y]])

        # self.rt_affine_matrix = np.hstack((relative_A_matrix,np.zeros((2,1)))) 
        # print(current_formation_array,relative_A_matrix,relative_B_matrix)
        return relative_A_matrix.dot(current_formation_array) + relative_B_matrix 

    def identity_template_generate(self,n_agents):
        angle_array = np.pi/2 - np.array(range(n_agents))*(2*np.pi)/n_agents
        points_array = np.vstack((np.cos(angle_array),np.sin(angle_array)))#*1.5
        # import pdb;pdb.set_trace()
        
        return points_array
    
    def random_template_generate(self):
        self.base_init_formation_coordinate = np.array([[0,0],[0,0.3],[0.3,0]]).T 
        self.base_init_formation_array = np.array([[1.0,1.0],[1.0,-1.0],[-1.0,-1.0],[-1.0,1.0]]).T #four vertices of the inital square foramtion
        self.pre_affine_array = np.array([0.3,0,np.log(3),np.log(3),0,0])
        self.pre_affine_array = np.array([np.random.uniform(-np.pi,np.pi),\
        np.random.uniform(-1,1),\
        np.random.uniform(-np.log(1.5),np.log(1.5)),\
        np.random.uniform(-np.log(1.5),np.log(1.5)),0,0])
        temp = np.hstack((self.base_init_formation_array,self.base_init_formation_coordinate))
        init_formation_all = self.relative_rt_formation_array_calculation(temp,self.pre_affine_array)
        return init_formation_all

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def generate_random_obstacles(self, X, start, end, n ,obs_extend):
        """
        Generates n random obstacles without disrupting world connectivity.
        It also respects start and end points so that they don't lie inside of an obstacle.
        """
        # Note: Current implementation only supports hyperrectangles.
        # i = 0
        obstacles = []
        obstacle_dimension = 2
        # p = index.Property()
        # p.dimension = 2#只在二维上产生障碍物self.dimensions
        # temp_obs = index.Index(interleaved=True, properties=p)
        # obs_extend = 0.5#3#3#0.7#0.5
        for j in range(1):
            i = 0
            counter = 0 
            temp_switch = random.choice([1,1,1,1,-1,-1,-1-1,0])
            while i < n:
                counter += 1
                if counter>n*2:#10
                    break
                obstacle_center = np.empty(obstacle_dimension, np.float)
                scollision = True
                fcollision = True
                edge_lengths = []
                for j in range(obstacle_dimension):
                    # None of the sides of a hyperrectangle can be higher than 0.1 of the total span
                    # in that particular X.dimensions
                    # if i<1:
                    max_edge_length = 1.5#1.5#1.5#1.5#+obs_extend#15#(X.dimension_lengths[4+j][1] - X.dimension_lengths[4+j][0]) / 10.0
                    min_edge_length = 0.5#1#+obs_extend#10#(X.dimension_lengths[4+j][1] - X.dimension_lengths[4+j][0]) / 100.0
                    # else:
                    #     max_edge_length = 1#1.5#1.5#1.5#+obs_extend#15#(X.dimension_lengths[4+j][1] - X.dimension_lengths[4+j][0]) / 10.0
                    #     min_edge_length = 0.5#1#+obs_extend#10#(X.dimension_lengths[4+j][1] - X.dimension_lengths[4+j][0]) / 100.0
    
                    edge_length = random.uniform(min_edge_length, max_edge_length)
                    edge_lengths.append(edge_length)
                    if start[-2+j] < 0:
                        obstacle_center[j] = random.uniform(-self.center2target-2,0+self.obs_surround_area)#-self.center2target-1,2#-self.center2target-3,3
                    else:
                        obstacle_center[j] = random.uniform(0-self.obs_surround_area,self.center2target+2)#-2,self.center2target+1##-3,self.center2target+3
                        # 
                    
                ###保证障碍物中心不在编队内部
                if self.fast_obstacle_collision_detect(self.rt_formation_array,np.expand_dims(obstacle_center,0)) or self.fast_obstacle_collision_detect(self.target_formation_array,np.expand_dims(obstacle_center,0)):
                    continue

                # Check if any part of the obstacle is inside of another obstacle.
                min_corner = np.empty(obstacle_dimension, np.float)
                max_corner = np.empty(obstacle_dimension, np.float)

                for j in range(obstacle_dimension):
                    min_corner[j] = obstacle_center[j] - edge_lengths[j]
                    max_corner[j] = obstacle_center[j] + edge_lengths[j]
                

                
                obstacle_real = np.append(min_corner, max_corner)

            
            
                obstacle = obstacle_real.copy()
                obstacle[:2] = obstacle_real[:2] - obs_extend#障碍物膨胀
                obstacle[-2:] = obstacle_real[-2:] + obs_extend

                # temp = np.zeros((5,2))
                # temp[0,:] = obstacle_center
                # temp[1,:] = np.array(obstacle[:2])
                # temp[2,:] = np.array([obstacle[0],obstacle[3]])
                # temp[3,:] = np.array(obstacle[-2:])
                # temp[4,:] = np.array([obstacle[2],obstacle[1]])
                
                # ###保证障碍物中心不在编队内部
                # if self.fast_obstacle_collision_detect(self.rt_formation_array,temp) or self.fast_obstacle_collision_detect(self.target_formation_array,temp):
                #     continue
                
                # # import pdb;pdb.set_trace()
                # Check newly generated obstacle intersects any former ones. Also respect start and end points
                if len(list(X.extend_obs.intersection(obstacle))) > 0 :
                    continue

                temp_id = uuid.uuid4()
                X.extend_obs.insert(temp_id, tuple(obstacle))

                print("obstacle1", i,counter)

                X.obs.insert(temp_id, tuple(obstacle_real))
                if X.formation_obstacle_free(start) and X.formation_obstacle_free(end):
                    obstacles.append(obstacle_real)
                else:
                    X.extend_obs.delete(temp_id, tuple(obstacle))
                    X.obs.delete(temp_id, tuple(obstacle_real))
                    continue
                i += 1
                print("obstacle2",i)
            
            if i==n:
                break 
            p1 = index.Property()
            p1.dimension = 2#只在二维上产生障碍物self.dimensions
            p2 = index.Property()
            p2.dimension = 2
            X.obs = index.Index(interleaved=True, properties=p1)
            X.extend_obs =  index.Index(interleaved=True, properties=p2)
            obstacles = []

        return obstacles

    def reset(self,level=0):
        # import pdb;pdb.set_trace()
        plt.cla()
        self.counter = 0
        self.n_agents = self.init_n_agents
        self.x = np.zeros((self.n_agents, self.nx_system)) #keep this to track position
        # team_center = [np.random.uniform(-0,0),np.random.uniform(-50,50)]
        # for i in range(self.n_agents):
        #     self.x[i,0] = np.random.uniform(-50,50)+team_center[0]
        #     self.x[i,1] = np.random.uniform(-0,0)+team_center[1]
        # self.target = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),0,0])
        # self.center = np.array([np.random.uniform(-30,30)+self.target[0],np.random.uniform(-30,30)+self.target[1],0,0]) 
        
        

        
        self.target = np.array([np.random.uniform(0,0),np.random.uniform(0,0),0,0])

        self.level = level
        if self.level ==0:#by default
            self.obstacle_density = 1.85
            self.center2target = 5
            self.n_obstacle = 0
            self.obs_extend = 1.25
            self.obs_surround_area = 2
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3/2),0.3,0.3,0.3,1])
        elif self.level ==1:
            self.obstacle_density = 2.5
            self.center2target = 7
            self.n_obstacle = 2
            self.obs_extend = 1.0#0.5
            self.obs_surround_area = 2
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3),0.5,0.5,0.5,2])
        elif self.level ==2:
            # self.obstacle_density = 2.5
            # self.center2target = 10
            # self.n_obstacle = 4
            # self.obs_extend = 1#0.75
            # self.obs_surround_area = 2

            self.obstacle_density = 4
            self.center2target = 10
            self.n_obstacle = 5#5
            self.obs_extend = 1
            self.obs_surround_area = 4
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3/2),0.3,0.3,0.3,1])

        elif self.level ==3:
            self.obstacle_density = 4
            self.center2target = 10
            self.n_obstacle = 5
            self.obs_extend = 1
            self.obs_surround_area = 4
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3/2),0.3,0.3,0.3,1])

        elif self.level ==7:
            self.obstacle_density = 4
            self.center2target = 20
            self.n_obstacle = 15
            self.obs_extend = 1
            self.obs_surround_area = 5
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3),0.5,0.5,0.5,2])

        elif self.level ==8:
            self.obstacle_density = 4
            self.center2target = 30
            self.n_obstacle = 20
            self.obs_extend = 1
            self.obs_surround_area = 5
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3),0.5,0.5,0.5,2])

        self.r_max = self.center2target*2.5
        # if random.choice([1,0]):
        #     self.center = np.array([np.random.uniform(self.center2target-2,self.center2target)*random.choice([1,-1])+self.target[0],np.random.uniform(-self.center2target,self.center2target) +self.target[1],0,0]) 
        # else:
        #     self.center = np.array([np.random.uniform(-self.center2target,self.center2target) +self.target[0],np.random.uniform(self.center2target-2,self.center2target)*random.choice([1,-1])+self.target[1],0,0])
        if random.choice([1,0]):
            self.center = np.array([np.random.uniform(self.center2target-1.5,self.center2target)*random.choice([1,-1])+self.target[0],np.random.uniform(-self.center2target,self.center2target)*random.choice([1,-1]) +self.target[1],0,0]) 
        else:
            self.center = np.array([np.random.uniform(-self.center2target,self.center2target)*random.choice([1,-1]) +self.target[0],np.random.uniform(self.center2target-1.5,self.center2target)*random.choice([1,-1])+self.target[1],0,0])

        
        # self.center = np.array([-30,20,0,0])



        


        # self.u = np.zeros((self.n_agents, self.dim_action))
        self.obs_dict = np.zeros(int(self.observation_space.shape[1]))
        self.copy_obs_dict = np.zeros(int(self.observation_space.shape[1]))
        self.pre_obs_dict = np.zeros(int(self.observation_space.shape[1]))

        #done:
        self.done_dict = 0#np.zeros(self.n_agents)     



        
        self.n_agents = 4#np.random.randint(3,8)
        # self.template_encode = np.zeros(4)
        # self.template_encode[self.n_agents-3] = 1
        # self.init_formation_array = self.identity_template_generate(self.n_agents)*2#np.hstack((self.identity_template_generate(self.n_agents),np.zeros((2,1))))
        while True:
            random_theta = np.array([np.random.uniform(-np.pi,-np.pi/2),np.random.uniform(-np.pi/2,0),np.random.uniform(0,np.pi/2),np.random.uniform(np.pi/2,np.pi)])
            self.init_formation_array = np.vstack((np.cos(random_theta),np.sin(random_theta)))*2#1.5
            temp_random_rotation = np.random.uniform(-np.pi,np.pi)
            temp_random_rotation_matrix = np.array([[np.cos(temp_random_rotation),np.sin(temp_random_rotation)],[-np.sin(temp_random_rotation),np.cos(temp_random_rotation)]])
            self.init_formation_array = temp_random_rotation_matrix.dot(self.init_formation_array) 
            if self.inter_agent_collision_free(self.init_formation_array, self.inter_agent_collision_threshold):
                break

        self.init_formation_coordinate = np.array([[0,0],[0,0.3],[0.3,0]]).T
        self.init_formation_all = np.hstack((self.init_formation_array,self.init_formation_coordinate))
        # self.init_formation_image = self.formation2image(self.init_formation_array.T,np.zeros(2))
        # self.init_formation_image_transform = self.init_formation_image.T 

        
        
        # self.init_state_dict = np.array([0,0,0,0,self.center[0],self.center[1]])
        while True:#

            while True:
                self.target_affine_param = np.array([np.random.uniform(-np.pi,np.pi),\
                np.random.uniform(-2,2),\
                np.random.uniform(-np.log(2),np.log(2)),\
                np.random.uniform(-np.log(2),np.log(2)),self.target[0],self.target[1]])
                temp_target = self.relative_rt_formation_array_calculation(self.init_formation_all,self.target_affine_param)

                if self.inter_agent_collision_free(temp_target[:,:-3], self.inter_agent_collision_threshold):
                    break

            self.target_formation_coordinate = temp_target[:,-3:]
            self.target_formation_array = temp_target[:,:-3]



            # i = 0
            # while(i<40):

            #     # if i>0:
            #     #     self.target = np.array([np.random.uniform(-10,10),np.random.uniform(-10,10),0,0])     
            #     collison_flag, temp_dist_array = self.fast_obstacle_collision_detect(self.target_formation_array,self.obstacle_state[:,:2])
            #     if collison_flag==False and temp_dist_array.min()>5:
            #         break
            #     # import pdb;pdb.set_trace() 
            #     self.obstacle_state[temp_dist_array.argmin(),0] +=  np.random.uniform(-7,-3)*random.choice([1,-1])
            #     self.obstacle_state[temp_dist_array.argmin(),1] +=  np.random.uniform(-7,-3)*random.choice([1,-1])
        
            #     i += 1

            while True:
                self.state_dict = np.array([np.random.uniform(-np.pi,np.pi),\
                np.random.uniform(-2,2),\
                np.random.uniform(-np.log(2),np.log(2)),\
                np.random.uniform(-np.log(2),np.log(2)),\
                self.center[0],self.center[1]]) 

                #init same as target 
                self.state_dict[:4] = self.target_affine_param[:4]

                temp = self.relative_rt_formation_array_calculation(self.init_formation_all,self.state_dict)

                if self.inter_agent_collision_free(temp[:,:-3], self.inter_agent_collision_threshold):
                    break
            
            self.rt_formation_coordinate = temp[:,-3:]
            self.rt_formation_array = temp[:,:-3]
            self.sub_goal_formation_array = self.rt_formation_array.copy() 


            #####生成障碍物#####
            self.rtree = SearchSpace(self.init_formation_array,self.X_dimensions,self.search_space_resolution)
            #len(self.obstacle_state)

            temp_obstacle_list = self.generate_random_obstacles(self.rtree,self.state_dict,self.target_affine_param,self.n_obstacle,self.obs_extend)
            if len(temp_obstacle_list)>=self.n_obstacle:
                break
        
        temp_list = []
        for obstacle in temp_obstacle_list:
            temp = np.zeros((4,2))
            temp[0,:] = np.array(obstacle[:2])
            temp[1,:] = np.array([obstacle[0],obstacle[3]])
            temp[2,:] = np.array(obstacle[-2:])
            temp[3,:] = np.array([obstacle[2],obstacle[1]])
            temp_list.append(temp)
        self.rrt_obstacle = np.array(temp_list) 



        #init reward:
        self.dist_list = np.zeros(self.n_agents)
        self.pre_dist_list = np.zeros(self.n_agents)
        self.dist_agn2tar_list = np.zeros(self.n_agents)
        self.energy_list  = np.zeros(self.n_agents)
        self.on_circle_time = np.zeros(self.n_agents)
        
        
        self.pre_rt_formation_array = self.rt_formation_array.copy()
        self.pre_agents_vel_array = np.zeros_like(self.pre_rt_formation_array)
        self.dist2target = np.linalg.norm(self.target[:2]-self.center[:2])
        self.pre_dist2target = self.dist2target
        self.pre_dist_affine_to_target,_ = self.dist_affine_param()
        self.pre_action_dict = np.zeros(self.dim_action)
        self.pre_normlized_action_dict = np.zeros(self.dim_action)
        self.pre_reward_collision = 0

        
        # import pdb;pdb.set_trace()

        return self.get_obs()
    
    def rlto_reset(self,init_affine_param,init_total_traj):
        self.counter = 0
        self.done_dict = 0
        self.state_dict = init_affine_param
        self.center[:2] = init_affine_param[-2:]
        self.rt_formation_array = init_total_traj

        self.pre_rt_formation_array = self.rt_formation_array.copy()
        self.pre_agents_vel_array = np.zeros_like(self.pre_rt_formation_array)
        self.dist2target = np.linalg.norm(self.target[:2]-self.center[:2])
        self.pre_dist2target = self.dist2target
        self.pre_dist_affine_to_target,_ = self.dist_affine_param()
        self.pre_reward_collision = 0
        return self.get_obs()
    
    def rlto_step(self,normalized_action_dict,next_affine_param,next_total_traj):
        self.counter = self.counter+1
        self.info_dict = {}
        self.normlized_action_dict = normalized_action_dict#action_dict[:-1]
        self.action_dict = self.preprocess(normalized_action_dict)
        
        pre_state_dict = self.state_dict.copy() 
        self.state_dict = next_affine_param.copy() 

        temp_pre = self.postprocess(pre_state_dict,self.min_state,self.max_state)
        temp_now = self.postprocess(self.state_dict,self.min_state,self.max_state)
        if np.linalg.norm(temp_pre-pre_state_dict)>1e-6 or np.linalg.norm(temp_now-self.state_dict)>1e-6 or np.abs(normalized_action_dict).max()>(1+1e-6):#状态或者动作超出限制范围了则不能放入buffer中
            self.info_dict["valid"] = False 
        else:
            self.info_dict["valid"] = True

        self.center[:2] = next_affine_param[-2:]
        self.rt_formation_array = next_total_traj.copy() 
        

        return self.get_obs(), self.instant_reward(), self.done_dict, self.info_dict#


    def affine_param_identification(self,current_formation_array):
        # psudo_inverse = self.init_formation_array.T .dot(np.linalg.inv(self.init_formation_array.T.dot(self.init_formation_array)) )
        temp = np.ones((3,self.init_formation_array.shape[1])) 
        temp[:2,:] = self.init_formation_array[:,:]
        A_matrix = current_formation_array.dot(np.linalg.pinv(temp))
        a11 = A_matrix[0,0]
        a12 = A_matrix[0,1]
        a21 = A_matrix[1,0]
        a22 = A_matrix[1,1]
        b_x = A_matrix[0,2]
        b_y = A_matrix[1,2]
        
        s_x = np.linalg.norm([a11,a21])
        theta = np.arctan2(a21,a11)
        m_sy = a12*np.cos(theta)+a22*np.sin(theta)
        if np.sin(theta) != 0:
            s_y = (m_sy*np.cos(theta)-a12)/np.sin(theta)
        else:
            s_y = (a22 - m_sy*np.sin(theta))/np.cos(theta)
        
        m = m_sy/s_y

        return np.array([theta, m, np.log(s_x), np.log(s_y), b_x, b_y])         






    def fast_obstacle_collision_detect(self,formation_array,points):

        temp_coordinate = np.zeros((points.shape[0],2))
        distance_array = np.zeros(points.shape[0])
        ouput_flag = False
        for i in range(points.shape[0]):
            temp_vector_list = formation_array.T - points[i,:]
            # import pdb;pdb.set_trace()
            first_closet_id = np.linalg.norm(temp_vector_list,axis=1).argsort()[0]
            left_closet_id =  first_closet_id -1
            if left_closet_id < 0:
                left_closet_id = self.n_agents - 1
            right_closet_id =  first_closet_id+1
            if right_closet_id > self.n_agents - 1:
                right_closet_id = 0

            vector_1_to_left = temp_vector_list[first_closet_id,:]-temp_vector_list[left_closet_id,:]
            vector_1_to_right = temp_vector_list[first_closet_id,:]-temp_vector_list[right_closet_id,:]
            vector_1_to_point = temp_vector_list[first_closet_id,:]

            if vector_1_to_left.dot(vector_1_to_point.T)<0:
                left_distance = np.linalg.norm(vector_1_to_point)
            else:
                left_distance = np.linalg.norm(vector_1_to_point - vector_1_to_left.dot(vector_1_to_point.T)/np.linalg.norm(vector_1_to_left))
            if vector_1_to_right.dot(vector_1_to_point.T)<0:
                right_distance = np.linalg.norm(vector_1_to_point)
            else:
                right_distance = np.linalg.norm(vector_1_to_point - vector_1_to_right.dot(vector_1_to_point.T)/np.linalg.norm(vector_1_to_right))

            distance_array[i] = min(left_distance,right_distance)
            temp_A = np.vstack((vector_1_to_left,vector_1_to_right)).T
            temp_coordinate[i,:] = np.linalg.inv(temp_A).dot(vector_1_to_point)
            if temp_coordinate[i,:].min()>=0:
                ouput_flag = True
                
        return ouput_flag#,distance_array



    def distance2formation(self,points):
        distance_array = np.zeros(points.shape[0])
        among_polygon_array = np.zeros(points.shape[0])
        self.temp_coordinate = np.zeros((points.shape[0],2))
        self.cloest_vertices_id = np.zeros((points.shape[0],3))
        for i in range(points.shape[0]):
            temp_vector_list = self.rt_formation_array.T - points[i,:]
            # import pdb;pdb.set_trace()
            first_closet_id = np.linalg.norm(temp_vector_list,axis=1).argsort()[0]
            left_closet_id =  first_closet_id -1
            if left_closet_id < 0:
                left_closet_id = self.n_agents - 1
            right_closet_id =  first_closet_id+1
            if right_closet_id > self.n_agents - 1:
                right_closet_id = 0

            vector_1_to_left = temp_vector_list[first_closet_id,:]-temp_vector_list[left_closet_id,:]
            vector_1_to_right = temp_vector_list[first_closet_id,:]-temp_vector_list[right_closet_id,:]
            vector_1_to_point = temp_vector_list[first_closet_id,:]
            if vector_1_to_left.dot(vector_1_to_point.T)<0:
                left_distance = np.linalg.norm(vector_1_to_point)
            else:
                left_distance = np.linalg.norm(vector_1_to_point - vector_1_to_left.dot(vector_1_to_point.T)/np.linalg.norm(vector_1_to_left))
            if vector_1_to_right.dot(vector_1_to_point.T)<0:
                right_distance = np.linalg.norm(vector_1_to_point)
            else:
                right_distance = np.linalg.norm(vector_1_to_point - vector_1_to_right.dot(vector_1_to_point.T)/np.linalg.norm(vector_1_to_right))
            # import pdb;pdb.set_trace()
            distance_array[i] = min(left_distance,right_distance)
            temp_A = np.vstack((vector_1_to_left,vector_1_to_right)).T
            self.temp_coordinate[i,:] = np.linalg.inv(temp_A).dot(vector_1_to_point)
            self.cloest_vertices_id[i,:] = np.array([left_closet_id,first_closet_id,right_closet_id])
            among_polygon_array[i] = 0 if self.temp_coordinate[i,:].min()<0 else 1

        
        return distance_array,among_polygon_array

    def collision_detect(self,points):
        '''
        通过点障碍物与多边形各定点间夹角的最大值计算与多边形是否碰撞，返回碰撞ID和各障碍物到多边形的远近程度
        
        '''
        angle_array = np.zeros((points.shape[0],self.n_agents)) 
        for i in range(points.shape[0]):
            for j in range(self.n_agents):
                temp_vector = self.rt_formation_array[:,j] - points[i,:]
                if (temp_vector[0])==0:
                    temp = -90#math.pi/2 
                else:
                    temp = math.atan(temp_vector[1]/temp_vector[0])/math.pi*180
                angle_array[i,j] =  temp if temp_vector[1]<0 else temp+180
        angle_array.sort(axis=1)
        # import pdb;pdb.set_trace()
        next_angle_array = np.hstack((angle_array[:,1:],(angle_array[:,0]+360).reshape((-1,1))))
        intersection_angle_array = next_angle_array - angle_array
        absmax_intersection_angle_array = np.abs(intersection_angle_array).max(axis=1)
        collison_ids = np.where(absmax_intersection_angle_array<=180)[0]
        print(collison_ids,intersection_angle_array)
        return collison_ids,absmax_intersection_angle_array
        # self.distribution_std = np.std(np.array(next_angle_list)-np.array(angle_list))*0.1
 
    def inter_agent_collision_free(self, formation_array, inter_agent_collision_threshold):
        '''
        判断编队内部各节点间的相互碰撞情况,通过求最近邻距离矩阵中的最小值和碰撞阈值()相对比        
        '''
        self.neighbor_solver.fit(formation_array.T)
        self.adj_distance_matrix,self.neighbor_id_matrix = self.neighbor_solver.kneighbors()
        print("gaojinglei@@@###",self.adj_distance_matrix)
        return self.adj_distance_matrix.min() > inter_agent_collision_threshold





    def obsimg_render(self,center_point):

        fov_min_corner = center_point - self.FOV_radius
        fov_max_corner = center_point + self.FOV_radius
        
        fov_bbx = np.append(fov_min_corner, fov_max_corner)
        fov_obs_list = self.rtree.obs.intersection(fov_bbx, objects=True)


        output_image = np.zeros((self.FOV_pixel_size,self.FOV_pixel_size))
        image_origin = center_point - np.array([self.FOV_radius,self.FOV_radius])

        
        for obs_bbx in fov_obs_list:
            obs_min_corner = np.array(obs_bbx.bbox[:2])
            obs_max_corner = np.array(obs_bbx.bbox[2:])
            
            pix_obs_min_corner = ((obs_min_corner - image_origin)/self.FOV_resolution)
            pix_obs_max_corner = ((obs_max_corner - image_origin)/self.FOV_resolution)
            pix_obs_min_corner = np.maximum(pix_obs_min_corner,np.zeros(2)).astype(int)
            pix_obs_max_corner = np.minimum(pix_obs_max_corner,np.array([self.FOV_pixel_size,self.FOV_pixel_size])).astype(int)

            kernel = np.ones(tuple(pix_obs_max_corner-pix_obs_min_corner))
            output_image[pix_obs_min_corner[0] : pix_obs_max_corner[0], pix_obs_min_corner[1] : pix_obs_max_corner[1]] = kernel

        return output_image

    def points2image(self,state_array,center_point):
        radius = int(self.collision_threshold/self.FOV_resolution/2)

        def generate_circle_mask(img_height,img_width,radius,center_x,center_y):
            y,x=np.ogrid[0:img_height,0:img_width]
            # circle mask
            mask = (x-center_x)**2+(y-center_y)**2<=radius**2
            return mask
        
        
        # kernel = np.ones((radius*2+1,radius*2+1))
        # time1 = time.time()
        # kernel = np.ones((radius*2+1,radius*2+1))
        kernel = generate_circle_mask(radius*2+1,radius*2+1,radius,radius,radius)
        # time2 = time.time()
        output_image = np.zeros((self.FOV_pixel_size+radius*2,self.FOV_pixel_size+radius*2))
        image_origin = center_point - np.array([self.FOV_radius,self.FOV_radius])
        state_array_pixel = (state_array - image_origin)/self.FOV_resolution
        # print("time",time2-time1)
        # import pdb;pdb.set_trace()

        
        for point in state_array_pixel:
            # output_image[int(point[1]),int(point[0])] = 1
            output_image[radius+int(point[1])-radius:radius+int(point[1])+radius+1,radius+int(point[0])-radius:radius+int(point[0])+radius+1] = kernel
            


        return output_image[radius:radius+self.FOV_pixel_size,radius:radius+self.FOV_pixel_size]

    def formation2image(self,formation_array,center_point):
        output_image = np.zeros((self.FOV_pixel_size,self.FOV_pixel_size))
        image_origin = center_point - np.array([self.FOV_radius,self.FOV_radius])
        formation_array_pixel = (formation_array - image_origin)/self.FOV_resolution
        
        # time1 = time.time()
        # for i in range(self.FOV_pixel_size):
        #     for j in range(self.FOV_pixel_size):
        #         point = np.array([[i,j]])
        #         flag,_ = self.fast_obstacle_collision_detect(formation_array_pixel.T,point)
        #         if flag:
        #             output_image[j,i] = 1 
        # time2 = time.time()
        # output_image2 = np.zeros((self.FOV_pixel_size,self.FOV_pixel_size))
        output_image = cv2.fillPoly(output_image, [formation_array_pixel.astype(np.int32)], 1)
        # time3 = time.time()
        # print("time:",time2-time1,time3-time2)

        # import pdb;pdb.set_trace()
        # plt.figure()
        # plt.title("output_image1")
        # plt.imshow(output_image)
        # plt.figure()
        # plt.title("output_image2")
        # plt.imshow(output_image2)
        # plt.show()

        return output_image

    def get_obs(self):
        if self.image_flag:
            return self._get_image_obs()
        else:
            return None
            # return self._get_obs()


    def _get_obs(self):
        self.pre_obs_dict[:] = self.obs_dict[:]
        # self.total_state = self.obstacle_state#np.vstack((self.x[:,:self.n_features],self.obstacle_state))
        self.obstacle_distance_array,self.obstacle_collison_array =  self.distance2formation(self.obstacle_state[:,:2])
        self.dangerous_obstacle_ids = self.obstacle_distance_array.argsort()[:self.degree]
        # print("find bug","cloest_id",self.dangerous_obstacle_ids,"vertices id:",self.cloest_vertices_id[self.dangerous_obstacle_ids[0]],"coordinate:",self.temp_coordinate[self.dangerous_obstacle_ids[0]])
        

        # self.obstacle_collison_ids,self.obstacle2formation = self.collision_detect(self.obstacle_state[:,:2])
       
        # input_obstacle_ids = self.obstacle2formation.argsort()[:self.degree]
        # import pdb;pdb.set_trace()
        self.closet_obstacle_state = self.obstacle_state[self.dangerous_obstacle_ids]
        self.closet_obstacle_distance = self.obstacle_distance_array[self.dangerous_obstacle_ids]
        print("$$obstacle_distance_array",self.closet_obstacle_distance)
        temp_formation_array = self.rt_formation_array.T - self.center[:2]
        # temp_formation_array = 
        temp = self.closet_obstacle_state - self.center
        temp_target = self.target - self.center
        # self.obs_dict = np.hstack((self.state_dict[:4]-self.init_state_dict[:4],temp_target,temp.reshape((-1))))
        # self.obs_dict = np.hstack((temp_target,temp.reshape((-1))))#1-np.cos(self.target_affine_param[0]-self.state_dict[0])
        diff_theta = (self.target_affine_param[0]-self.state_dict[0])%(np.pi*2)
        if diff_theta > np.pi:
            diff_theta -= np.pi*2
        elif diff_theta < -np.pi:
            diff_theta += np.pi*2
        elif np.abs(diff_theta) == np.pi:
            diff_theta = self.action_dict[0]/np.abs(self.action_dict[0])*np.pi

        ##把实时队形配置和初始模板直接放进去
        
        self.obs_dict = np.hstack((diff_theta/np.pi,(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2]),temp_target[:-2]/6,temp_formation_array.reshape((-1))/6,self.init_formation_array.reshape((-1))/6,temp[:,:-2].reshape((-1))/6))

        # self.obs_dict = np.hstack((diff_theta,self.target_affine_param[1:-2]-self.state_dict[1:-2],self.init_formation_array.reshape((-1)),self.state_dict[:-2],temp[:,:-2].reshape((-1)),temp_target[:-2]))
        # self.obs_dict = np.hstack((diff_theta/np.pi,(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2]),self.init_formation_array.reshape((-1)),self.state_dict[:-2],temp[:,:-2].reshape((-1)),temp_target[:-2]))
        
        #这种方式要硬编码所有agent的实时位置，
        # self.obs_dict = np.hstack((temp_formation_array.reshape((-1))/4.5,diff_theta/np.pi,(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2]),temp_target/20,temp.reshape((-1))/20))
        # self.obs_dict = np.hstack((temp_formation_array.reshape((-1)),diff_theta,(self.target_affine_param[1:-2]-self.state_dict[1:-2]),temp_target/20,temp.reshape((-1))/20))
        

        # self.obs_dict = np.hstack((temp_formation_array.reshape((-1))/6,self.init_formation_array.reshape((-1)),diff_theta/np.pi,(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2]),temp_target/20,temp.reshape((-1))/20))
        # self.obs_dict = np.hstack((temp_formation_array.reshape((-1)),self.state_dict[:-2],diff_theta/np.pi*3,(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2])*3,temp_target,temp.reshape((-1))))
        # self.obs_dict = np.hstack((temp_formation_array.reshape((-1)),self.state_dict[:-2],diff_theta,(self.target_affine_param[1:-2]-self.state_dict[1:-2]),temp_target,temp.reshape((-1))))
        # self.obs_dict = np.hstack((temp_formation_array.reshape((-1)),self.target_affine_param[:-2],self.state_dict[:-2],temp_target,temp.reshape((-1))))
        
        print("self.obs_dict:",self.obs_dict)
        # for id in list(input_obstacle_ids):
        #     temp = list((self.obstacle_state[id] - self.center))#

        #     for j in list(self.neighbor_id_matrix[i]):
        #         temp_list = list((self.total_state[j,] - self.x[i,:self.n_features]).squeeze())#/3.6
        #         temp = temp + temp_list 

        #     self.obs_dict[i] = np.array(temp)
        # self.copy_obs_dict = self.obs_dict.copy() 
        # self.obs_dict = self.add_noise_to_obs(self.obs_dict,0.3,0.3)#0.03
        #return (state_values, state_network)
        # import pdb;pdb.set_trace()
        # self._get_image_obs()

        return self.obs_dict#np.hstack((self.obs_dict,self.pre_obs_dict))#/self.radius*5#self.feats.reshape(1,self.n_agents*self.n_features)


    

    def _get_image_obs(self):
        self.pre_obs_dict[:] = self.obs_dict[:]
         # self.obstacle_distance_array,self.obstacle_collison_array =  self.distance2formation(self.obstacle_state[:,:2])

        # time0 = time.time()

        # time1 = time.time()


        self.obstacle_image = self.obsimg_render(self.center[:2])



        # time3 = time.time()
        
        # time3 = time.time()
        # print("time:",time1-time0,time2-time1,time3-time2,time9-time1)
        # import pdb;pdb.set_trace()

        # plt.figure()
        # plt.title("obstacle_image")
        # plt.imshow(self.obstacle_image)
        # plt.figure()
        # plt.title("formation_image")
        # plt.imshow(self.formation_image_transform)
        # plt.figure()
        # plt.title("init_formation_image")
        # plt.imshow(self.ss_init_formation_image_transform)

        # plt.show()
        temp_formation_array = self.rt_formation_array.T - self.center[:2]
        temp_target = self.target - self.center
        diff_theta = (self.target_affine_param[0]-self.state_dict[0])#%(np.pi*2)
        # if diff_theta > np.pi:
        #     diff_theta -= np.pi*2
        # elif diff_theta < -np.pi:
        #     diff_theta += np.pi*2
        # elif np.abs(diff_theta) == np.pi:
        #     diff_theta = self.action_dict[0]/np.abs(self.action_dict[0])*np.pi

        print("diff_theta",diff_theta)
        # ##把实时队形配置和初始模板的vector放进去
        obs_vector = np.hstack((temp_formation_array.reshape((-1))/6,self.init_formation_array.reshape((-1))/6,np.sin(diff_theta/2),np.cos(diff_theta/2),(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2]),temp_target[:-2]/6))
        # obs_vector = np.hstack((temp_formation_array.reshape((-1))/6,self.init_formation_array.reshape((-1))/6,diff_theta/np.pi,(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2]),temp_target[:-2]/6))
        # obs_vector = np.hstack((self.state_dict[:4],diff_theta/np.pi,(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2]),temp_target[:-2]/6))
        # obs_vector = np.hstack((diff_theta/np.pi,(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2]),temp_target[:-2]/6))
        obs_images = np.expand_dims(self.obstacle_image,0)#np.concatenate((np.expand_dims(self.obstacle_image,0),np.expand_dims(self.formation_image_transform,0),np.expand_dims(self.formation_image_transform,0)),axis=0) 
        # self.obs_dict = np.hstack((diff_theta/np.pi,(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2]),temp_target[:-2]/6,temp_formation_array.reshape((-1))/6,self.init_formation_array.reshape((-1))/6,temp[:,:-2].reshape((-1))/6))

        return obs_vector,obs_images


    def add_noise_to_obs(self,obs_dict,pos_noise_scale,vel_noise_scale):
        noise_pos = np.random.normal(loc=0.0, scale=pos_noise_scale, size=(obs_dict.shape[0],int(obs_dict.shape[1]/2)))
        noise_vel = np.random.normal(loc=0.0, scale=vel_noise_scale, size=(obs_dict.shape[0],int(obs_dict.shape[1]/2)))
        temp =  np.hstack((noise_pos[:,:2],noise_vel[:,:2],noise_pos[:,2:4],noise_vel[:,2:4],noise_pos[:,4:6],noise_vel[:,4:6],noise_pos[:,6:8],noise_vel[:,6:8],noise_pos[:,8:10],noise_vel[:,8:10],noise_pos[:,10:12],noise_vel[:,10:12]))
        # for i in range(obs_dict.shape[1]/2):
        #     temp = np.hstack((,))
        noisy_obs_dict = obs_dict + temp
        return noisy_obs_dict


    def preprocess(self,action_dict):
        '''
        to rescale action to the real world scene
        '''
        action_dict = action_dict*self.max_action
        # action_dict[1:-2] = -action_dict[1:-2] 
        
        return action_dict

    def postprocess(self, input_list,min_list,max_list):

        state_dict = np.clip(input_list,min_list,max_list)

        return state_dict


    def step(self, action_dict):

        print("level",self.level)
        # self.CBF_flag = action_dict[-1]
        self.normlized_action_dict = action_dict#action_dict[:-1]
        self.info_dict = {}
        self.counter = self.counter+1
        # import pdb;pdb.set_trace()
        print("action_dict:",action_dict)
        self.action_dict = self.preprocess(action_dict)
        print("action_dict:",self.action_dict)

        # self.state_dict[4] += (self.action_dict[4]*np.cos(self.state_dict[0]) - self.action_dict[5]*np.sin(self.state_dict[0]) )*self.T_step
        # self.state_dict[5] += (self.action_dict[4]*np.sin(self.state_dict[0]) + self.action_dict[5]*np.cos(self.state_dict[0]) )*self.T_step        
        # self.state_dict[:4] += self.action_dict[:4]*self.T_step

        self.state_dict += self.action_dict*self.T_step
        

        self.state_dict = self.postprocess(self.state_dict,self.min_state,self.max_state)
        print("target_dict:",self.target_affine_param)
        print("state_dict:",self.state_dict)

        # self.rt_transition_x += self.action_dict[4]*self.T_step
        # self.rt_transition_y += self.action_dict[5]*self.T_step
        
        self.center = np.array([self.state_dict[-2],self.state_dict[-1],self.action_dict[-2],self.action_dict[-1]]) 
         
        temp = self.relative_rt_formation_array_calculation(self.init_formation_all,self.state_dict)
        self.rt_formation_coordinate = temp[:,-3:]
        self.rt_formation_array = temp[:,:-3]
        print("rt_formation_array:",self.rt_formation_array)

        # indentify_affine_param = self.affine_param_identification(self.rt_formation_array)
        # print(indentify_affine_param,"vs",self.state_dict)
        # import pdb;pdb.set_trace()
        ##dynamic obstacles
        # self.obstacle_state[:,2] =  self.obstacle_speed_amplitude[:,0]*math.sin(self.counter/20)
        # self.obstacle_state[:,3] =  self.obstacle_speed_amplitude[:,1]*math.cos(self.counter/20)
        # self.obstacle_state[:,:2] = self.obstacle_state[:,:2] + self.obstacle_state[:,2:4]*0.1


        return self.get_obs(), self.instant_reward(), self.done_dict, self.info_dict#
        # return None,None,None,None

            

    def dist_affine_param(self):
        theta_dist = 1-np.cos((self.target_affine_param[0]-self.state_dict[0])/2)
        shear_x_dist = np.abs(self.target_affine_param[1]-self.state_dict[1])
        scale_x_dist = np.abs(self.target_affine_param[2]-self.state_dict[2])
        scale_y_dist = np.abs(self.target_affine_param[3]-self.state_dict[3])
        traslation_dist = np.linalg.norm(self.target_affine_param[-2:]-self.state_dict[-2:])
        dist_array = np.array([theta_dist,shear_x_dist,scale_x_dist,scale_y_dist,traslation_dist])
        end_dist_array = np.maximum(dist_array-self.affine_finish_threshold_list,np.zeros(5))

        weights_array = np.array([2,1/self.max_shear_x,1/self.max_scale_x,1/self.max_scale_y,0])*8*15#15#[3,1,1,1,0]
        return np.dot(end_dist_array,weights_array),dist_array
    
    ###碰撞检测###
    def formation_collision_free(self):
        flag = self.rtree.formation_obstacle_free(self.state_dict)
        return flag

    def instant_reward(self):  # sum of differences in velocities
        reward_dict = 0#np.zeros(self.n_agents)
        self.done_dict = 0

        ###
        self.dist2target = np.linalg.norm(self.target[:2]-self.center[:2])
        
        reward_closer_to_target = -self.dist2target*0.08#*10#15#40

        # if abs(self.dist2target)<3:
        #     reward_closer_to_target = (self.pre_dist2target-self.dist2target)*6
        # else:
        #     reward_closer_to_target = (self.pre_dist2target-self.dist2target)*2#15#40


        self.agents_vel_array = np.abs(self.rt_formation_array - self.pre_rt_formation_array)
        self.agents_acl_array = np.abs(self.agents_vel_array - self.pre_agents_vel_array)
        # reward_energy = -np.abs(self.normlized_action_dict).dot(np.array([1,1,1,1,1,1]))*0.05#0.2#0.2  #np.array([1,0.5,0.5,0.5,0.1,0.1]))*0.2
        self.VEL_BOUND = 2
        self.ACCEL_BOUND = 3
        self.agents_vel_violation = np.clip(self.agents_vel_array - self.VEL_BOUND,0,np.inf)
        self.agents_acl_violation = np.clip(self.agents_acl_array - self.ACCEL_BOUND,0,np.inf)
        a = np.linalg.norm(self.agents_vel_violation,axis=0).sum()
        b = np.linalg.norm(self.agents_acl_violation,axis=0).sum()
        reward_limitation_violation = -(a + 5*b)/self.n_agents/2


        a = np.linalg.norm(self.agents_vel_array,axis=0).sum()
        b = np.linalg.norm(self.agents_acl_array,axis=0).sum()
        reward_energy = -(a + 20*b)/self.n_agents/2#-(a/3 + 5*b)/self.n_agents/1.5
        # reward_energy = -(a/3 + 5*b)/1.5#-(a + 5*b)/1.5
        
        #time punishment 
        reward_time = -0.2
        # if (self.pre_dist2target - self.dist2target)>0:
        #     reward_time = -0.56+2*(self.pre_dist2target - self.dist2target)#-0.3#-0.3#0.4
        # else:
        #     reward_time = -0.56#-0.3

        reward_action_energy = -np.abs(self.normlized_action_dict).dot(np.array([1,1,1,1,0.1,0.1]))*0.05#-np.abs(self.pre_normlized_action_dict-self.normlized_action_dict).dot(np.array([1,1,1,1,0.1,0.1]))*0.1

        self.dist_affine_to_target,self.dist_affine_array = self.dist_affine_param()
        print("dist_affine_array:theta,shear,scalex,scaley,translation:",self.dist_affine_array)
        
        
        # if self.dist2target < self.on_circle_threshold:
        # import pdb;pdb.set_trace()
        if (self.affine_finish_threshold_list-self.dist_affine_array).min()>=0:
            self.done_dict = 1 ##到点记录
            if self.level>3:
                reward_finish = 200*2*4
            else:
                reward_finish = 200*2#*4#200*8#5
        else:  
            reward_finish =  0

        # reward_dis_from_target =(self.pre_dist_affine_to_target-self.dist_affine_to_target)*0.1
        reward_dis_from_target = -self.dist_affine_to_target*0.01/3#3*0.004#(self.pre_dist_affine_to_target-self.dist_affine_to_target)*1.5

        #######障碍物碰撞检测######
        
        flag_formation_collision_free = self.formation_collision_free() 
        if flag_formation_collision_free: #and flag_inter_agent_collision_free:
            reward_collision = 0
        else:
            # if self.obstacle_collison_array.max()==1 or (self.closet_obstacle_distance.min() < self.collision_threshold and self.obstacle_collison_array.max()==0):
            # if flag_formation_collision_free:
            #     reward_collision = -0.3
            #     
            # else:
            if self.level>3:
                reward_collision = -100*2*4
            else:
                reward_collision = -100*2#-100*2#-100*2#*4#-100*4
            self.done_dict = -1#碰撞障碍物记录
            # else:
            #     # if self.closet_obstacle_distance.min()< self.collision_free_threshold and self.CBF_flag==0 :
            #     #     reward_collision = -np.sum(15/self.closet_obstacle_distance)*0.15#0.05
            #     # reward_collision = -100*(1- (self.closet_obstacle_distance.min() - self.collision_threshold)/(self.collision_free_threshold - self.collision_threshold))*0.01#0.005

            #     reward_collision = -40*(1- (self.closet_obstacle_distance - self.collision_threshold)/(self.collision_free_threshold - self.collision_threshold)).sum()*0.01#0.005

        ###agent间相互避障
        flag_inter_agent_collision_free = self.inter_agent_collision_free(self.rt_formation_array, self.inter_agent_collision_threshold)
        self.info_dict["inter_col_free"] = flag_inter_agent_collision_free 
        # if flag_inter_agent_collision_free==False:##punish the inter-agent collision
        #     reward_collision += -1


        


        # if  self.dist2target<6:#10#reward_collision > self.pre_reward_collision and reward_closer_to_target>=0:
        #     reward_dis_from_target = (self.pre_dist_affine_to_target-self.dist_affine_to_target)*1.5#-np.abs(self.state_dict[:4]-self.init_state_dict[:4]).dot(index_list)*0.05
        # else:
        #     reward_dis_from_targ et = 0

        # import pdb;pdb.set_trace()
        # if self.CBF_flag:
        #     reward_dict = reward_closer_to_target  +  reward_energy + reward_collision + reward_finish + reward_dis_from_target 
        # else:
        #     # reward_dict = reward_closer_to_target + reward_collision +  reward_energy + reward_finish + reward_far_from_init
        reward_dict = reward_closer_to_target +  reward_energy + reward_collision + reward_finish + reward_dis_from_target+reward_action_energy +reward_time + reward_limitation_violation
        
        self.pre_agents_vel_array = self.agents_vel_array.copy()
        self.pre_rt_formation_array = self.rt_formation_array.copy()
        self.pre_dist2target = self.dist2target
        self.pre_dist_affine_to_target = self.dist_affine_to_target
        self.pre_action_dict = self.action_dict
        # self.pre_normlized_action_dict = self.normlized_action_dict
        self.pre_reward_collision = reward_collision

        print("step:",self.counter,"reward:",reward_dict,"reward_closer_to_target",reward_closer_to_target,"reward_dis_from_target",reward_dis_from_target,"reward_collision",reward_collision,"reward_energy",reward_energy,"reward_action_energy",reward_action_energy,"reward_finish",reward_finish,"reward_time",reward_time,"reward_limitation_violation",reward_limitation_violation)
        return reward_dict



    def render(self):
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            # self.eva_ax = fig.add_subplot(121)
            self.ax = fig.add_subplot(111)#fig.add_subplot(122)
            plt.title('MavenRL Simulator')
            self.fig = fig
            # self.r_max = 40

        if self.counter ==1:
            global ax,annotator
            ax = self.ax
            self.formation_point, = ax.plot(self.rt_formation_array[0, :], self.rt_formation_array[1, :], 'bo')  # Returns a tuple of line objects, thus the comma
            self.coordinate, = ax.plot(self.rt_formation_coordinate[0, :], self.rt_formation_coordinate[1, :], 'bo',lw=1)  # Returns a tuple of line objects, thus the comma
            self.formation_boundary, = ax.plot([], [], '-', color='b', lw=1)


            self.target_center, = ax.plot(self.target[0], self.target[1], 'gx')
            

            # for i in range(self.target_formation_array.shape[0]):
            # self.target_boundary, = ax.plot([], [], '-', color='y', lw=1)
            # self.target_boundary.set_xdata(np.append(self.target_formation_array[i,0,:],[self.target_formation_array[i,0,0]]))
            # self.target_boundary.set_ydata(np.append(self.target_formation_array[i,1,:],[self.target_formation_array[i,1,0]]))
            ax.plot(np.append(self.target_formation_array[0,:],[self.target_formation_array[0,0]]), np.append(self.target_formation_array[1,:],[self.target_formation_array[1,0]]), '-', color='y', lw=1)
            
            
            self.display_step = ax.annotate(str(self.counter),(self.r_max,self.r_max))

            #############obstacle####################
            for i in range(self.rrt_obstacle.shape[0]):
                plt.fill(self.rrt_obstacle[i,:,0], self.rrt_obstacle[i,:,1], color = "cornflowerblue")
            
            # plt.ylim(-10, self.r_max)
            # plt.xlim(-10, self.r_max)
            plt.ylim(-self.r_max/2,self.r_max/2)
            plt.xlim(-self.r_max/2,self.r_max/2)

            a = gca()

        # self.eva_ax.cla()
        # self.eva_ax.imshow(self.obstacle_image)

        #########################需要实时更新的绘制内容########################
        ##trajectory
        self.formation_point.set_xdata(self.rt_formation_array[0, :])
        self.formation_point.set_ydata(self.rt_formation_array[1, :])
        self.coordinate.set_xdata(self.rt_formation_coordinate[0, :])
        self.coordinate.set_ydata(self.rt_formation_coordinate[1, :])
        self.formation_boundary.set_xdata(np.append(self.rt_formation_array[0,:],[self.rt_formation_array[0,0]]))
        self.formation_boundary.set_ydata(np.append(self.rt_formation_array[1,:],[self.rt_formation_array[1,0]]))
            
        ax.plot(self.rt_formation_array[0, 0], self.rt_formation_array[1, 0], 'ro', markersize=2)
        ax.plot(self.rt_formation_array[0, 1], self.rt_formation_array[1, 1], 'go', markersize=2)
        ax.plot(self.rt_formation_array[0, 2], self.rt_formation_array[1, 2], 'bo', markersize=2)
        ax.plot(self.rt_formation_array[0, 3], self.rt_formation_array[1, 3], 'yo', markersize=2)

        self.target_center.set_xdata(self.target[0])
        self.target_center.set_ydata(self.target[1])

        self.display_step.remove()
        self.display_step = ax.annotate(str(self.counter),(self.r_max/2,self.r_max/2))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass

    def RRT_init(self,obstacle_inter, obstacle_list, path,scene_kind):
        plt.cla()
        self.counter = 0
        self.obstacle_inter = obstacle_inter
        self.state_dict = np.array(path[0])#self.target_path[0,:]
        self.center = np.zeros(4) 
        self.center[:2] = self.state_dict[-2:]
        self.init_formation_all = np.hstack((self.init_formation_array,self.init_formation_coordinate))
        temp = self.relative_rt_formation_array_calculation(self.init_formation_all,self.state_dict)
        self.rt_formation_coordinate = temp[:,-3:]
        self.rt_formation_array = temp[:,:-3]


        temp_list = []
        for target in path:
            target_formation = self.relative_rt_formation_array_calculation(self.init_formation_array,np.array(target))
            temp_list.append(target_formation)
        self.target_formation_array = np.array(temp_list)
        temp_list = []
        for obstacle in obstacle_list:
            temp = np.zeros((4,2))
            temp[0,:] = np.array(obstacle[:2])
            temp[1,:] = np.array([obstacle[0],obstacle[3]])
            temp[2,:] = np.array(obstacle[-2:])
            temp[3,:] = np.array([obstacle[2],obstacle[1]])
            temp_list.append(temp)
        self.rrt_obstacle = np.array(temp_list) 
        
        if scene_kind=="spa":#trick for improve performance in sparse env, reducing deformation cost
            self.max_det_theta = math.pi/12
            self.max_det_shear_x = 0.1#0.2
            self.max_det_scale_x = np.log(1.1)#np.log(1.5)#np.log(1.1)
            self.max_det_scale_y = np.log(1.1)#np.log(1.5)#np.log(1.1)
            self.max_action = np.array([self.max_det_theta,self.max_det_shear_x,self.max_det_scale_x,self.max_det_scale_y,self.max_det_transition_x,self.max_det_transition_y])
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3),0.5,0.5,0.5,2])
        
        if scene_kind=="real":
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3),0.3,0.3,0.3,0.5])
            # self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3/2),0.3,0.3,0.3,1])

    def RRT_reset(self, target):
        self.target_affine_param = np.array(target)
        self.target = np.zeros(4)
        self.target[:2] = self.target_affine_param[-2:]

        return self.RRT_obs()


    def RRT_obs(self):

        center_point = self.center[:2]
        fov_min_corner = center_point - self.FOV_radius
        fov_max_corner = center_point + self.FOV_radius
        
        fov_bbx = np.append(fov_min_corner, fov_max_corner)
        fov_obs_list = self.obstacle_inter.obs.intersection(fov_bbx, objects=True)

        output_image = np.zeros((self.FOV_pixel_size,self.FOV_pixel_size))
        image_origin = center_point - np.array([self.FOV_radius,self.FOV_radius])

        for obs_bbx in fov_obs_list:
            obs_min_corner = np.array(obs_bbx.bbox[:2])
            obs_max_corner = np.array(obs_bbx.bbox[2:]) 
            
            pix_obs_min_corner = ((obs_min_corner - image_origin)/self.FOV_resolution)
            pix_obs_max_corner = ((obs_max_corner - image_origin)/self.FOV_resolution)
            pix_obs_min_corner = np.maximum(pix_obs_min_corner,np.zeros(2)).astype(int)
            pix_obs_max_corner = np.minimum(pix_obs_max_corner,np.array([self.FOV_pixel_size,self.FOV_pixel_size])).astype(int)

            kernel = np.ones(tuple(pix_obs_max_corner-pix_obs_min_corner))
            output_image[pix_obs_min_corner[0] : pix_obs_max_corner[0], pix_obs_min_corner[1] : pix_obs_max_corner[1]] = kernel


        self.output_image = output_image
        obs_images = np.expand_dims(output_image,0)
        temp_formation_array = self.rt_formation_array.T - self.center[:2]
        temp_target = self.target - self.center
        diff_theta = (self.target_affine_param[0]-self.state_dict[0])#%(np.pi*2)
        # if diff_theta > np.pi:
        #     diff_theta -= np.pi*2
        # elif diff_theta < -np.pi:
        #     diff_theta += np.pi*2
        # elif np.abs(diff_theta) == np.pi:
        #     diff_theta = self.action_dict[0]/np.abs(self.action_dict[0])*np.pi
        obs_vector = np.hstack((temp_formation_array.reshape((-1))/6,self.init_formation_array.reshape((-1))/6,np.sin(diff_theta/2),np.cos(diff_theta/2),(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2]),temp_target[:-2]/6))
             
        return obs_vector,obs_images

    def RRT_step(self,action_dict):
        self.counter = self.counter+1
        self.action_dict = self.preprocess(action_dict)
        # print("action_dict:",self.action_dict)
        self.state_dict += self.action_dict*self.T_step
        self.state_dict = self.postprocess(self.state_dict,self.min_state,self.max_state)
        # print("target_dict:",self.target_affine_param)
        # print("state_dict:",self.state_dict)        
        self.center = np.array([self.state_dict[-2],self.state_dict[-1],self.action_dict[-2],self.action_dict[-1]]) 
         
        temp = self.relative_rt_formation_array_calculation(self.init_formation_all,self.state_dict)
        self.rt_formation_coordinate = temp[:,-3:]
        self.rt_formation_array = temp[:,:-3]

        _,dist_affine_array = self.dist_affine_param()
        finish_flag = (self.affine_finish_threshold_list-dist_affine_array).min()>=-0.3
        # print("affine_dis:", (self.affine_finish_threshold_list-dist_affine_array))   
        return self.RRT_obs(), finish_flag


    def RRT_render(self):
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            # self.eva_ax = fig.add_subplot(121)
            self.ax = fig.add_subplot(111)#fig.add_subplot(122)
            plt.title('MavenRL Simulator')
            self.fig = fig
            self.r_max = 40

        if self.counter ==1:
            global ax,annotator
            ax = self.ax
            self.formation_point, = ax.plot(self.rt_formation_array[0, :], self.rt_formation_array[1, :], 'bo')  # Returns a tuple of line objects, thus the comma
            self.coordinate, = ax.plot(self.rt_formation_coordinate[0, :], self.rt_formation_coordinate[1, :], 'bo',lw=1)  # Returns a tuple of line objects, thus the comma
            self.formation_boundary, = ax.plot([], [], '-', color='b', lw=1)


            self.target_center, = ax.plot(self.target[0], self.target[1], 'gx')
            

            for i in range(self.target_formation_array.shape[0]):
                # self.target_boundary, = ax.plot([], [], '-', color='y', lw=1)
                # self.target_boundary.set_xdata(np.append(self.target_formation_array[i,0,:],[self.target_formation_array[i,0,0]]))
                # self.target_boundary.set_ydata(np.append(self.target_formation_array[i,1,:],[self.target_formation_array[i,1,0]]))
                ax.plot(np.append(self.target_formation_array[i,0,:],[self.target_formation_array[i,0,0]]), np.append(self.target_formation_array[i,1,:],[self.target_formation_array[i,1,0]]), '-', color='y', lw=1)
            
            
            self.display_step = ax.annotate(str(self.counter),(self.r_max,self.r_max))

            #############obstacle####################
            for i in range(self.rrt_obstacle.shape[0]):
                plt.fill(self.rrt_obstacle[i,:,0], self.rrt_obstacle[i,:,1], color = "cornflowerblue")
            
            # plt.ylim(-10, self.r_max)
            # plt.xlim(-10, self.r_max)
            plt.ylim(-self.r_max,20 )
            plt.xlim(-self.r_max,20 )

            a = gca()

        # self.eva_ax.cla()
        # self.eva_ax.imshow(self.output_image)

        #########################需要实时更新的绘制内容########################
        ##trajectory
        self.formation_point.set_xdata(self.rt_formation_array[0, :])
        self.formation_point.set_ydata(self.rt_formation_array[1, :])
        self.coordinate.set_xdata(self.rt_formation_coordinate[0, :])
        self.coordinate.set_ydata(self.rt_formation_coordinate[1, :])
        self.formation_boundary.set_xdata(np.append(self.rt_formation_array[0,:],[self.rt_formation_array[0,0]]))
        self.formation_boundary.set_ydata(np.append(self.rt_formation_array[1,:],[self.rt_formation_array[1,0]]))
            
        ax.plot(self.rt_formation_array[0, 0], self.rt_formation_array[1, 0], 'ro', markersize=2)
        ax.plot(self.rt_formation_array[0, 1], self.rt_formation_array[1, 1], 'go', markersize=2)
        ax.plot(self.rt_formation_array[0, 2], self.rt_formation_array[1, 2], 'bo', markersize=2)
        ax.plot(self.rt_formation_array[0, 3], self.rt_formation_array[1, 3], 'yo', markersize=2)

        self.target_center.set_xdata(self.target[0])
        self.target_center.set_ydata(self.target[1])

        self.display_step.remove()
        self.display_step = ax.annotate(str(self.counter),(self.r_max,self.r_max))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        

    def render_save(self):
        self.save_path  =  "../output/rrt_affine_trajectory.png"
        plt.savefig(self.save_path) 


