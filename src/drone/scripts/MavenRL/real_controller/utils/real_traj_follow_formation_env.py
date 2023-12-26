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
import torch
from scipy.ndimage import distance_transform_edt
from queue import Queue
###################rrt related########### 
# import sys
# sys.path.append("../../../..")

from rtree import index
import uuid
from real_controller.utils.env_utils import SearchSpace,GJK,Polytope
from real_controller.utils.Experience_Graph import generate_fully_connected_graph_with_edge_attr_with_formation_aware,generate_egocentric_graph_with_edge_attr_with_formation_aware


font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class trajFollowFormationEnv(gym.Env):

    def __init__(self,test_flag=True,image_flag=False):

        # config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
        # config = configparser.ConfigParser()
        # config.read(config_file)

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
        self.dim_action = 2
        self.future_ref_frame = 5
        self.node_feature_dim = (1+1 + self.future_ref_frame)*2
        self.past_obs_frame = 5

        # problem parameters from file
        self.max_agents = 6
        self.init_n_agents = 4
        self.n_agents = self.init_n_agents



        self.target = np.array([0.0,0.0,0.0,0.0])
        self.center = np.array([0.0,0.0,0.0,0.0])
        self.counter = 0   
        self.a_net = np.zeros((self.n_agents, self.n_agents)) 
        self.x = np.zeros((self.n_agents, self.nx_system))
        self.u = np.zeros((self.n_agents, self.dim_action))

        
        self.radius = 10#5
        self.adj_distance_matrix = np.zeros((self.n_agents,2))
        self.closest_obs_dist = np.inf

        ###threshold
        self.collision_threshold = 0.1#0.5#1.5#1.5#1.5#1.5#.5#0.8
        self.collision_free_threshold = 1.5#1.5#3#1.5#5#10
        self.inter_agent_collision_threshold = 0.3#0.3
        self.finish_threshold = 0.5
        self.obstacle_size = 0.3
        
        
        # self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3),0.5,0.5,0.5,1])
        # self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3/2),0.3,0.3,0.3,1])#np.array([1-np.cos(np.pi/3),0.5,0.5,0.5,2])#np.array([np.cos(np.pi/3),0.5,0.5,0.5,2]) #np.array([np.cos(np.pi/3)-0.2,0.3,0.3,0.3,2]) 
        
        self.max_accel = 1 

        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(1,self.dim_action),dtype=float)
        # self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(1,(1+self.degree)*self.n_features+8*1+4),#+16

        #                                     dtype=np.float32)#(1+self.degree)*self.n_features*2
        # ###任意初始模板
        # self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(1,8+4*2+(1+self.degree)*(self.n_features-2)),#+16
        #                             dtype=np.float32)#(1+self.degree)*self.n_features*2

        ###实际队形+任意初始模板
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(1,4),#+16
                                    dtype=float)#(1+self.degree)*self.n_features*2

        #render init     
        self.r_min = -40
        self.r_max = 40#50#float(config['max_rad_init'])
        self.fig = None
        self.line1 = None
        self.line2 = None
        self.line3 = None
        self.seed(1)

        ###initialize obstacle set:
        self.n_obstacle = 15#15#5#15
        self.obstacle_state = np.zeros((self.n_obstacle,4))

        ######Constant######
        self.VEL_BOUND = 2
        self.ACCEL_BOUND = 3
        self.T_step = 0.1
        self.horizon_N = 10#20
        self.lqr_kp = 1#10#10#10
        

        ##FOV
        self.global_map_center = np.zeros(2)
        self.world_range = 25#30#50
        self.global_map_origin = self.global_map_center-self.world_range
        self.FOV_radius = 10
        self.cen_FOV_radius = 10
        self.dec_FOV_radius = 6

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
        self.max_det_transition_y = 1.5
        self.max_transition_x = np.inf
        self.min_transition_x = -np.inf
        self.max_transition_y = np.inf
        self.min_transition_y = -np.inf

        self.max_action = 2
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
                obstacle_center = np.empty(obstacle_dimension, float)
                scollision = True
                fcollision = True
                edge_lengths = []
                for j in range(obstacle_dimension):
                    # None of the sides of a hyperrectangle can be higher than 0.1 of the total span
                    # in that particular X.dimensions
                    # if i<1:
                    max_edge_length = 1.2#1.5#1.5#1.5#+obs_extend#15#(X.dimension_lengths[4+j][1] - X.dimension_lengths[4+j][0]) / 10.0
                    min_edge_length = 0.3#1#+obs_extend#10#(X.dimension_lengths[4+j][1] - X.dimension_lengths[4+j][0]) / 100.0
                    # else:
                    #     max_edge_length = 1#1.5#1.5#1.5#+obs_extend#15#(X.dimension_lengths[4+j][1] - X.dimension_lengths[4+j][0]) / 10.0
                    #     min_edge_length = 0.5#1#+obs_extend#10#(X.dimension_lengths[4+j][1] - X.dimension_lengths[4+j][0]) / 100.0
    
                    edge_length = random.uniform(min_edge_length, max_edge_length)
                    edge_lengths.append(edge_length)
                    self.obs_surround_area = self.center2target/2.0
                    if start[-2+j] < 0:
                        obstacle_center[j] = random.uniform(-self.center2target-3,0+self.obs_surround_area)#-self.center2target-1,2#-self.center2target-3,3
                    else:
                        obstacle_center[j] = random.uniform(0-self.obs_surround_area,self.center2target+3)#-2,self.center2target+1##-3,self.center2target+3
                        # 
                    
                ###保证障碍物中心不在编队内部
                if self.fast_obstacle_collision_detect(self.rt_formation_array,np.expand_dims(obstacle_center,0)) or self.fast_obstacle_collision_detect(self.target_formation_array,np.expand_dims(obstacle_center,0)):
                    continue

                # Check if any part of the obstacle is inside of another obstacle.
                min_corner = np.empty(obstacle_dimension, float)
                max_corner = np.empty(obstacle_dimension, float)

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
                # temp[4,:] = np.array([obstacle[2,obstacle[1]])
                
                # ###保证障碍物中心不在编队内部
                # if self.fast_obstacle_collision_detect(self.rt_formation_array,temp) or self.fast_obstacle_collision_detect(self.target_formation_array,temp):
                #     continue
                
                # # import pdb;pdb.set_trace()
                # Check newly generated obstacle intersects any former ones. Also respect start and end points
                if len(list(X.extend_obs.intersection(obstacle))) > 0 :
                    continue

                temp_id = i#uuid.uuid4()
                X.extend_obs.insert(temp_id, tuple(obstacle))

                # print("obstacle1", i,counter)

                # import pdb;pdb.set_trace()
                if X.formation_obstacle_free(start) and X.formation_obstacle_free(end):
                    # obstacles.append((temp_id,obstacle_real))
                    obstacles.append(obstacle_real)
                    X.obs.insert(temp_id, tuple(obstacle_real))
                else:
                    X.extend_obs.delete(temp_id, tuple(obstacle))
                    # X.obs.delete(temp_id, tuple(obstacle_real))
                    continue
                i += 1
                # print("obstacle2",i)
            
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

    def reset(self, trajs_for_following, static_obs, rrt_obstacle, level=0):
        
        plt.cla()
        # plt.figure()
        # plt.title("self.global_map")
        self.level = level
        self.counter = 0
        self.trajs_for_following = trajs_for_following.copy()
        self.total_traj_steps = self.trajs_for_following.shape[0]
        self.n_agents = self.trajs_for_following.shape[2]
        # import pdb;pdb.set_trace()

        self.target_formation_array = self.trajs_for_following[-1,:,:self.n_agents]
        self.rt_formation_array = self.trajs_for_following[0,:,:self.n_agents]
        self.center = self.rt_formation_array.mean(axis=1)
        
        self.next_ref_pos_agn = []
        for i in range(self.future_ref_frame):
            # if self.counter+1+i < self.total_traj_steps-1:
            self.next_ref_pos_agn.append( self.trajs_for_following[min(self.counter+1+i, self.total_traj_steps-1), :, :self.n_agents] )

        self.cur_ref_pos_agn = self.trajs_for_following[0,:,:self.n_agents]
        
        self.rtree = static_obs


        if self.level ==0:#by default
            self.collision_free_threshold = 2
            self.obstacle_density = 2.5
            self.center2target = 7
            self.n_obstacle = 0
            self.obs_extend = 0.5#1.0#0.5
            self.obs_surround_area = 2
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3),0.5,0.5,0.5,1])

        elif self.level ==1:
            self.collision_free_threshold = 2
            self.obstacle_density = 2.5
            self.center2target = 7
            self.n_obstacle = 1
            self.obs_extend = 0.5#1.0#0.5
            self.obs_surround_area = 4
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/6),0.5,0.5,0.5,1])
            # self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/3),0.5,0.5,0.5,2])
            # self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/12),0.3,0.3,0.3,1])
            # self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/6),0.3,0.3,0.3,1])
        
        elif self.level ==2:
            # self.obstacle_density = 2.5
            # self.center2target = 10
            # self.n_obstacle = 4
            # self.obs_extend = 1#0.75
            # self.obs_surround_area = 2
            self.collision_free_threshold = 2

            self.obstacle_density = 4
            self.center2target = 10
            self.n_obstacle = 1#6#5#5
            self.obs_extend = 0.6
            self.obs_surround_area = 4
            # self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/12),0.3,0.3,0.3,0.5])
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/6),0.5,0.5,0.5,1])
        
        elif self.level ==3:
            self.collision_free_threshold = 1.5
            self.obstacle_density = 4
            self.center2target = 10
            self.n_obstacle = 7#6#5#5
            self.obs_extend = 0.6
            self.obs_surround_area = 7
            self.affine_finish_threshold_list = np.array([1-np.cos(np.pi/12),0.3,0.3,0.3,0.5])
        

        # temp_list = []
        GJK_polytope_list = []
        self.obstacle_state = np.zeros((self.n_obstacle,4))
        for i in range(self.n_obstacle):
            start_step = np.random.randint(int(self.total_traj_steps*2/4),int(self.total_traj_steps*3/4))
            start_id = np.random.randint(0,self.n_agents-1)
            self.obstacle_state[i,:2] = self.trajs_for_following[start_step,:,start_id]
            self.obstacle_state[i,2] = start_step
            self.obstacle_state[i,3] = start_id

            obstacle_id = i
            temp_gjk = np.zeros((1,3))
            temp_gjk[0,:2] = self.obstacle_state[i,:2] 
            GJK_polytope_list.append((str(obstacle_id),Polytope(temp_gjk)))

        # self.global_map = self.obsimg_render(self.obstacle_state[:,:2],self.global_map_center,self.world_range)
        self.historical_obs_queue = Queue()
        for i in range(self.past_obs_frame):
            self.historical_obs_queue.put(self.obstacle_state.copy())


        self.rrt_obstacle = rrt_obstacle#np.array(temp_list) 
        temp = np.zeros((self.n_agents,3))
        temp[:,:2] = self.rt_formation_array.T
        GJK_polytope_list.append(("formation",Polytope(temp)))
        self.gjk_solver = GJK(GJK_polytope_list)

        #init param
        self.r_max = self.center2target*5#2.5
        self.obs_dict = np.zeros(int(self.observation_space.shape[1]))
        self.pre_obs_dict = np.zeros(int(self.observation_space.shape[1]))

        #done:
        self.done_dict = 0#np.zeros(self.n_agents) 

        #init reward:
        self.dist_list = np.zeros(self.n_agents)
        self.pre_dist_list = np.zeros(self.n_agents)
        self.dist_agn2tar_list = np.zeros(self.n_agents)
        self.energy_list  = np.zeros(self.n_agents)
        self.on_circle_time = np.zeros(self.n_agents)
        
        
        self.pre_rt_formation_array = self.rt_formation_array.copy()
        self.pre_agents_vel_array = np.zeros_like(self.pre_rt_formation_array)
        # self.dist2target = np.linalg.norm(self.target[:2]-self.center[:2])
        # self.pre_dist2target = self.dist2target

        self.pre_action_dict = np.zeros(self.dim_action)
        self.pre_normlized_action_dict = np.zeros(self.dim_action)
        self.pre_reward_collision = 0

        
        # import pdb;pdb.set_trace()

        return self.get_obs()



    def step(self, action_dict):

        self.normlized_action_dict = action_dict#action_dict[:-1]
        self.info_dict = {}
        self.counter = self.counter+1
        # import pdb;pdb.set_trace()
        # print("action_dict:",action_dict)
        self.action_dict = self.preprocess(action_dict)

        print("action_dict:",self.action_dict)
        ###update formation agents
        self.rt_formation_array += self.action_dict*self.T_step
        self.rt_formation_array = self.postprocess(self.rt_formation_array, -self.world_range, self.world_range)

        self.center[:2] = self.rt_formation_array.mean(axis=1)
        # print("rt_formation_array:",self.rt_formation_array)
        self_id = self.gjk_solver.ObjID["formation"]
        temp = np.zeros((self.n_agents,3))
        temp[:,:2] = self.rt_formation_array.T
        self.gjk_solver.ConvObject[self_id] = Polytope(temp.copy())

        self.next_ref_pos_agn = []
        for i in range(self.future_ref_frame):
            # if self.counter+1+i < self.total_traj_steps-1:
            self.next_ref_pos_agn.append( self.trajs_for_following[min(self.counter+1+i, self.total_traj_steps-1), :, :self.n_agents] )

        # self.next_ref_pos_agn = self.trajs_for_following[min(self.counter+1,self.total_traj_steps-1),:,:self.n_agents] 
        self.cur_ref_pos_agn = self.trajs_for_following[min(self.counter,self.total_traj_steps-1),:,:self.n_agents] 

        ###update dynamic obstacles
        for i in range(self.n_obstacle):
            start_step = int(self.obstacle_state[i,2])
            start_id = int(self.obstacle_state[i,3]) 
            temp_step = max(start_step - self.counter,0)
            self.obstacle_state[i,:2] = self.trajs_for_following[temp_step, : ,start_id]

            temp_gjk = np.zeros((1,3))
            temp_gjk[0,:2] = self.obstacle_state[i,:2] 
            self.gjk_solver.ConvObject[i] = Polytope(temp_gjk)

        self.historical_obs_queue.get()
        self.historical_obs_queue.put(self.obstacle_state.copy())


        return self.get_obs(), self.instant_reward(), self.done_dict, self.info_dict#
        # return None,None,None,None


    def load_static_obs(self, two_points_static_obs_list):
        self.two_points_static_obs_list = two_points_static_obs_list
        temp_list = []
        GJK_polytope_list = []
        for i in range(len(self.two_points_static_obs_list)):
            obstacle_id = i
            obstacle = self.two_points_static_obs_list[i]
            temp = np.zeros((4,2))
            temp[0,:] = np.array(obstacle[:2])
            temp[1,:] = np.array([obstacle[0],obstacle[3]])
            temp[2,:] = np.array(obstacle[-2:])
            temp[3,:] = np.array([obstacle[2],obstacle[1]])
            temp_list.append(temp)
            temp_gjk = np.zeros((4,3))
            temp_gjk[:,:2] = temp 
            GJK_polytope_list.append((str(obstacle_id),Polytope(temp_gjk)))
        
        self.gjk_solver = GJK(GJK_polytope_list)
        self.rrt_obstacle = np.array(temp_list) 


    def reset_dummy_env(self, center_point, update_drone_state_array, update_dynamic_obs_state_array, two_points_static_obs_list):
        
        ###load agent state
        self.n_agents = update_drone_state_array.shape[0]
        self.rt_formation_array = update_drone_state_array[:,:2].T
        self.agents_vel_array = update_drone_state_array[:,3:-1].T
        
        ###load_dynamic_obs
        self.n_dyn_obs = update_dynamic_obs_state_array.shape[0]
        self.obstacle_state = update_dynamic_obs_state_array.copy()
        self.historical_obs_queue = Queue()
        for i in range(self.past_obs_frame):
            self.historical_obs_queue.put(self.obstacle_state.copy())
        

        ###load_static_obs
        self.n_sta_obs = len(two_points_static_obs_list)
        self.global_map_center = center_point[:2]
        self.global_map_origin = self.global_map_center-self.world_range
        
        self.load_static_obs(two_points_static_obs_list)
        self.static_global_map = self.static_obsimg_render(self.global_map_center, self.world_range)

        self.pre_rt_formation_array = self.rt_formation_array.copy()
        self.pre_agents_vel_array = self.agents_vel_array.copy()

        # return 



    def update_dummy_env(self, update_drone_state_array, update_dyn_obs_state_array, cur_ref_pos_agn, next_ref_pos_agn):

        ###update agent reference 
        self.cur_ref_pos_agn = cur_ref_pos_agn.copy()
        self.next_ref_pos_agn = next_ref_pos_agn.copy()
        
        ###update agent state
        self.rt_formation_array = update_drone_state_array[:,:2].T
        self.agents_vel_array = update_drone_state_array[:,3:-1].T
        
        ###update_dynamic_obs
        self.obstacle_state[:,:] = update_dyn_obs_state_array.copy()
        self.historical_obs_queue.get()
        self.historical_obs_queue.put(self.obstacle_state.copy())

        ###update_pre_agent_state
        self.pre_rt_formation_array = self.rt_formation_array.copy()
        self.pre_agents_vel_array = self.agents_vel_array.copy()

        return self.render_dummy_env()



    def render_dummy_env(self):

        if self.n_dyn_obs>0:
            self.dynamic_global_map = self.dynamic_obsimg_render(self.global_map_center, self.world_range)
        else:
            self.dynamic_global_map = self.static_global_map.copy()        
        # obstacle_map = self.dynamic_global_map.copy()
        # obstacle_field = distance_transform_edt(1-obstacle_map, self.FOV_resolution)
        # # time4 = time.time()
        # # print("time-4",time4-time3)
        # formation_map = self.formation2image(self.rt_formation_array.T, self.global_map_center,self.world_range)
        # # time5 = time.time()
        # # print("time-5",time5-time4)
        # formation_field = distance_transform_edt(1-formation_map, self.FOV_resolution)
        # # time6 = time.time()
        # # print("time-6",time6-time5)
        # comprehensive_field = obstacle_field + formation_field
        # self.closest_obs_dist = comprehensive_field.min()
        # print("self.closest_obs_dist",self.closest_obs_dist)
        
        # plt.figure()
        # plt.title("comprehensive_field")
        # plt.imshow(comprehensive_field)

        # # plt.show()
        
        cur_act_pos_agn = self.rt_formation_array.T 
        cur_ref_pos_agn = self.cur_ref_pos_agn.T
        
        

        # cen_state_img = np.expand_dims(self.fov_query_global_map(self.center[:2], self.cen_FOV_radius),0)
        cen_next_ref_pos_agn = np.zeros((self.n_agents,0))
        
        # for temp in self.next_ref_pos_agn:
        #     cen_next_ref_pos_agn = np.hstack((cen_next_ref_pos_agn, temp.T - self.center[:2]))
        # cen_state_graph = generate_fully_connected_graph(np.hstack((cur_act_pos_agn- self.center[:2], cur_ref_pos_agn- self.center[:2], cen_next_ref_pos_agn ))/6)

        for temp in self.next_ref_pos_agn:
            cen_next_ref_pos_agn = np.hstack((cen_next_ref_pos_agn, temp.T - cur_act_pos_agn))

        node_features = np.hstack((self.agents_vel_array.T, cur_ref_pos_agn- cur_act_pos_agn, cen_next_ref_pos_agn ))
        
        # cen_state_graph = generate_fully_connected_graph_with_edge_attr(node_features, cur_act_pos_agn)

        # cen_state_graph = generate_fully_connected_graph_with_edge_attr_with_formation_aware(node_features, cur_act_pos_agn)
        
        dec_state_graph =  []
        dec_state_img = []
        # time3 = time.time()
        # print("time-2",time3-time2)


        for i in range(self.n_agents):
            dec_state_graph.append(generate_egocentric_graph_with_edge_attr_with_formation_aware(node_features, cur_act_pos_agn, i))
            
            temp_img = self.fov_query_global_map(self.rt_formation_array[:,i], self.dec_FOV_radius, self.dynamic_global_map)
            dec_state_img.append(np.expand_dims(temp_img,0))
            




        try:
            dec_state_img = np.array(dec_state_img)
        except:
            import pdb;pdb.set_trace()
            dec_state_img = np.array([np.expand_dims(np.zeros((120,120)),0) for i in range(self.n_agents)])

        # cen_state_img = dec_state_img.copy()
        dec_state_img = torch.from_numpy(dec_state_img).float()#.unsqueeze(0)
        # cen_state_img = torch.from_numpy(cen_state_img).float().unsqueeze(0)
        # cen_state_img = torch.from_numpy(cen_state_img).float()

        return dec_state_graph, dec_state_img



    def static_obsimg_render(self, center_point, FOV_radius):

        FOV_pixel_size = int(2*FOV_radius/self.FOV_resolution)
        fov_min_corner = center_point - FOV_radius
        fov_max_corner = center_point + FOV_radius
        
        fov_bbx = np.append(fov_min_corner, fov_max_corner)
        
        
        # output_image = self.points2image(self.obstacle_state[:,:2], 1, center_point, FOV_radius)
        output_image = np.zeros((FOV_pixel_size,FOV_pixel_size))

        image_origin = center_point - np.array([FOV_radius,FOV_radius])
        # fov_obs_list = self.rtree.obs.intersection(fov_bbx, objects=True)
        for i in range(len(self.two_points_static_obs_list)):
            obstacle = np.array(self.two_points_static_obs_list[i])            
            obs_min_corner = obstacle[:2]
            obs_max_corner = obstacle[2:]
            
            pix_obs_min_corner = ((obs_min_corner - image_origin)/self.FOV_resolution)
            pix_obs_max_corner = ((obs_max_corner - image_origin)/self.FOV_resolution)
            pix_obs_min_corner = np.maximum(pix_obs_min_corner,np.zeros(2)).astype(int)
            pix_obs_max_corner = np.minimum(pix_obs_max_corner,np.array([FOV_pixel_size,FOV_pixel_size])).astype(int)

            kernel = np.ones(tuple(pix_obs_max_corner-pix_obs_min_corner))
            output_image[pix_obs_min_corner[0] : pix_obs_max_corner[0], pix_obs_min_corner[1] : pix_obs_max_corner[1]] = kernel

        return output_image

    def dynamic_obsimg_render(self, center_point, FOV_radius):
        
        dynamic_global_map = self.static_global_map.copy()
        for i in range(self.past_obs_frame):
            fill_value = (i+1)/self.past_obs_frame
            dynamic_global_map = self.points2image(self.historical_obs_queue.queue[i][:,:2], fill_value, self.global_map_center, self.world_range, dynamic_global_map)

        return dynamic_global_map
        


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


    def inter_agent_collision_free(self, formation_array, inter_agent_collision_threshold):
        '''
        判断编队内部各节点间的相互碰撞情况,通过求最近邻距离矩阵中的最小值和碰撞阈值()相对比        
        '''
        
        try:
            self.neighbor_solver.fit(formation_array.T)
        except:
            import pdb;pdb.set_trace()
        self.adj_distance_matrix,self.neighbor_id_matrix = self.neighbor_solver.kneighbors()
        print("gaojinglei@@@###",self.adj_distance_matrix)
        return self.adj_distance_matrix.min() > inter_agent_collision_threshold

    def points2image(self,state_array, fill_value, center_point, FOV_radius, input_image=None):
        FOV_pixel_size = int(FOV_radius*2/self.FOV_resolution)
        ###to inflate the obstacle
        radius = int(self.obstacle_size* 1.5 /self.FOV_resolution/2)

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
        output_image = np.zeros((FOV_pixel_size+radius*2,FOV_pixel_size+radius*2))
        if input_image is not None:
            output_image[radius:radius+FOV_pixel_size,radius:radius+FOV_pixel_size] = input_image[:,:]
        image_origin = center_point - np.array([FOV_radius,FOV_radius])
        state_array_pixel = (state_array - image_origin)/self.FOV_resolution
        # print("time",time2-time1)
        # import pdb;pdb.set_trace()

        
        for point in state_array_pixel:
            # output_image[int(point[1]),int(point[0])] = 1
            temp = output_image[radius+int(point[0])-radius:radius+int(point[0])+radius+1,radius+int(point[1])-radius:radius+int(point[1])+radius+1]
            # output_image[radius+int(point[0])-radius:radius+int(point[0])+radius+1,radius+int(point[1])-radius:radius+int(point[1])+radius+1] = np.maximum(temp,kernel*fill_value)
            try:
                output_image[radius+int(point[0])-radius:radius+int(point[0])+radius+1,radius+int(point[1])-radius:radius+int(point[1])+radius+1] = np.maximum(temp,kernel*fill_value)
            except:
                pass



        return output_image[radius:radius+FOV_pixel_size,radius:radius+FOV_pixel_size]
    
    def formation2image(self,formation_array, center_point, FOV_radius):
        FOV_pixel_size = int(2*FOV_radius/self.FOV_resolution)
        output_image = np.zeros((FOV_pixel_size,FOV_pixel_size))
        image_origin = center_point - np.array([FOV_radius,FOV_radius])
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

        return output_image.T




    def obsimg_render(self, center_point, FOV_radius):

        FOV_pixel_size = int(2*FOV_radius/self.FOV_resolution)
        fov_min_corner = center_point - FOV_radius
        fov_max_corner = center_point + FOV_radius
        
        fov_bbx = np.append(fov_min_corner, fov_max_corner)
        
        
        output_image = self.points2image(self.obstacle_state[:,:2], 1, center_point, FOV_radius)

        image_origin = center_point - np.array([FOV_radius,FOV_radius])
        fov_obs_list = self.rtree.obs.intersection(fov_bbx, objects=True)

        for obs_bbx in fov_obs_list:
            # time2 = time.time()
            # print("time",time2-time1)
            # import pdb;pdb.set_trace()
            obs_min_corner = np.array(obs_bbx.bbox[:2])
            obs_max_corner = np.array(obs_bbx.bbox[2:])
            
            pix_obs_min_corner = ((obs_min_corner - image_origin)/self.FOV_resolution)
            pix_obs_max_corner = ((obs_max_corner - image_origin)/self.FOV_resolution)
            pix_obs_min_corner = np.maximum(pix_obs_min_corner,np.zeros(2)).astype(int)
            pix_obs_max_corner = np.minimum(pix_obs_max_corner,np.array([FOV_pixel_size,FOV_pixel_size])).astype(int)

            kernel = np.ones(tuple(pix_obs_max_corner-pix_obs_min_corner))
            output_image[pix_obs_min_corner[0] : pix_obs_max_corner[0], pix_obs_min_corner[1] : pix_obs_max_corner[1]] = kernel

        return output_image

    def fov_query_global_map(self, center_coord, FOV_radius, global_map):
        FOV_pixel_size = int(2*FOV_radius/self.FOV_resolution)
        output_image = np.zeros((FOV_pixel_size,FOV_pixel_size))
        # map_origin = global_map_origin
        # output_image = np.zeros((self.FOV_pixel_size,self.FOV_pixel_size))
        # fov_min_corner = center_coord - self.FOV_radius
        # fov_max_corner = center_coord + self.FOV_radius
        pix_center = ((center_coord - self.global_map_origin)/self.FOV_resolution).astype(np.int32)
        pix_fov_min_corner = pix_center-int(FOV_pixel_size/2)
        pix_fov_max_corner = pix_center+int(FOV_pixel_size/2)

        # pix_fov_min_corner = ((fov_min_corner - self.global_map_origin)/self.FOV_resolution)
        # pix_fov_max_corner = ((fov_max_corner - self.global_map_origin)/self.FOV_resolution)
        
        # kernel = np.ones(tuple(pix_fov_max_corner-pix_fov_min_corner))
        # output_image[pix_obs_min_corner[0] : pix_obs_max_corner[0], pix_obs_min_corner[1] : pix_obs_max_corner[1]] = kernel
        # import pdb;pdb.set_trace()
        if pix_fov_min_corner[1]==pix_fov_max_corner[1]:
            import pdb;pdb.set_trace()
        
        shape = np.array(global_map.shape)
        min_index  = 0 - np.minimum(pix_fov_min_corner,0)
        max_index  = FOV_pixel_size - np.maximum(pix_fov_max_corner-shape,0)

        try:
            output_image[min_index[0]:max_index[0],min_index[1]:max_index[1]] = global_map[ max(pix_fov_min_corner[0],0) : min(pix_fov_max_corner[0],shape[0]), max(pix_fov_min_corner[1],0): min(pix_fov_max_corner[1],shape[1])] 
        except:
            pass
        # return self.global_map[pix_fov_min_corner[0]:pix_fov_max_corner[0],pix_fov_min_corner[1]:pix_fov_max_corner[1]] 
        return output_image

    def get_obs(self):
        if self.image_flag:

            return self._get_image_obs()
        else:
            # return None
            return self._get_obs_node()

    

    def _get_image_obs(self):   

        self.dynamic_obsimg_render(self.global_map_center, self.world_range)
        
        obstacle_map = self.dynamic_global_map.copy()
        obstacle_field = distance_transform_edt(1-obstacle_map, self.FOV_resolution)
        # time4 = time.time()
        # print("time-4",time4-time3)
        formation_map = self.formation2image(self.rt_formation_array.T, self.global_map_center,self.world_range)
        # time5 = time.time()
        # print("time-5",time5-time4)
        formation_field = distance_transform_edt(1-formation_map, self.FOV_resolution)
        # time6 = time.time()
        # print("time-6",time6-time5)
        comprehensive_field = obstacle_field + formation_field
        self.closest_obs_dist = comprehensive_field.min()
        print("self.closest_obs_dist",self.closest_obs_dist)
        
        # plt.figure()
        # plt.title("comprehensive_field")
        # plt.imshow(comprehensive_field)

        # # plt.show()
        
        cur_act_pos_agn = self.rt_formation_array.T 
        cur_ref_pos_agn = self.cur_ref_pos_agn.T
        
        
        
        self.ids_obs_in_fov = []
        for i in range(self.n_obstacle): 
            if abs(self.obstacle_state[i,0]-self.center[0])<self.cen_FOV_radius and abs(self.obstacle_state[i,1]-self.center[1])<self.cen_FOV_radius: 
                start_step = int(self.obstacle_state[i,2])
                start_id = int(self.obstacle_state[i,3])
                temp_step = max(start_step - self.counter - 1,0)
                self.ids_obs_in_fov.append(i)




        # cen_state_img = np.expand_dims(self.fov_query_global_map(self.center[:2], self.cen_FOV_radius),0)
        cen_next_ref_pos_agn = np.zeros((self.n_agents,0))
        
        # for temp in self.next_ref_pos_agn:
        #     cen_next_ref_pos_agn = np.hstack((cen_next_ref_pos_agn, temp.T - self.center[:2]))
        # cen_state_graph = generate_fully_connected_graph(np.hstack((cur_act_pos_agn- self.center[:2], cur_ref_pos_agn- self.center[:2], cen_next_ref_pos_agn ))/6)

        for temp in self.next_ref_pos_agn:
            cen_next_ref_pos_agn = np.hstack((cen_next_ref_pos_agn, temp.T - cur_act_pos_agn))

        node_features = np.hstack((cur_ref_pos_agn- cur_act_pos_agn, cen_next_ref_pos_agn ))
        cen_state_graph = generate_fully_connected_graph_with_edge_attr(node_features, cur_act_pos_agn)

        dec_state_graph =  []
        dec_state_img = []
        # time3 = time.time()
        # print("time-2",time3-time2)


        for i in range(self.n_agents):
            # dec_next_ref_pos_agn = np.zeros((self.n_agents,0))
            # for temp in self.next_ref_pos_agn:
            #     # import pdb;pdb.set_trace()
            #     dec_next_ref_pos_agn = np.hstack((dec_next_ref_pos_agn, temp.T - self.rt_formation_array[:,i]))
            # temp_graph = np.hstack((cur_act_pos_agn - self.rt_formation_array[:,i], cur_ref_pos_agn - self.rt_formation_array[:,i], dec_next_ref_pos_agn))/6
            
            # print("counter",self.counter,"graph_i",i,temp_graph)
            # dec_state_graph.append(generate_egocentric_graph(temp_graph, i))
            dec_state_graph.append(generate_egocentric_graph_with_edge_attr(node_features, cur_act_pos_agn, i))
            temp_img = self.fov_query_global_map(self.rt_formation_array[:,i], self.dec_FOV_radius)
            dec_state_img.append(np.expand_dims(temp_img,0))
            

        try:
            dec_state_img = np.array(dec_state_img)
        except:
            import pdb;pdb.set_trace()
            dec_state_img = np.array([np.expand_dims(np.zeros((120,120)),0) for i in range(self.n_agents)])
            

        cen_state_img = dec_state_img.copy()
        dec_state_img = torch.from_numpy(dec_state_img).float()#.unsqueeze(0)
        # cen_state_img = torch.from_numpy(cen_state_img).float().unsqueeze(0)
        cen_state_img = torch.from_numpy(cen_state_img).float()

        return dec_state_graph, dec_state_img

    def _get_obs_node(self):        
        cur_act_pos_agn = self.rt_formation_array.T 
        next_ref_pos_agn = self.next_ref_pos_agn.T
        cur_act_pos_obs = []
        next_ref_pos_obs = []#np.zeros((2,self.n_obstacle))
        
        self.ids_obs_in_fov = []
        for i in range(self.n_obstacle):
            if abs(self.obstacle_state[i,0]-self.center[0])<self.cen_FOV_radius and abs(self.obstacle_state[i,1]-self.center[1])<self.cen_FOV_radius: 
                start_step = int(self.obstacle_state[i,2])
                start_id = int(self.obstacle_state[i,3])
                temp_step = max(start_step - self.counter - 1,0)
                next_ref_pos_obs.append(self.trajs_for_following[temp_step, :, start_id]) 
                cur_act_pos_obs.append(self.obstacle_state[i,:2])
                self.ids_obs_in_fov.append(i)
        
        n_obs_in_fov = len(cur_act_pos_obs)
        node_id_array = np.zeros(self.n_agents + n_obs_in_fov)
        node_id_array[:self.n_agents] = 1
        cur_act_pos_obs = np.array(cur_act_pos_obs)
        next_ref_pos_obs = np.array(next_ref_pos_obs)

        dec_obs =  []
        for i in range(self.n_agents):
            dec_agent_comp = np.hstack((cur_act_pos_agn - self.rt_formation_array[:,i], next_ref_pos_agn - self.rt_formation_array[:,i]))
            dec_obstacle_comp = np.hstack((np.array(cur_act_pos_obs)-self.rt_formation_array[:,i], np.array(next_ref_pos_obs)-self.rt_formation_array[:,i]))
            dec_obs.append( np.vstack(( np.hstack((dec_agent_comp.T, dec_obstacle_comp.T)), node_id_array)) )
        
        cen_agent_comp = np.hstack((cur_act_pos_agn- self.center[:2],next_ref_pos_agn- self.center[:2]))
        cen_obstacle_comp = np.hstack((np.array(cur_act_pos_obs)-self.center[:2], np.array(next_ref_pos_obs)-self.center[:2]))
        
        cen_obs =  np.vstack(( np.hstack((cen_agent_comp.T,cen_obstacle_comp.T)), node_id_array))

        return dec_obs, cen_obs


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
        # temp_mask = 1-self.agent_finish_mask
        action_dict.resize((self.n_agents,2))
        action_dict = action_dict.T *self.max_action#*temp_mask


        # action_dict[1:-2] = -action_dict[1:-2] 
        
        return action_dict

    def postprocess(self, input_list,min_list,max_list):

        state_dict = np.clip(input_list,min_list,max_list)

        return state_dict


    
    def dist_affine_param(self):
        theta_dist = 1-np.cos((self.target_affine_param[0]-self.state_dict[0]))#1-np.cos((self.target_affine_param[0]-self.state_dict[0])/2)
        shear_x_dist = np.abs(self.target_affine_param[1]-self.state_dict[1])
        scale_x_dist = np.abs(self.target_affine_param[2]-self.state_dict[2])
        scale_y_dist = np.abs(self.target_affine_param[3]-self.state_dict[3])
        traslation_dist = np.linalg.norm(self.target_affine_param[-2:]-self.state_dict[-2:])
        dist_array = np.array([theta_dist,shear_x_dist,scale_x_dist,scale_y_dist,traslation_dist])
        # end_dist_array = np.maximum(dist_array-self.affine_finish_threshold_list,np.zeros(5))

        weights_array = np.array([3,1/self.max_shear_x,1/self.max_scale_x,1/self.max_scale_y,0])*8*15#15#[3,1,1,1,0]
        return np.dot(dist_array,weights_array),dist_array
    
    
    ###碰撞检测###
    def formation_collision_free(self):
        # flag = self.rtree.formation_obstacle_free(self.state_dict)
        flag = True
        # self.closest_obs_dist = np.inf


        # fov_min_corner = self.center[:2] - self.cen_FOV_radius
        # fov_max_corner = self.center[:2] + self.cen_FOV_radius
        
        # fov_bbx = np.append(fov_min_corner, fov_max_corner)
        # fov_obs_list = self.rtree.obs.intersection(fov_bbx, objects=True)
        # for obs_bbx in fov_obs_list:
        #     self.closest_obs_dist = min(self.closest_obs_dist,self.gjk_solver.GetDist("formation",str(obs_bbx.id)))


        # for id in self.ids_obs_in_fov:
        #     self.closest_obs_dist = min(self.closest_obs_dist, max(self.gjk_solver.GetDist("formation",str(id)-self.obstacle_size, 0) ))

        if self.closest_obs_dist < self.collision_threshold:
        # if self.closest_obs_dist < self.obstacle_size:
            flag = False 

        return flag

    def instant_reward(self):  # sum of differences in velocities

        reward_dict = 0#np.zeros(self.n_agents)
        self.done_dict = 0


        distance_array = np.linalg.norm(self.cur_ref_pos_agn - self.rt_formation_array, axis = 0)
        ###到自身目标轨迹点的距离做惩罚
        self.dist2target =  distance_array.mean()
        
        # reward_dis_from_target = - np.clip(self.dist2target*0.5,0,2.5)#*0.08#*10#15#40
        reward_dis_from_target = - np.clip( 2*(np.exp(self.dist2target*0.05)-1)/(np.exp(1*0.1)-1) ,0, 2)
        # reward_dis_from_target = 0


        self.agents_vel_array = np.abs(self.rt_formation_array - self.pre_rt_formation_array)/self.T_step
        # self.agents_acl_array = np.abs(self.agents_vel_array - self.pre_agents_vel_array)
        
        # reward_energy = -np.abs(self.normlized_action_dict).dot(np.array([1,1,1,1,1,1]))*0.05#0.2#0.2  #np.array([1,0.5,0.5,0.5,0.1,0.1]))*0.2
        self.VEL_BOUND = 2
        self.ACCEL_BOUND = 3
        # self.agents_vel_violation = np.clip(self.agents_vel_array - self.VEL_BOUND, 0, np.inf)
        # self.agents_acl_violation = np.clip(self.agents_acl_array - self.ACCEL_BOUND,0,np.inf)
        # total_agents_vel_violation = np.linalg.norm(self.agents_vel_violation, axis=0).sum()
        # b = np.linalg.norm(self.agents_acl_violation,axis=0).sum()
        # reward_limitation_violation = - np.clip((np.exp(total_agents_vel_violation/self.n_agents)-1),0,1.5)#-a/self.n_agents/2#-(a + 5*b)/self.n_agents/2


        # a = np.linalg.norm(self.agents_vel_array, axis=0).sum()
        # b = np.linalg.norm(self.agents_acl_array,axis=0).sum()
        reward_energy = 0#- a/self.n_agents/10#-(a + 20*b)/self.n_agents/2
        # reward_energy = -(a/3 + 5*b)/1.5#-(a + 5*b)/1.5
        
        #time punishment 
        reward_time = 0#-0.2#-0.1#-0.2
        # if (self.pre_dist2target - self.dist2target)>0:
        #     reward_time = -0.56+2*(self.pre_dist2target - self.dist2target)#-0.3#-0.3#0.4
        # else:
        #     reward_time = -0.56#-0.3

        reward_action_energy = 0#-np.abs(self.normlized_action_dict).dot(np.array([1,1,1,1,0.1,0.1]))*0.05#-np.abs(self.pre_normlized_action_dict-self.normlized_action_dict).dot(np.array([1,1,1,1,0.1,0.1]))*0.1


        # print("dist_affine_array:theta,shear,scalex,scaley,translation:",self.dist_affine_array)
        
        
        # if self.dist2target < self.on_circle_threshold:
        # import pdb;pdb.set_trace()
        
        ###时间结束并且到终点附近
        # if (self.affine_finish_threshold_list-self.dist_affine_array).min()>=0:
        if distance_array.max() < self.finish_threshold:
            reward_temp_finish = 2
        else:
            reward_temp_finish = 0


        if self.counter >= self.total_traj_steps-2 and distance_array.max() < self.finish_threshold*2:
            self.done_dict = 1 ##到点记录
            # if self.level>10:
            #     reward_finish = 200*2*4
            # else:
            #     reward_finish = 80#30#200*2#*4#200*8#5
            reward_finish =  100
        else:  
            reward_finish =  0



        ####障碍物碰撞检测和惩罚
        # flag_formation_collision_free = True        
        flag_formation_collision_free = self.formation_collision_free() 
        if flag_formation_collision_free: #and flag_inter_agent_collision_free:
            if self.closest_obs_dist>self.collision_free_threshold:
                reward_collision = 0
            else:
                # reward_collision = -100*(1- (self.closest_obs_dist - self.collision_threshold)/(self.collision_free_threshold - self.collision_threshold))*0.01#0.005
                reward_collision = -2.5*np.log(self.closest_obs_dist/self.collision_free_threshold) / np.log(self.collision_threshold/self.collision_free_threshold)#0.005
            
                if self.closest_obs_dist<1:
                    reward_dis_from_target = reward_dis_from_target/5
                    reward_temp_finish = 0
                else:
                    reward_dis_from_target = reward_dis_from_target/3
                    reward_temp_finish = reward_temp_finish/3

            
            self.info_dict["formation_col_free"] = True
        else:
            # if self.obstacle_collison_array.max()==1 or (self.closet_obstacle_distance.min() < self.collision_threshold and self.obstacle_collison_array.max()==0):
            # if flag_formation_collision_free:
            #     reward_collision = -0.3
            #     
            # else:
            if self.level>10:
                reward_collision = -100*2*4
            else:
                reward_collision = -80#-15#-100*2#*2#-100*2#-100*2#*4#-100*4
            
            self.done_dict = -1#碰撞障碍物记录
            
            self.info_dict["formation_col_free"] = False 
            # else:
            #     # if self.closet_obstacle_distance.min()< self.collision_free_threshold and self.CBF_flag==0 :
            #     #     reward_collision = -np.sum(15/self.closet_obstacle_distance)*0.15#0.05
            #     # reward_collision = -100*(1- (self.closet_obstacle_distance.min() - self.collision_threshold)/(self.collision_free_threshold - self.collision_threshold))*0.01#0.005

            #     reward_collision = -40*(1- (self.closet_obstacle_distance - self.collision_threshold)/(self.collision_free_threshold - self.collision_threshold)).sum()*0.01#0.005

        
        ###agent间相互避障
        flag_inter_agent_collision_free = self.inter_agent_collision_free(self.rt_formation_array, self.inter_agent_collision_threshold)
        self.info_dict["inter_col_free"] = flag_inter_agent_collision_free 
        if flag_inter_agent_collision_free==False:##punish the inter-agent collision
            reward_inter_collision = -80
            self.done_dict = -1
        else:
            reward_inter_collision = 0

        reward_dict = reward_dis_from_target +  reward_energy + reward_collision + reward_inter_collision + reward_finish + reward_temp_finish + reward_action_energy +reward_time 
        
        reward_dict = reward_dict/5.0

        self.pre_agents_vel_array = self.agents_vel_array.copy()
        self.pre_rt_formation_array = self.rt_formation_array.copy()
        self.pre_dist2target = self.dist2target

        self.pre_action_dict = self.action_dict
        # self.pre_normlized_action_dict = self.normlized_action_dict
        # self.pre_reward_collision = reward_collision

        print("step:",self.counter,"reward:",reward_dict,"reward_dis_from_target",reward_dis_from_target,"reward_inter_collision",reward_inter_collision,"reward_collision",reward_collision,"closest_obs_dist",self.closest_obs_dist,"reward_energy",reward_energy,"reward_action_energy",reward_action_energy,"reward_finish",reward_finish,"reward_temp_finish",reward_temp_finish,"reward_time",reward_time)
        return reward_dict


    def render(self):
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
            # self.coordinate, = ax.plot(self.rt_formation_coordinate[0, :], self.rt_formation_coordinate[1, :], 'bo',lw=1)  # Returns a tuple of line objects, thus the comma
            self.obstacle_points, = ax.plot(self.obstacle_state[:,0],self.obstacle_state[:,1], 'gs')
            self.formation_boundary, = ax.plot([], [], '-', color='b', lw=1)
            
            ###绘制轨迹
            for i in range(self.n_agents):
                ax.plot(self.trajs_for_following[:,0,i], self.trajs_for_following[:,1,i], 'b-', markersize=1)


            self.target_center, = ax.plot(self.target[0], self.target[1], 'gx')
            

            # for i in range(self.target_formation_array.shape[0]):
            # self.target_boundary, = ax.plot([], [], '-', color='y', lw=1)
            # self.target_boundary.set_xdata(np.append(self.target_formation_array[i,0,:],[self.target_formation_array[i,0,0]]))
            # self.target_boundary.set_ydata(np.append(self.target_formation_array[i,1,:],[self.target_formation_array[i,1,0]]))
            ax.plot(np.append(self.target_formation_array[0,:],[self.target_formation_array[0,0]]), np.append(self.target_formation_array[1,:],[self.target_formation_array[1,0]]), '-', color='y', lw=1)
            
            self.display_step = ax.annotate(str(self.counter),(self.r_max,self.r_max))
            self.display_finish = ax.annotate(str(self.done_dict),(self.r_max/2,self.r_max))
            

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
        # self.coordinate.set_xdata(self.rt_formation_coordinate[0, :])
        # self.coordinate.set_ydata(self.rt_formation_coordinate[1, :])
        self.formation_boundary.set_xdata(np.append(self.rt_formation_array[0,:],[self.rt_formation_array[0,0]]))
        self.formation_boundary.set_ydata(np.append(self.rt_formation_array[1,:],[self.rt_formation_array[1,0]]))
        
        self.obstacle_points.set_xdata(self.obstacle_state[:,0])
        self.obstacle_points.set_ydata(self.obstacle_state[:,1])
            
        # ax.plot(self.rt_formation_array[0, 0], self.rt_formation_array[1, 0], 'ro', markersize=2)
        # ax.plot(self.rt_formation_array[0, 1], self.rt_formation_array[1, 1], 'go', markersize=2)
        # ax.plot(self.rt_formation_array[0, 2], self.rt_formation_array[1, 2], 'bo', markersize=2)
        # ax.plot(self.rt_formation_array[0, 3], self.rt_formation_array[1, 3], 'yo', markersize=2)
        ax.plot(self.center[0], self.center[1], marker="o",color="black", markersize=2)
        color_list = ['ro','go','bo','yo','co','mo','ro','go','bo','yo','co','mo','ko']   
        for i in range(self.n_agents):
            ax.plot(self.rt_formation_array[0, i], self.rt_formation_array[1, i], color_list[i], markersize=2)

        self.target_center.set_xdata(self.target[0])
        self.target_center.set_ydata(self.target[1])

        self.display_step.remove()
        self.display_step = ax.annotate(str(self.counter),(self.r_max/2,self.r_max/2))

        self.display_finish.remove()
        self.display_finish = ax.annotate(str(self.done_dict),(self.r_max/2,self.r_max/4))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass

    def RRT_init(self,obstacle_inter, obstacle_list, path, scene_kind):
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
        # obs_images = np.expand_dims(output_image,0)



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

        # print("diff_theta",diff_theta)
        # ##把实时队形配置和初始模板的vector放进去
        obs_vector = np.hstack((np.sin(diff_theta/2),np.cos(diff_theta/2),(self.target_affine_param[1:-2]-self.state_dict[1:-2])/(self.max_state[1:-2]-self.min_state[1:-2]),temp_target[:-2]/6))
        # import pdb;pdb.set_trace()
        obs_graph = np.hstack((temp_formation_array,self.init_formation_array.T))
        
        obs_images = np.expand_dims(self.output_image,0)
             
        return obs_vector,obs_graph,obs_images

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
            plt.ylim(-self.r_max,10 )
            plt.xlim(-self.r_max,10 )

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

        color_list = ['ro','go','bo','yo','co','mo','ko']    
        for i in range(self.n_agents):
            ax.plot(self.rt_formation_array[0, i], self.rt_formation_array[1, i], color_list[i], markersize=2)
        # ax.plot(self.rt_formation_array[0, 1], self.rt_formation_array[1, 1], 'go', markersize=2)
        # ax.plot(self.rt_formation_array[0, 2], self.rt_formation_array[1, 2], 'bo', markersize=2)
        # ax.plot(self.rt_formation_array[0, 3], self.rt_formation_array[1, 3], 'yo', markersize=2)

        self.target_center.set_xdata(self.target[0])
        self.target_center.set_ydata(self.target[1])

        self.display_step.remove()
        self.display_step = ax.annotate(str(self.counter),(self.r_max,self.r_max))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        

    def render_save(self):
        self.save_path  =  "../output/rrt_affine_trajectory.png"
        plt.savefig(self.save_path) 



