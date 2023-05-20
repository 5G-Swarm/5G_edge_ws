import math
import numpy as np
from numpy import random
import os
from sys import platform
import time
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary,ImageType

from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
from sklearn.neighbors import NearestNeighbors
import pybullet as p
import pybullet_data

class Formation_containment(BaseMultiagentAviary):
    """Multi-agent RL problem: flocking."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=10,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=True,#False
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.VEL):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        # initial_xyzs = np.zeros((num_drones+1,3))
        # team_center = [np.random.uniform(-0.08,0.08),np.random.uniform(-0.08,0.08)]
        # for i in range(num_drones):
        #     initial_xyzs[i,0] = np.random.uniform(-0.1,0.1)+team_center[0]
        #     initial_xyzs[i,1] = np.random.uniform(-0.1,0.1)+team_center[1]
        #     initial_xyzs[i,2] = 0.05
        


        super().__init__(drone_model=drone_model,
                         num_drones=num_drones+1,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )
        self.n_agents = num_drones#定义为，agent数量， 而self.NUM_DRONES 则包括了target数量在内
        self.target_center = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),1.0,0.0,0.0,0.0])#np.array([0,0,1])
        self.radius = 1
        self.counter = 0
        self.n_features = 2
        self.degree = 2

        self.state_drones = np.zeros((self.n_agents,3))
        self.obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))
        self.pre_obs_dict = np.zeros((self.n_agents,(1+self.degree)*self.n_features))
        
        self.ag_threshold = 2*self.radius*math.sin(math.pi/self.n_agents)
        # import pdb;pdb.set_trace()
        self.collision_threshold = self.COLLISION_R*2
        self.MAX_LIN_VEL_XY = 3 
        self.MAX_LIN_VEL_Z = 1
        



    ###############################################################
    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.last_action = -1*np.ones((self.NUM_DRONES, 4))
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.INIT_XYZS = np.vstack([np.zeros(self.NUM_DRONES), \
                                        np.array([y*8*self.L for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * 0.25]).transpose().reshape(self.NUM_DRONES, 3)
        
        if random.randint(2):
            self.INIT_XYZS[-1,:] =  np.array([np.random.uniform(1.5,2),np.random.uniform(8*self.L*self.n_agents ,0),0.25])
        else:
            self.INIT_XYZS[-1,:] =  np.array([np.random.uniform(-1.5,-2),np.random.uniform(8*self.L*self.n_agents ,0),0.25])

        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        self.DRONE_IDS =[p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../../assets/"+self.URDF,
                                              self.INIT_XYZS[i,:],
                                              p.getQuaternionFromEuler(self.INIT_RPYS[i,:]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES-1)]
        temp = DroneModel.CF2P
        target_URDF = temp.value + ".urdf"

        self.DRONE_IDS.append(p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../../assets/"+target_URDF,
                                              self.INIT_XYZS[-1,:],
                                              p.getQuaternionFromEuler(self.INIT_RPYS[-1,:]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ))
        self.DRONE_IDS = np.array(self.DRONE_IDS)
        for i in range(self.NUM_DRONES):
            #### Show the frame of reference of the drone, note that ###
            #### It severly slows down the GUI #########################
            if self.GUI and self.USER_DEBUG:
                self._showDroneLocalAxes(i)
            #### Disable collisions between drones' and the ground plane
            #### E.g., to start a drone at [0,0,0] #####################
            # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles()



    ##############################################################################

    def reset(self):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.

        """
        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        self.counter = 0
        self.center_speed_x = float(np.random.uniform(-3,3,1))
        self.center_speed_y = float(np.random.uniform(-3,3,1))

        #init reward:
        self.dist_list = np.zeros(self.n_agents)
        self.pre_dist_list = np.zeros(self.n_agents)
        self.dist_agn2tar_list = np.zeros(self.n_agents)
        self.energy_list  = np.zeros(self.n_agents)
        self.done_dict = np.zeros(self.n_agents)
        self.on_circle_time = np.zeros(self.n_agents)
        

        return self._computeObs()

    ################################################################################
    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
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
                cost_dist[i] = (self.dist_list[i]-self.pre_dist_list[i])*100#15

            if self.dist_list[i]<self.collision_threshold:#0.25
                self.on_circle_time[i] -= 1
                # self.on_circle_time[i] = -10#-5
            else:
                self.on_circle_time[i] = 0#5
            
            cost_on_circle[i] = self.on_circle_time[i]*0.1
            temp_dist_1 = np.linalg.norm(self.obs_dict[i,:2])
            if temp_dist_1<self.collision_threshold:
                cost_dist_agn2tar[i] = 50 
                self.done_dict[i] = 1
            elif temp_dist_1<self.radius*0.5:
                cost_dist_agn2tar[i] = 0.1/(temp_dist_1+0.01)#0.07 
            else: 
                cost_dist_agn2tar[i] = 0
                # import pdb;pdb.set_trace() 
                
            #[1/(dist+0.01) if dist<self.ag_threshold else 0 for dist in self.adj_distance_matrix.reshape(self.NUM_DRONES*self.degree)]
            cost_energy[i] = np.linalg.norm(self.u[i])*0.3
            temp_dist_list = []

            for temp_dist_2 in list(self.adj_distance_matrix[i]):
                if temp_dist_2>self.ag_threshold*1.5:#self.radius*2#self.ag_threshold*2:#*1.5
                    k = 0
                else:
                    k = 2.5#2.5
                if temp_dist_2>self.collision_threshold:#0.5
                    temp_dist_list.append(k/(temp_dist_2*temp_dist_2+0.01)) 
                else:
                    # import pdb;pdb.set_trace() 
                    temp_dist_list.append(200)
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



        #cost_accumulate_dist = 0#np.sum(self.dist_list)*0.3/self.NUM_DRONES
        print("adj_distance_matrix",self.adj_distance_matrix)


        # angle_list.sort()
        # next_angle_list = angle_list[1:] + [angle_list[0]+360]
        # self.distribution_std = np.std(np.array(next_angle_list)-np.array(angle_list))*0.1
        # if  self.counter>1:#np.sum(self.dist_list) <(0.25*self.NUM_DRONES) and 
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

##############################################
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Dict({i: spaces.Box(low=0,
                                              high=255,
                                              shape=(self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8
                                              ) for i in range(self.NUM_DRONES)})
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
            # obs_lower_bound = np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
            # obs_upper_bound = np.array([1,       1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])          
            # return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )
            ############################################################
            #### OBS SPACE OF SIZE 12
            return spaces.Dict({i: spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1]),
                                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1]),
                                              dtype=np.float32
                                              ) for i in range(self.NUM_DRONES)})
            ############################################################
        else:
            print("[ERROR] in BaseMultiagentAviary._observationSpace()")
    
    ################################################################################
    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        self.counter = self.counter+1 
        self.pre_obs_dict[:] = self.obs_dict[:]
        ############################################################
        #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
        # return {   i   : self._clipAndNormalizeState(self._getDroneStateVector(i)) for i in range(self.NUM_DRONES) }
        ############################################################
        #### OBS SPACE OF SIZE 12
        
        obs = self._getDroneStateVector(self.NUM_DRONES-1)
        self.target_center[:3] = obs[:3]
        # self.target_center[:2] = -0.2*np.ones(2)#np.zeros(2) 

        for i in range(self.n_agents):
            # obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
            obs = self._getDroneStateVector(i)
            self.state_drones[i, :3] = obs[:3]
        # import pdb;pdb.set_trace()
        adj_matrix = self.get_connectivity(self.state_drones[:,:self.n_features])
        for i in range(self.n_agents):
            temp = list(self.target_center[:self.n_features] - self.state_drones[i,:self.n_features])

            a = list(np.argwhere(adj_matrix[i]==1.0).squeeze())
            
            list0 = list((self.state_drones[a[0],:self.n_features] - self.state_drones[i,:self.n_features]).squeeze())
            list1 = list((self.state_drones[a[1],:self.n_features] - self.state_drones[i,:self.n_features]).squeeze())
            
            dist0 = np.linalg.norm((self.state_drones[a[0],:self.n_features] - self.state_drones[i,:self.n_features]).squeeze())
            dist1 = np.linalg.norm((self.state_drones[a[1],:self.n_features] - self.state_drones[i,:self.n_features]).squeeze())
            
            if dist0<=dist1: 
                temp = temp + list0
                temp = temp + list1
            else:
                temp = temp + list1
                temp = temp + list0

            self.obs_dict[i] = np.array(temp)
            # self.obs_err[i, :] = obs[0:3]-self.target_center[:3]#np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
        
        print("obs_dict",self.obs_dict)
        return np.hstack((self.obs_dict,self.pre_obs_dict))#*3#
 
    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and 
            one additional boolean value for key "__all__".

        """
        # bool_val = True if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        # done = {i: bool_val for i in range(self.n_agents)}
        # done["__all__"] = True if True in done.values() else False
        # done = False
        print("done_dict:",self.done_dict)

        return self.done_dict

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.n_agents)}

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
    #######################################
    
    def dist2_mat(self, x):

        x_loc = np.reshape(x[:, 0:2], (self.NUM_DRONES,2,1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net
    
    def get_connectivity(self, x):
        # import pdb;pdb.set_trace()
        if self.degree == 0:
            a_net = self.dist2_mat(x)
            # a_net = (a_net < self.comm_radius2).astype(float)
            self.adj_distance_matrix = self._getAdjacencyMatrix()
        else:
            neigh = NearestNeighbors(n_neighbors=self.degree)
            neigh.fit(x)
            # neigh.fit(x[:,2:4])
            self.adj_distance_matrix,_ = neigh.kneighbors()
            a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())

            # import pdb;pdb.set_trace()

        # if self.mean_pooling:
        #     # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        #     n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.NUM_DRONES,1)) # TODO or axis=0? Is the mean in the correct direction?
        #     n_neighbors[n_neighbors == 0] = 1
        #     a_net = a_net / n_neighbors 

        return a_net


#############################################
    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : dict[str, ndarray]
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """

        rpm = np.zeros((self.NUM_DRONES,4))
        self.u = action
        self.target_center[3] = self.center_speed_x*math.sin(self.counter/90)
        self.target_center[4] = self.center_speed_y*math.cos(self.counter/90)
        self.target_center[5] = 0.01
        
        action[self.NUM_DRONES-1] = self.target_center[3:] #target的速度也融入action指令中
        # import pdb;pdb.set_trace()
        print("action:",action)
        for k, v in action.items():
            state = self._getDroneStateVector(int(k))
            rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * v # target the desired velocity vector
                                                        ) 
            # import pdb;pdb.set_trace()
            # self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
            #                                         cur_pos=state[0:3],
            #                                         cur_quat=state[3:7],
            #                                         cur_vel=state[10:13],
            #                                         cur_ang_vel=state[13:16],
            #                                         target_pos=state[0:3]+0.1*v
            #                                         )
            rpm[int(k),:] = rpm_k
        
        return rpm