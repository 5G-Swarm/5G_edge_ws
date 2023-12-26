from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped,Pose,Twist,TwistStamped
from std_msgs.msg import Float64MultiArray,Header   
from autoware_msgs.msg import DroneSyn
import rospy
import math
import numpy as np
from rtree import index
import uuid
from pyquaternion import Quaternion
import sys
from gazebo_msgs.srv import *
from scipy.optimize import linear_sum_assignment
import torch
from torch_geometric.data.batch import Batch

sys.path.append("..")
##########################
from gym_formation_particle.gym_flock.envs.env_affine_formation_plan.Img_rrtobs_formation_plan_env import Img_rrtobs_formationPlanEnv as affine_formation 


import rllib
from utils.real_traj_follow_formation_env import trajFollowFormationEnv

from rl_code.real_matd3_gnn_formation_plan import MATD3_DEC as MA_Method 
from rl_code.args import EnvParams,generate_args

from real_controller.utils.Experience_Graph import Experience_Graph,generate_ring_graph,generate_fully_connected_graph,data_graph_to_cuda

ma_load_path = "../rl_code/results/affine_formation/distributed_traj_follow/delay_system/saved_models"
ma_model_num = -1

class Trajectory_manager(object):

    def __init__(self,down_sampling_rate,num_agent,real_flag,init_total_waypoints=[],init_template=[], flag_use_rtk_sta_obs=False):
        
        ####task relevant param
        self.dynamic_obs_index_list = []
        self.static_obs_index_list = []
        self.drone_index_list = [0,1,2,3]#12,13,3,4,5
        self.VEL_LIMIT = 1
        self.traj_step_jump = 20
        
        
        self.total_rtk_index_list = self.static_obs_index_list + self.dynamic_obs_index_list + self.drone_index_list 
        self.formation_origin = np.array([0,0,3.0])   
        self.real_flag = real_flag


        self.down_sampling_rate = down_sampling_rate
        self.num_agent = num_agent
        self.init_total_waypoints = init_total_waypoints.copy()
        self.current_total_waypoints = init_total_waypoints.copy()#np.zeros((int((init_total_waypoints.shape[0]-1)/self.down_sampling_rate)+1,2,self.num_agent+1))#init_total_waypoints.copy()
        self.optimized_affine_param = []#init_affine_param
        self.optimized_total_waypoints = [] 
        self.init_template = init_template.copy()
        self.traj_record = []
        self.accumulate_real_pos_err = np.zeros(self.num_agent)
        # for i in range(self.num_agent):
        #     self.traj_record.append([])

        self.vehicle_type = "solo"#"iris"

        #####
        self.traj_step = 0
        self.stage = "Prepare"#"Prepare","Init_formation","Formation_navigation","Target_formation"
        self.traj2drone_id_map = []#np.zeros(self.num_agent)

        
        self.traj_cmd_publisher_list = {}
        self.pose_sub_list = {}
        self.goal_publisher_list = {}
        self.traj_follow_err_publisher_list = {}
        self.drone_state = {}#np.zeros((self.num_agent,7))
        for i in range(len(self.total_rtk_index_list)):
            drone_id = self.total_rtk_index_list[i]
            # self.traj_cmd_publisher_list.append(rospy.Publisher("/xtdrone/"+self.vehicle_type+'_'+str(i)+"/cmd_pose_enu", Pose, queue_size = 10)) 
            if self.real_flag:
                self.traj_cmd_publisher = rospy.Publisher("/teamrl_controller_vel", TwistStamped, queue_size = 10)
                self.pose_sub_list[drone_id] = rospy.Subscriber("/drone_state_"+str(drone_id), DroneSyn, self.callback_drone_state_real,(drone_id),queue_size=1)            
            else:
                self.traj_cmd_publisher_list[drone_id] = rospy.Publisher("/xtdrone/"+self.vehicle_type+'_'+str(drone_id)+"/cmd_vel_enu", Twist, queue_size = 10)
                self.pose_sub_list[drone_id] = rospy.Subscriber(self.vehicle_type+'_'+str(drone_id)+"/mavros/local_position/pose", PoseStamped, self.callback_drone_state_sim,(drone_id),queue_size=1)

            self.traj_follow_err_publisher_list[drone_id] = rospy.Publisher("/xtdrone/"+self.vehicle_type+'_'+str(drone_id)+"/pos_err", PoseStamped, queue_size = 10) 
            self.goal_publisher_list[drone_id] = rospy.Publisher(self.vehicle_type+"_"+str(drone_id)+"/move_base_simple/goal", PoseStamped, queue_size=1)
            

        #####################new
        self.two_points_static_obs_list = []
        self.flag_use_rtk_sta_obs = flag_use_rtk_sta_obs

        self.num_dyn_obs = len(self.dynamic_obs_index_list)
        
        # if self.two_points_static_obs_list is not None:
        #     self.num_sta_obs = len(self.two_points_static_obs_list)
        # else:
        #     self.num_sta_obs = len(self.static_obs_index_list)

        self.dummy_env = trajFollowFormationEnv()
        self.load_model(ma_load_path, ma_model_num)
        self.finish_threshold = 0.5

            

    ###########################################################################################
    def load_model(self, ma_load_path, ma_model_num):
        self.ma_config = rllib.basic.YamlConfig()
        args = generate_args()
        self.ma_config.update(args)
        self.ma_config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.ma_config.set("model_dir",ma_load_path)
        self.ma_config.set("model_num",ma_model_num)
        self.ma_config.set("restore",False)
        self.ma_config.set("test_flag",True)
        self.ma_config.env = "Target_FormationPlanEnv-v0"#"Target_FormationPlanEnv-v0"#"Test_FormationPlanEnv-v0"#'Relative_affine_FormationPlanEnv-v0'#"FormationPlanEnv-v0"#		
        self.ma_config.set("dim_state",self.dummy_env.node_feature_dim)
        self.ma_config.set("dim_action",self.dummy_env.dim_action)
        self.ma_config.n_agents = self.dummy_env.max_agents
        ma_model_name = "test"
        self.ma_writer = rllib.basic.create_dir(self.ma_config, ma_model_name)

        self.ma_method = MA_Method(self.ma_config, self.ma_writer)
        self.ma_method.models_to_load = [self.ma_method.actor,self.ma_method.critic]
        self.ma_method._load_model()

    def marl_distributed_reset(self):
        update_drone_state_array = np.zeros((self.num_agent,6))
        update_dyn_obs_state_array = np.zeros((self.num_dyn_obs,6))
        for i in range(self.num_agent):
            drone_id = self.traj2drone_id_map[i]
            update_drone_state_array[i,:] =  self.drone_state[drone_id][:-1]

        for i in range(self.num_dyn_obs):
            drone_id = self.dynamic_obs_index_list[i]
            update_dyn_obs_state_array[i,:] = self.drone_state[drone_id][:-1]



        self.dummy_env.reset_dummy_env(self.formation_origin,update_drone_state_array, update_dyn_obs_state_array, self.two_points_static_obs_list)
        

    def marl_distributed_controller(self, step, optimized_total_waypoints):
        final_action_dict = np.zeros((self.num_agent,3))
        update_drone_state_array = np.zeros((self.num_agent,6))
        update_dyn_obs_state_array = np.zeros((self.num_dyn_obs,6))

        cur_ref_pos_agn = optimized_total_waypoints[:,step,:3] + self.formation_origin

        next_ref_pos_agn = []
        for i in range(self.dummy_env.future_ref_frame):
            # if self.counter+1+i < self.total_traj_steps-1:
            next_ref_pos_agn.append( (optimized_total_waypoints[:self.num_agent, min(step+(1+i)*self.traj_step_jump, optimized_total_waypoints.shape[1]-1), :2]+self.formation_origin[:2]).T )



        for i in range(self.num_agent):
            drone_id = self.traj2drone_id_map[i]
            update_drone_state_array[i,:] =  self.drone_state[drone_id][:-1]

        for i in range(self.num_dyn_obs):
            drone_id = self.dynamic_obs_index_list[i]
            update_dyn_obs_state_array[i,:] = self.drone_state[drone_id][:-1]
        
        ###参考轨迹本身的顺序就是和dummy env中的agent编号一致
        xy_cur_ref_pos_agn = cur_ref_pos_agn[:,:2]
        z_cur_ref_pos_agn = cur_ref_pos_agn[:,-1]
        # for i in range(self.num_agent):
        #     z_cur_ref_pos_agn[i] += i 

        obs_dict = self.dummy_env.update_dummy_env(update_drone_state_array, update_dyn_obs_state_array, xy_cur_ref_pos_agn.T, next_ref_pos_agn)
        dec_state_graph = Batch.from_data_list([data_graph_to_cuda(local_graph) for local_graph in obs_dict[0]])
        dec_state_img = obs_dict[1].cuda()
        dec_state = (dec_state_graph, dec_state_img, [self.num_agent])

        action_dict = self.ma_method.actor(dec_state)
        action_dict = action_dict.cpu().detach().numpy().squeeze()

        final_action_dict[:,:2] = self.dummy_env.preprocess(action_dict).T

        ##z轴
        kp = 0.6
        final_action_dict[:,-1] = kp*(z_cur_ref_pos_agn - update_drone_state_array[:,2]) 

        return np.clip(final_action_dict,-self.VEL_LIMIT,self.VEL_LIMIT)

    ################################################################

    def get_crossing(self,s1,s2):
        xa,ya = s1[0],s1[1]
        xb,yb = s1[2],s1[3]
        xc,yc = s2[0],s2[1]
        xd,yd = s2[2],s2[3]
        #判断两条直线是否相交，矩阵行列式计算
        a = np.matrix(
            [
                [xb-xa,-(xd-xc)],
                [yb-ya,-(yd-yc)]
            ]
        )
        delta = np.linalg.det(a)
        #不相交,返回两线段
        if np.fabs(delta) < 1e-6:
            print(delta)
            return None        
        #求两个参数lambda和miu
        c = np.matrix(
            [
                [xc-xa,-(xd-xc)],
                [yc-ya,-(yd-yc)]
            ]
        )
        d = np.matrix(
            [
                [xb-xa,xc-xa],
                [yb-ya,yc-ya]
            ]
        )
        lamb = np.linalg.det(c)/delta
        miu = np.linalg.det(d)/delta
        #相交
        if lamb <= 1 and lamb >= 0 and miu >= 0 and miu <= 1:
            x = xc + miu*(xd-xc)
            y = yc + miu*(yd-yc)
            return np.array([x,y])
        #相交在延长线上
        else:
            return None

    def obstacle_generator(self,obstacles):
        """
        Add obstacles to r-tree
        :param obstacles: list of obstacles
        """
        self.two_points_static_obs_list = obstacles.copy() 
        self.num_sta_obs = len(self.two_points_static_obs_list)
        for i in range(len(obstacles)):
            obstacle = obstacles[i]
            # obstacle[:2] = obstacle[:2]-0.3
            # obstacle[-2:] = obstacle[-2:]+0.3
            yield (i, obstacle, obstacle)


    def pub_front_end_traj(self,frontend_publisher,front_end_traj,frame_id):
        msg = Path()
        msg.header.frame_id = frame_id
        msg.header.stamp =  rospy.Time.now()
        # if(frame_id=="center"):    
        #     id = -1
        # else:
        id = int(frame_id[-1])
            
        for i in range(0,front_end_traj.shape[0],self.down_sampling_rate):
            pose = PoseStamped()
            pose.pose.position.x = front_end_traj[i,0]
            pose.pose.position.y = front_end_traj[i,1]
            pose.pose.position.z = 0#front_end_traj["time_series"][i]
            msg.poses.append(pose)     
            # self.current_total_waypoints[int(i/self.down_sampling_rate),:,id] = front_end_traj[i,:]
        if (front_end_traj.shape[0]-1)%self.down_sampling_rate != 0:
            pose = PoseStamped()
            pose.pose.position.x = front_end_traj[-1,0]
            pose.pose.position.y = front_end_traj[-1,1]
            pose.pose.position.z = 0#front_end_traj["time_series"][i]
            msg.poses.append(pose)  

        frontend_publisher.publish(msg)
        print("publish front_end_traj of "+frame_id,int((front_end_traj.shape[0]-1)/self.down_sampling_rate)+1)

    def pub_target_affine_param(self,publisher,target_affine_param):
        pose = PoseStamped()
        pose.pose.orientation.x = target_affine_param[0]
        pose.pose.orientation.y = target_affine_param[1]
        pose.pose.orientation.z = target_affine_param[2]
        pose.pose.orientation.w = target_affine_param[3]
         
        publisher.publish(pose)
        # print("publish target_affine_param ")

    def pub_front_end_affine_param(self,frontend_publisher,front_end_traj,frame_id):
        msg = Path()
        msg.header.frame_id = frame_id
        msg.header.stamp =  rospy.Time.now()
        # if(frame_id=="center"):    
        #     id = -1
        # else:
        #     id = int(frame_id[-1])
            
        for i in range(0,front_end_traj.shape[0],self.down_sampling_rate):
            pose = PoseStamped()
            pose.pose.position.x = front_end_traj[i,0]
            pose.pose.position.y = front_end_traj[i,1]
            pose.pose.position.z = front_end_traj[i,2]#front_end_traj["time_series"][i]
            pose.pose.orientation.x = front_end_traj[i,3]
            pose.pose.orientation.y = front_end_traj[i,4]
            pose.pose.orientation.z = front_end_traj[i,5]
            pose.pose.orientation.w = self.down_sampling_rate#传递降采样率到优化器
            msg.poses.append(pose)

        if (front_end_traj.shape[0]-1)/self.down_sampling_rate != 0:
            pose = PoseStamped()
            pose.pose.position.x = front_end_traj[-1,0]
            pose.pose.position.y = front_end_traj[-1,1]
            pose.pose.position.z = front_end_traj[-1,2]#front_end_traj["time_series"][i]
            pose.pose.orientation.x = front_end_traj[-1,3]
            pose.pose.orientation.y = front_end_traj[-1,4]
            pose.pose.orientation.z = front_end_traj[-1,5]
            pose.pose.orientation.w = self.down_sampling_rate#传递降采样率到优化器
            msg.poses.append(pose)  
        
        frontend_publisher.publish(msg)
        print("publish front_end_affine_param ",int((front_end_traj.shape[0]-1)/self.down_sampling_rate)+1)

    def pub_basepoint_direction(self,publisher,base_point_array,direction_array,frame_id):
        msg = Path()
        msg.header.frame_id = frame_id
        msg.header.stamp =  rospy.Time.now()
        # if(frame_id=="center"):    
        #     id = -1
        # else:
        #     id = int(frame_id[-1])
            
        for i in range(base_point_array.shape[0]):
            pose = PoseStamped()
            pose.pose.position.x = base_point_array[i,0]
            pose.pose.position.y = base_point_array[i,1]
            pose.pose.position.z = 0
            pose.pose.orientation.x = direction_array[i,0]
            pose.pose.orientation.y = direction_array[i,0]
            pose.pose.orientation.z = 0

            msg.poses.append(pose)     
        publisher.publish(msg)
        print("publish basepoint_direction of "+frame_id)

    def pub_init_template(self,publisher):
        msg = Path()
        msg.header.frame_id = "init_template"
        msg.header.stamp =  rospy.Time.now()
        # if(frame_id=="center"):    
        #     id = -1
        # else:
        #     id = int(frame_id[-1])
            
        for i in range(self.init_template.shape[1]):
            pose = PoseStamped()
            pose.pose.position.x = self.init_template[0,i]
            pose.pose.position.y = self.init_template[1,i]
            msg.poses.append(pose)     
        publisher.publish(msg)
        print("publish init_template")

    def callback_optimized_affine_param(self,msg):
        global final_cost
        temp = []
        num_step = len(msg.poses)#int(len(msg.poses)/3)
        if msg.header.frame_id =="sparse":
            # import pdb;pdb.set_trace()
            self.current_total_waypoints = np.zeros((num_step,2,num_agent+1)) 
            for i in range(num_step):
                # if i >= self.current_total_waypoints.shape[0]:
                #     break
                pos = msg.poses[i].pose.position
                ori = msg.poses[i].pose.orientation
                # vel = msg.poses[3*i+1].pose.position
                # acc = msg.poses[3*i+2].pose.position
                cur_affine = [pos.x,pos.y,pos.z,ori.x,ori.y,ori.z]
                # print("shahao",cur_affine)
                temp.append(cur_affine)
                temp_formation_array = affine_formation.relative_rt_formation_array_calculation(self.init_template, np.array(cur_affine))
                self.current_total_waypoints[i,:,:-1] = temp_formation_array.copy()
                self.current_total_waypoints[i,:,-1] = cur_affine[-2:].copy()

            self.optimized_affine_param  = np.array(temp)
        else:
            self.optimized_total_waypoints = np.ones((num_agent,num_step,3)) 
            for i in range(num_step):
                pos = msg.poses[i].pose.position
                ori = msg.poses[i].pose.orientation
                # vel = msg.poses[3*i+1].pose.position
                # acc = msg.poses[3*i+2].pose.position
                cur_affine = [pos.x,pos.y,pos.z,ori.x,ori.y,ori.z]
                # print("shahao",cur_affine)
                temp.append(cur_affine)
                temp_formation_array = affine_formation.relative_rt_formation_array_calculation(self.init_template, np.array(cur_affine))
                self.optimized_total_waypoints[:,i,:2] = temp_formation_array.T.copy()
                # self.optimized_total_waypoints[-1,i,:2] = cur_affine[-2:].copy()
            final_cost = msg.poses[0].pose.orientation.w      
        
        print("recieve optimized_cps of "+msg.header.frame_id,len(temp))

    def check_center_collision(self, obs_checker,center_cps):
        for i in range(0,center_cps.shape[0]):
            # import pdb;pdb.set_trace()
            cp = center_cps[i,:2]#np.array(center_cps[i][:2]) #control_point
            collision_list = list(obs_checker.intersection(np.hstack((cp,cp)),objects=True))
            if collision_list != []:
                print("###oh my god, center is in collision,optimization will definitely fail!!",i,cp)
                return True
        
        return False
    
    def generate_basepts_direction(self,obs_checker,init_cps,center_cps):
        base_point_array = np.ones((len(init_cps),2))*2333#代表约定无效值
        direction_array = np.ones((len(init_cps),2))*2333#代表约定无效值
        # collision_time_id_list = []
        for i in range(0,len(init_cps)):
            cp = np.array(init_cps[i][:2]) #control_point
            gp = np.array(center_cps[i][:2])#guidance_point
            
            collision_list = list(obs_checker.intersection(np.hstack((cp,cp)),objects=True))
            if collision_list == []:
                continue
            # collision_time_id_list

            vertex = np.zeros((4,2))
            vertex[0,:] = np.array(collision_list[0].bbox[:2])
            vertex[1,:] = np.array([collision_list[0].bbox[0],collision_list[0].bbox[3]])
            vertex[2,:] = np.array(collision_list[0].bbox[-2:])
            vertex[3,:] = np.array([collision_list[0].bbox[2],collision_list[0].bbox[1]])

            guidance_vec = gp-cp
            if(i==0):
                tagent_vec = np.array(init_cps[i+1][:2]) - cp
            elif(i==len(init_cps)-1):
                tagent_vec = cp - np.array(init_cps[i-1][:2]) 
            else:
                tagent_vec = np.array(init_cps[i+1][:2]) - np.array(init_cps[i-1][:2]) 

            R = np.array([[np.cos(np.pi/2),-np.sin(np.pi/2)],[np.sin(np.pi/2),np.cos(np.pi/2)]])
            direction = R.dot(tagent_vec/np.linalg.norm(tagent_vec))
            if guidance_vec.dot(direction)<0:
                direction = -direction

            line_1 = np.hstack((cp,cp+direction* np.linalg.norm(vertex[2]-vertex[0])))
            
            for j in range(4):
                if j==3:
                    line_2 = np.hstack((vertex[0],vertex[j]))
                else:
                    # import pdb;pdb.set_trace()
                    line_2 = np.hstack((vertex[j+1],vertex[j]))
                base_point = self.get_crossing(line_1,line_2)
                if base_point is not None:
                    break
            # if direction[0] and direction[1]:
            #     if direction[0]>0:
            #         if direction[1]>0:
                        
            #         else:

            #     else:


            # else:
            #     if direction[0]==0:
            #         base_point = np.array([cp[0],obs_min_corner[1] if direction[1]>0 else obs_max_corner[1]])
            #     if direction[1]==0:
            #         base_point = np.array([obs_min_corner[0] if direction[0]>0 else obs_max_corner[0],cp[1]])
            direction_array[i,:] = direction
            base_point_array[i,:] = base_point
            print("finish calc base_point and direction of"+str(i),base_point,direction)
            
        
        return base_point_array,direction_array
            
    def find_and_publish_near_obstacles(self,obs_checker,publisher):
        find_range = 3
        msg = Path()
        # msg.header.frame_id = frame_id
        msg.header.stamp =  rospy.Time.now()
        
        for i in range(self.current_total_waypoints.shape[0]):
            near_obstacle_list = []
            for j in range(num_agent):
                point = self.current_total_waypoints[i,:,j]
                near_obstacle_list += list(obs_checker.intersection(np.hstack((point,point)),objects=True))

            # cp = np.array(center_cps[i][:2]) #control_point
            
            # obstacle_list = list(obs_checker.intersection(np.hstack((cp-find_range,cp+find_range)),objects=True))
            
            if near_obstacle_list == []:
                continue
            # import pdb;pdb.set_trace()
            for obstacle in near_obstacle_list:
                pose = PoseStamped()
                pose.header.frame_id = str(i)
                pose.pose.orientation.x = np.array(obstacle.bbox[:2])[0]
                pose.pose.orientation.y = np.array(obstacle.bbox[:2])[1]
                pose.pose.orientation.z = np.array(obstacle.bbox[-2:])[0]
                pose.pose.orientation.w = np.array(obstacle.bbox[-2:])[1]

                msg.poses.append(pose)  

        publisher.publish(msg)
        print("totally have collision obstacles:",len(msg.poses))
        # return base_point_array,direction_array
            
    def pub_map_obstacles(self,obs_list,publisher):
        height = 10
        msg = Path()
        # msg.header.frame_id = frame_id
        # msg.header.stamp =  rospy.Time.now()
    
        for obstacle in obs_list:
            pose = PoseStamped()
            # pose.header.frame_id = str(i)
            pose.pose.position.x = obstacle[0]+self.formation_origin[0]
            pose.pose.position.y = obstacle[1]+self.formation_origin[1]
            pose.pose.position.z = 0
            pose.pose.orientation.x = obstacle[2]+self.formation_origin[0]
            pose.pose.orientation.y = obstacle[3]+self.formation_origin[1]
            pose.pose.orientation.z = height
            msg.poses.append(pose)  

        publisher.publish(msg)
        print("publish map obstacles:",len(msg.poses))

    # def callback_control_loop(self,event):
    #     if self.optimized_total_waypoints==[]:return
    #     if self.drone_state[0,2] < 0.3: return
    #     step = min(self.traj_step,self.optimized_total_waypoints.shape[1]-1)
    #     flag_step = True
    #     for i in range(self.num_agent):
    #         pos = self.optimized_total_waypoints[i,step,:]  
    #         current_pos = self.drone_state[i,:3]
    #         error = pos-current_pos
    #         flag_step = flag_step&(np.linalg.norm(error)<0.2)
    #         msg = Pose()
    #         msg.position.x = pos[0]
    #         msg.position.y = pos[1]
    #         msg.position.z = 1#pos[2] 
    #         self.traj_cmd_publisher_list[i].publish(msg)
    #         print("finish pub traj cmd to "+self.vehicle_type+"_"+str(i),flag_step,error,"at time_step:",step)

    #     if flag_step: self.traj_step += 1
    def prepare_stage(self):
        if self.stage != "Prepare":
            return

        while not rospy.is_shutdown():
            print("current stage:",self.stage,"existing_robot_id",self.drone_state.keys())
            ###计算初始队形中心
            flag_get_drone_state = True
            sum_pos = np.zeros(3)
            for drone_id in self.drone_index_list:
                flag_get_drone_state = flag_get_drone_state & (drone_id in self.drone_state.keys()) 
                if flag_get_drone_state== False:
                    break
                sum_pos += self.drone_state[drone_id][:3].copy()                         
            
            ###没有拿到初始状态则继续等待
            if flag_get_drone_state== False:
                continue

            self.formation_origin[:3] = self.formation_origin[:3] + sum_pos/num_agent

            enu_init_formation = TM.optimized_total_waypoints[:,0,:2] + self.formation_origin[:2]  

            ###如果用的是提前预设好的地图，则障碍物的位置应以formation_origin为基准重新设置位置
            if self.flag_use_rtk_sta_obs is False:
                for i in range(len(self.two_points_static_obs_list)):
                    self.two_points_static_obs_list[i][0] += self.formation_origin[0]
                    self.two_points_static_obs_list[i][2] += self.formation_origin[0]
                    self.two_points_static_obs_list[i][1] += self.formation_origin[1]
                    self.two_points_static_obs_list[i][3] += self.formation_origin[1]
                    

            ###使用匈牙利算法计算轨迹起点匹配,行序列为formation_id,列序列为drone_id
            cost_matrix = np.zeros((num_agent,num_agent))
            for i in range(num_agent):
                for j in range(num_agent):
                    drone_id = self.drone_index_list[j]
                    cost_matrix[i,j] = np.linalg.norm(enu_init_formation[i,:]-self.drone_state[drone_id][:2])
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            self.traj2drone_id_map = np.zeros(self.num_agent)
            for k in range(num_agent):
                ###记录下轨迹分配到各机器人的编号映射关系
                self.traj2drone_id_map[k] = int(self.drone_index_list[col_ind[k]])
            
            break

        self.stage = "Init_formation"



    def real_callback_control_loop(self,event):
        if self.stage is "Prepare" or self.optimized_total_waypoints==[]:return
        # print("here",self.drone_state[0,2])
        # if self.drone_state[0,2] < 0.3: return
        print("at stage:",self.stage,"assignment",self.traj2drone_id_map,"===============")
        if self.stage=="Init_formation":
            flag_next_stage = True
            for i in range(self.num_agent):
                drone_id = self.traj2drone_id_map[i]
                temp = self.optimized_total_waypoints[3,0,:].copy() 
                temp[-1] = 0
                init_target_pos = self.formation_origin + temp
                current_pos = self.drone_state[drone_id][:3]

                error = init_target_pos-current_pos
                flag_next_stage = flag_next_stage and (np.linalg.norm(error)<0.15)#0.5
                
                output = error*0.3#2.5
                output = np.clip(output,-1,1)

                drone_vel_msg = TwistStamped() 
                drone_vel_msg.header = Header()
                drone_vel_msg.header.stamp = rospy.Time.now()
                drone_vel_msg.header.frame_id = str(int(drone_id))
                drone_vel_msg.twist.linear.x = output[0]
                drone_vel_msg.twist.linear.y = output[1]
                drone_vel_msg.twist.linear.z = output[2]
                drone_vel_msg.twist.angular.z = 0
                self.traj_cmd_publisher.publish(drone_vel_msg) 

                print("self.formation_origin:",self.formation_origin,"current_pos",current_pos)
                print("drone"+"_"+str(drone_id),"error",error,np.linalg.norm(error),"output",output)
            
            if flag_next_stage:
                self.stage = "Formation_navigation"   


        elif self.stage=="Formation_navigation":
            step = min(self.traj_step,self.optimized_total_waypoints.shape[1]-1)
            flag_step = True
            flag_next_stage = True
            
            for i in range(self.num_agent):
                drone_id = self.traj2drone_id_map[i]
                pos_step = step+0
                if pos_step >= self.optimized_total_waypoints.shape[1]-2:
                    temp = self.optimized_total_waypoints[i,-2,:].copy()
                    pos_step = self.optimized_total_waypoints.shape[1]-2
                else:    
                    temp = self.optimized_total_waypoints[i,pos_step,:].copy()
                temp[-1] = 0
                follow_target_pos = self.formation_origin + temp
                if step == self.optimized_total_waypoints.shape[1]-1:
                    target_vel = np.zeros(3)
                else:    
                    target_vel = (self.optimized_total_waypoints[3,step+1,:] - self.optimized_total_waypoints[3,step,:])/control_ts
                current_pos = self.drone_state[drone_id][:3]
                follow_target_pos[-1] = current_pos[-1]
                pos_error = follow_target_pos-current_pos
                flag_step = flag_step and (np.linalg.norm(pos_error)<0.5)
                
                kp=1.0#0.8#1#0.5
                kv=1

                output = pos_error*kp + target_vel*kv#2.5
                output = np.clip(output,-2,2)#np.clip(output,-1.5,1.5)
                ###publish
                drone_vel_msg = TwistStamped() 
                drone_vel_msg.header = Header()
                drone_vel_msg.header.stamp = rospy.Time.now()
                drone_vel_msg.header.frame_id = str(int(drone_id))
                drone_vel_msg.twist.linear.x = output[0]
                drone_vel_msg.twist.linear.y = output[1]
                drone_vel_msg.twist.linear.z = output[2]
                drone_vel_msg.twist.angular.z = 0
                self.traj_cmd_publisher.publish(drone_vel_msg) 
                # print("finish pub traj cmd to "+"drone"+"_"+str(drone_id),"output",output,"pos_error",np.linalg.norm(pos_error),"target_vel",target_vel,"at stage:",self.stage,"at time_step:",step,"assignment",self.traj2drone_id_map)
                

                temp = self.optimized_total_waypoints[3,-1,:] 
                temp[-1] = 0
                final_target_pos = self.formation_origin + temp
                final_target_pos[-1] = current_pos[-1]
                target_error = final_target_pos - current_pos
                flag_next_stage = flag_next_stage and (np.linalg.norm(target_error)<0.15)

                print("drone"+"_"+str(drone_id),"pos_error",np.linalg.norm(pos_error),"target_error",target_error)
                print("target_vel",target_vel,"output",output,"at time_step:",step)


                real_temp = self.optimized_total_waypoints[i,step,:].copy() 
                real_temp[-1] = 0
                real_follow_target_pos = self.formation_origin + real_temp
                real_error = real_follow_target_pos-current_pos
                msg = PoseStamped()
                msg.pose.position.x = pos_error[0]
                msg.pose.position.y = pos_error[1]
                msg.pose.position.z = pos_error[2]
                msg.pose.orientation.x = np.linalg.norm(pos_error)
                self.traj_follow_err_publisher_list[drone_id].publish(msg)
                self.accumulate_real_pos_err[i] += np.linalg.norm(real_error)
                
                # print("pos of drone"+str(i)+":",pos_err,flag_record,self.drone_state[i,:3])

            # if flag_step: self.traj_step += 20
            self.traj_step += self.traj_step_jump#20

            # print("flag_next_stage",flag_next_stage)
            if flag_next_stage:
                self.stage = "Target_formation" 


        elif self.stage=="Target_formation":
            for i in range(self.num_agent):
                drone_id = self.traj2drone_id_map[i]
                temp = self.optimized_total_waypoints[3,-1,:].copy() 
                temp[-1] = 0
                final_target_pos = self.formation_origin + temp
                current_pos = self.drone_state[drone_id][:3]
                final_target_pos[-1] = current_pos[-1]
                error = final_target_pos-current_pos
                output = error*0.3#2.5
                output = np.clip(output,-1,1)
                ###publish
                drone_vel_msg = TwistStamped() 
                drone_vel_msg.header = Header()
                drone_vel_msg.header.stamp = rospy.Time.now()
                drone_vel_msg.header.frame_id = str(int(drone_id))
                drone_vel_msg.twist.linear.x = output[0]
                drone_vel_msg.twist.linear.y = output[1]
                drone_vel_msg.twist.linear.z = output[2]
                drone_vel_msg.twist.angular.z = 0
                self.traj_cmd_publisher.publish(drone_vel_msg) 
                print("accumulate_real_pos_err of"," robot_",drone_id,":", self.accumulate_real_pos_err[i])
                
                # print()


 

    def sim_callback_control_loop(self,event):
        if self.stage is "Prepare" or self.optimized_total_waypoints==[]:return
        # print("here",self.drone_state[0,2])
        # if self.drone_state[0,2] < 0.3: return
        if self.stage=="Init_formation":
            flag_next_stage = True
            for i in range(self.num_agent):
                drone_id = self.traj2drone_id_map[i]
                temp = self.optimized_total_waypoints[i,0,:].copy() 
                temp[-1] = 0
                init_target_pos = self.formation_origin + temp
                current_pos = self.drone_state[drone_id][:3]

                error = init_target_pos-current_pos
                flag_next_stage = flag_next_stage and (np.linalg.norm(error)<0.5)
                
                output = error*0.3#2.5
                output = np.clip(output,-1,1)
                drone_vel = Twist() 
                drone_vel.linear.x = output[0]
                drone_vel.linear.y = output[1]
                drone_vel.linear.z = output[2]
                drone_vel.angular.z = 0
                self.traj_cmd_publisher_list[drone_id].publish(drone_vel)
                print("finish pub traj cmd to "+"drone"+"_"+str(drone_id),"output",output,"error",np.linalg.norm(error),"at stage:",self.stage)
            
            if flag_next_stage:
                self.stage = "Formation_navigation"   
                
                ###TODO:
                self.marl_distributed_reset()

                


        elif self.stage=="Formation_navigation":
            step = min(self.traj_step,self.optimized_total_waypoints.shape[1]-1)
            flag_step = True
            flag_next_stage = True
            
            
            ######new

            final_action_array = self.marl_distributed_controller(step, self.optimized_total_waypoints.copy())

            ####
            
            for i in range(self.num_agent):
                drone_id = self.traj2drone_id_map[i]
                pos_step = step+20
                if pos_step >= self.optimized_total_waypoints.shape[1]-2:
                    temp = self.optimized_total_waypoints[i,-2,:].copy()
                    pos_step = self.optimized_total_waypoints.shape[1]-2
                else:    
                    temp = self.optimized_total_waypoints[i,pos_step,:].copy() 
                temp[-1] = 0
                follow_target_pos = self.formation_origin + temp
                if step == self.optimized_total_waypoints.shape[1]-1:
                    target_vel = np.zeros(3)
                else:    
                    target_vel = (self.optimized_total_waypoints[i,step+1,:] - self.optimized_total_waypoints[i,step,:])/control_ts
                current_pos = self.drone_state[drone_id][:3]
                pos_error = follow_target_pos-current_pos
                flag_step = flag_step and (np.linalg.norm(pos_error)<0.5)
                
                kp=0.6#0.5
                kv=1.05

                #####pid_controller
                # output = pos_error*kp + target_vel*kv#2.5
                # output = np.clip(output,-self.VEL_LIMIT,self.VEL_LIMIT)

                #####marl controller
                output = final_action_array[i,:]

                drone_vel = Twist() 
                drone_vel.linear.x = output[0]
                drone_vel.linear.y = output[1]
                drone_vel.linear.z = output[2]
                drone_vel.angular.z = 0
                self.traj_cmd_publisher_list[drone_id].publish(drone_vel)
                # print("finish pub traj cmd to "+"drone"+"_"+str(drone_id),"output",output,"pos_error",np.linalg.norm(pos_error),"target_vel",target_vel,"at stage:",self.stage,"at time_step:",step,"assignment",self.traj2drone_id_map)
                print("pos_error",np.linalg.norm(pos_error),"target_vel",target_vel,"output",output)



                temp = self.optimized_total_waypoints[i,-1,:] 
                temp[-1] = 0
                final_target_pos = self.formation_origin + temp
                target_error = final_target_pos - current_pos

                # flag_next_stage = flag_next_stage and (np.linalg.norm(target_error)<0.05)
                flag_next_stage = flag_next_stage and (np.linalg.norm(target_error)<self.finish_threshold*2)


                real_temp = self.optimized_total_waypoints[i,step,:].copy() 
                real_temp[-1] = 0
                real_follow_target_pos = self.formation_origin + real_temp
                real_error = real_follow_target_pos-current_pos
                msg = PoseStamped()
                msg.pose.position.x = real_error[0]
                msg.pose.position.y = real_error[1]
                msg.pose.position.z = real_error[2]
                msg.pose.orientation.x = np.linalg.norm(real_error)

                self.traj_follow_err_publisher_list[drone_id].publish(msg)
                self.accumulate_real_pos_err[i] += np.linalg.norm(real_error)

                
                # print("pos of drone"+str(i)+":",pos_err,flag_record,self.drone_state[i,:3])

            # if flag_step: self.traj_step += 20
            self.traj_step += self.traj_step_jump#1

            print("flag_next_stage",flag_next_stage)
            if flag_next_stage:
                self.stage = "Target_formation" 


        elif self.stage=="Target_formation":
            for i in range(self.num_agent):
                drone_id = self.traj2drone_id_map[i]
                temp = self.optimized_total_waypoints[i,-1,:].copy() 
                temp[-1] = 0
                final_target_pos = self.formation_origin + temp
                current_pos = self.drone_state[drone_id][:3]
                error = final_target_pos-current_pos
                output = error*0.3#2.5
                output = np.clip(output,-1,1)
                drone_vel = Twist() 
                drone_vel.linear.x = output[0]
                drone_vel.linear.y = output[1]
                drone_vel.linear.z = output[2]
                drone_vel.angular.z = 0
                self.traj_cmd_publisher_list[drone_id].publish(drone_vel)
                print("accumulate_real_pos_err of"," robot_",drone_id,":", self.accumulate_real_pos_err[i])


 

    def callback_traj_record_loop(self,event):
        # if self.optimized_total_waypoints==[]:return
        # step = min(self.traj_step,self.optimized_total_waypoints.shape[1]-1)
        if self.stage is not "Formation_navigation":
            return 
        temp = []
        for i in range(self.num_agent):
            drone_id = self.traj2drone_id_map[i]
            traj_goal = TM.optimized_total_waypoints[i,-1,:]
            # if self.drone_state[drone_id][2] < 0.3:
            #     return

            pos_err = np.linalg.norm(self.drone_state[drone_id][:3]-traj_goal) 
            temp.append(list(self.drone_state[drone_id][:3]-self.formation_origin))

        self.traj_record.append(temp) 
        
        # if self.stage == "Target_formation":
            
            # print("finish pub traj cmd to "+self.vehicle_type+"_"+str(i),"at time_step:",step)


    def q2yaw(self, q):
        if isinstance(q, Quaternion):
            rotate_z_rad = q.yaw_pitch_roll[0]
        else:
            q_ = Quaternion(q.w, q.x, q.y, q.z)
            rotate_z_rad = q_.yaw_pitch_roll[0]

        return rotate_z_rad
    
    def q2yaw_split(self, qx, qy, qz, qw):
        q_ = Quaternion(qw, qx, qy, qz)
        rotate_z_rad = q_.yaw_pitch_roll[0]

        return rotate_z_rad
    


    
    def callback_drone_state_sim(self,msg,id):
        # if self.traj_step%20 != 0: return 

        self.drone_state[id] = np.zeros(7)
        self.drone_state[id][0] = msg.pose.position.x
        self.drone_state[id][1] = msg.pose.position.y
        self.drone_state[id][2] = msg.pose.position.z
        self.drone_state[id][6] = self.q2yaw(msg.pose.orientation)

    def callback_drone_state_real(self,msg,id):
        # id = args[0]
        self.drone_state[id] = np.zeros(7)
        self.drone_state[id][0] = msg.gps[0]
        self.drone_state[id][1] = msg.gps[1]
        self.drone_state[id][2] = msg.gps[2]
        self.drone_state[id][3] = msg.gps[3]
        self.drone_state[id][4] = msg.gps[4]
        self.drone_state[id][5] = msg.gps[5]
        self.drone_state[id][6] = self.q2yaw_split(msg.imu[0],msg.imu[1],msg.imu[2],msg.imu[3])



if __name__ == "__main__": 
    rospy.init_node("real_affine_formation", anonymous=True)
    rate = rospy.Rate(5)
    # rospy.wait_for_service('/gazebo/set_model_state')
    # set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    
    ###constant######parameter###
    num_agent = 4
    down_sampling_rate = 1
    control_ts = 0.005
    real_flag = False


    
    TM = Trajectory_manager(down_sampling_rate,num_agent,real_flag)
    frontend_publisher = rospy.Publisher("frontend_traj", Path, queue_size = 10)
    basepoint_direction_publisher = rospy.Publisher("basepoint_direction", Path, queue_size = 10)
    init_template_publisher = rospy.Publisher("init_template", Path, queue_size = 10)
    near_obstacles_publisher = rospy.Publisher("near_obstacles", Path, queue_size = 10)
    map_obstacles_publisher = rospy.Publisher("map_obstacles", Path, queue_size = 10)
    target_affine_param_publisher = rospy.Publisher("target_affine_param", PoseStamped, queue_size = 10)
    rospy.Subscriber('optimized_affine_param', Path, TM.callback_optimized_affine_param, queue_size=10)
    if real_flag:
        rospy.Timer(rospy.Duration(0.1), TM.real_callback_control_loop)#control_ts
    else:
        rospy.Timer(rospy.Duration(0.005), TM.sim_callback_control_loop)#control_ts
    rospy.Timer(rospy.Duration(control_ts*10), TM.callback_traj_record_loop)
    


    ### debug param
    for scene_kind in ["real"]:#["spa","mid","den"]:
        for scene_id in [4]:#[1,2,3,4,5,6,7,8,9,10]
            fname_folder = "/home/ubuntu/Shahao/MavenRL/rrt_algorithms/output/"+"obs_"+scene_kind+"/"+scene_kind+str(scene_id)
            fname_scene_bag = fname_folder+"/Scene_bag.npy"
            
            Scene_bag = np.load(fname_scene_bag,allow_pickle='TRUE').item()
            
            obstacle_list = Scene_bag["Obstacles"]
            init_template = Scene_bag["init_template"]
            target_affine_param = Scene_bag["x_goal"][:4]
            
            # num_agent = init_template.shape[1]
            p = index.Property()
            p.dimension = 2#只在二维上产生障碍物self.dimensions
            obs_checker = index.Index(TM.obstacle_generator(obstacle_list), interleaved=True, properties=p) 
            ###
            TM.num_agent = num_agent
            TM.init_template = init_template.copy()

            

            # fname_method = "oc_apo"#"bo"
            ########
            for fname_method in ["oc_apo"]:#["oc_apo","agc_apo","ags_apo"]:#["oc","agc","ags","al","br","ba"]
                if rospy.is_shutdown():
                    break
                prefix,_ = fname_method.split("_",1)
                fname_waypoints_bag =fname_folder+"/Waypoints_bag_"+prefix+".npy"
                fname_affine_param_waypoints_bag = fname_folder+"/Ap_waypoints_bag_"+fname_method+".npy"
                Waypoints_bag = np.load(fname_waypoints_bag,allow_pickle='TRUE').item()
                Affine_param_waypoints_bag = np.load(fname_affine_param_waypoints_bag,allow_pickle='TRUE').item()

                for time_id in [5]:#[1,2,3,4,5]
                    if rospy.is_shutdown():
                        break
                    print("start to optimize traj of ",scene_kind,"-",scene_id,"-",fname_method,"-",time_id)
                    fname_optimal_affine_param = fname_folder+"/Optimized_affine_param_"+fname_method+"_t"+str(time_id)+".npy"
                    fname_optimal_affine_param_traj = fname_folder+"/Optimized_traj_"+fname_method+"_t"+str(time_id)+".npy"
                    fname_egoswarm_record_traj = fname_folder+"/Egoswarm_record_traj"+".npy"

                    total_waypoints = Waypoints_bag[str(time_id)]
                    affine_param_waypoints = Affine_param_waypoints_bag[str(time_id)]
                    if type(total_waypoints)==type(None) or type(affine_param_waypoints)==type(None):
                        print("no init traj of ",scene_kind,"-",scene_id,"-",fname_method,"-",time_id)
                        continue
                    ###init for optimization
                    TM.traj_step = 0
                    wait_time = 0
                    final_cost = -1
                    flag_save = False
                    TM.init_total_waypoints = total_waypoints.copy()
                    TM.current_total_waypoints = total_waypoints.copy()
                    TM.optimized_affine_param = []

                    # temp_traj = np.load(fname_optimal_affine_param_traj)
                    TM.optimized_total_waypoints = np.load(fname_optimal_affine_param_traj)
                    

                    # TM.optimized_total_waypoints = temp_traj.copy()

                    # if scene_kind=="spa":
                    #     COST_THRESHOLD = 15
                    # elif scene_kind=="mid":
                    #     COST_THRESHOLD = 30
                    # elif scene_kind=="den":
                    #     COST_THRESHOLD = 9#50
                
                    ### loop for preparation
                    print("current stage:",TM.stage,"existing_robot_id",TM.drone_state.keys())
                    TM.prepare_stage()

                    while not rospy.is_shutdown():
                        # print("stress_vector:",stress_vector)
                        # stress_vector_publisher.publish(Float64MultiArray(data=list(stress_vector)))
                        TM.pub_map_obstacles(obstacle_list,map_obstacles_publisher)
                        # for i in range(TM.num_agent):

                        #     print("pos of drone"+str(i)+":",TM.drone_state[i,:3])
                        
                        print("current stage:",TM.stage,"existing_robot_id",TM.drone_state.keys())
                        #####202##code for reactive ego-swarm

                        for i in range(num_agent):
                            # if self.traj2drone_id_map ==[]:

                            drone_id = TM.traj2drone_id_map[i]
                            # if drone_id is not in TM.drone_state.keys():
                            #     continue 
                            temp = TM.optimized_total_waypoints[3,-1,:].copy() 
                            temp[-1] = 0
                            traj_goal = temp+TM.formation_origin

                            goal_point=PoseStamped()
                            goal_point.header.frame_id = 'map'
                            goal_point.pose.position.x=traj_goal[0]
                            goal_point.pose.position.y=traj_goal[1]
                            goal_point.pose.position.z=traj_goal[2]
                            goal_point.pose.orientation.x=0
                            goal_point.pose.orientation.y=0
                            goal_point.pose.orientation.z=0
                            goal_point.pose.orientation.w=1  
                            goal_point.header.stamp = rospy.Time.now()
                            TM.goal_publisher_list[drone_id].publish(goal_point) 
                            
                            # print("pos of drone"+str(drone_id)+":",TM.drone_state[drone_id][:3])


                        if TM.stage == "Target_formation":
                            np.save(fname_egoswarm_record_traj,np.array(TM.traj_record))
                            print("finish_save_record")
                            print("shape",np.array(TM.traj_record).shape)
                            # break

                        

                        # TM.pub_target_affine_param(target_affine_param_publisher,target_affine_param)
                        
                        
                        # if TM.optimized_affine_param==[]:
                        #     if TM.check_center_collision(obs_checker,TM.current_total_waypoints[:,:,-1]):
                        #         break
                        #     for i in range(num_agent):
                        #         TM.pub_front_end_traj(frontend_publisher,total_waypoints[:,:,i],"drone_"+str(i))
                        #     TM.pub_front_end_affine_param(frontend_publisher,affine_param_waypoints,"affine_param")


                        # for i in range(num_agent):
                        #     # base_point_array,direction_array = TM.generate_basepts_direction(obs_checker,TM.current_total_waypoints[:,:,i],TM.current_total_waypoints[:,:,-1]) 
                        #     # base_point_array = np.ones((TM.current_total_waypoints.shape[0],2))*2333#代表约定无效值
                        #     # direction_array = np.ones((TM.current_total_waypoints.shape[0],2))*2333#代表约定无效值
                        #     # TM.pub_basepoint_direction(basepoint_direction_publisher,base_point_array,direction_array,"drone_"+str(i))
                        #     # 
                        #     TM.find_and_publish_near_obstacles(obs_checker,near_obstacles_publisher)
                        
                        # TM.pub_init_template(init_template_publisher)


                        # if TM.traj_step>TM.optimized_total_waypoints.shape[1]+5:
                        #     break



                        rate.sleep()

                        # if wait_time==5: COST_THRESHOLD=100
                        # if TM.optimized_total_waypoints != [] and final_cost>0 and final_cost<COST_THRESHOLD:
                        #     flag_save = True
                        # # elif TM.optimized_total_waypoints != [] and final_cost>0 and final_cost<30 and wait_time==10:



                        # if flag_save:
                        #     np.save(fname_optimal_affine_param,TM.optimized_affine_param)
                        #     np.save(fname_optimal_affine_param_traj,TM.optimized_total_waypoints)            
                        #     # TM.optimized_affine_param = []
                        #     TM.optimized_total_waypoints = []
                        #     print("###finish save ma_optimized_cps###final_cost:",final_cost)
                        #     break
                        # else:
                        #     print("wait for optimization of ",scene_kind,"-",scene_id,"-",fname_method,"-",time_id,"wait time:",wait_time)


                        # if fname_method=="oc_apo":
                        #     if wait_time>5:
                        #         print("fail to optimize the traj of ",scene_kind,"-",scene_id,"-",fname_method,"-",time_id)
                        #         TM.optimized_affine_param=[]
                        #         break 
                        # else:
                        #     if wait_time>5:
                        #         print("fail to optimize the traj of ",scene_kind,"-",scene_id,"-",fname_method,"-",time_id)
                        #         TM.optimized_affine_param=[]
                        #         break
                        # wait_time += 1

                        








