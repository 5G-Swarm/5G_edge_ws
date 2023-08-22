from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped,Pose,Twist
from std_msgs.msg import Float64MultiArray   
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
sys.path.append("..")
##########################
from gym_formation_particle.gym_flock.envs.env_affine_formation_plan.Img_rrtobs_formation_plan_env import Img_rrtobs_formationPlanEnv as affine_formation 






# class call_back_template_state(object):
# 	def __init__(self, name):
# 		self._name = name
# 		self._id = int(name[-1])

# 	def __call__(self,drone_state):
# 		# if real_env.state_drones_dict.has_key(self._id):
# 		temp = np.array(drone_state.gps+drone_state.imu)
# 		# temp[0] = temp[0]-33425788.8125
# 		# temp[1] = -(temp[1]-7650161.05)
# 		real_env.state_drones_dict[self._id] = temp

# class FunctorFactory_state(object):
#     def create(self,name):
#         globals()[name] = call_back_template_state(name)


class Trajectory_manager(object):

    def __init__(self,down_sampling_rate,num_agent=4,init_total_waypoints=[],init_template=[]):
        
        ####task relevant param
        self.obstacle_index_list = [8,9,10,13,14]
        self.drone_index_list = [0,1,2,3]#12,13,3,4,5
        self.formation_origin = np.array([0,0,3])   


        self.down_sampling_rate = down_sampling_rate
        self.num_agent = num_agent
        self.init_total_waypoints = init_total_waypoints.copy()
        self.current_total_waypoints = init_total_waypoints.copy()#np.zeros((int((init_total_waypoints.shape[0]-1)/self.down_sampling_rate)+1,2,self.num_agent+1))#init_total_waypoints.copy()
        self.optimized_affine_param = []#init_affine_param
        self.optimized_total_waypoints = [] 
        self.init_template = init_template.copy()
        self.traj_record = []
        # for i in range(self.num_agent):
        #     self.traj_record.append([])

        self.vehicle_type = "iris"

        #####
        self.traj_step = 0
        self.stage = "Prepare"#"Prepare","Init_formation","Formation_navigation","Target_formation"
        self.traj2drone_id_map = []#np.zeros(self.num_agent)
        self.drone_id_list = []
        
        self.traj_cmd_publisher_list = {}
        self.pose_sub_list = {}
        self.goal_publisher_list = {}
        self.traj_follow_err_publisher_list = {}
        self.drone_state = {}#np.zeros((self.num_agent,7))
        for i in range(self.num_agent):
            drone_id = self.drone_index_list[i]
            # self.traj_cmd_publisher_list.append(rospy.Publisher("/xtdrone/"+self.vehicle_type+'_'+str(i)+"/cmd_pose_enu", Pose, queue_size = 10)) 
            self.traj_cmd_publisher_list[drone_id] = rospy.Publisher("/xtdrone/"+self.vehicle_type+'_'+str(drone_id)+"/cmd_vel_enu", Twist, queue_size = 10)
            self.pose_sub_list[drone_id] = rospy.Subscriber(self.vehicle_type+'_'+str(drone_id)+"/mavros/local_position/pose", PoseStamped, self.callback_drone_state_sim,(drone_id),queue_size=1)
            self.traj_follow_err_publisher_list[drone_id] = rospy.Publisher("/xtdrone/"+self.vehicle_type+'_'+str(drone_id)+"/pos_err", PoseStamped, queue_size = 10) 
            
            # self.traj_cmd_publisher_list[drone_id] = rospy.Publisher("/xtdrone/"+self.vehicle_type+'_'+str(drone_id)+"/cmd_vel_enu", Twist, queue_size = 10)
            # self.pose_sub_list[drone_id] = rospy.Subscriber("/drone_state_"+str(drone_id), DroneSyn, self.callback_drone_state_real,(drone_id),queue_size=1)
            # self.traj_cmd_publisher_list.append(rospy.Publisher("/xtdrone/"+self.vehicle_type+'_'+str(i)+"/cmd_vel_enu", Twist, queue_size = 10)) 
            # self.pose_sub_list.append(rospy.Subscriber("/drone_state_"+str(i), DroneSyn, self.callback_drone_state_real,(i),queue_size=1))
            
            self.goal_publisher_list[drone_id] = rospy.Publisher(self.vehicle_type+"_"+str(drone_id)+"/move_base_simple/goal", PoseStamped, queue_size=1)
            

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
        for obstacle in obstacles:
            yield (uuid.uuid4(), obstacle, obstacle)


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
            sum_pos = np.zeros(2)
            for drone_id in self.drone_index_list:
                flag_get_drone_state = flag_get_drone_state & (drone_id in self.drone_state.keys()) 
                if flag_get_drone_state== False:
                    break
                sum_pos += self.drone_state[drone_id][:2]                         
            
            ###没有拿到初始状态则继续等待
            if flag_get_drone_state== False:
                continue

            self.formation_origin[:2] = sum_pos/num_agent
            enu_init_formation = TM.optimized_total_waypoints[:,0,:2] + self.formation_origin[:2]  

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


    def callback_control_loop(self,event):
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


        elif self.stage=="Formation_navigation":
            step = min(self.traj_step,self.optimized_total_waypoints.shape[1]-1)
            flag_step = True
            flag_next_stage = True
            for i in range(self.num_agent):
                drone_id = self.traj2drone_id_map[i]
                pos_step = step+5
                if pos_step >= self.optimized_total_waypoints.shape[1]-1:
                    temp = self.optimized_total_waypoints[i,-1,:].copy()
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
                
                kp=1#0.5
                kv=1

                output = pos_error*kp + target_vel*kv#2.5
                output = np.clip(output,-1.5,1.5)
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
                flag_next_stage = flag_next_stage and (np.linalg.norm(target_error)<0.1)


                msg = PoseStamped()
                msg.pose.position.x = pos_error[0]
                msg.pose.position.y = pos_error[1]
                msg.pose.position.z = pos_error[2]
                msg.pose.orientation.x = np.linalg.norm(pos_error)
                self.traj_follow_err_publisher_list[drone_id].publish(msg)
                
                # print("pos of drone"+str(i)+":",pos_err,flag_record,self.drone_state[i,:3])

            # if flag_step: self.traj_step += 20
            self.traj_step += 20

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
                # print()


 

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
    
    def callback_drone_state_sim(self,msg,id):
        if self.traj_step%20 != 0: return 

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
        self.drone_state[id][3] = msg.imu[3]
        self.drone_state[id][4] = msg.imu[4]
        self.drone_state[id][5] = msg.imu[5]
        # self.drone_state[id][6] = self.q2yaw(msg.pose.orientation)



if __name__ == "__main__": 
    rospy.init_node("real_affine_formation", anonymous=True)
    rate = rospy.Rate(5)
    # rospy.wait_for_service('/gazebo/set_model_state')
    # set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    
    ###constant######parameter###
    num_agent = 4
    down_sampling_rate = 1
    control_ts = 0.005


    
    TM = Trajectory_manager(down_sampling_rate,num_agent)
    frontend_publisher = rospy.Publisher("frontend_traj", Path, queue_size = 10)
    basepoint_direction_publisher = rospy.Publisher("basepoint_direction", Path, queue_size = 10)
    init_template_publisher = rospy.Publisher("init_template", Path, queue_size = 10)
    near_obstacles_publisher = rospy.Publisher("near_obstacles", Path, queue_size = 10)
    map_obstacles_publisher = rospy.Publisher("map_obstacles", Path, queue_size = 10)
    target_affine_param_publisher = rospy.Publisher("target_affine_param", PoseStamped, queue_size = 10)
    rospy.Subscriber('optimized_affine_param', Path, TM.callback_optimized_affine_param, queue_size=10)
    rospy.Timer(rospy.Duration(0.1), TM.callback_control_loop)#control_ts
    rospy.Timer(rospy.Duration(control_ts*10), TM.callback_traj_record_loop)
    


    ### debug param
    for scene_kind in ["real"]:#["spa","mid","den"]:
        for scene_id in [6]:#[1,2,3,4,5,6,7,8,9,10]
            fname_folder = "../output/"+"obs_"+scene_kind+"/"+scene_kind+str(scene_id)
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
                            
                            traj_goal = TM.optimized_total_waypoints[i,-1,:]
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
                            print("pos of drone"+str(drone_id)+":",TM.drone_state[drone_id][:3])


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

                        








