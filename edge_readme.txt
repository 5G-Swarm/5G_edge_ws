Edge端操作手册：
##一，程序组成和信息流：
~/5G_edge_ws:
1.5g转发承接部分（上行：/drone_state_x,/drone_image_x,/drone_state_x;下行：/teamrl_controller_vel(十架飞机共用，header.frame_id = robot_id区分) x:0-9）
2.气球检测部分，基于yolov5检测气球中心点（输入/drone_image_x,输出/target_bbx_x）
3.气球位置估计：基于三角化初始化和卡尔曼滤波（输入/target_bbx_x,输出/balloon_estimation/odom）
4.teamrl_controller:


##二，各环节配置方式：



##三, edge端运行
1. wait for all ms.
2. start the edge-5g-transfer: 
cd ~/5G_edge_ws/src/ros-edge-transfer/ros-edge-transfer/scripts && python edge-drone.py
#check state topics: 
rostopic hz /drone_state_x 
#open rviz to check images from drones: subscribe to /drone_image_x(x:0-9)

3. open the balloon detection program:
cd ~/5G_edge_ws/src/drone/scripts/balloon_test && python detect_balloon.py

4. before drones taking off(to get the initiate altitudes), open the balloon location estimation program:
#remember to adjust the ros parameter of "uav_num" in ~/5g-ws/src/balloon_estimation/launch/balloon_estimation.launch line 6
cd  ~/5G_edge_ws && roslaunch balloon_estimation balloon_estimation.launch

5. open the teamrl_controller program:
#Note:check the loaded model,  
cd ~/5G_edge_ws/src/drone/scripts/MavenRL/real_controller && python teamrl_controller.py

6. check the command download:
check each drone from the output of their 5g-transfer program for "TAKE OFF"
if no, restart the step 2 and check again


#####################################
exploration, target detection and localization, formation with target tracking 

1. before drones taking off(to get the initiate altitudes), open the balloon location estimation program:
#remember to adjust the ros parameter of "uav_num" in ~/5g-ws/src/balloon_estimation/launch/balloon_estimation.launch line 6
cd  ~/5G_edge_ws && roslaunch balloon_estimation balloon_estimation.launch

2. open the teamrl_controller_explore program:
#Note:check the loaded model,  
cd ~/5G_edge_ws/src/drone/scripts/MavenRL/real_controller && python teamrl_controller_explore.py




#####################################
formation with obstacle avoidance

1. open the trajectory planning:
cd ~/5G_edge_ws/ && roslaunch global_path_planning global_path_planning.launch
use 2D Nav Goal to set the destination of planning 

2. open the teamrl_controller_obs_avoid program:
cd ~/5G_edge_ws/src/drone/scripts/MavenRL/real_controller && python teamrl_controller_obs_avoid.py


#######final exam##################

###communication between three zones: yuquan--ssh->huzhou--ssh->gongyuan

###huzhou: server
ssh server@172.16.20.15
pwd:zju104104
###gongyuan:5g-new
ssh ubuntu@172.17.10.3
###yuquan:5g
ssh ubuntu@172.16.10.3


client send to server

###1. operation on huzhou
ssh server@172.16.20.15
pwd:zju104104
cd ~/DTD && python server.py

###2. operation on yuquan104
cd ~/DTD && python c2s.py -c

###3.operation on gongyuan
ssh server@172.16.20.15
pwd:zju104104
ssh ubuntu@172.17.10.3
cd ~ &&  python c2s.py -s



