飞机操作手册：
##一，硬件备注：
1.安装好两块电池，先短按一次再长按一次直到四灯全亮（关闭电池同理，可以不关机在线换电池，关闭其中一块拔下，替换新电池开启，再同理替换另一块），上电后电机滴四声自检
2.如果想要UI，则需要插着屏幕上电
3.5g模块连接好（四根通信线和一根网线到树莓派，网线和电源线都用扎带固定）

##二，需要预先拷贝到树莓派的内容：
将内容拷贝至drone对应目录：
/catkin_ws: 
/bash各种工具脚本:
    launch.sh:总启动脚本
    wifi.sh:树莓派切换wifi优先级使其高于5g从而连接外网（默认连接QKWLAN，route -n看优先级）
    install_ntp.sh:第一次配置ntp的脚本
    ntp.sh:常态化开启ntp时间同步的脚本（已经在launch.sh中调用）
    camera.sh:单独启动ros realsense
    50-cloud-init.yaml:设置静态ip所需要的配置文件
/src:
    /autoware_msgs:5g转发相关的自定义消息类型和protobuf
    /informer2: 5g转发相关的底层库
    /realsense-ros:相机ros库，没必要全拷，其实只需要向/realsense2_camera/launch中添加rs_camera_drone.launch
    /ros_transfer_drone: 5g转发节点和用到的proto
    /test:只需要copy其中的 /launch 和/test_drone (飞机端单机测试代码，包括各种单机基础运动控制的python程序示例)

##三，配置5g通讯相关：
1，配置ntp:连接QKWLAN并sh wifi.sh，连接好外网后， sh install_ntp.sh , 
2,配置飞机和edge端的ssh: 
2.1,vscode ssh插件配置：
飞机端和edge端连接同一外网情况下，使用wlan的动态ip建立vscode远程连接，以使得drone端完成vscode server的联网配置,而后双双连回5g，关闭wifi, ping 172.16.10.3(edge端ip)
2.2,5g模块端口映射：
首先，设置5g有线网的静态ip： cd ~/catkin_ws/bash; sudo cp 50-cloud-init.yaml /etc/netplan ; cd /etc/netplan ; sudo netplan apply ; ifconfig检查有线网ip是否成功修改为192.168.8.104 
其次，浏览器登录192.168.8.1（帐号密码均为：admin）,运行状态-》大地球图标：确定5g模块ip地址，并登记在edge端vscode remote ssh插件中。
最后，应用设置-转发设置-NAT(最下一栏点加号新建):接口设为（modem），初始端口和映射端口均为（22），映射地址为（192.168.8.104），点击右上角保存
2.3,测试edge端能否启动drone端launch.sh


##四，配置catkin_ws:（三完成后均可以在edge端远程进行）
1,编译：将二中内容拷贝至/catkin_ws后，catkin_make
2,关联飞机编号：修改转发5g程序配置文件 cd ~/catkin_ws/src/ros_transfer_drone/scripts,修改config.yaml第三行robot_id
3,本地安装informer库: cd ~/catkin_ws/src/informer2 ; pip3 install -e . (注意最后有一个“.”); 
4,验证是否成功：
启动：cd ~/catkin_ws/bash; sh launch.sh(px4飞控启动，/mavros topics出现，相机成功启动/camera/color/image_raw)
转发：cd ~/catkin_ws/src/ros_transfer_drone/scripts; python3 robot_drone_rosimg_control.py(绿色提示连接到172.16.10.3)


##五, drone端运行
启动：cd ~/catkin_ws/bash && sh launch.sh(rostopic echo /mavros/global_position/global;rostopic echo /camera/color/image_raw)
转发：cd ~/catkin_ws/src/ros_transfer_drone/scripts && python3 robot_drone_rosimg_control.py(分别有gps,image信息的上传和相应时间戳)

