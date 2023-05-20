#!/home/ubuntu/miniconda3/envs/5G_37/bin/python
#https://zhuanlan.zhihu.com/p/50219346
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist



def drone_test_publisher():
	# ROS节点初始化
    rospy.init_node('drone_test', anonymous=True)
    global target_pos_pub
    drone_vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped',Twist, queue_size=10)
    
    rate = rospy.Rate(10) 
    drone_vel = Twist()


    while not rospy.is_shutdown():
        # 初始化learning_topic::Person类型的消息
        drone_vel.linear = [0.05,0.05,0.05]
        drone_vel.angular = [0.05,0.05,0.05]
        # drone_vel.angular[0] = 0.1
        # drone_vel.angular[1] = 0.1
        # drone_vel.angular[2] = 0.1
        drone_vel_pub.publish(drone_vel)

        # 按照循环频率延时
        rate.sleep()


if __name__ == '__main__':
    try:
        drone_test_publisher()
    except rospy.ROSInterruptException:
        pass
