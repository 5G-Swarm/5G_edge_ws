cmake_minimum_required(VERSION 2.8.3)
project(vehicle-message-transfer)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
)

generate_messages(
  DEPENDENCIES
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime   
)