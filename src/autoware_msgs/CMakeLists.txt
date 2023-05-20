cmake_minimum_required(VERSION 2.8.3)
project(autoware_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation
  std_msgs
  geometry_msgs
  sensor_msgs
  # visualization_msgs
  )

add_message_files(
  FILES
  # TrackingObjectMarker.msg
  # TrackingObjectMarkerArray.msg
  # Pose2DArray.msg
  DroneSyn.msg
)

#catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  # visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime   
)

#install(PROGRAMS detect
#        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
