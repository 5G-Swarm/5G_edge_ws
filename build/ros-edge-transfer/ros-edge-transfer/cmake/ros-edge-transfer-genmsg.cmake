# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(WARNING "Invoking generate_messages() without having added any message or service file before.
You should either add add_message_files() and/or add_service_files() calls or remove the invocation of generate_messages().")
message(STATUS "ros-edge-transfer: 0 messages, 0 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/melodic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/melodic/share/geometry_msgs/cmake/../msg;-Iautoware_msgs:/home/ubuntu/5g-ws/src/autoware_msgs/msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(ros-edge-transfer_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services

### Generating Module File
_generate_module_cpp(ros-edge-transfer
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros-edge-transfer
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(ros-edge-transfer_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(ros-edge-transfer_generate_messages ros-edge-transfer_generate_messages_cpp)

# add dependencies to all check dependencies targets

# target for backward compatibility
add_custom_target(ros-edge-transfer_gencpp)
add_dependencies(ros-edge-transfer_gencpp ros-edge-transfer_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ros-edge-transfer_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services

### Generating Module File
_generate_module_eus(ros-edge-transfer
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros-edge-transfer
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(ros-edge-transfer_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(ros-edge-transfer_generate_messages ros-edge-transfer_generate_messages_eus)

# add dependencies to all check dependencies targets

# target for backward compatibility
add_custom_target(ros-edge-transfer_geneus)
add_dependencies(ros-edge-transfer_geneus ros-edge-transfer_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ros-edge-transfer_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services

### Generating Module File
_generate_module_lisp(ros-edge-transfer
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros-edge-transfer
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(ros-edge-transfer_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(ros-edge-transfer_generate_messages ros-edge-transfer_generate_messages_lisp)

# add dependencies to all check dependencies targets

# target for backward compatibility
add_custom_target(ros-edge-transfer_genlisp)
add_dependencies(ros-edge-transfer_genlisp ros-edge-transfer_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ros-edge-transfer_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services

### Generating Module File
_generate_module_nodejs(ros-edge-transfer
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros-edge-transfer
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(ros-edge-transfer_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(ros-edge-transfer_generate_messages ros-edge-transfer_generate_messages_nodejs)

# add dependencies to all check dependencies targets

# target for backward compatibility
add_custom_target(ros-edge-transfer_gennodejs)
add_dependencies(ros-edge-transfer_gennodejs ros-edge-transfer_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ros-edge-transfer_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services

### Generating Module File
_generate_module_py(ros-edge-transfer
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros-edge-transfer
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(ros-edge-transfer_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(ros-edge-transfer_generate_messages ros-edge-transfer_generate_messages_py)

# add dependencies to all check dependencies targets

# target for backward compatibility
add_custom_target(ros-edge-transfer_genpy)
add_dependencies(ros-edge-transfer_genpy ros-edge-transfer_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ros-edge-transfer_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros-edge-transfer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ros-edge-transfer
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(ros-edge-transfer_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(ros-edge-transfer_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(ros-edge-transfer_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET autoware_msgs_generate_messages_cpp)
  add_dependencies(ros-edge-transfer_generate_messages_cpp autoware_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros-edge-transfer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ros-edge-transfer
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(ros-edge-transfer_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(ros-edge-transfer_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(ros-edge-transfer_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET autoware_msgs_generate_messages_eus)
  add_dependencies(ros-edge-transfer_generate_messages_eus autoware_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros-edge-transfer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ros-edge-transfer
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(ros-edge-transfer_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(ros-edge-transfer_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(ros-edge-transfer_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET autoware_msgs_generate_messages_lisp)
  add_dependencies(ros-edge-transfer_generate_messages_lisp autoware_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros-edge-transfer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ros-edge-transfer
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(ros-edge-transfer_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(ros-edge-transfer_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(ros-edge-transfer_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET autoware_msgs_generate_messages_nodejs)
  add_dependencies(ros-edge-transfer_generate_messages_nodejs autoware_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros-edge-transfer)
  install(CODE "execute_process(COMMAND \"/usr/bin/python\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros-edge-transfer\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ros-edge-transfer
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(ros-edge-transfer_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(ros-edge-transfer_generate_messages_py sensor_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(ros-edge-transfer_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET autoware_msgs_generate_messages_py)
  add_dependencies(ros-edge-transfer_generate_messages_py autoware_msgs_generate_messages_py)
endif()