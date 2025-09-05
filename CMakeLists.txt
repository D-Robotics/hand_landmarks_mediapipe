# Copyright (c) 2024，D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.5)
project(hand_landmarks_mediapipe)

# For Debug
set(CMAKE_BUILD_TYPE Debug)
add_definitions(-DDEBUG)
add_compile_options(
    -fdebug-prefix-map=/mnt/workspace/cc_ws=/mnt/deyu.li/cc_ws
)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ai_msgs REQUIRED)
find_package(dnn_node REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(hobot_cv REQUIRED)
find_package(rclcpp_components REQUIRED)


find_package(hbm_img_msgs REQUIRED)
add_definitions(-DSHARED_MEM_ENABLED)

include_directories(include
  ${PROJECT_SOURCE_DIR}
)

if(PLATFORM_S100)
  message("build ${PROJECT_NAME} for platform S100")
  set(SYS_ROOT ${CMAKE_SYSROOT})
  set(PREFIX_PATH s100)
else()
  message("invalid platform, build platform s100 default")
  set(PREFIX_PATH s100)
  set(SYS_ROOT ${CMAKE_SYSROOT})
endif()

message("CMAKE_SYSROOT is " ${CMAKE_SYSROOT})
include_directories(
  ${SYS_ROOT}/usr/include/
  ${SYS_ROOT}/usr/hobot/include/
  ${SYS_ROOT}/usr/include/hobot/
  ${SYS_ROOT}/usr/include/opencv/
)

link_directories(
  ${SYS_ROOT}/usr/lib/hbbpu/
  ${SYS_ROOT}/usr/hobot/lib/
  ${SYS_ROOT}/usr/lib/hbmedia/
)


add_library(${PROJECT_NAME}_component SHARED
  src/main.cpp
  src/utils.cpp
  src/mono2d_hand_lmk_node.cpp
  src/post_process/hand_lmk_parser.cpp
  src/post_process/gesture.cpp
)

target_compile_definitions(${PROJECT_NAME}_component
  PRIVATE "COMPOSITION_BUILDING_DLL")
  ament_target_dependencies(
    ${PROJECT_NAME}_component
  )

ament_target_dependencies(
  ${PROJECT_NAME}_component
  rclcpp
  rclcpp_components
  dnn_node
  sensor_msgs
  ai_msgs
  cv_bridge
  hobot_cv
  hbm_img_msgs
)

rclcpp_components_register_node(${PROJECT_NAME}_component
  PLUGIN "Mono2dHandLmkNode"
  EXECUTABLE ${PROJECT_NAME}
)

include_directories(include
${PROJECT_SOURCE_DIR}
)

# Install executables
install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
    TARGETS ${PROJECT_NAME}_component
    DESTINATION lib/
  )

install(DIRECTORY
${PROJECT_SOURCE_DIR}/config/${PREFIX_PATH}/
DESTINATION lib/${PROJECT_NAME}/config/
)

install(FILES
  ${PROJECT_SOURCE_DIR}/config/example.jpg
  DESTINATION lib/${PROJECT_NAME}/config/
)

install(DIRECTORY
${PROJECT_SOURCE_DIR}/launch/
DESTINATION share/${PROJECT_NAME}/launch)

if(PLATFORM_S100)
install(FILES
  ${PROJECT_SOURCE_DIR}/config/s100/hand_224_224.hbm
  DESTINATION lib/${PROJECT_NAME}/config/
)
endif()

ament_package()
