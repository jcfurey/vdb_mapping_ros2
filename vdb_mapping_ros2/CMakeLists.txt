cmake_minimum_required(VERSION 3.16.3)
project(vdb_mapping_ros2)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  message("${PROJECT_NAME}: You did not request a specific build type: selecting 'Release'.")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# vdb_mapping is an exported imported target rather than a plain ament dep
find_package(vdb_mapping REQUIRED)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/VDBMappingROS2.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}
  vdb_mapping::vdb_mapping
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

rclcpp_components_register_nodes(${PROJECT_NAME}
  "vdb_mapping_ros2::VDBMappingROS2"
)

ament_auto_add_executable(vdb_mapping_ros_node
  src/vdb_mapping_ros_node.cpp
)
target_link_libraries(vdb_mapping_ros_node
  ${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME})


if(BUILD_TESTING)
endif()

ament_auto_package()
