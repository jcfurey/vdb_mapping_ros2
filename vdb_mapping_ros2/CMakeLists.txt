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

# Graph-anchored map assembler: renders the map from per-keyframe evidence
# at SLAM-optimized poses (see src/vdb_map_assembler_node.cpp)
ament_auto_add_executable(vdb_map_assembler_node
  src/vdb_map_assembler_node.cpp
)
target_link_libraries(vdb_map_assembler_node
  ${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME})


if(BUILD_TESTING)
  # Survey voxel reduction: every field must survive (pcl::VoxelGrid zeroed
  # the custom ones), sentinels must neither dilute nor fabricate.
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_survey_voxel test/unit/test_survey_voxel.cpp)
  if(TARGET test_survey_voxel)
    target_include_directories(test_survey_voxel PRIVATE include)
    target_link_libraries(test_survey_voxel ${PCL_LIBRARIES})
    ament_target_dependencies(test_survey_voxel pcl_conversions)
  endif()

  ament_add_gtest(test_sonar_reconstruction test/unit/test_sonar_reconstruction.cpp)
  if(TARGET test_sonar_reconstruction)
    target_include_directories(test_sonar_reconstruction PRIVATE include)
    target_link_libraries(test_sonar_reconstruction ${PCL_LIBRARIES})
    ament_target_dependencies(test_sonar_reconstruction pcl_conversions)
  endif()

  # End-to-end launch tests: start a vdb_mapping_ros_node, feed it a synthetic
  # wall cloud and validate the published occupancy grid / marker and the
  # raytrace + get_map_section services against the known geometry.
  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/smoke.test.py TIMEOUT 150)
  set_tests_properties(test_smoke.test.py PROPERTIES ENVIRONMENT "ROS_DOMAIN_ID=152")
  # End-to-end assembler test: one keyframe of synthetic evidence through the
  # real node; pins survey metadata survival, odom-delta re-anchoring, and
  # the z-only pose-correction re-render.
  add_launch_test(test/assembler.test.py TIMEOUT 200)
  set_tests_properties(test_assembler.test.py PROPERTIES ENVIRONMENT "ROS_DOMAIN_ID=153")
  # Remote section sync: mapper A -> sections -> remote B, including the
  # clear-after-reset transition through pruned tiles.
  add_launch_test(test/section_sync.test.py TIMEOUT 250)
  set_tests_properties(test_section_sync.test.py PROPERTIES ENVIRONMENT "ROS_DOMAIN_ID=154")
  add_launch_test(test/raytrace.test.py TIMEOUT 150)
  set_tests_properties(test_raytrace.test.py PROPERTIES ENVIRONMENT "ROS_DOMAIN_ID=155")

  install(PROGRAMS
    test/smoke_test.py
    test/raytrace_test.py
    test/assembler_test.py
    test/section_sync_test.py
    DESTINATION lib/${PROJECT_NAME})
endif()

ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR)
