// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
// -- BEGIN LICENSE BLOCK ----------------------------------------------
// Copyright 2022 FZI Forschungszentrum Informatik
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// -- END LICENSE BLOCK ------------------------------------------------

#include "rclcpp/rclcpp.hpp"
#include "vdb_mapping_ros2/VDBMappingROS2.hpp"

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto vdb_mapping = std::make_shared<vdb_mapping_ros2::VDBMappingROS2>(rclcpp::NodeOptions());
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(vdb_mapping);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
