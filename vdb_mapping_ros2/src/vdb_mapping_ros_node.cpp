#include "rclcpp/rclcpp.hpp"
#include "vdb_mapping_ros2/VDBMappingROS2.hpp"

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto vdb_mapping = std::make_shared<vdb_mapping_ros2::VDBMappingROS2>(rclcpp::NodeOptions());
  rclcpp::spin(vdb_mapping);
  rclcpp::shutdown();
  return 0;
}
