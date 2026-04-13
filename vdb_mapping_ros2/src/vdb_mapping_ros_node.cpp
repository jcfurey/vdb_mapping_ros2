#include "rclcpp/rclcpp.hpp"
#include "vdb_mapping/OccupancyVDBMapping.hpp"
#include "vdb_mapping_ros2/VDBMappingROS2.hpp"


int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto vdb_mapping =
    std::make_shared<VDBMappingROS2<vdb_mapping::OccupancyVDBMapping>>();

  rclcpp::spin(vdb_mapping);

  rclcpp::shutdown();
  return 0;
}
