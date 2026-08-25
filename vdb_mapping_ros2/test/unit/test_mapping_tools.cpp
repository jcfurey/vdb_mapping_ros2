#include <gtest/gtest.h>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <vdb_mapping/OccupancyVDBMapping.hpp>
#include <vdb_mapping_ros2/VDBMappingTools.hpp>

TEST(MappingTools, ProjectionThresholdIsInclusive)
{
  using MapT = vdb_mapping::OccupancyVDBMapping;
  MapT map(0.1);
  auto grid = map.getGrid();
  auto accessor = grid->getAccessor();

  // Two adjacent one-cell-wide columns keep the occupancy smoother from
  // deliberately demoting an isolated lethal cell. Each column has exactly
  // three occupied z voxels and must therefore pass a threshold of three.
  for (int x = 0; x <= 1; ++x)
  {
    for (int z = 0; z < 3; ++z)
    {
      accessor.setValueOn(openvdb::Coord(x, 0, z), 1.0F);
    }
  }

  visualization_msgs::msg::Marker marker;
  sensor_msgs::msg::PointCloud2 cloud;
  nav_msgs::msg::OccupancyGrid occupancy;
  VDBMappingTools<MapT>::createMappingOutput(
    grid, "map", marker, cloud, occupancy,
    /*create_marker=*/false, /*create_pointcloud=*/false,
    /*create_occupancy_grid=*/true,
    /*lower_z_limit=*/0.0, /*upper_z_limit=*/0.0,
    /*resolution=*/0.1F, /*two_dim_proj_threshold=*/3);

  ASSERT_GT(occupancy.info.width, 1U);
  ASSERT_GT(occupancy.info.height, 0U);
  // The active leaf starts at index (0,0), and output bounds are chunk
  // aligned, so these are the first two cells in the first row.
  EXPECT_EQ(occupancy.data[0], 100);
  EXPECT_EQ(occupancy.data[1], 100);
}

TEST(MappingTools, SmoothingDoesNotDuplicateEdgeCells) {
  using MapT = vdb_mapping::OccupancyVDBMapping;

  nav_msgs::msg::OccupancyGrid isolated;
  isolated.info.width = 1;
  isolated.info.height = 1;
  isolated.data.assign(1, -1);
  std::vector<int> isolated_projection{100};
  VDBMappingTools<MapT>::smoothOccGrid(isolated, isolated_projection);
  EXPECT_EQ(isolated.data[0], -1);

  nav_msgs::msg::OccupancyGrid corner;
  corner.info.width = 2;
  corner.info.height = 2;
  corner.data.assign(4, -1);
  std::vector<int> corner_projection{-1, 0, 0, 100};
  VDBMappingTools<MapT>::smoothOccGrid(corner, corner_projection);
  EXPECT_EQ(corner.data[0], 0);
}
