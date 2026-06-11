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
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Marvin Große Besselmann grosse@fzi.de
 * \date    2022-05-09
 *
 */
//----------------------------------------------------------------------
#ifndef VDB_MAPPING_ROS2_VDBMAPPINGTOOLS_H_INCLUDED
#define VDB_MAPPING_ROS2_VDBMAPPINGTOOLS_H_INCLUDED
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <openvdb/openvdb.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>

/*!
 * \brief Collection of VDBMapping helper functions and tools
 */
template <typename VDBMappingT>
class VDBMappingTools
{
public:
  VDBMappingTools(){};
  virtual ~VDBMappingTools(){};
  /*!
   * \brief Creates output msgs for pointcloud and marker arrays
   *
   * \param grid Map grid
   * \param resolution Resolution of the grid
   * \param frame_id Frame ID of the grid
   * \param marker_msg Output Marker message
   * \param cloud_msg Output Pointcloud message
   * \param create_marker Flag specifying to create a marker message
   * \param create_pointcloud Flag specifying to create a pointcloud message
   */
  static void createMappingOutput(const typename VDBMappingT::GridT::Ptr grid,
                                  const std::string& frame_id,
                                  visualization_msgs::msg::Marker& marker_msg,
                                  sensor_msgs::msg::PointCloud2& cloud_msg,
                                  nav_msgs::msg::OccupancyGrid& occupancy_grid_msg,
                                  const bool create_marker,
                                  const bool create_pointcloud,
                                  const bool create_occupancy_grid,
                                  double lower_z_limit             = 0.0,
                                  double upper_z_limit             = 0.0,
                                  const float resolution           = 0.05,
                                  const int two_dim_proj_threshold = 5,
                                  const int occupancy_chunk        = 32)
  {
    typename VDBMappingT::PointCloudT::Ptr cloud(new typename VDBMappingT::PointCloudT);
    // The active bbox only spans *occupied* voxels. Observed-free voxels
    // (inactive, non-background) regularly lie outside it — e.g. raytraced
    // space in front of the outermost obstacle — and must still be covered by
    // the occupancy grid. Their extent is bounded by the allocated leaf nodes
    // (8^3 granularity; evalLeafBoundingBox cannot be used since it only
    // evaluates *active* per-leaf bounds) plus any non-background tiles
    // (pruned constant regions).
    openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
    for (auto leaf_iter = grid->tree().cbeginLeaf(); leaf_iter; ++leaf_iter)
    {
      const openvdb::CoordBBox leaf_bbox = leaf_iter->getNodeBoundingBox();
      bbox.expand(leaf_bbox.min());
      bbox.expand(leaf_bbox.max());
    }
    {
      auto tile_iter = grid->tree().cbeginValueAll();
      tile_iter.setMaxDepth(VDBMappingT::GridT::TreeType::DEPTH - 2);
      for (; tile_iter; ++tile_iter)
      {
        if (tile_iter.getValue() != 0)
        {
          openvdb::CoordBBox tile_bbox;
          tile_iter.getBoundingBox(tile_bbox);
          bbox.expand(tile_bbox.min());
          bbox.expand(tile_bbox.max());
        }
      }
    }
    if (bbox.empty())
    {
      // An empty map yields an inverted bbox (min = Coord::max(), max =
      // Coord::min()). The aligned-bounds arithmetic below overflows on those
      // values, so emit empty/delete outputs instead of garbage.
      if (create_marker)
      {
        marker_msg.header.frame_id = frame_id;
        marker_msg.id              = 0;
        marker_msg.type            = visualization_msgs::msg::Marker::CUBE_LIST;
        marker_msg.action          = visualization_msgs::msg::Marker::DELETE;
      }
      if (create_pointcloud)
      {
        cloud->width  = 0;
        cloud->height = 1;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = frame_id;
      }
      if (create_occupancy_grid)
      {
        occupancy_grid_msg.info.resolution           = resolution;
        occupancy_grid_msg.info.origin.orientation.w = 1.0;
      }
      return;
    }
    double min_z, max_z;
    openvdb::Vec3d min_world_coord = grid->indexToWorld(bbox.min());
    openvdb::Vec3d max_world_coord = grid->indexToWorld(bbox.max());
    min_z                          = min_world_coord.z();
    max_z                          = max_world_coord.z();

    if (lower_z_limit != upper_z_limit && lower_z_limit < upper_z_limit)
    {
      min_z = min_z < lower_z_limit ? lower_z_limit : min_z;
      max_z = max_z > upper_z_limit ? upper_z_limit : max_z;
    }

    // Snap the 2D occupancy grid bounds to a fixed multiple of chunk voxels so
    // info.origin only jumps in chunk-sized steps as the active bbox grows.
    // Without this, downstream consumers (nav2 static_layer, AMCL) would see
    // the origin shift every visualization tick and rebuild their costmaps.
    auto floor_chunk = [occupancy_chunk](int v) {
      int r = v % occupancy_chunk;
      return v - (r < 0 ? r + occupancy_chunk : r);
    };
    auto ceil_chunk = [occupancy_chunk](int v) {
      int r = v % occupancy_chunk;
      if (r > 0) return v + occupancy_chunk - r;
      if (r < 0) return v - r;
      return v;
    };

    int aligned_min_x = floor_chunk(bbox.min().x());
    int aligned_min_y = floor_chunk(bbox.min().y());
    int aligned_max_x = ceil_chunk(bbox.max().x() + 1);
    int aligned_max_y = ceil_chunk(bbox.max().y() + 1);
    int aligned_width  = aligned_max_x - aligned_min_x;
    int aligned_height = aligned_max_y - aligned_min_y;

    // Reuse scratch buffers across calls on the same visualization thread.
    // assign() preserves capacity when the size is stable (the chunk-aligned
    // bounds change rarely), avoiding the per-tick large allocations.
    thread_local std::vector<int> occ_voxel_projection_grid;
    thread_local std::vector<bool> occ_observed_grid;
    if (create_occupancy_grid)
    {
      const size_t cells = static_cast<size_t>(aligned_width) * aligned_height;
      occupancy_grid_msg.info.height     = aligned_height;
      occupancy_grid_msg.info.width      = aligned_width;
      occupancy_grid_msg.info.resolution = resolution;
      occupancy_grid_msg.data.assign(cells, -1);
      occ_voxel_projection_grid.assign(cells, 0);
      occ_observed_grid.assign(cells, false);

      geometry_msgs::msg::Pose origin_pose;
      // vdb_mapping discretizes sensor data cell-centered (its worldToIndex
      // rounds to the nearest lattice point), so indexToWorld(i) is the voxel
      // *center*. nav_msgs/MapMetaData defines origin as the bottom-left
      // *corner* of cell (0,0), hence the half-voxel shift.
      origin_pose.position.x    = (aligned_min_x - 0.5) * resolution;
      origin_pose.position.y    = (aligned_min_y - 0.5) * resolution;
      origin_pose.position.z    = 0.00;
      origin_pose.orientation.w = 1.0;

      occupancy_grid_msg.info.origin = origin_pose;
    }

    // Pre-compute the z range in index space. The grid uses a uniform-scale
    // linear transform, so world_z and idx_z differ only by resolution.
    // Checking the int z first lets us skip the indexToWorld call entirely
    // for voxels outside the visualization band.
    int min_z_idx = static_cast<int>(std::floor(min_z / resolution));
    int max_z_idx = static_cast<int>(std::ceil(max_z / resolution));

    auto process_voxel = [&](const openvdb::Coord& coord, const bool is_on) {
      if (coord.z() < min_z_idx || coord.z() > max_z_idx)
      {
        return;
      }

      if (create_occupancy_grid && bbox.isInside(coord))
      {
        int vdb_index_to_occ_index =
          (coord.y() - aligned_min_y) * aligned_width + (coord.x() - aligned_min_x);
        occ_observed_grid[vdb_index_to_occ_index] = true;
        if (is_on)
        {
          // Active voxel = occupied — count toward lethal threshold
          occ_voxel_projection_grid[vdb_index_to_occ_index] += 1;
        }
      }

      // Marker and pointcloud only show occupied (active) voxels
      if (is_on)
      {
        openvdb::Vec3d world_coord = grid->indexToWorld(coord);
        if (create_marker)
        {
          geometry_msgs::msg::Point cube_center;
          cube_center.x = world_coord.x();
          cube_center.y = world_coord.y();
          cube_center.z = world_coord.z();
          marker_msg.points.push_back(cube_center);
          // Guard against a single-layer map (max_z == min_z) and clamp:
          // the index-space z filter can admit voxels slightly outside the
          // clamped [min_z, max_z] band.
          double z_span = max_z - min_z;
          double h      = z_span > 0.0 ? 1.0 - ((world_coord.z() - min_z) / z_span) : 0.0;
          h             = std::clamp(h, 0.0, 1.0);
          marker_msg.colors.push_back(heightColorCoding(h));
        }
        if (create_pointcloud)
        {
          cloud->points.push_back(
            typename VDBMappingT::PointT(world_coord.x(), world_coord.y(), world_coord.z()));
        }
      }
    };

    // Use cbeginValueAll() to iterate ALL voxels with non-background values:
    //   - Active voxels (value > logodds_thres_max): occupied — count toward 2D projection
    //   - Inactive voxels (value != 0 background): observed free — mark column as "seen"
    // Previously cbeginValueOn() only visited occupied voxels, so raytraced free
    // space was indistinguishable from never-observed space in the 2D grid.
    for (typename VDBMappingT::GridT::ValueAllCIter iter = grid->cbeginValueAll(); iter; ++iter)
    {
      // Skip voxels that still hold the background value (0.0 = never observed).
      // This filters out untouched tiles/voxels efficiently.
      if (!iter.isValueOn() && iter.getValue() == 0)
      {
        continue;
      }

      if (iter.isVoxelValue())
      {
        process_voxel(iter.getCoord(), iter.isValueOn());
      }
      else
      {
        // Tile value: pruned constant regions (e.g. uniform free space after a
        // PCD load, since vdb_mapping calls pruneGrid()) are visited as a
        // single iterator item covering their whole extent. Expand the
        // footprint, clamped to the active bbox and the visualization z band,
        // so free-space tiles are not projected as a single cell.
        openvdb::CoordBBox tile_bbox;
        iter.getBoundingBox(tile_bbox);
        tile_bbox.intersect(bbox);
        const bool is_on = iter.isValueOn();
        const int z0     = std::max(tile_bbox.min().z(), min_z_idx);
        const int z1     = std::min(tile_bbox.max().z(), max_z_idx);
        for (int z = z0; z <= z1; ++z)
        {
          for (int y = tile_bbox.min().y(); y <= tile_bbox.max().y(); ++y)
          {
            for (int x = tile_bbox.min().x(); x <= tile_bbox.max().x(); ++x)
            {
              process_voxel(openvdb::Coord(x, y, z), is_on);
            }
          }
        }
      }
    }
    if (create_marker)
    {
      double size                = grid->transform().voxelSize()[0];
      marker_msg.header.frame_id = frame_id;
      // marker_msg.header.stamp       = ros::Time::now();
      marker_msg.id                 = 0;
      marker_msg.type               = visualization_msgs::msg::Marker::CUBE_LIST;
      marker_msg.scale.x            = size;
      marker_msg.scale.y            = size;
      marker_msg.scale.z            = size;
      marker_msg.color.a            = 1.0;
      marker_msg.pose.orientation.w = 1.0;
      marker_msg.frame_locked       = true;
      if (marker_msg.points.size() > 0)
      {
        marker_msg.action = visualization_msgs::msg::Marker::ADD;
      }
      else
      {
        marker_msg.action = visualization_msgs::msg::Marker::DELETE;
      }
    }
    if (create_pointcloud)
    {
      cloud->width  = cloud->points.size();
      cloud->height = 1;
      pcl::toROSMsg(*cloud, cloud_msg);
      cloud_msg.header.frame_id = frame_id;
      // cloud_msg.header.stamp    = ros::Time::now();
    }

    if (create_occupancy_grid)
    {
      for (size_t i = 0; i < occ_voxel_projection_grid.size(); i++)
      {
        if (occ_voxel_projection_grid[i] > two_dim_proj_threshold)
        {
          occ_voxel_projection_grid[i] = 100;   // lethal: enough occupied voxels
        }
        else if (occ_observed_grid[i])
        {
          occ_voxel_projection_grid[i] = 0;     // free: observed but few/no occupied voxels
        }
        else
        {
          occ_voxel_projection_grid[i] = -1;    // unknown: never observed
        }
      }
      smoothOccGrid(occupancy_grid_msg, occ_voxel_projection_grid);
    }
  }

  static void smoothOccGrid(nav_msgs::msg::OccupancyGrid& occupancy_grid_msg,
                            std::vector<int>& occ_voxel_projection_grid)
  {
    auto get_index = [&](int i, int j) -> int {
      // Clamp
      i = std::max(0, std::min((int)occupancy_grid_msg.info.height - 1, i));
      j = std::max(0, std::min((int)occupancy_grid_msg.info.width - 1, j));
      return i * occupancy_grid_msg.info.width + j;
    };

    for (size_t i = 0; i < occupancy_grid_msg.info.height; ++i)
    {
      for (size_t j = 0; j < occupancy_grid_msg.info.width; ++j)
      {
        int current_index = get_index(i, j);
        if (occ_voxel_projection_grid[current_index] == -1)
        {
          std::vector<int> counts = {0, 0, 0};
          for (int di = -1; di <= 1; ++di)
          {
            for (int dj = -1; dj <= 1; ++dj)
            {
              if (di == 0 && dj == 0)
              {
                continue;
              }
              int value = occ_voxel_projection_grid[get_index(i + di, j + dj)];
              if (value == -1)
              {
                counts[0]++;
              }
              else if (value == 0)
              {
                counts[1]++;
              }
              else if (value == 100)
              {
                counts[2]++;
              }
            }
          }
          int most_count_index =
            std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()));
          if (most_count_index == 0)
          {
            // occupancy_grid_msg.data[current_index] = -1;
          }
          else if (most_count_index == 1)
          {
            occupancy_grid_msg.data[current_index] = 0;
          }
          else if (most_count_index == 2)
          {
            occupancy_grid_msg.data[current_index] = 100;
          }
        }
        else if (occ_voxel_projection_grid[current_index] == 100)
        {
          int count = 0;
          for (int di = -1; di <= 1; ++di)
          {
            for (int dj = -1; dj <= 1; ++dj)
            {
              if (di == 0 && dj == 0)
              {
                continue;
              }
              if (occ_voxel_projection_grid[get_index(i + di, j + dj)] == 100)
              {
                count++;
              }
            }
          }
          // Only demote truly isolated lethal cells (no lethal neighbor at
          // all): one-cell-wide walls have exactly 2 lethal neighbors and
          // line endpoints just 1, so any stricter rule erases real thin
          // obstacles. Demote to unknown rather than free — the column did
          // exceed the occupancy threshold, so claiming it is traversable
          // would hide poles or trunks from planners.
          if (count > 0)
          {
            occupancy_grid_msg.data[current_index] = occ_voxel_projection_grid[current_index];
          }
          else
          {
            occupancy_grid_msg.data[current_index] = -1;
          }
        }
        else
        {
          occupancy_grid_msg.data[current_index] = occ_voxel_projection_grid[current_index];
        }
      }
    }
  }


  static void createMappingOutput(const typename VDBMappingT::GridT::Ptr grid,
                                  const std::string& frame_id,
                                  visualization_msgs::msg::Marker& marker_msg,
                                  double lower_z_limit   = 0.0,
                                  double upper_z_limit   = 0.0,
                                  const float resolution = 0.05)
  {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    nav_msgs::msg::OccupancyGrid occupancy_grid_msg;
    createMappingOutput(grid,
                        frame_id,
                        marker_msg,
                        cloud_msg,
                        occupancy_grid_msg,
                        true,
                        false,
                        false,
                        lower_z_limit,
                        upper_z_limit,
                        resolution);
  }
  static void createMappingOutput(const typename VDBMappingT::GridT::Ptr grid,
                                  const std::string& frame_id,
                                  sensor_msgs::msg::PointCloud2& cloud_msg,
                                  double lower_z_limit   = 0.0,
                                  double upper_z_limit   = 0.0,
                                  const float resolution = 0.05)
  {
    visualization_msgs::msg::Marker marker_msg;
    nav_msgs::msg::OccupancyGrid occupancy_grid_msg;
    createMappingOutput(grid,
                        frame_id,
                        marker_msg,
                        cloud_msg,
                        occupancy_grid_msg,
                        false,
                        true,
                        false,
                        lower_z_limit,
                        upper_z_limit,
                        resolution);
  }
  static void createMappingOutput(const typename VDBMappingT::GridT::Ptr grid,
                                  const std::string& frame_id,
                                  nav_msgs::msg::OccupancyGrid& occupancy_grid_msg,
                                  double lower_z_limit             = 0.0,
                                  double upper_z_limit             = 0.0,
                                  const float resolution           = 0.05,
                                  const int two_dim_proj_threshold = 5)
  {
    visualization_msgs::msg::Marker marker_msg;
    sensor_msgs::msg::PointCloud2 cloud_msg;
    createMappingOutput(grid,
                        frame_id,
                        marker_msg,
                        cloud_msg,
                        occupancy_grid_msg,
                        false,
                        false,
                        true,
                        lower_z_limit,
                        upper_z_limit,
                        resolution,
                        two_dim_proj_threshold);
  }

  /*!
   * \brief Calculates a height correlating color coding using HSV color space
   *
   * \param height Gridcell height relativ to the min and max height of the complete grid. Parameter
   * can take values between 0 and 1
   *
   * \returns RGBA color of the grid cell
   */
  static std_msgs::msg::ColorRGBA heightColorCoding(const double height)
  {
    // The factor of 0.8 is only for a nicer color range
    double h = height * 0.8;
    int i    = (int)(h * 6.0);
    double f = (h * 6.0) - i;
    double q = (1.0 - f);
    i %= 6;
    auto toMsg = [](double v1, double v2, double v3) {
      std_msgs::msg::ColorRGBA rgba;
      rgba.a = 1.0;
      rgba.r = v1;
      rgba.g = v2;
      rgba.b = v3;
      return rgba;
    };
    switch (i)
    {
      case 0:
        return toMsg(1.0, f, 0.0);
        break;
      case 1:
        return toMsg(q, 1.0, 0.0);
        break;
      case 2:
        return toMsg(0.0, 1.0, f);
        break;
      case 3:
        return toMsg(0.0, q, 1.0);
        break;
      case 4:
        return toMsg(f, 0.0, 1.0);
        break;
      case 5:
        return toMsg(1.0, 0.0, q);
        break;
      default:
        return toMsg(1.0, 0.5, 0.5);
        break;
    }
  }
};
#endif /* VDB_MAPPING_ROS2_VDBMAPPINGTOOLS_H_INCLUDED */
