// -- BEGIN LICENSE BLOCK ----------------------------------------------
// Copyright 2022 FZI Forschungszentrum Informatik
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
// -- END LICENSE BLOCK ------------------------------------------------
#ifndef VDB_MAPPING_ROS2_VDBMAPPINGROS2_HPP_INCLUDED
#define VDB_MAPPING_ROS2_VDBMAPPINGROS2_HPP_INCLUDED

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <Eigen/Core>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <vdb_mapping/OccupancyVDBMapping.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <vdb_mapping_interfaces/msg/update_grid.hpp>
#include <vdb_mapping_interfaces/srv/add_artificial_areas.hpp>
#include <vdb_mapping_interfaces/srv/add_points_to_grid.hpp>
#include <vdb_mapping_interfaces/srv/batch_raytrace.hpp>
#include <vdb_mapping_interfaces/srv/get_map_section.hpp>
#include <vdb_mapping_interfaces/srv/load_map.hpp>
#include <vdb_mapping_interfaces/srv/load_map_from_pcd.hpp>
#include <vdb_mapping_interfaces/srv/raytrace.hpp>
#include <vdb_mapping_interfaces/srv/remove_points_from_grid.hpp>
#include <vdb_mapping_interfaces/srv/toggle_remote_source.hpp>
#include <vdb_mapping_interfaces/srv/trigger_map_section_update.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace vdb_mapping_ros2 {

struct RemoteSource
{
  rclcpp::Subscription<vdb_mapping_interfaces::msg::UpdateGrid>::SharedPtr map_section_sub;
  rclcpp::Subscription<vdb_mapping_interfaces::msg::UpdateGrid>::SharedPtr map_full_section_sub;
  rclcpp::Client<vdb_mapping_interfaces::srv::GetMapSection>::SharedPtr get_map_section_client;
  rclcpp::Client<vdb_mapping_interfaces::srv::GetMapSection>::SharedPtr
    get_map_full_section_client;
  bool apply_remote_sections;
  bool apply_remote_full_sections;
  // Written by the toggle service, read by subscription callbacks that may
  // run on a different executor thread.
  std::atomic<bool> active{true};
};

struct SensorSource
{
  std::string source_id;
  std::string topic;
  std::string sensor_origin_frame;
  double max_range;
  double max_rate;
  bool reliable;
};

class VDBMappingROS2 : public rclcpp::Node
{
public:
  using VDBMapT = vdb_mapping::OccupancyVDBMapping;

  explicit VDBMappingROS2(const rclcpp::NodeOptions& options);
  ~VDBMappingROS2() override = default;

  void resetMap();
  bool saveMap(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
               const std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  bool saveMapToPCD(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                    const std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  bool loadMap(const std::shared_ptr<vdb_mapping_interfaces::srv::LoadMap::Request> req,
               const std::shared_ptr<vdb_mapping_interfaces::srv::LoadMap::Response> res);
  bool
  loadMapFromPCD(const std::shared_ptr<vdb_mapping_interfaces::srv::LoadMapFromPCD::Request> req,
                 const std::shared_ptr<vdb_mapping_interfaces::srv::LoadMapFromPCD::Response> res);

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg,
                     const SensorSource& sensor_source);
  void publishMap() const;

  void mapSectionCallback(const vdb_mapping_interfaces::msg::UpdateGrid::SharedPtr update_msg,
                          const std::shared_ptr<RemoteSource>& remote_source);
  void mapFullSectionCallback(const vdb_mapping_interfaces::msg::UpdateGrid::SharedPtr update_msg,
                              const std::shared_ptr<RemoteSource>& remote_source);

  const std::string& getMapFrame() const { return m_map_frame; }
  std::shared_ptr<VDBMapT> getMap() { return m_vdb_map; }
  const std::shared_ptr<VDBMapT> getMap() const { return m_vdb_map; }

  bool resetMapCallback(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                        const std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  bool getMapSectionCallback(
    const std::shared_ptr<vdb_mapping_interfaces::srv::GetMapSection::Request> req,
    const std::shared_ptr<vdb_mapping_interfaces::srv::GetMapSection::Response> res);
  bool triggerMapSectionUpdateCallback(
    const std::shared_ptr<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate::Request> req,
    const std::shared_ptr<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate::Response> res);
  bool triggerMapFullSectionUpdateCallback(
    const std::shared_ptr<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate::Request> req,
    const std::shared_ptr<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate::Response> res);
  bool addPointsToGridCallback(
    const std::shared_ptr<vdb_mapping_interfaces::srv::AddPointsToGrid::Request> req,
    const std::shared_ptr<vdb_mapping_interfaces::srv::AddPointsToGrid::Response> res);
  bool removePointsFromGridCallback(
    const std::shared_ptr<vdb_mapping_interfaces::srv::RemovePointsFromGrid::Request> req,
    const std::shared_ptr<vdb_mapping_interfaces::srv::RemovePointsFromGrid::Response> res);
  bool raytraceCallback(const std::shared_ptr<vdb_mapping_interfaces::srv::Raytrace::Request> req,
                        const std::shared_ptr<vdb_mapping_interfaces::srv::Raytrace::Response> res);
  bool batchRaytraceCallback(
    const std::shared_ptr<vdb_mapping_interfaces::srv::BatchRaytrace::Request> req,
    const std::shared_ptr<vdb_mapping_interfaces::srv::BatchRaytrace::Response> res);
  bool addArtificialAreasCallback(
    const std::shared_ptr<vdb_mapping_interfaces::srv::AddArtificialAreas::Request> req,
    const std::shared_ptr<vdb_mapping_interfaces::srv::AddArtificialAreas::Response> res);
  bool removeArtificialAreasCallback(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                                     const std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  bool toggleRemoteSource(
    const std::shared_ptr<vdb_mapping_interfaces::srv::ToggleRemoteSource::Request> req,
    const std::shared_ptr<vdb_mapping_interfaces::srv::ToggleRemoteSource::Response> res);

  void visualizationTimerCallback();
  void sectionTimerCallback();
  void fullSectionTimerCallback();

private:
  void setUpVDBMap();
  void setUpLocalSources();
  void setUpRemoteSources();
  void setUpVisualization();
  void setUpServices();
  void setUpPublishers();
  void setUpMapServer();

  std::vector<SensorSource> m_sensor_sources;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> m_cloud_subs;

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr m_visualization_marker_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_pointcloud_pub;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr m_occupancy_grid_pub;
  rclcpp::Publisher<vdb_mapping_interfaces::msg::UpdateGrid>::SharedPtr m_map_section_pub;
  rclcpp::Publisher<vdb_mapping_interfaces::msg::UpdateGrid>::SharedPtr m_map_full_section_pub;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr m_save_map_service;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr m_save_map_to_pcd_service;
  rclcpp::Service<vdb_mapping_interfaces::srv::LoadMap>::SharedPtr m_load_map_service;
  rclcpp::Service<vdb_mapping_interfaces::srv::LoadMapFromPCD>::SharedPtr
    m_load_map_from_pcd_service;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr m_reset_map_service;
  rclcpp::Service<vdb_mapping_interfaces::srv::Raytrace>::SharedPtr m_raytrace_service;
  rclcpp::Service<vdb_mapping_interfaces::srv::BatchRaytrace>::SharedPtr m_batch_raytrace_service;
  rclcpp::Service<vdb_mapping_interfaces::srv::GetMapSection>::SharedPtr m_get_map_section_service;
  rclcpp::Service<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate>::SharedPtr
    m_trigger_map_section_update_service;
  rclcpp::Service<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate>::SharedPtr
    m_trigger_map_full_section_update_service;
  rclcpp::Service<vdb_mapping_interfaces::srv::AddPointsToGrid>::SharedPtr
    m_add_points_to_grid_service;
  rclcpp::Service<vdb_mapping_interfaces::srv::RemovePointsFromGrid>::SharedPtr
    m_remove_points_from_grid_service;
  rclcpp::Service<vdb_mapping_interfaces::srv::AddArtificialAreas>::SharedPtr
    m_add_artificial_areas_service;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr m_remove_artificial_areas_service;
  rclcpp::Service<vdb_mapping_interfaces::srv::ToggleRemoteSource>::SharedPtr
    m_toggle_remote_source_service;

  std::unique_ptr<tf2_ros::Buffer> m_tf_buffer;
  std::shared_ptr<tf2_ros::TransformListener> m_tf_listener{nullptr};

  double m_resolution;
  std::string m_map_frame;
  std::string m_robot_frame;
  std::shared_ptr<VDBMapT> m_vdb_map;
  vdb_mapping::Config m_config;

  bool m_publish_pointcloud;
  bool m_publish_vis_marker;
  bool m_publish_occupancy_grid;
  bool m_publish_sections;
  bool m_publish_full_sections;
  bool m_apply_raw_sensor_data;
  bool m_smooth_remote_sections;
  // Only declared/read when apply_raw_sensor_data is true; keep a defined
  // value on the pure-remote path.
  bool m_accumulate_updates = false;
  int m_remote_section_smoothing_iterations;

  std::map<std::string, std::shared_ptr<RemoteSource>> m_remote_sources;

  rclcpp::TimerBase::SharedPtr m_visualization_timer;
  rclcpp::TimerBase::SharedPtr m_section_timer;
  rclcpp::TimerBase::SharedPtr m_full_section_timer;

  Eigen::Matrix<double, 3, 1> m_section_min_coord;
  Eigen::Matrix<double, 3, 1> m_section_max_coord;
  std::string m_section_update_frame;
  int m_two_dim_projection_threshold;
  double m_tf_lookup_timeout;
  double m_artificial_negative_height;
  double m_artificial_positive_height;

  std::shared_ptr<rclcpp::ParameterEventHandler> m_param_sub;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> m_z_min_param_handle;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> m_z_max_param_handle;
  // Written by parameter callbacks (default callback group), read by
  // publishMap on the visualization callback group.
  std::atomic<double> m_lower_visualization_z_limit{0.0};
  std::atomic<double> m_upper_visualization_z_limit{0.0};

  rclcpp::CallbackGroup::SharedPtr m_accumulation_cb_group;
  rclcpp::CallbackGroup::SharedPtr m_visualization_cb_group;
  rclcpp::CallbackGroup::SharedPtr m_remote_cb_group;
};

}  // namespace vdb_mapping_ros2

#endif  // VDB_MAPPING_ROS2_VDBMAPPINGROS2_HPP_INCLUDED
