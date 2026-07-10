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

#include <vdb_mapping_ros2/VDBMappingROS2.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <utility>

#if __has_include(<tf2/exceptions.hpp>)
#include <tf2/exceptions.hpp>
#else
#include <tf2/exceptions.h>
#endif
#include <tf2_eigen/tf2_eigen.hpp>

#define BOOST_BIND_NO_PLACEHOLDERS
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>

#include <vdb_mapping_ros2/VDBMappingTools.hpp>

namespace vdb_mapping_ros2 {

namespace {
// Newer vdb_mapping versions (e.g. forks past FZI devel) expose
// setLogCallback to route library log output into a host logging framework.
// Detect it at compile time so the wrapper keeps building against cores
// without it; with it, library messages reach the ROS log instead of stderr.
template <typename MapT>
auto trySetLogCallback(MapT& map, const rclcpp::Logger& logger, int)
  -> decltype(map.setLogCallback(nullptr), void())
{
  map.setLogCallback([logger](typename MapT::LogLevel level, const std::string& msg) {
    switch (level)
    {
      case MapT::LogLevel::Info:
        RCLCPP_INFO(logger, "%s", msg.c_str());
        break;
      case MapT::LogLevel::Warning:
        RCLCPP_WARN(logger, "%s", msg.c_str());
        break;
      default:
        RCLCPP_ERROR(logger, "%s", msg.c_str());
        break;
    }
  });
}
template <typename MapT>
void trySetLogCallback(MapT&, const rclcpp::Logger&, long)
{
}

// pcl::fromROSMsg indexes the data buffer via row_step/point_step without any
// bounds checking, and silently leaves coordinates uninitialized when x/y/z
// fields are absent — so a lying or incompatible publisher could crash the
// node or inject garbage points. Returns nullptr if the message is usable,
// otherwise a description of the defect.
const char* cloudMsgError(const sensor_msgs::msg::PointCloud2& msg)
{
  bool has_x = false;
  bool has_y = false;
  bool has_z = false;
  for (const auto& field : msg.fields)
  {
    if (field.name != "x" && field.name != "y" && field.name != "z")
    {
      continue;
    }
    if (field.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        field.offset + sizeof(float) > msg.point_step)
    {
      return "x/y/z field is not a FLOAT32 lying within point_step";
    }
    has_x |= field.name == "x";
    has_y |= field.name == "y";
    has_z |= field.name == "z";
  }
  if (!(has_x && has_y && has_z))
  {
    return "missing x/y/z FLOAT32 fields";
  }
  if (msg.width != 0 && msg.height != 0)
  {
    const size_t required = static_cast<size_t>(msg.height - 1) * msg.row_step +
                            static_cast<size_t>(msg.width) * msg.point_step;
    if (msg.data.size() < required)
    {
      return "data buffer smaller than the declared width/height/point_step extent";
    }
  }
  return nullptr;
}
}  // namespace

VDBMappingROS2::VDBMappingROS2(const rclcpp::NodeOptions& options)
  : Node("vdb_mapping_ros2", options)
{
  m_tf_buffer   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  m_tf_listener = std::make_shared<tf2_ros::TransformListener>(*m_tf_buffer);

  m_accumulation_cb_group =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  m_visualization_cb_group =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  m_remote_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  setUpVDBMap();
  setUpRemoteSources();
  setUpVisualization();
  setUpServices();
  setUpPublishers();
  setUpMapServer();
  // Local cloud subscriptions go live last: cloudCallback runs on
  // m_accumulation_cb_group on a separate executor thread the moment the
  // subscription is created, so every member it reads (m_vdb_map config,
  // m_accumulate_updates, m_publish_*, sensor_source entries) must already
  // be populated. Anything earlier in the ctor and a cloud arriving mid-init
  // races against partially-constructed state and SIGSEGVs the container.
  setUpLocalSources();
}

void VDBMappingROS2::resetMap()
{
  RCLCPP_INFO(this->get_logger(), "Resetting Map");
  m_vdb_map->resetMap();
  publishMap();
}

bool VDBMappingROS2::saveMap(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                             const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  (void)req;
  RCLCPP_INFO(this->get_logger(), "Saving Map");
  res->success = m_vdb_map->saveMap();
  return res->success;
}

bool VDBMappingROS2::saveMapToPCD(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                                  const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  (void)req;
  RCLCPP_INFO(this->get_logger(), "Saving Map to PCD");
  res->success = m_vdb_map->saveMapToPCD();
  return res->success;
}

bool VDBMappingROS2::loadMap(
  const std::shared_ptr<vdb_mapping_interfaces::srv::LoadMap::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::LoadMap::Response> res)
{
  RCLCPP_INFO(this->get_logger(), "Loading Map");
  bool success = m_vdb_map->loadMap(req->path);
  publishMap();
  res->success = success;
  return success;
}

bool VDBMappingROS2::loadMapFromPCD(
  const std::shared_ptr<vdb_mapping_interfaces::srv::LoadMapFromPCD::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::LoadMapFromPCD::Response> res)
{
  RCLCPP_INFO(this->get_logger(), "Loading Map from PCD file");
  bool success = m_vdb_map->loadMapFromPCD(req->path, req->set_background, req->clear_map);
  publishMap();
  res->success = success;
  return success;
}

void VDBMappingROS2::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg,
                                   const SensorSource& sensor_source)
{
  if (cloud_msg->width * cloud_msg->height == 0)
  {
    return;
  }
  if (const char* error = cloudMsgError(*cloud_msg))
  {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "Dropping cloud from source %s: %s",
                          sensor_source.source_id.c_str(),
                          error);
    return;
  }
  VDBMapT::PointCloudT::Ptr cloud(new VDBMapT::PointCloudT);
  pcl::fromROSMsg(*cloud_msg, *cloud);
  geometry_msgs::msg::TransformStamped cloud_origin_tf;

  std::string sensor_frame = sensor_source.sensor_origin_frame.empty()
                               ? cloud_msg->header.frame_id
                               : sensor_source.sensor_origin_frame;

  try
  {
    cloud_origin_tf =
      m_tf_buffer->lookupTransform(m_map_frame,
                                   sensor_frame,
                                   cloud_msg->header.stamp,
                                   rclcpp::Duration::from_seconds(m_tf_lookup_timeout));
  }
  catch (tf2::TransformException& ex)
  {
    RCLCPP_ERROR(this->get_logger(),
                 "MapToSensor: Could not transform %s to %s: %s",
                 m_map_frame.c_str(),
                 sensor_frame.c_str(),
                 ex.what());
    return;
  }
  if (m_map_frame != cloud_msg->header.frame_id)
  {
    if (sensor_frame == cloud_msg->header.frame_id)
    {
      pcl::transformPointCloud(*cloud, *cloud, tf2::transformToEigen(cloud_origin_tf).matrix());
    }
    else
    {
      geometry_msgs::msg::TransformStamped origin_to_map_tf;
      try
      {
        origin_to_map_tf =
          m_tf_buffer->lookupTransform(m_map_frame,
                                       cloud_msg->header.frame_id,
                                       cloud_msg->header.stamp,
                                       rclcpp::Duration::from_seconds(m_tf_lookup_timeout));
      }
      catch (tf2::TransformException& ex)
      {
        RCLCPP_ERROR(this->get_logger(),
                     "MapToMessage: Could not transform %s to %s: %s",
                     m_map_frame.c_str(),
                     cloud_msg->header.frame_id.c_str(),
                     ex.what());
        return;
      }
      pcl::transformPointCloud(*cloud, *cloud, tf2::transformToEigen(origin_to_map_tf).matrix());
    }
    cloud->header.frame_id = m_map_frame;
  }
  m_vdb_map->addDataToAccumulate(
    cloud, tf2::transformToEigen(cloud_origin_tf).translation(), sensor_source.source_id);
  if (!m_accumulate_updates)
  {
    m_vdb_map->integrateUpdate();
  }
}

void VDBMappingROS2::publishMap() const
{
  if (!(m_publish_pointcloud || m_publish_vis_marker || m_publish_occupancy_grid))
  {
    return;
  }
  // Ask the publisher handles rather than count_subscribers(name): the latter
  // expands the node-relative name but does NOT apply remap rules, so it would
  // report 0 forever if the topic is remapped at launch.
  bool publish_vis_marker =
    (m_publish_vis_marker && m_visualization_marker_pub->get_subscription_count() > 0);
  bool publish_pointcloud =
    (m_publish_pointcloud && m_pointcloud_pub->get_subscription_count() > 0);
  bool publish_occupancy_grid =
    (m_publish_occupancy_grid && m_occupancy_grid_pub->get_subscription_count() > 0);

  if (!(publish_vis_marker || publish_pointcloud || publish_occupancy_grid))
  {
    return;
  }

  // The z-limits are interpreted relative to the robot frame, so the robot
  // height is only needed when z-clipping is actually enabled (the limits
  // differ). When it is disabled the robot TF would not influence the output
  // at all, so we must not gate the whole visualization on it — otherwise a
  // pure map-server or a not-yet-localized instance would publish nothing.
  const double lower_z_limit = m_lower_visualization_z_limit;
  const double upper_z_limit = m_upper_visualization_z_limit;
  double robot_z             = 0.0;
  if (lower_z_limit < upper_z_limit)
  {
    // Skip until the TF tree is connected so we don't spam ERROR-level logs
    // during startup before localization comes up.
    if (!m_tf_buffer->canTransform(m_map_frame, m_robot_frame, tf2::TimePointZero))
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                           "VisMapToRobot: Waiting for TF %s -> %s (localization not yet active)",
                           m_map_frame.c_str(), m_robot_frame.c_str());
      return;
    }
    try
    {
      robot_z = m_tf_buffer->lookupTransform(m_map_frame, m_robot_frame, tf2::TimePointZero)
                  .transform.translation.z;
    }
    catch (tf2::TransformException& ex)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "VisMapToRobot: Could not transform %s to %s: %s",
                           m_map_frame.c_str(),
                           m_robot_frame.c_str(),
                           ex.what());
      return;
    }
  }

  visualization_msgs::msg::Marker visualization_marker_msg;
  sensor_msgs::msg::PointCloud2 cloud_msg;
  nav_msgs::msg::OccupancyGrid occupancy_grid_msg;

  // fetch the handle BEFORE locking — getGrid() locks internally, and a
  // recursive shared acquisition from the same thread is UB (deadlocks
  // outright when a writer is queued between the two acquisitions)
  auto grid = m_vdb_map->getGrid();
  std::shared_lock map_lock(*m_vdb_map->getMapMutex());
  VDBMappingTools<VDBMapT>::createMappingOutput(
    grid,
    m_map_frame,
    visualization_marker_msg,
    cloud_msg,
    occupancy_grid_msg,
    publish_vis_marker,
    publish_pointcloud,
    publish_occupancy_grid,
    robot_z + lower_z_limit,
    robot_z + upper_z_limit,
    m_resolution,
    m_two_dim_projection_threshold);
  map_lock.unlock();
  if (publish_vis_marker)
  {
    visualization_marker_msg.header.stamp = this->now();
    m_visualization_marker_pub->publish(visualization_marker_msg);
  }
  if (publish_pointcloud)
  {
    cloud_msg.header.stamp = this->now();
    m_pointcloud_pub->publish(cloud_msg);
  }
  if (publish_occupancy_grid)
  {
    occupancy_grid_msg.header.stamp    = this->now();
    occupancy_grid_msg.header.frame_id = m_map_frame;
    occupancy_grid_msg.info.resolution = m_resolution;
    m_occupancy_grid_pub->publish(occupancy_grid_msg);
  }
}

void VDBMappingROS2::mapSectionCallback(
  const vdb_mapping_interfaces::msg::UpdateGrid::SharedPtr update_msg,
  const std::shared_ptr<RemoteSource>& remote_source)
{
  if (!remote_source->active)
  {
    return;
  }
  // Deserialization throws on corrupt/truncated payloads and yields null on a
  // grid-type mismatch (e.g. a full section wired to a section subscription).
  // An uncaught throw in a subscription callback would take down the whole
  // component container, so drop bad sections instead.
  VDBMapT::UpdateGridT::Ptr section;
  try
  {
    section = m_vdb_map->byteArrayToGrid<VDBMapT::UpdateGridT>(update_msg->map);
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "MapSection: dropping undeserializable section: %s", ex.what());
    return;
  }
  if (!section)
  {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "MapSection: dropping section: payload is not an update grid");
    return;
  }
  if (m_map_frame == update_msg->header.frame_id)
  {
    m_vdb_map->applyMapSectionUpdateGrid(
      section, m_smooth_remote_sections, m_remote_section_smoothing_iterations);
  }
  else
  {
    geometry_msgs::msg::TransformStamped transform;
    try
    {
      transform =
        m_tf_buffer->lookupTransform(m_map_frame,
                                     update_msg->header.frame_id,
                                     update_msg->header.stamp,
                                     rclcpp::Duration::from_seconds(m_tf_lookup_timeout));
    }
    catch (tf2::TransformException& ex)
    {
      RCLCPP_ERROR(this->get_logger(),
                   "MapSection: Could not transform %s to %s: %s",
                   m_map_frame.c_str(),
                   update_msg->header.frame_id.c_str(),
                   ex.what());
      return;
    }
    m_vdb_map->transformAndApplyMapSectionUpdateGrid(
      section,
      tf2::transformToEigen(transform).matrix(),
      m_smooth_remote_sections,
      m_remote_section_smoothing_iterations);
  }
}

void VDBMappingROS2::mapFullSectionCallback(
  const vdb_mapping_interfaces::msg::UpdateGrid::SharedPtr update_msg,
  const std::shared_ptr<RemoteSource>& remote_source)
{
  if (!remote_source->active)
  {
    return;
  }
  // See mapSectionCallback: drop corrupt or type-mismatched payloads instead
  // of letting a deserialization throw terminate the container.
  VDBMapT::GridT::Ptr section;
  try
  {
    section = m_vdb_map->byteArrayToGrid<VDBMapT::GridT>(update_msg->map);
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "MapFullSection: dropping undeserializable section: %s", ex.what());
    return;
  }
  if (!section)
  {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "MapFullSection: dropping section: payload is not a map grid");
    return;
  }
  if (m_map_frame == update_msg->header.frame_id)
  {
    m_vdb_map->applyMapSectionGrid(
      section, m_smooth_remote_sections, m_remote_section_smoothing_iterations);
  }
  else
  {
    geometry_msgs::msg::TransformStamped transform;
    try
    {
      transform =
        m_tf_buffer->lookupTransform(m_map_frame,
                                     update_msg->header.frame_id,
                                     update_msg->header.stamp,
                                     rclcpp::Duration::from_seconds(m_tf_lookup_timeout));
    }
    catch (tf2::TransformException& ex)
    {
      RCLCPP_ERROR(this->get_logger(),
                   "MapFullSection: Could not transform %s to %s: %s",
                   m_map_frame.c_str(),
                   update_msg->header.frame_id.c_str(),
                   ex.what());
      return;
    }
    m_vdb_map->transformAndApplyMapSectionGrid(
      section,
      tf2::transformToEigen(transform).matrix(),
      m_smooth_remote_sections,
      m_remote_section_smoothing_iterations);
  }
}

bool VDBMappingROS2::resetMapCallback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
  const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  (void)req;
  resetMap();
  res->success = true;
  res->message = "Reset map successful.";
  return true;
}

bool VDBMappingROS2::getMapSectionCallback(
  const std::shared_ptr<vdb_mapping_interfaces::srv::GetMapSection::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::GetMapSection::Response> res)
{
  geometry_msgs::msg::TransformStamped source_to_map_tf;
  try
  {
    // The request's own stamp is used for the lookup; callers that want the
    // latest transform send a zero stamp, which tf2 treats as "latest
    // available".
    source_to_map_tf =
      m_tf_buffer->lookupTransform(m_map_frame,
                                   req->header.frame_id,
                                   rclcpp::Time(req->header.stamp),
                                   rclcpp::Duration::from_seconds(m_tf_lookup_timeout));
  }
  catch (tf2::TransformException& ex)
  {
    RCLCPP_ERROR(this->get_logger(),
                 "GetMapSection: Could not transform %s to %s: %s",
                 m_map_frame.c_str(),
                 req->header.frame_id.c_str(),
                 ex.what());
    res->success = false;
    return true;
  }
  res->section.map = m_vdb_map->gridToByteArray<VDBMapT::UpdateGridT>(
    m_vdb_map->getMapSectionUpdateGrid(
      Eigen::Matrix<double, 3, 1>(req->bounding_box.min_corner.x,
                                  req->bounding_box.min_corner.y,
                                  req->bounding_box.min_corner.z),
      Eigen::Matrix<double, 3, 1>(req->bounding_box.max_corner.x,
                                  req->bounding_box.max_corner.y,
                                  req->bounding_box.max_corner.z),
      tf2::transformToEigen(source_to_map_tf).matrix()));
  res->section.header.frame_id = m_map_frame;
  res->section.header.stamp    = this->now();
  res->success                 = true;
  return true;
}

bool VDBMappingROS2::getMapFullSectionCallback(
  const std::shared_ptr<vdb_mapping_interfaces::srv::GetMapSection::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::GetMapSection::Response> res)
{
  geometry_msgs::msg::TransformStamped source_to_map_tf;
  try
  {
    source_to_map_tf =
      m_tf_buffer->lookupTransform(m_map_frame,
                                   req->header.frame_id,
                                   rclcpp::Time(req->header.stamp),
                                   rclcpp::Duration::from_seconds(m_tf_lookup_timeout));
  }
  catch (tf2::TransformException& ex)
  {
    RCLCPP_ERROR(this->get_logger(),
                 "GetMapFullSection: Could not transform %s to %s: %s",
                 m_map_frame.c_str(),
                 req->header.frame_id.c_str(),
                 ex.what());
    res->success = false;
    return true;
  }
  // Full sections carry the probabilistic GridT (not the binary UpdateGridT
  // returned by get_map_section), matching what mapFullSectionCallback /
  // applyMapSectionGrid expect on the receiving side.
  res->section.map = m_vdb_map->gridToByteArray<VDBMapT::GridT>(
    m_vdb_map->getMapSectionGrid(
      Eigen::Matrix<double, 3, 1>(req->bounding_box.min_corner.x,
                                  req->bounding_box.min_corner.y,
                                  req->bounding_box.min_corner.z),
      Eigen::Matrix<double, 3, 1>(req->bounding_box.max_corner.x,
                                  req->bounding_box.max_corner.y,
                                  req->bounding_box.max_corner.z),
      tf2::transformToEigen(source_to_map_tf).matrix()));
  res->section.header.frame_id = m_map_frame;
  res->section.header.stamp    = this->now();
  res->success                 = true;
  return true;
}

bool VDBMappingROS2::triggerMapSectionUpdateCallback(
  const std::shared_ptr<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate::Response> res)
{
  auto remote_source = m_remote_sources.find(req->remote_source);
  if (remote_source == m_remote_sources.end())
  {
    std::stringstream ss;
    ss << "Key " << req->remote_source << " not found. Available sources are: ";
    for (auto& source : m_remote_sources)
    {
      ss << source.first << ", ";
    }
    RCLCPP_WARN(this->get_logger(), "%s", ss.str().c_str());
    res->success = false;
    return true;
  }

  if (!remote_source->second->get_map_section_client)
  {
    RCLCPP_WARN(this->get_logger(),
                "Remote source %s has apply_remote_sections=false; cannot trigger update",
                req->remote_source.c_str());
    res->success = false;
    return true;
  }
  if (!remote_source->second->active)
  {
    RCLCPP_WARN(this->get_logger(),
                "Remote source %s is currently deactivated; cannot trigger update",
                req->remote_source.c_str());
    res->success = false;
    return true;
  }

  auto request = std::make_shared<vdb_mapping_interfaces::srv::GetMapSection::Request>();
  request->header       = req->header;
  request->bounding_box = req->bounding_box;

  if (!remote_source->second->get_map_section_client->service_is_ready())
  {
    RCLCPP_WARN(this->get_logger(),
                "get_map_section service of remote source %s is not available",
                req->remote_source.c_str());
    res->success = false;
    return true;
  }

  // The response is a binary occupancy section snapshot (see
  // getMapSectionUpdateGrid in vdb_mapping). It must be applied with
  // replace-section semantics and honoring the remote map frame, exactly like
  // the passively subscribed sections, so reuse mapSectionCallback. Feeding it
  // to updateMap would treat every voxel as a single probabilistic hit, which
  // never crosses the occupancy threshold and bypasses the map mutex.
  auto remote = remote_source->second;
  remote_source->second->get_map_section_client->async_send_request(
    request,
    [this, remote](
      rclcpp::Client<vdb_mapping_interfaces::srv::GetMapSection>::SharedFuture future) {
      auto response = future.get();
      if (response->success)
      {
        auto update =
          std::make_shared<vdb_mapping_interfaces::msg::UpdateGrid>(response->section);
        mapSectionCallback(update, remote);
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "Remote get_map_section returned success=false");
      }
    });
  res->success = true;
  return true;
}

bool VDBMappingROS2::triggerMapFullSectionUpdateCallback(
  const std::shared_ptr<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate::Response> res)
{
  auto remote_source = m_remote_sources.find(req->remote_source);
  if (remote_source == m_remote_sources.end())
  {
    std::stringstream ss;
    ss << "Key " << req->remote_source << " not found. Available sources are: ";
    for (auto& source : m_remote_sources)
    {
      ss << source.first << ", ";
    }
    RCLCPP_WARN(this->get_logger(), "%s", ss.str().c_str());
    res->success = false;
    return true;
  }

  if (!remote_source->second->get_map_full_section_client)
  {
    RCLCPP_WARN(this->get_logger(),
                "Remote source %s has apply_remote_full_sections=false; cannot trigger update",
                req->remote_source.c_str());
    res->success = false;
    return true;
  }
  if (!remote_source->second->active)
  {
    RCLCPP_WARN(this->get_logger(),
                "Remote source %s is currently deactivated; cannot trigger update",
                req->remote_source.c_str());
    res->success = false;
    return true;
  }

  auto request = std::make_shared<vdb_mapping_interfaces::srv::GetMapSection::Request>();
  request->header       = req->header;
  request->bounding_box = req->bounding_box;

  if (!remote_source->second->get_map_full_section_client->service_is_ready())
  {
    RCLCPP_WARN(this->get_logger(),
                "get_map_full_section service of remote source %s is not available",
                req->remote_source.c_str());
    res->success = false;
    return true;
  }

  // Reuse mapFullSectionCallback so the response honors the remote map frame
  // (transformAndApply* when frames differ) instead of being applied blindly.
  auto remote = remote_source->second;
  remote_source->second->get_map_full_section_client->async_send_request(
    request,
    [this, remote](
      rclcpp::Client<vdb_mapping_interfaces::srv::GetMapSection>::SharedFuture future) {
      auto response = future.get();
      if (response->success)
      {
        auto update =
          std::make_shared<vdb_mapping_interfaces::msg::UpdateGrid>(response->section);
        mapFullSectionCallback(update, remote);
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "Remote get_map_full_section returned success=false");
      }
    });
  res->success = true;
  return true;
}

bool VDBMappingROS2::addPointsToGridCallback(
  const std::shared_ptr<vdb_mapping_interfaces::srv::AddPointsToGrid::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::AddPointsToGrid::Response> res)
{
  if (const char* error = cloudMsgError(req->points))
  {
    RCLCPP_ERROR(this->get_logger(), "AddPointsToGrid: rejecting request: %s", error);
    res->success = false;
    return true;
  }
  VDBMapT::PointCloudT::Ptr cloud(new VDBMapT::PointCloudT);
  pcl::fromROSMsg(req->points, *cloud);
  res->success = m_vdb_map->addPointsToGrid(cloud);
  return true;
}

bool VDBMappingROS2::removePointsFromGridCallback(
  const std::shared_ptr<vdb_mapping_interfaces::srv::RemovePointsFromGrid::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::RemovePointsFromGrid::Response> res)
{
  if (const char* error = cloudMsgError(req->points))
  {
    RCLCPP_ERROR(this->get_logger(), "RemovePointsFromGrid: rejecting request: %s", error);
    res->success = false;
    return true;
  }
  VDBMapT::PointCloudT::Ptr cloud(new VDBMapT::PointCloudT);
  pcl::fromROSMsg(req->points, *cloud);
  res->success = m_vdb_map->removePointsFromGrid(cloud);
  return true;
}

bool VDBMappingROS2::raytraceCallback(
  const std::shared_ptr<vdb_mapping_interfaces::srv::Raytrace::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::Raytrace::Response> res)
{
  auto batch_req = std::make_shared<vdb_mapping_interfaces::srv::BatchRaytrace::Request>();
  auto batch_res = std::make_shared<vdb_mapping_interfaces::srv::BatchRaytrace::Response>();

  batch_req->header = req->header;
  batch_req->rays.push_back(req->ray);
  batchRaytraceCallback(batch_req, batch_res);
  res->header    = batch_res->header;
  res->success   = batch_res->successes[0];
  res->end_point = batch_res->end_points[0];
  return true;
}

bool VDBMappingROS2::batchRaytraceCallback(
  const std::shared_ptr<vdb_mapping_interfaces::srv::BatchRaytrace::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::BatchRaytrace::Response> res)
{
  geometry_msgs::msg::TransformStamped reference_tf;
  res->header.frame_id = m_map_frame;
  res->header.stamp    = req->header.stamp;
  res->successes.resize(req->rays.size());
  res->end_points.resize(req->rays.size());
  try
  {
    reference_tf =
      m_tf_buffer->lookupTransform(m_map_frame,
                                   req->header.frame_id.c_str(),
                                   req->header.stamp,
                                   rclcpp::Duration::from_seconds(m_tf_lookup_timeout));
  }
  catch (tf2::TransformException& ex)
  {
    RCLCPP_ERROR_STREAM(this->get_logger(),
                        "BatchRaytrace: Transform to map frame failed: " << ex.what());
    for (size_t i = 0; i < req->rays.size(); i++)
    {
      res->successes[i]  = false;
      res->end_points[i] = geometry_msgs::msg::Point();
    }
    return true;
  }

  Eigen::Matrix<double, 4, 4> m = tf2::transformToEigen(reference_tf).matrix();

  // This intentionally does not use vdb_mapping::raytrace, which depends on
  // the volume ray intersector and therefore requires fast_mode plus a prior
  // integration. FZI devel additionally null-derefs without one (node crash),
  // skips the first voxel of each marched segment (missing one-voxel-thick
  // obstacles) and reports success with the segment end on a miss; newer
  // forks fix those but keep the fast_mode requirement. Walking the grid with
  // a plain DDA is exact, works in every mode and is entirely sufficient at
  // service rates.
  using RayT = openvdb::math::Ray<double>;
  using DDAT = openvdb::math::DDA<RayT, 0>;

  // handle first, then lock — see the note in setUpVDBMap()/publishMap()
  auto grid = m_vdb_map->getGrid();
  std::shared_lock map_lock(*m_vdb_map->getMapMutex());
  auto acc  = grid->getConstAccessor();
  for (size_t i = 0; i < req->rays.size(); i++)
  {
    Eigen::Matrix<double, 4, 1> origin, direction;
    origin << req->rays[i].origin.x, req->rays[i].origin.y, req->rays[i].origin.z, 1;
    direction << req->rays[i].direction.x, req->rays[i].direction.y, req->rays[i].direction.z, 0;

    origin    = m * origin;
    direction = m * direction;

    Eigen::Matrix<double, 3, 1> dir = direction.head<3>();
    const double max_ray_length     = req->rays[i].max_ray_length;
    bool success                    = false;
    geometry_msgs::msg::Point end_point;

    // Non-finite inputs would put NaN/inf coordinates into the DDA, and an
    // unbounded length walks the grid one voxel per step while the shared map
    // lock is held — a huge value would starve out map integration. Hence the
    // finite check and the max_raytrace_length clamp.
    if (!origin.allFinite() || !dir.allFinite() || !std::isfinite(max_ray_length))
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "BatchRaytrace: dropping ray with non-finite origin/direction/length");
      res->successes[i]  = false;
      res->end_points[i] = end_point;
      continue;
    }

    if (dir.norm() > 0.0 && max_ray_length > 0.0)
    {
      dir = dir.normalized() * std::min(max_ray_length, m_max_raytrace_length);
      const openvdb::Vec3d origin_world(origin.x(), origin.y(), origin.z());
      const openvdb::Vec3d dir_world(dir.x(), dir.y(), dir.z());
      // Voxels are cell-centered (vdb_mapping's worldToIndex rounds), while
      // the DDA cell convention is [i, i+1) — shift by half a voxel so
      // dda.voxel() yields proper cell-centered indices.
      const openvdb::Vec3d origin_index =
        grid->worldToIndex(origin_world) + openvdb::Vec3d(0.5);
      const openvdb::Vec3d dir_index = grid->worldToIndex(dir_world);

      RayT ray(origin_index, dir_index, 0.0, 1.0);
      DDAT dda(ray);
      do
      {
        const openvdb::Coord voxel = dda.voxel();
        if (acc.isValueOn(voxel))
        {
          const openvdb::Vec3d world = grid->indexToWorld(voxel);
          end_point.x                = world.x();
          end_point.y                = world.y();
          end_point.z                = world.z();
          success                    = true;
          break;
        }
      } while (dda.step());
    }
    else
    {
      dir.setZero();
    }

    if (!success)
    {
      end_point.x = origin.x() + dir.x();
      end_point.y = origin.y() + dir.y();
      end_point.z = origin.z() + dir.z();
    }
    res->successes[i]  = success;
    res->end_points[i] = end_point;
  }
  return true;
}

bool VDBMappingROS2::addArtificialAreasCallback(
  const std::shared_ptr<vdb_mapping_interfaces::srv::AddArtificialAreas::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::AddArtificialAreas::Response> res)
{
  std::vector<std::vector<Eigen::Matrix<double, 4, 1>>> artificial_areas;
  if (req->artificial_areas.size() > 0)
  {
    geometry_msgs::msg::TransformStamped source_to_map_tf;
    try
    {
      // The request's own stamp is used for the lookup; callers that want the
      // latest transform send a zero stamp, which tf2 treats as "latest
      // available".
      source_to_map_tf =
        m_tf_buffer->lookupTransform(m_map_frame,
                                     req->artificial_areas[0].header.frame_id,
                                     rclcpp::Time(req->artificial_areas[0].header.stamp),
                                     rclcpp::Duration::from_seconds(m_tf_lookup_timeout));
    }
    catch (tf2::TransformException& ex)
    {
      RCLCPP_ERROR(this->get_logger(),
                   "ArtificialArea: Could not transform %s to %s: %s",
                   m_map_frame.c_str(),
                   req->artificial_areas[0].header.frame_id.c_str(),
                   ex.what());
      res->success = false;
      return true;
    }
    Eigen::Matrix<double, 4, 4> transform = tf2::transformToEigen(source_to_map_tf).matrix();

    for (auto& artificial_area : req->artificial_areas)
    {
      std::vector<Eigen::Matrix<double, 4, 1>> area;
      for (auto& p : artificial_area.polygon.points)
      {
        area.push_back(transform * Eigen::Matrix<double, 4, 1>(p.x, p.y, p.z, 1.0));
      }
      artificial_areas.push_back(area);
    }
  }
  m_vdb_map->addArtificialAreas(
    artificial_areas, m_artificial_negative_height, m_artificial_positive_height);
  res->success = true;
  return true;
}

bool VDBMappingROS2::removeArtificialAreasCallback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
  const std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  (void)req;
  m_vdb_map->restoreMapIntegrity();
  res->success = true;
  return true;
}

bool VDBMappingROS2::toggleRemoteSource(
  const std::shared_ptr<vdb_mapping_interfaces::srv::ToggleRemoteSource::Request> req,
  const std::shared_ptr<vdb_mapping_interfaces::srv::ToggleRemoteSource::Response> res)
{
  auto remote_source = m_remote_sources.find(req->remote_source);
  if (remote_source == m_remote_sources.end())
  {
    std::stringstream ss;
    ss << "Key " << req->remote_source << " not found. Available sources are: ";
    for (auto& source : m_remote_sources)
    {
      ss << source.first << ", ";
    }
    RCLCPP_WARN(this->get_logger(), "%s", ss.str().c_str());
    res->success = false;
    return true;
  }
  remote_source->second->active = req->toggle;
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Remote source " << req->remote_source << " set to "
                                      << (remote_source->second->active ? "active" : "inactive"));
  res->success = true;
  return true;
}

void VDBMappingROS2::visualizationTimerCallback() { publishMap(); }

void VDBMappingROS2::sectionTimerCallback()
{
  geometry_msgs::msg::TransformStamped map_to_robot_tf;
  try
  {
    map_to_robot_tf =
      m_tf_buffer->lookupTransform(m_map_frame,
                                   m_section_update_frame,
                                   rclcpp::Time(0),
                                   rclcpp::Duration::from_seconds(m_tf_lookup_timeout));
  }
  catch (tf2::TransformException& ex)
  {
    RCLCPP_ERROR(this->get_logger(),
                 "SectionTimer: Could not transform %s to %s: %s",
                 m_map_frame.c_str(),
                 m_section_update_frame.c_str(),
                 ex.what());
    return;
  }
  VDBMapT::UpdateGridT::Ptr section = m_vdb_map->getMapSectionUpdateGrid(
    m_section_min_coord, m_section_max_coord, tf2::transformToEigen(map_to_robot_tf).matrix());
  vdb_mapping_interfaces::msg::UpdateGrid msg;
  msg.header.frame_id = m_map_frame;
  msg.header.stamp    = map_to_robot_tf.header.stamp;
  msg.map             = m_vdb_map->gridToByteArray<VDBMapT::UpdateGridT>(section);
  m_map_section_pub->publish(msg);
}

void VDBMappingROS2::fullSectionTimerCallback()
{
  geometry_msgs::msg::TransformStamped map_to_robot_tf;
  try
  {
    map_to_robot_tf =
      m_tf_buffer->lookupTransform(m_map_frame,
                                   m_section_update_frame,
                                   rclcpp::Time(0),
                                   rclcpp::Duration::from_seconds(m_tf_lookup_timeout));
  }
  catch (tf2::TransformException& ex)
  {
    RCLCPP_ERROR(this->get_logger(),
                 "FullSectionTimer: Could not transform %s to %s: %s",
                 m_map_frame.c_str(),
                 m_section_update_frame.c_str(),
                 ex.what());
    return;
  }

  VDBMapT::GridT::Ptr section = m_vdb_map->getMapSectionGrid(
    m_section_min_coord, m_section_max_coord, tf2::transformToEigen(map_to_robot_tf).matrix());
  vdb_mapping_interfaces::msg::UpdateGrid msg;
  msg.header.frame_id = m_map_frame;
  msg.header.stamp    = map_to_robot_tf.header.stamp;
  msg.map             = m_vdb_map->gridToByteArray<VDBMapT::GridT>(section);
  m_map_full_section_pub->publish(msg);
}

void VDBMappingROS2::setUpVDBMap()
{
  this->declare_parameter<bool>("fast_mode", false);
  this->get_parameter("fast_mode", m_config.fast_mode);
  this->declare_parameter<double>("accumulation_period", 1);
  this->get_parameter("accumulation_period", m_config.accumulation_period);
  this->declare_parameter<double>("resolution", 0.05);
  this->get_parameter("resolution", m_resolution);
  m_vdb_map = std::make_shared<VDBMapT>(m_resolution);
  trySetLogCallback(*m_vdb_map, this->get_logger(), 0);

  this->declare_parameter<double>("max_range", 10.0);
  this->get_parameter("max_range", m_config.max_range);
  this->declare_parameter<double>("prob_hit", 0.7);
  this->get_parameter("prob_hit", m_config.prob_hit);
  this->declare_parameter<double>("prob_miss", 0.4);
  this->get_parameter("prob_miss", m_config.prob_miss);
  this->declare_parameter<double>("prob_thres_min", 0.12);
  this->get_parameter("prob_thres_min", m_config.prob_thres_min);
  this->declare_parameter<double>("prob_thres_max", 0.97);
  this->get_parameter("prob_thres_max", m_config.prob_thres_max);
  this->declare_parameter<std::string>("map_directory_path", "");
  this->get_parameter("map_directory_path", m_config.map_directory_path);
  this->declare_parameter<int>("two_dim_projection_threshold", 5);
  this->get_parameter("two_dim_projection_threshold", m_two_dim_projection_threshold);
  this->declare_parameter<double>("tf_lookup_timeout", 0.1);
  this->get_parameter("tf_lookup_timeout", m_tf_lookup_timeout);
  this->declare_parameter<double>("max_raytrace_length", 1000.0);
  this->get_parameter("max_raytrace_length", m_max_raytrace_length);
  if (m_max_raytrace_length <= 0.0)
  {
    RCLCPP_WARN(this->get_logger(),
                "max_raytrace_length must be positive; falling back to 1000 m");
    m_max_raytrace_length = 1000.0;
  }
  this->declare_parameter<bool>("smooth_remote_sections", false);
  this->get_parameter("smooth_remote_sections", m_smooth_remote_sections);
  this->declare_parameter<int>("remote_section_smoothing_iterations", 2);
  this->get_parameter("remote_section_smoothing_iterations",
                      m_remote_section_smoothing_iterations);
  this->declare_parameter<double>("artificial_negative_height", -0.5);
  this->get_parameter("artificial_negative_height", m_artificial_negative_height);
  this->declare_parameter<double>("artificial_positive_height", 1.5);
  this->get_parameter("artificial_positive_height", m_artificial_positive_height);

  m_vdb_map->setConfig(m_config);

  this->declare_parameter<std::string>("map_frame", "");
  this->get_parameter("map_frame", m_map_frame);
  if (m_map_frame.empty())
  {
    RCLCPP_WARN(this->get_logger(), "No map frame specified");
  }
  // getGrid() takes the map mutex internally — fetch the handle BEFORE
  // locking (locking first recursively acquires the non-recursive
  // shared_mutex from this thread: EDEADLK, constructor throws)
  {
    auto grid = m_vdb_map->getGrid();
    std::unique_lock map_lock(*m_vdb_map->getMapMutex());
    grid->insertMeta("ros/map_frame", openvdb::StringMetadata(m_map_frame));
  }
  this->declare_parameter<std::string>("robot_frame", "");
  this->get_parameter("robot_frame", m_robot_frame);
  if (m_robot_frame.empty())
  {
    RCLCPP_WARN(this->get_logger(), "No robot frame specified");
  }
}

void VDBMappingROS2::setUpLocalSources()
{
  this->declare_parameter<bool>("apply_raw_sensor_data", true);
  this->get_parameter("apply_raw_sensor_data", m_apply_raw_sensor_data);

  if (!m_apply_raw_sensor_data)
  {
    return;
  }

  // Read accumulate_updates BEFORE any subscription goes live: cloudCallback
  // dereferences this on every message and a stray uninitialised bool here
  // is undefined behaviour the moment the first cloud arrives.
  this->declare_parameter<bool>("accumulate_updates", false);
  this->get_parameter("accumulate_updates", m_accumulate_updates);

  std::vector<std::string> source_ids;
  this->declare_parameter<std::vector<std::string>>("sources", std::vector<std::string>());
  this->get_parameter("sources", source_ids);

  // The subscriptions created in pass 2 capture references to elements of
  // m_sensor_sources, so the vector must not be modified after that point.
  m_sensor_sources.reserve(source_ids.size());

  // Pass 1 — declare params, populate m_sensor_sources, register input sources
  // with the VDB map. No subscriptions yet: we want addInputSource() to be in
  // place for every source before any cloudCallback can fire for that source.
  for (auto& source_id : source_ids)
  {
    SensorSource sensor_source;
    sensor_source.source_id = source_id;
    this->declare_parameter<std::string>(source_id + ".topic", "");
    this->get_parameter(source_id + ".topic", sensor_source.topic);
    this->declare_parameter<std::string>(source_id + ".sensor_origin_frame", "");
    this->get_parameter(source_id + ".sensor_origin_frame", sensor_source.sensor_origin_frame);
    this->declare_parameter<double>(source_id + ".max_range", 0);
    this->get_parameter(source_id + ".max_range", sensor_source.max_range);
    this->declare_parameter<double>(source_id + ".max_rate", 0);
    this->get_parameter(source_id + ".max_rate", sensor_source.max_rate);
    this->declare_parameter<bool>(source_id + ".reliable", false);
    this->get_parameter(source_id + ".reliable", sensor_source.reliable);
    this->declare_parameter<bool>(source_id + ".ray_clearing", true);
    this->get_parameter(source_id + ".ray_clearing", sensor_source.ray_clearing);
    this->declare_parameter<bool>(source_id + ".endpoint_hits", true);
    this->get_parameter(source_id + ".endpoint_hits", sensor_source.endpoint_hits);
    this->declare_parameter<double>(source_id + ".prob_hit", -1.0);
    this->get_parameter(source_id + ".prob_hit", sensor_source.prob_hit);
    this->declare_parameter<double>(source_id + ".prob_miss", -1.0);
    this->get_parameter(source_id + ".prob_miss", sensor_source.prob_miss);
    RCLCPP_INFO_STREAM(this->get_logger(), "Setting up source: " << source_id);

    if (sensor_source.topic.empty())
    {
      RCLCPP_ERROR_STREAM(this->get_logger(),
                          "No input topic specified for source: " << source_id);
      continue;
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "Topic: " << sensor_source.topic);
    if (sensor_source.sensor_origin_frame.empty())
    {
      RCLCPP_INFO(this->get_logger(), "Using frame id of topic as raycast origin");
    }
    else
    {
      RCLCPP_INFO_STREAM(this->get_logger(),
                         "Using " << sensor_source.sensor_origin_frame << " as raycast origin");
    }

    m_sensor_sources.push_back(std::move(sensor_source));
    const SensorSource& stored = m_sensor_sources.back();
    RCLCPP_INFO(this->get_logger(),
                "Source behavior: ray_clearing=%s endpoint_hits=%s prob_hit=%.2f prob_miss=%.2f",
                stored.ray_clearing ? "true" : "false",
                stored.endpoint_hits ? "true" : "false",
                stored.prob_hit,
                stored.prob_miss);
    m_vdb_map->addInputSource(stored.source_id,
                              stored.max_range,
                              stored.max_rate,
                              stored.ray_clearing,
                              stored.endpoint_hits,
                              stored.prob_hit,
                              stored.prob_miss);
  }

  // Pass 2 — create the cloud subscriptions. After this, cloudCallback can
  // fire on m_accumulation_cb_group from a different executor thread.
  rclcpp::SubscriptionOptions opt;
  opt.callback_group = m_accumulation_cb_group;
  for (const SensorSource& stored : m_sensor_sources)
  {
    rclcpp::QoS qos_profile(1);
    if (stored.reliable)
    {
      qos_profile = qos_profile.durability_volatile().reliable();
    }
    else
    {
      qos_profile = qos_profile.durability_volatile().best_effort();
    }
    m_cloud_subs.push_back(this->create_subscription<sensor_msgs::msg::PointCloud2>(
      stored.topic,
      qos_profile,
      [this, &stored](const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
        cloudCallback(cloud_msg, stored);
      },
      opt));
  }
}

void VDBMappingROS2::setUpRemoteSources()
{
  std::vector<std::string> source_ids;
  this->declare_parameter<std::vector<std::string>>("remote_sources", std::vector<std::string>());
  this->get_parameter("remote_sources", source_ids);

  for (auto& source_id : source_ids)
  {
    RCLCPP_INFO_STREAM(this->get_logger(), "Setting up remote source: " << source_id);

    std::string remote_namespace;
    this->declare_parameter<std::string>(source_id + ".namespace", "");
    this->get_parameter(source_id + ".namespace", remote_namespace);
    if (remote_namespace.empty())
    {
      RCLCPP_WARN_STREAM(this->get_logger(),
                         "Remote source " << source_id
                           << " has no namespace configured; its topics and services will "
                              "resolve at the root namespace");
    }

    auto remote_source = std::make_shared<RemoteSource>();
    this->declare_parameter<bool>(source_id + ".apply_remote_sections", false);
    this->get_parameter(source_id + ".apply_remote_sections",
                        remote_source->apply_remote_sections);
    this->declare_parameter<bool>(source_id + ".apply_remote_full_sections", false);
    this->get_parameter(source_id + ".apply_remote_full_sections",
                        remote_source->apply_remote_full_sections);
    bool autostart = true;
    this->declare_parameter<bool>(source_id + ".autostart", true);
    this->get_parameter(source_id + ".autostart", autostart);
    remote_source->active = autostart;

    if (remote_source->apply_remote_sections)
    {
      remote_source->map_section_sub =
        this->create_subscription<vdb_mapping_interfaces::msg::UpdateGrid>(
          remote_namespace + "/vdb_map_sections",
          rclcpp::QoS(10).durability_volatile().best_effort(),
          [this, remote_source](
            const vdb_mapping_interfaces::msg::UpdateGrid::SharedPtr msg) {
            mapSectionCallback(msg, remote_source);
          });
      RCLCPP_INFO_STREAM(this->get_logger(),
                         "Subscribing to Section: " << remote_namespace + "/vdb_map_sections");
    }
    if (remote_source->apply_remote_full_sections)
    {
      remote_source->map_full_section_sub =
        this->create_subscription<vdb_mapping_interfaces::msg::UpdateGrid>(
          remote_namespace + "/vdb_map_full_sections",
          rclcpp::QoS(10).durability_volatile().best_effort(),
          [this, remote_source](
            const vdb_mapping_interfaces::msg::UpdateGrid::SharedPtr msg) {
            mapFullSectionCallback(msg, remote_source);
          });
      RCLCPP_INFO_STREAM(this->get_logger(),
                         "Subscribing to Full Section: " << remote_namespace +
                                                              "/vdb_map_full_sections");
    }
    if (remote_source->apply_remote_sections)
    {
      remote_source->get_map_section_client =
        this->create_client<vdb_mapping_interfaces::srv::GetMapSection>(remote_namespace +
                                                                        "/get_map_section");
    }
    if (remote_source->apply_remote_full_sections)
    {
      remote_source->get_map_full_section_client =
        this->create_client<vdb_mapping_interfaces::srv::GetMapSection>(remote_namespace +
                                                                        "/get_map_full_section");
    }
    m_remote_sources.insert(std::make_pair(source_id, remote_source));
  }
}

void VDBMappingROS2::setUpVisualization()
{
  double z_limit_min = 0.0;
  double z_limit_max = 0.0;
  this->declare_parameter<double>("z_limit_min", 0);
  this->get_parameter("z_limit_min", z_limit_min);
  this->declare_parameter<double>("z_limit_max", 0);
  this->get_parameter("z_limit_max", z_limit_max);
  m_lower_visualization_z_limit = z_limit_min;
  m_upper_visualization_z_limit = z_limit_max;

  m_param_sub = std::make_shared<rclcpp::ParameterEventHandler>(this);

  auto min_z_cb = [this](const rclcpp::Parameter& p) {
    m_lower_visualization_z_limit = p.as_double();
  };
  auto max_z_cb = [this](const rclcpp::Parameter& p) {
    m_upper_visualization_z_limit = p.as_double();
  };

  m_z_min_param_handle = m_param_sub->add_parameter_callback("z_limit_min", min_z_cb);
  m_z_max_param_handle = m_param_sub->add_parameter_callback("z_limit_max", max_z_cb);

  double visualization_rate;
  this->declare_parameter<double>("visualization_rate", 1.0);
  this->get_parameter("visualization_rate", visualization_rate);
  if (visualization_rate > 0.0)
  {
    m_visualization_timer = this->create_wall_timer(
      std::chrono::milliseconds(std::max(1, (int)(1000.0 / visualization_rate))),
      std::bind(&VDBMappingROS2::visualizationTimerCallback, this),
      m_visualization_cb_group);
  }
}

void VDBMappingROS2::setUpServices()
{
  using namespace std::placeholders;
  m_reset_map_service = this->create_service<std_srvs::srv::Trigger>(
    "~/reset_map", std::bind(&VDBMappingROS2::resetMapCallback, this, _1, _2));
  m_save_map_service = this->create_service<std_srvs::srv::Trigger>(
    "~/save_map", std::bind(&VDBMappingROS2::saveMap, this, _1, _2));
  m_save_map_to_pcd_service = this->create_service<std_srvs::srv::Trigger>(
    "~/save_map_to_pcd", std::bind(&VDBMappingROS2::saveMapToPCD, this, _1, _2));
  m_load_map_service = this->create_service<vdb_mapping_interfaces::srv::LoadMap>(
    "~/load_map", std::bind(&VDBMappingROS2::loadMap, this, _1, _2));
  m_load_map_from_pcd_service = this->create_service<vdb_mapping_interfaces::srv::LoadMapFromPCD>(
    "~/load_map_from_pcd", std::bind(&VDBMappingROS2::loadMapFromPCD, this, _1, _2));
  m_get_map_section_service = this->create_service<vdb_mapping_interfaces::srv::GetMapSection>(
    "~/get_map_section", std::bind(&VDBMappingROS2::getMapSectionCallback, this, _1, _2));
  m_get_map_full_section_service = this->create_service<vdb_mapping_interfaces::srv::GetMapSection>(
    "~/get_map_full_section", std::bind(&VDBMappingROS2::getMapFullSectionCallback, this, _1, _2));
  m_trigger_map_section_update_service =
    this->create_service<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate>(
      "~/trigger_map_section_update",
      std::bind(&VDBMappingROS2::triggerMapSectionUpdateCallback, this, _1, _2));
  m_trigger_map_full_section_update_service =
    this->create_service<vdb_mapping_interfaces::srv::TriggerMapSectionUpdate>(
      "~/trigger_map_full_section_update",
      std::bind(&VDBMappingROS2::triggerMapFullSectionUpdateCallback, this, _1, _2));
  m_raytrace_service = this->create_service<vdb_mapping_interfaces::srv::Raytrace>(
    "~/raytrace", std::bind(&VDBMappingROS2::raytraceCallback, this, _1, _2));
  m_batch_raytrace_service = this->create_service<vdb_mapping_interfaces::srv::BatchRaytrace>(
    "~/batch_raytrace", std::bind(&VDBMappingROS2::batchRaytraceCallback, this, _1, _2));
  m_add_points_to_grid_service =
    this->create_service<vdb_mapping_interfaces::srv::AddPointsToGrid>(
      "~/add_points_to_grid", std::bind(&VDBMappingROS2::addPointsToGridCallback, this, _1, _2));
  m_remove_points_from_grid_service =
    this->create_service<vdb_mapping_interfaces::srv::RemovePointsFromGrid>(
      "~/remove_points_from_grid",
      std::bind(&VDBMappingROS2::removePointsFromGridCallback, this, _1, _2));
  m_add_artificial_areas_service =
    this->create_service<vdb_mapping_interfaces::srv::AddArtificialAreas>(
      "~/add_artificial_areas",
      std::bind(&VDBMappingROS2::addArtificialAreasCallback, this, _1, _2));
  m_remove_artificial_areas_service = this->create_service<std_srvs::srv::Trigger>(
    "~/remove_artificial_areas",
    std::bind(&VDBMappingROS2::removeArtificialAreasCallback, this, _1, _2));
  m_toggle_remote_source_service =
    this->create_service<vdb_mapping_interfaces::srv::ToggleRemoteSource>(
      "~/toggle_remote_source", std::bind(&VDBMappingROS2::toggleRemoteSource, this, _1, _2));
}

void VDBMappingROS2::setUpPublishers()
{
  this->declare_parameter<bool>("publish_pointcloud", true);
  this->get_parameter("publish_pointcloud", m_publish_pointcloud);
  this->declare_parameter<bool>("publish_vis_marker", true);
  this->get_parameter("publish_vis_marker", m_publish_vis_marker);
  this->declare_parameter<bool>("publish_occupancy_grid", true);
  this->get_parameter("publish_occupancy_grid", m_publish_occupancy_grid);
  this->declare_parameter<bool>("publish_sections", false);
  this->get_parameter("publish_sections", m_publish_sections);
  this->declare_parameter<bool>("publish_full_sections", false);
  this->get_parameter("publish_full_sections", m_publish_full_sections);

  if (m_publish_pointcloud)
  {
    m_pointcloud_pub =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("~/vdb_map_pointcloud", 1);
  }
  if (m_publish_vis_marker)
  {
    m_visualization_marker_pub =
      this->create_publisher<visualization_msgs::msg::Marker>("~/vdb_map_visualization", 1);
  }
  if (m_publish_occupancy_grid)
  {
    m_occupancy_grid_pub =
      this->create_publisher<nav_msgs::msg::OccupancyGrid>("~/vdb_map_occupancy", 1);
  }

  // section_update.* params are shared between the sparse and full section
  // timers; declare once so both flags can be enabled together.
  double section_update_rate = 1.0;
  if (m_publish_sections || m_publish_full_sections)
  {
    this->declare_parameter<double>("section_update.rate", 1.0);
    this->get_parameter("section_update.rate", section_update_rate);
    this->declare_parameter<double>("section_update.min_coord.x", -10);
    this->get_parameter("section_update.min_coord.x", m_section_min_coord.x());
    this->declare_parameter<double>("section_update.min_coord.y", -10);
    this->get_parameter("section_update.min_coord.y", m_section_min_coord.y());
    this->declare_parameter<double>("section_update.min_coord.z", -10);
    this->get_parameter("section_update.min_coord.z", m_section_min_coord.z());
    this->declare_parameter<double>("section_update.max_coord.x", 10);
    this->get_parameter("section_update.max_coord.x", m_section_max_coord.x());
    this->declare_parameter<double>("section_update.max_coord.y", 10);
    this->get_parameter("section_update.max_coord.y", m_section_max_coord.y());
    this->declare_parameter<double>("section_update.max_coord.z", 10);
    this->get_parameter("section_update.max_coord.z", m_section_max_coord.z());
    this->declare_parameter<std::string>("section_update.frame", m_robot_frame);
    this->get_parameter("section_update.frame", m_section_update_frame);
  }

  if ((m_publish_sections || m_publish_full_sections) && section_update_rate <= 0.0)
  {
    RCLCPP_WARN(this->get_logger(),
                "section_update.rate must be positive; section publishing disabled");
    m_publish_sections      = false;
    m_publish_full_sections = false;
  }
  auto section_update_period = std::chrono::milliseconds(
    section_update_rate > 0.0 ? std::max(1, (int)(1000.0 / section_update_rate)) : 1);

  if (m_publish_sections)
  {
    m_map_section_pub = this->create_publisher<vdb_mapping_interfaces::msg::UpdateGrid>(
      "~/vdb_map_sections", rclcpp::QoS(1).durability_volatile().best_effort());
    m_section_timer =
      this->create_wall_timer(section_update_period,
                              std::bind(&VDBMappingROS2::sectionTimerCallback, this),
                              m_remote_cb_group);
  }
  if (m_publish_full_sections)
  {
    m_map_full_section_pub = this->create_publisher<vdb_mapping_interfaces::msg::UpdateGrid>(
      "~/vdb_map_full_sections", rclcpp::QoS(1).durability_volatile().best_effort());
    m_full_section_timer =
      this->create_wall_timer(section_update_period,
                              std::bind(&VDBMappingROS2::fullSectionTimerCallback, this),
                              m_remote_cb_group);
  }
}

void VDBMappingROS2::setUpMapServer()
{
  std::string initial_map_file;
  bool set_background;
  bool clear_map;
  this->declare_parameter<std::string>("map_server.initial_map_file", "");
  this->get_parameter("map_server.initial_map_file", initial_map_file);
  this->declare_parameter<bool>("map_server.set_background", false);
  this->get_parameter("map_server.set_background", set_background);
  this->declare_parameter<bool>("map_server.clear_map", false);
  this->get_parameter("map_server.clear_map", clear_map);
  if (!initial_map_file.empty())
  {
    RCLCPP_INFO_STREAM(this->get_logger(), "Loading initial Map " << initial_map_file);
    m_vdb_map->loadMapFromPCD(initial_map_file, set_background, clear_map);
    publishMap();
  }
}

}  // namespace vdb_mapping_ros2

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vdb_mapping_ros2::VDBMappingROS2)
