// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// Copyright 2026 (fork addition)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// -- END LICENSE BLOCK ------------------------------------------------

/*!
 * \brief Graph-anchored map assembler (rtabmap GridGlobal-inspired).
 *
 * The monolithic odom-frame map integrates evidence once, at whatever pose
 * odometry believed at the time: drift smears voxels irreversibly and SLAM
 * corrections never reach them. This node instead snapshots per-keyframe
 * evidence (a fill cloud + a clearing cloud, stored in the robot frame) and
 * renders the global map in the SLAM map frame from the CURRENT optimized
 * keyframe poses — appending new keyframes cheaply and re-rendering from
 * scratch when the graph moves, so loop closures and scan-match corrections
 * retroactively straighten the map.
 *
 * Inputs:
 *  - hits cloud (fill evidence; e.g. sonar candidates with elevation)
 *  - clear cloud (clearing-ray endpoints; e.g. per-beam first-return rays)
 *  - SLAM trajectory cloud with fields [x y z roll pitch yaw i t]; `t` is
 *    each keyframe's stamp relative to the message stamp — used to associate
 *    buffered evidence clouds (same source-ping stamps) to keyframes.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Geometry>
// SurveyPoint is a custom point type: the prebuilt PCL libraries only carry
// explicit template instantiations for the built-in types, so this TU must
// instantiate PCLBase/VoxelGrid<SurveyPoint> itself from the impl headers
#define PCL_NO_PRECOMPILE
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/impl/pcl_base.hpp>
// PCD read/write for the custom point types: same PCL_NO_PRECOMPILE reason as
// the filter impls above -- the prebuilt libraries carry no instantiation for
// SurveyPoint/SurveyExportPoint/SurfaceExportPoint, so this TU must supply its
// own.
#include <pcl/io/pcd_io.h>
#include <pcl/io/impl/pcd_io.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

#include <vdb_mapping/OccupancyVDBMapping.hpp>
#include <vdb_mapping_ros2/VDBMappingTools.hpp>

// SurveyPoint and its all-fields voxel reduction live in survey_voxel.hpp:
// pcl::VoxelGrid's closed accumulator set silently zeroed every custom field
// (texture moments, elevation bounds, range, incidence) during reduction,
// and the zeros passed every downstream validity guard.
#include <vdb_mapping_ros2/survey_voxel.hpp>
#include <vdb_mapping_ros2/sonar_reconstruction.hpp>

// Consolidated survey product written to PCD: the same thirteen fields the
// ~/survey_pointcloud topic carries.
//
// This struct is deliberately NOT layout-compatible with that topic. The
// message is tightly packed while PCL_ADD_POINT4D puts a 4-byte hole after z.
// That is fine and not worth "fixing": a PCD file declares its own field table
// in the header and every reader addresses fields by name, so the file is
// self-describing regardless of in-memory padding. The topic keeps
// hand-packing its rows.
struct SurveyExportPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float range;
  float incidence;
  float support;
  float pose_sigma;
  float texture;
  float texture_variance;
  float elevation_lo_offset;
  float elevation_hi_offset;
  float elevation_resolved_fraction;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
  SurveyExportPoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    float, range, range)(float, incidence, incidence)(float, support, support)(
    float, pose_sigma, pose_sigma)(float, texture, texture)(
    float, texture_variance, texture_variance)(
    float, elevation_lo_offset, elevation_lo_offset)(
    float, elevation_hi_offset, elevation_hi_offset)(
    float, elevation_resolved_fraction, elevation_resolved_fraction))

// Dense graph-corrected navigation returns written to PCD. This intentionally
// has the same named fields as the live ~/navigation_pointcloud topic: survey
// ranging/uncertainty remains present on every admitted return, while the
// normal fields are explicitly optional through normal_valid.
struct NavigationExportPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float range;
  float incidence;
  float support;
  float confidence;
  float pose_sigma;
  float texture;
  float texture_variance;
  float elevation_lo_offset;
  float elevation_hi_offset;
  float elevation_resolved_fraction;
  float normal_x;
  float normal_y;
  float normal_z;
  float curvature;
  float residual;
  float normal_valid;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
  NavigationExportPoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    float, range, range)(float, incidence, incidence)(float, support, support)(
    float, confidence, confidence)(float, pose_sigma, pose_sigma)(
    float, texture, texture)(float, texture_variance, texture_variance)(
    float, elevation_lo_offset, elevation_lo_offset)(
    float, elevation_hi_offset, elevation_hi_offset)(
    float, elevation_resolved_fraction, elevation_resolved_fraction)(
    float, normal_x, normal_x)(float, normal_y, normal_y)(
    float, normal_z, normal_z)(float, curvature, curvature)(
    float, residual, residual)(float, normal_valid, normal_valid))

// Strict graph-corrected navigation surfels written to PCD. This keeps the
// former navigation schema on the new ~/navigation_surfel_pointcloud topic for
// registration consumers that require every row to have a trustworthy normal.
struct SurfaceExportPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float support;
  float view_span_deg;
  float confidence;
  float normal_x;
  float normal_y;
  float normal_z;
  float curvature;
  float residual;
  float range_sigma;
  float echo_width;
  float echo_prominence;
  float peak_prominence;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
  SurfaceExportPoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    float, support, support)(float, view_span_deg, view_span_deg)(
    float, confidence, confidence)(float, normal_x, normal_x)(
    float, normal_y, normal_y)(float, normal_z, normal_z)(
    float, curvature, curvature)(float, residual, residual)(
    float, range_sigma, range_sigma)(float, echo_width, echo_width)(
    float, echo_prominence, echo_prominence)(
    float, peak_prominence, peak_prominence))

namespace vdb_mapping_ros2 {

constexpr std::size_t kSurveyOutputFields = 13;
using SurveyRow = std::array<float, kSurveyOutputFields>;

struct NavigationRow
{
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
  float intensity = 0.0F;
  float range = 0.0F;
  float incidence = -1.0F;
  float support = 0.0F;
  float confidence = 0.0F;
  float pose_sigma = 0.0F;
  float texture = 0.0F;
  float texture_variance = 0.0F;
  float elevation_lo_offset = 0.0F;
  float elevation_hi_offset = 0.0F;
  float elevation_resolved_fraction = 0.0F;
  float normal_x = 0.0F;
  float normal_y = 0.0F;
  float normal_z = 0.0F;
  // Negative curvature/residual plus normal_valid=0 are deliberate, finite
  // invalid markers. They keep generic PointCloud2 readers from deleting the
  // measured row under skip_nans while making normal availability explicit.
  float curvature = -1.0F;
  float residual = -1.0F;
  float normal_valid = 0.0F;
};

class VDBMapAssembler : public rclcpp::Node
{
public:
  using VDBMapT   = vdb_mapping::OccupancyVDBMapping;
  using PointT    = pcl::PointXYZ;
  using CloudT    = pcl::PointCloud<PointT>;
  using CloudPtrT = CloudT::Ptr;
  using SurveyCloudT    = pcl::PointCloud<SurveyPoint>;
  using SurveyCloudPtrT = SurveyCloudT::Ptr;
  using ReconstructionCloudT = ReconstructionCloud;
  using ReconstructionCloudPtrT = ReconstructionCloudPtr;

  VDBMapAssembler()
    : Node("vdb_map_assembler")
  {
    declare_parameter<double>("resolution", 0.1);
    declare_parameter<double>("max_range", 25.0);
    declare_parameter<double>("prob_hit", 0.75);
    declare_parameter<double>("prob_miss", 0.40);
    declare_parameter<double>("prob_thres_min", 0.12);
    declare_parameter<double>("prob_thres_max", 0.85);
    declare_parameter<std::string>("map_frame", "map");
    declare_parameter<std::string>("robot_frame", "base_link");
    declare_parameter<std::string>("hits_topic", "");
    declare_parameter<std::string>("clear_topic", "");
    declare_parameter<std::string>("traj_topic", "/bruce/slam/slam/traj");
    // Dense graph-anchored SURVEY product: aggregate ALL inter-keyframe
    // survey clouds into each keyframe's frame (odom-delta transforms), then
    // re-render intensity, texture moments and elevation uncertainty at the
    // optimized poses alongside the occupancy map. Empty disables.
    declare_parameter<std::string>("survey_topic", "");
    declare_parameter<double>("survey_resolution", 0.05);
    // Keep representation resolution independent from correspondence
    // tolerance. A centimetre output must not require graph-corrected returns
    // from separate keyframes to quantize into the identical centimetre cell.
    declare_parameter<double>("survey_support_resolution", 0.05);
    // Operator-facing graph survey: publish a second view containing only
    // voxels supported by this many distinct keyframes. Each keyframe's
    // inter-ping aggregate is voxel-reduced before global accumulation, so
    // support cannot be inflated by adjacent bins or repeated pings within
    // one keyframe interval.
    declare_parameter<int>("survey_min_support", 2);
    // publish only survey points whose occupancy voxel is occupied — the
    // clearing evidence then scrubs transients out of the survey product too
    declare_parameter<bool>("survey_occupancy_mask", true);
    // The lossless tile owns the graph-corrected image mosaic. A separate
    // range-consolidated return stream owns multi-view 3-D reconstruction, so
    // one pulse spanning many pixels cannot become a thick radial shell.
    declare_parameter<std::string>("tile_topic", "");
    declare_parameter<std::string>("reconstruction_topic", "");
    declare_parameter<double>("tile_resolution", 0.05);
    declare_parameter<double>("surface_resolution", 0.10);
    declare_parameter<int>("surface_min_observations", 3);
    declare_parameter<double>("surface_min_view_span_deg", 6.0);
    declare_parameter<int>("surface_max_samples_per_return", 31);
    declare_parameter<int>("surface_peak_radius_voxels", 0);
    declare_parameter<double>("surface_min_return_intensity", 0.0);
    declare_parameter<double>("surface_min_confidence", 0.45);
    // Operator-facing subset of the centimetre survey: retain only supported,
    // sufficiently strong returns for an uncluttered live navigation aid.
    // Elevation-resolved returns receive full confidence; unresolved returns
    // must accumulate more independent keyframe support. The broad aperture
    // surface and full tile products remain separately available for diagnosis
    // and conservative occupancy.
    declare_parameter<double>("navigation_min_confidence", 0.45);
    declare_parameter<double>("navigation_min_intensity", 0.10);
    declare_parameter<double>("navigation_unresolved_confidence_scale", 0.50);
    // Lidar-style surface-element refinement for the operator cloud. The 1 cm
    // representation preserves sonar detail; an independent metric-radius
    // neighborhood supplies stable local normals without 10 cm output voxels.
    declare_parameter<double>("surfel_radius_m", 0.10);
    declare_parameter<int>("surfel_min_neighbors", 5);
    declare_parameter<double>("surfel_max_surface_variation", 0.12);
    declare_parameter<double>("surfel_max_projection", 0.01);
    // Which obstacle evidence owns the graph-corrected 2-D planning map:
    //   hits              legacy capped-curtain VDB projection
    //   union             legacy projection plus confirmed surface cells
    //   confirmed_surface preserve VDB observed/free space, but only the
    //                     multi-view surface may mark lethal cells
    // The 3-D navigation_pointcloud is built independently from the supported
    // centimetre survey; these modes only select the 2-D occupancy owner.
    declare_parameter<std::string>("global_occupancy_mode", "hits");
    declare_parameter<std::string>("odom_frame", "odom");
    // evidence association
    declare_parameter<double>("buffer_seconds", 6.0);
    declare_parameter<double>("stamp_tolerance", 0.06);
    declare_parameter<int>("input_queue_depth", 5);
    declare_parameter<bool>("input_reliable", false);
    declare_parameter<bool>("allow_latest_tf_fallback", true);
    declare_parameter<double>("tf_buffer_duration", 10.0);
    declare_parameter<bool>("reset_on_time_rewind", true);
    declare_parameter<double>("time_rewind_tolerance", 0.5);
    // rendering policy
    declare_parameter<double>("render_min_period", 2.0);
    declare_parameter<double>("pose_epsilon_xy", 0.05);
    declare_parameter<double>("pose_epsilon_yaw", 0.02);
    declare_parameter<double>("pose_epsilon_z", 0.05);
    declare_parameter<int>("two_dim_projection_threshold", 3);
    // Evidence spill. Keyframe clouds are write-once -- after append only the
    // POSE is ever mutated -- so they can live on disk and be streamed back at
    // render time, which is what keeps RAM flat over a long survey instead of
    // growing linearly with keyframe count. Empty disables (clouds stay in
    // RAM, the original behaviour).
    declare_parameter<std::string>("spill_dir", "");
    // Consolidated survey product, written by ~/export_survey and (optionally)
    // at shutdown. Empty disables.
    declare_parameter<std::string>("export_path", "");
    declare_parameter<bool>("export_on_shutdown", true);
    // Dense navigation returns and strict normal-bearing surfels, written
    // independently of the broad survey export. Empty paths disable their
    // corresponding products.
    declare_parameter<std::string>("navigation_export_path", "");
    declare_parameter<bool>("navigation_export_on_shutdown", true);
    declare_parameter<std::string>("navigation_surfel_export_path", "");
    declare_parameter<bool>("navigation_surfel_export_on_shutdown", true);

    get_parameter("resolution", m_resolution);
    get_parameter("map_frame", m_map_frame);
    get_parameter("robot_frame", m_robot_frame);
    get_parameter("buffer_seconds", m_buffer_seconds);
    get_parameter("stamp_tolerance", m_stamp_tolerance);
    get_parameter("input_queue_depth", m_input_queue_depth);
    get_parameter("input_reliable", m_input_reliable);
    get_parameter("allow_latest_tf_fallback", m_allow_latest_tf_fallback);
    get_parameter("tf_buffer_duration", m_tf_buffer_duration);
    get_parameter("reset_on_time_rewind", m_reset_on_time_rewind);
    get_parameter("time_rewind_tolerance", m_time_rewind_tolerance);
    m_input_queue_depth = std::max(1, m_input_queue_depth);
    m_tf_buffer_duration = std::max(0.1, m_tf_buffer_duration);
    m_time_rewind_tolerance = std::max(0.0, m_time_rewind_tolerance);
    get_parameter("render_min_period", m_render_min_period);
    get_parameter("pose_epsilon_xy", m_pose_eps_xy);
    get_parameter("pose_epsilon_yaw", m_pose_eps_yaw);
    get_parameter("pose_epsilon_z", m_pose_eps_z);
    get_parameter("two_dim_projection_threshold", m_two_dim_projection_threshold);
    get_parameter("survey_resolution", m_survey_resolution);
    get_parameter("survey_support_resolution", m_survey_support_resolution);
    get_parameter("survey_min_support", m_survey_min_support);
    get_parameter("survey_occupancy_mask", m_survey_occupancy_mask);
    get_parameter("tile_resolution", m_tile_resolution);
    get_parameter("surface_resolution", m_surface_resolution);
    get_parameter("surface_min_observations", m_surface_min_observations);
    get_parameter("surface_min_view_span_deg", m_surface_min_view_span_deg);
    get_parameter("surface_max_samples_per_return", m_surface_max_samples_per_return);
    get_parameter("surface_peak_radius_voxels", m_surface_peak_radius_voxels);
    get_parameter("surface_min_return_intensity", m_surface_min_return_intensity);
    get_parameter("surface_min_confidence", m_surface_min_confidence);
    get_parameter("navigation_min_confidence", m_navigation_min_confidence);
    get_parameter("navigation_min_intensity", m_navigation_min_intensity);
    get_parameter(
      "navigation_unresolved_confidence_scale",
      m_navigation_unresolved_confidence_scale);
    get_parameter("surfel_radius_m", m_surfel_radius_m);
    get_parameter("surfel_min_neighbors", m_surfel_min_neighbors);
    get_parameter("surfel_max_surface_variation",
                  m_surfel_max_surface_variation);
    get_parameter("surfel_max_projection", m_surfel_max_projection);
    {
      std::string mode;
      get_parameter("global_occupancy_mode", mode);
      if (mode == "hits")
      {
        m_global_occupancy_mode = GlobalOccupancyMode::Hits;
      }
      else if (mode == "union")
      {
        m_global_occupancy_mode = GlobalOccupancyMode::Union;
      }
      else if (mode == "confirmed_surface")
      {
        m_global_occupancy_mode = GlobalOccupancyMode::ConfirmedSurface;
      }
      else
      {
        throw std::invalid_argument(
          "global_occupancy_mode must be hits, union, or confirmed_surface; got '" +
          mode + "'");
      }
    }
    m_tile_resolution = std::max(1e-3, m_tile_resolution);
    m_survey_resolution = std::max(1e-3, m_survey_resolution);
    m_survey_support_resolution =
      std::max(m_survey_resolution, m_survey_support_resolution);
    m_survey_min_support = std::max(1, m_survey_min_support);
    m_surface_resolution = std::max(1e-3, m_surface_resolution);
    m_surface_min_observations = std::max(1, m_surface_min_observations);
    // Zero is the quality-first mode: sample every elevation ribbon densely
    // enough for surface_resolution at the measured range. A positive cap is
    // an explicit compute trade, but values 1/2 cannot retain both aperture
    // edges plus the measured centre and are promoted to 3.
    if (m_surface_max_samples_per_return < 0)
    {
      throw std::invalid_argument(
        "surface_max_samples_per_return must be non-negative");
    }
    if (m_surface_max_samples_per_return > 0)
    {
      m_surface_max_samples_per_return =
        std::max(3, m_surface_max_samples_per_return);
    }
    m_surface_peak_radius_voxels = std::max(0, m_surface_peak_radius_voxels);
    m_surface_min_return_intensity = std::clamp(
      m_surface_min_return_intensity, 0.0, 1.0);
    m_surface_min_confidence = std::clamp(
      m_surface_min_confidence, 0.0, 1.0);
    m_navigation_min_confidence = std::clamp(
      m_navigation_min_confidence, 0.0, 1.0);
    m_navigation_min_intensity = std::clamp(
      m_navigation_min_intensity, 0.0, 1.0);
    m_navigation_unresolved_confidence_scale = std::clamp(
      m_navigation_unresolved_confidence_scale, 0.0, 1.0);
    m_surfel_radius_m = std::max(1e-3, m_surfel_radius_m);
    m_surfel_min_neighbors = std::max(1, m_surfel_min_neighbors);
    m_surfel_max_surface_variation = std::clamp(
      m_surfel_max_surface_variation, 1e-6, 1.0);
    m_surfel_max_projection = std::max(0.0, m_surfel_max_projection);
    m_surface_accumulator = std::make_unique<MultiViewSurfaceAccumulator>(
      static_cast<float>(m_surface_resolution), m_surface_min_observations,
      static_cast<float>(std::max(0.0, m_surface_min_view_span_deg) * M_PI / 180.0),
      m_surface_max_samples_per_return, m_surface_peak_radius_voxels,
      static_cast<float>(m_surface_min_return_intensity));
    get_parameter("odom_frame", m_odom_frame);
    get_parameter("export_path", m_export_path);
    get_parameter("export_on_shutdown", m_export_on_shutdown);
    get_parameter("navigation_export_path", m_navigation_export_path);
    get_parameter(
      "navigation_export_on_shutdown", m_navigation_export_on_shutdown);
    get_parameter(
      "navigation_surfel_export_path", m_navigation_surfel_export_path);
    get_parameter(
      "navigation_surfel_export_on_shutdown",
      m_navigation_surfel_export_on_shutdown);
    setUpSpill();

    m_map = std::make_unique<VDBMapT>(m_resolution);
    vdb_mapping::Config cfg;
    get_parameter("max_range", cfg.max_range);
    get_parameter("prob_hit", cfg.prob_hit);
    get_parameter("prob_miss", cfg.prob_miss);
    get_parameter("prob_thres_min", cfg.prob_thres_min);
    get_parameter("prob_thres_max", cfg.prob_thres_max);
    cfg.map_directory_path  = "";
    cfg.fast_mode           = false;  // batch re-render: DDA is fine and simplest
    cfg.accumulation_period = 1.0;
    if (!m_map->setConfig(cfg))
      RCLCPP_FATAL(get_logger(),
                   "assembler map config REJECTED — re-renders will produce "
                   "empty maps until the parameters are fixed");
    const auto ros_clock = get_clock();
    m_map->setTimeCallback([ros_clock]() -> uint64_t {
      return static_cast<uint64_t>(
        std::max<int64_t>(0, ros_clock->now().nanoseconds()));
    });
    // fill evidence: endpoints paint, rays never carve
    m_map->addInputSource("hits", 0.0, 0.0, /*ray_clearing=*/false, /*endpoint_hits=*/true);
    // clearing evidence: rays carve their full length, endpoints never paint
    m_map->addInputSource("clear", 0.0, 0.0, /*ray_clearing=*/true, /*endpoint_hits=*/false);

    m_tf_buffer = std::make_unique<tf2_ros::Buffer>(
      get_clock(), tf2::durationFromSec(m_tf_buffer_duration));
    m_tf_listener = std::make_shared<tf2_ros::TransformListener>(*m_tf_buffer);

    // Ingestion and rendering are each internally serialized, but occupy
    // different executor lanes. The renderer takes a brief keyframe metadata
    // snapshot under m_ingest_mutex and then releases it before any disk read,
    // global rebuild, message construction, or normal fitting.
    m_ingest_callback_group =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    m_render_callback_group =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions ingest_options;
    ingest_options.callback_group = m_ingest_callback_group;

    std::string hits_topic, clear_topic, traj_topic;
    get_parameter("hits_topic", hits_topic);
    get_parameter("clear_topic", clear_topic);
    get_parameter("traj_topic", traj_topic);
    if (hits_topic.empty() || clear_topic.empty())
    {
      RCLCPP_ERROR(get_logger(), "hits_topic / clear_topic must be set");
    }

    auto qos = rclcpp::QoS(
      rclcpp::KeepLast(static_cast<std::size_t>(m_input_queue_depth)));
    if (m_input_reliable)
    {
      qos.reliable();
    }
    else
    {
      qos.best_effort();
    }
    m_hits_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
      hits_topic, qos, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(m_ingest_mutex);
        bufferCloud(*msg, m_hits_buffer, true);
      }, ingest_options);
    m_clear_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
      clear_topic, qos, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(m_ingest_mutex);
        bufferCloud(*msg, m_clear_buffer, false);
      }, ingest_options);
    // trajectory is latched by the SLAM node
    m_traj_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
      traj_topic,
      rclcpp::QoS(rclcpp::KeepLast(static_cast<std::size_t>(m_input_queue_depth)))
        .reliable().transient_local(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(m_ingest_mutex);
        onTrajectory(*msg);
      }, ingest_options);

    // These are complete map snapshots, not observations. Latch the latest
    // render so Nav2/RViz consumers that start later receive current state.
    const auto map_qos = rclcpp::QoS(1).reliable().transient_local();
    m_cloud_pub = create_publisher<sensor_msgs::msg::PointCloud2>(
      "~/vdb_map_pointcloud", map_qos);
    m_grid_pub = create_publisher<nav_msgs::msg::OccupancyGrid>(
      "~/vdb_map_occupancy", map_qos);

    std::string survey_topic;
    get_parameter("survey_topic", survey_topic);
    if (!survey_topic.empty())
    {
      m_survey_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
        survey_topic, qos, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          std::lock_guard<std::mutex> lock(m_ingest_mutex);
          bufferSurvey(*msg);
        }, ingest_options);
      m_survey_pub = create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/survey_pointcloud", map_qos);
      m_supported_survey_pub = create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/supported_survey_pointcloud", map_qos);
      m_navigation_pub = create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/navigation_pointcloud", map_qos);
      m_navigation_surfel_pub = create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/navigation_surfel_pointcloud", map_qos);
    }

    std::string tile_topic, reconstruction_topic;
    get_parameter("tile_topic", tile_topic);
    get_parameter("reconstruction_topic", reconstruction_topic);
    if (!tile_topic.empty())
    {
      m_tile_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
        tile_topic, qos, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          std::lock_guard<std::mutex> lock(m_ingest_mutex);
          bufferTile(*msg);
        }, ingest_options);
      m_tile_pub = create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/tile_pointcloud", map_qos);
    }
    if (!reconstruction_topic.empty())
    {
      m_reconstruction_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
        reconstruction_topic, qos,
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          std::lock_guard<std::mutex> lock(m_ingest_mutex);
          bufferReconstructionReturns(*msg);
        }, ingest_options);
      m_surface_pub = create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/surface_pointcloud", map_qos);
    }
    else if (m_global_occupancy_mode != GlobalOccupancyMode::Hits)
    {
      throw std::invalid_argument(
        "global_occupancy_mode requires reconstruction_topic unless mode is hits");
    }

    m_export_srv = create_service<std_srvs::srv::Trigger>(
      "~/export_survey",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        res->success = exportSurvey(res->message);
      }, rclcpp::ServicesQoS(), m_render_callback_group);
    m_navigation_export_srv = create_service<std_srvs::srv::Trigger>(
      "~/export_navigation_surface",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        res->success = exportNavigationSurface(res->message);
      }, rclcpp::ServicesQoS(), m_render_callback_group);
    m_navigation_surfel_export_srv = create_service<std_srvs::srv::Trigger>(
      "~/export_navigation_surfels",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        res->success = exportNavigationSurfels(res->message);
      }, rclcpp::ServicesQoS(), m_render_callback_group);

    m_render_timer = create_timer(std::chrono::milliseconds(500),
                                  [this] { renderIfNeeded(); },
                                  m_render_callback_group);
    m_navigation_fit_thread =
      std::thread([this] { navigationFitWorker(); });

    RCLCPP_INFO(get_logger(),
                "Map assembler up: hits=%s clear=%s traj=%s tile=%s "
                "reconstruction=%s res=%.2f",
                hits_topic.c_str(), clear_topic.c_str(), traj_topic.c_str(),
                tile_topic.empty() ? "disabled" : tile_topic.c_str(),
                  reconstruction_topic.empty() ? "disabled" :
                  reconstruction_topic.c_str(), m_resolution);
  }

  ~VDBMapAssembler() override
  {
    stopNavigationFitWorker();
  }

private:
  enum class GlobalOccupancyMode
  {
    Hits,
    Union,
    ConfirmedSurface,
  };

  // Snapshot subscriber demand once at the beginning of a render. Complete
  // products are expensive and transient-local publishers exist even when no
  // consumer is connected, so publisher existence is not a work request.
  struct RenderDemand
  {
    bool tile = false;
    bool surface = false;
    bool full_survey = false;
    bool supported_survey = false;
    bool navigation = false;
    bool navigation_surfels = false;
    bool vdb_cloud = false;

    bool surveyProducts() const
    {
      return full_survey || supported_survey || navigation ||
        navigation_surfels;
    }

    bool navigationProducts() const
    {
      return navigation || navigation_surfels;
    }
  };

  struct ProductPublishMetrics
  {
    double tile_ms = 0.0;
    double surface_ms = 0.0;
    double navigation_ms = 0.0;
    double surfel_ms = 0.0;
  };

  using SteadyClock = std::chrono::steady_clock;

  struct NavigationFitJob
  {
    std::uint64_t generation = 0;
    std::uint64_t epoch = 0;
    rclcpp::Time stamp;
    bool publish_navigation = false;
    bool publish_surfels = false;
    std::size_t free_space_rejected = 0;
    std::vector<NavigationRow> navigation;
    SteadyClock::time_point queued_at;
  };

  static double elapsedWallMs(const SteadyClock::time_point& start)
  {
    return std::chrono::duration<double, std::milli>(
      SteadyClock::now() - start).count();
  }

  struct BufferedCloud
  {
    double stamp;
    CloudPtrT cloud;                 // in robot frame
    Eigen::Vector3d sensor_origin;   // in robot frame
  };

  struct KeyframeEvidence
  {
    double stamp = 0.0;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();  // map <- robot
    // Cloud handles are null once the keyframe has been spilled; the counts
    // below stay valid either way, so callers can skip a load that would
    // return nothing. Everything else here is the ~150 bytes per keyframe
    // that MUST stay resident: the pose is rewritten by the optimizer and is
    // what makes a re-render possible at all.
    CloudPtrT hits;                  // robot frame
    CloudPtrT clear;
    // dense survey aggregate (ALL inter-keyframe survey_points clouds,
    // odom-delta transformed into THIS keyframe's robot frame)
    SurveyCloudPtrT survey;
    // Lossless centre-plane image pixels for Product A. Kept distinct from
    // range-consolidated reconstruction so image density cannot thicken the
    // inferred 3-D surface.
    ReconstructionCloudPtrT tile;
    ReconstructionCloudPtrT reconstruction;
    Eigen::Vector3d hits_origin  = Eigen::Vector3d::Zero();
    Eigen::Vector3d clear_origin = Eigen::Vector3d::Zero();
    bool has_evidence = false;
    bool spilled = false;
    // Persist the namespace with the keyframe. A render snapshot may outlive
    // an input-time rewind, which starts a new spill segment on the ingestion
    // thread; looking old evidence up through the mutable current directory
    // would otherwise splice or lose a generation.
    std::string spill_dir;
    size_t n_hits = 0;
    size_t n_clear = 0;
    size_t n_survey = 0;
    size_t n_tile = 0;
    size_t n_reconstruction = 0;
  };

  // Evidence resolved for one keyframe: either the resident handles or clouds
  // just read back from disk. Held only for the duration of one keyframe's
  // integration, which is what bounds render-time RAM to a single keyframe.
  struct LoadedEvidence
  {
    CloudPtrT hits;
    CloudPtrT clear;
    SurveyCloudPtrT survey;
    ReconstructionCloudPtrT tile;
    ReconstructionCloudPtrT reconstruction;
  };

  struct BufferedSurvey
  {
    double stamp;
    SurveyCloudPtrT cloud;  // in robot frame at `stamp`
  };

  struct BufferedReconstruction
  {
    double stamp;
    ReconstructionCloudPtrT cloud;  // robot frame at `stamp`
  };

  // Per-voxel reduction of the survey product. At class scope because the
  // accumulator now outlives a single render (see renderIfNeeded).
  struct VoxAcc
  {
    double x = 0, y = 0, z = 0, i = 0, r = 0, ci = 0;
    double texture = 0, texture_squared = 0;
    double elevation_lo = 0, elevation_hi = 0;
    double elevation_resolved = 0;
    int n = 0, nci = 0, ntexture = 0, nelevation = 0;
  };

  struct TileVoxAcc
  {
    double x = 0.0, y = 0.0, z = 0.0;
    double intensity = 0.0, range = 0.0, half_angle = 0.0;
    int n = 0;
  };

  void setUpSpill()
  {
    std::string dir;
    get_parameter("spill_dir", dir);
    if (dir.empty())
    {
      RCLCPP_INFO(get_logger(),
                  "Evidence spill disabled (spill_dir unset): keyframe clouds "
                  "stay resident and RAM grows with survey length.");
      return;
    }
    // Namespace every run. These files ARE the persistence, so overwriting a
    // previous survey's keyframes is data loss; worse, a reused index would
    // splice two trajectories into one map at render time.
    const std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
    localtime_r(&now, &tm_buf);
    std::ostringstream sub;
    sub << "run_" << std::put_time(&tm_buf, "%Y%m%d_%H%M%S")
        << "_seg" << std::setw(3) << std::setfill('0') << m_replay_segment;

    const std::filesystem::path root = std::filesystem::path(dir) / sub.str();
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec)
    {
      RCLCPP_ERROR(get_logger(),
                   "Cannot create spill directory %s (%s); keeping evidence in "
                   "RAM instead.",
                   root.string().c_str(), ec.message().c_str());
      return;
    }
    m_spill_dir = root.string();
    RCLCPP_INFO(get_logger(), "Spilling keyframe evidence to %s",
                m_spill_dir.c_str());
  }

  static std::string spillPath(
    const std::string& directory, const size_t idx, const char* kind)
  {
    std::ostringstream p;
    p << directory << "/kf_" << std::setw(6) << std::setfill('0') << idx
      << '_' << kind << ".pcd";
    return p.str();
  }

  std::string spillPath(const size_t idx, const char* kind) const
  {
    return spillPath(m_spill_dir, idx, kind);
  }

  template <typename CloudPtr>
  bool writeSpill(
    const CloudPtr& cloud, const std::string& path,
    const bool compressed = false)
  {
    // An unorganized cloud whose width/height do not match size() writes a
    // header that disagrees with the payload, and the read back silently
    // returns the wrong point count.
    cloud->width    = static_cast<uint32_t>(cloud->size());
    cloud->height   = 1;
    cloud->is_dense = false;
    try
    {
      pcl::PCDWriter writer;
      // Full lossless tile evidence dominates spill volume (millions of
      // points per keyframe, with highly repetitive origin/axis/observation
      // fields). PCL's binary-compressed encoding is lossless and is read
      // transparently by PCDReader, so it preserves the tile product while
      // substantially reducing long-survey disk bandwidth and capacity.
      const int status = compressed
        ? writer.writeBinaryCompressed(path, *cloud)
        : writer.writeBinary(path, *cloud);
      if (status == 0)
      {
        return true;
      }
    }
    catch (const std::exception& e)
    {
      RCLCPP_ERROR(get_logger(), "spill write %s failed: %s", path.c_str(),
                   e.what());
      return false;
    }
    RCLCPP_ERROR(get_logger(), "spill write %s failed", path.c_str());
    return false;
  }

  template <typename CloudTT>
  typename CloudTT::Ptr readSpill(const std::string& path)
  {
    typename CloudTT::Ptr out(new CloudTT);
    try
    {
      pcl::PCDReader reader;
      if (reader.read(path, *out) == 0)
      {
        return out;
      }
    }
    catch (const std::exception& e)
    {
      RCLCPP_ERROR(get_logger(), "spill read %s failed: %s", path.c_str(),
                   e.what());
      out->clear();
      return out;
    }
    RCLCPP_ERROR(get_logger(), "spill read %s failed", path.c_str());
    out->clear();
    return out;
  }

  void spillKeyframe(KeyframeEvidence& kf, const size_t idx)
  {
    if (m_spill_dir.empty())
    {
      return;
    }
    kf.n_hits   = kf.hits ? kf.hits->size() : 0;
    kf.n_clear  = kf.clear ? kf.clear->size() : 0;
    kf.n_survey = kf.survey ? kf.survey->size() : 0;
    kf.n_tile = kf.tile ? kf.tile->size() : 0;
    kf.n_reconstruction =
      kf.reconstruction ? kf.reconstruction->size() : 0;

    bool ok = true;
    if (kf.n_hits > 0)
    {
      ok = writeSpill(kf.hits, spillPath(idx, "hits")) && ok;
    }
    if (kf.n_clear > 0)
    {
      ok = writeSpill(kf.clear, spillPath(idx, "clear")) && ok;
    }
    if (kf.n_survey > 0)
    {
      ok = writeSpill(kf.survey, spillPath(idx, "survey")) && ok;
    }
    if (kf.n_reconstruction > 0)
    {
      ok = writeSpill(
        kf.reconstruction, spillPath(idx, "reconstruction")) && ok;
    }
    if (kf.n_tile > 0)
    {
      ok = writeSpill(
        kf.tile, spillPath(idx, "tile"), /*compressed=*/true) && ok;
    }
    if (!ok)
    {
      // Dropping the handles after a partial write would delete evidence that
      // never reached disk, and every future re-render would be quietly
      // missing this keyframe. Keep it resident: one keyframe of RAM is a
      // cheaper failure than a map with a hole in it.
      ++m_spill_failures;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                           "keyframe %zu spill failed; keeping it in RAM "
                           "(%zu failures so far)",
                           idx, m_spill_failures);
      return;
    }
    kf.spill_dir = m_spill_dir;
    kf.hits.reset();
    kf.clear.reset();
    kf.survey.reset();
    kf.tile.reset();
    kf.reconstruction.reset();
    kf.spilled = true;
  }

  // Resolve one keyframe's evidence, reading it back only if it was spilled
  // and only the parts the caller will actually use.
  LoadedEvidence loadEvidence(const KeyframeEvidence& kf, const size_t idx,
                              const bool want_occupancy, const bool want_survey,
                              const bool want_reconstruction = false,
                              const bool want_tile = false)
  {
    LoadedEvidence e;
    if (!kf.spilled)
    {
      if (want_occupancy)
      {
        e.hits  = kf.hits;
        e.clear = kf.clear;
      }
      if (want_survey)
      {
        e.survey = kf.survey;
      }
      if (want_reconstruction)
      {
        e.reconstruction = kf.reconstruction;
      }
      if (want_tile)
      {
        e.tile = kf.tile;
      }
      return e;
    }
    if (want_occupancy)
    {
      if (kf.n_hits > 0)
      {
        e.hits = readSpill<CloudT>(spillPath(kf.spill_dir, idx, "hits"));
      }
      if (kf.n_clear > 0)
      {
        e.clear = readSpill<CloudT>(spillPath(kf.spill_dir, idx, "clear"));
      }
    }
    if (want_survey && kf.n_survey > 0)
    {
      e.survey = readSpill<SurveyCloudT>(
        spillPath(kf.spill_dir, idx, "survey"));
    }
    if (want_reconstruction && kf.n_reconstruction > 0)
    {
      e.reconstruction = readSpill<ReconstructionCloudT>(
        spillPath(kf.spill_dir, idx, "reconstruction"));
    }
    if (want_tile && kf.n_tile > 0)
    {
      e.tile = readSpill<ReconstructionCloudT>(
        spillPath(kf.spill_dir, idx, "tile"));
    }
    return e;
  }

  // transform an incoming cloud into the robot frame AT THE CLOUD'S STAMP —
  // the sonar frame rides the live pivot_head (cameraHead.tilt, +/-54 deg),
  // so the old static-TF cache froze the head angle into every snapshot
  void bufferCloud(const sensor_msgs::msg::PointCloud2& msg,
                   std::deque<BufferedCloud>& buffer,
                   const bool downsample)
  {
    const double stamp = rclcpp::Time(msg.header.stamp).seconds();
    double& last_stamp = downsample ? m_last_hits_stamp : m_last_clear_stamp;
    if (observeInputStamp(stamp, last_stamp, downsample ? "hits" : "clear"))
    {
      // resetReplaySession invalidated references to the old buffers, but the
      // member selected by the caller remains the same object and is safe to
      // continue filling for the new replay segment.
      last_stamp = stamp;
    }
    Eigen::Isometry3d t_robot_sensor;
    if (msg.width * msg.height == 0 ||
        !lookupAtStamp(msg.header.frame_id, msg.header.stamp, t_robot_sensor))
    {
      return;
    }
    CloudPtrT raw(new CloudT);
    pcl::fromROSMsg(msg, *raw);
    CloudPtrT in_robot(new CloudT);
    pcl::transformPointCloud(*raw, *in_robot, t_robot_sensor.cast<float>());
    if (downsample && !in_robot->empty())
    {
      pcl::VoxelGrid<PointT> vg;
      vg.setLeafSize(static_cast<float>(m_resolution),
                     static_cast<float>(m_resolution),
                     static_cast<float>(m_resolution));
      vg.setInputCloud(in_robot);
      CloudPtrT ds(new CloudT);
      vg.filter(*ds);
      in_robot = ds;
    }
    BufferedCloud entry;
    entry.stamp         = stamp;
    entry.cloud         = in_robot;
    entry.sensor_origin = t_robot_sensor.translation();
    buffer.push_back(std::move(entry));
    while (!buffer.empty() && buffer.back().stamp - buffer.front().stamp > m_buffer_seconds)
    {
      buffer.pop_front();
    }
  }

  // TF lookup at a specific stamp with latest-sample fallback (the sonar
  // frame is DYNAMIC — pivot head — so no caching). allow_fallback=false
  // forces stamp-exact: odom-DELTA anchoring (keyframe vs survey entry) is
  // only meaningful between two poses at their own stamps — "latest" there
  // is wrong by the robot's motion since, and a mis-anchored batch bakes
  // into kf.survey permanently, silently bypassing the designed
  // hold-for-next-keyframe failure path.
  bool lookupAtStamp(const std::string& frame,
                     const builtin_interfaces::msg::Time& stamp,
                     Eigen::Isometry3d& out,
                     const std::string& target = "",
                     bool allow_fallback = true)
  {
    const std::string& tgt = target.empty() ? m_robot_frame : target;
    geometry_msgs::msg::TransformStamped tfs;
    try
    {
      tfs = m_tf_buffer->lookupTransform(tgt, frame, rclcpp::Time(stamp),
                                         rclcpp::Duration::from_seconds(0.05));
    }
    catch (const tf2::TransformException&)
    {
      if (!m_allow_latest_tf_fallback || !allow_fallback)
      {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                             "No exact TF %s <- %s at %.9f; dropping cloud "
                             "(latest-TF fallback disabled)",
                             tgt.c_str(), frame.c_str(), rclcpp::Time(stamp).seconds());
        return false;
      }
      try
      {
        tfs = m_tf_buffer->lookupTransform(tgt, frame, tf2::TimePointZero);
      }
      catch (const tf2::TransformException& ex)
      {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                             "No TF %s <- %s yet (%s); dropping cloud",
                             tgt.c_str(), frame.c_str(), ex.what());
        return false;
      }
    }
    const auto& t = tfs.transform.translation;
    const auto& q = tfs.transform.rotation;
    out = Eigen::Isometry3d::Identity();
    out.translate(Eigen::Vector3d(t.x, t.y, t.z));
    out.rotate(Eigen::Quaterniond(q.w, q.x, q.y, q.z));
    return true;
  }

  // Buffer a rich survey_points cloud in the robot frame at its stamp. Scalar
  // texture/elevation attributes ride through the coordinate transform.
  void bufferSurvey(const sensor_msgs::msg::PointCloud2& msg)
  {
    const double stamp = rclcpp::Time(msg.header.stamp).seconds();
    observeInputStamp(stamp, m_last_survey_stamp, "survey");
    Eigen::Isometry3d t_robot_sensor;
    if (msg.width * msg.height == 0 ||
        !lookupAtStamp(msg.header.frame_id, msg.header.stamp, t_robot_sensor))
    {
      return;
    }
    SurveyCloudPtrT raw(new SurveyCloudT);
    pcl::fromROSMsg(msg, *raw);
    SurveyCloudPtrT in_robot(new SurveyCloudT);
    pcl::transformPointCloud(*raw, *in_robot, t_robot_sensor.cast<float>());
    BufferedSurvey entry;
    entry.stamp = stamp;
    entry.cloud = in_robot;
    m_survey_buffer.push_back(std::move(entry));
    while (!m_survey_buffer.empty() &&
           m_survey_buffer.back().stamp - m_survey_buffer.front().stamp >
             m_buffer_seconds)
    {
      m_survey_buffer.pop_front();
    }
  }

  // Attach the exact ping-time pivot-head pose to every centre-plane tile
  // sample. The full tile is retained only for the graph-corrected intensity
  // mosaic; surface reconstruction consumes the lobe-collapsed stream below.
  void bufferTile(const sensor_msgs::msg::PointCloud2& msg)
  {
    const double stamp = rclcpp::Time(msg.header.stamp).seconds();
    observeInputStamp(stamp, m_last_tile_stamp, "tile");
    Eigen::Isometry3d t_robot_sensor;
    if (msg.width * msg.height == 0 ||
        // Reconstruction geometry is allowed to fail closed.  A latest TF is
        // adequate for some display-oriented products, but here it would
        // silently attach the wrong encoder angle to an entire sonar ribbon
        // and manufacture multi-view support.  The production TF comes from
        // cameraHead.tilt.position (live or regenerated from bridge telemetry
        // during replay); the Oculus AHRS is deliberately not consulted.
        !lookupAtStamp(msg.header.frame_id, msg.header.stamp, t_robot_sensor,
                       /*target=*/"", /*allow_fallback=*/false))
    {
      return;
    }

    pcl::PointCloud<SonarTilePoint>::Ptr raw(
      new pcl::PointCloud<SonarTilePoint>);
    pcl::fromROSMsg(msg, *raw);
    if (raw->empty())
    {
      return;
    }

    ReconstructionCloudPtrT in_robot(new ReconstructionCloudT);
    in_robot->reserve(raw->size());
    const Eigen::Isometry3f tf = t_robot_sensor.cast<float>();
    Eigen::Vector3f elevation_axis = tf.linear() * Eigen::Vector3f::UnitX();
    Eigen::Vector3f boresight = tf.linear() * Eigen::Vector3f::UnitZ();
    elevation_axis.normalize();
    boresight.normalize();
    const Eigen::Vector3f origin = tf.translation();
    const std::uint32_t observation = ++m_observation_sequence;
    for (const auto& p : raw->points)
    {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
          !std::isfinite(p.intensity) || !std::isfinite(p.range) ||
          !std::isfinite(p.vertical_uncertainty) || !(p.range > 0.0F))
      {
        continue;
      }
      SonarReconstructionPoint out{};
      setPointPosition(out, tf * Eigen::Vector3f(p.x, p.y, p.z));
      out.intensity = p.intensity;
      out.range = p.range;
      out.azimuth = p.azimuth;
      out.elevation_half_angle = std::atan2(
        std::fabs(p.vertical_uncertainty), p.range);
      setPointOrigin(out, origin);
      setPointElevationAxis(out, elevation_axis);
      setPointBoresight(out, boresight);
      out.observation = observation;
      in_robot->push_back(out);
    }
    in_robot->width = static_cast<std::uint32_t>(in_robot->size());
    in_robot->height = 1;
    in_robot->is_dense = false;
    if (in_robot->empty())
    {
      return;
    }
    m_tile_buffer.push_back({stamp, in_robot});
    while (!m_tile_buffer.empty() &&
           m_tile_buffer.back().stamp - m_tile_buffer.front().stamp >
             m_buffer_seconds)
    {
      m_tile_buffer.pop_front();
    }
  }

  // Attach the same exact pose to sonar_proc's one-return-per-echo stream.
  // Keeping this callback separate from bufferTile is the key contract: the
  // immutable image mosaic stays lossless while pulse thickness cannot become
  // independent votes in the elevation reconstruction.
  void bufferReconstructionReturns(const sensor_msgs::msg::PointCloud2& msg)
  {
    const double stamp = rclcpp::Time(msg.header.stamp).seconds();
    observeInputStamp(stamp, m_last_reconstruction_stamp, "reconstruction");
    Eigen::Isometry3d t_robot_sensor;
    if (msg.width * msg.height == 0 ||
        !lookupAtStamp(msg.header.frame_id, msg.header.stamp, t_robot_sensor,
                       /*target=*/"", /*allow_fallback=*/false))
    {
      return;
    }

    pcl::PointCloud<SonarReconstructionReturnPoint>::Ptr raw(
      new pcl::PointCloud<SonarReconstructionReturnPoint>);
    pcl::fromROSMsg(msg, *raw);
    if (raw->empty()) return;

    ReconstructionCloudPtrT in_robot(new ReconstructionCloudT);
    in_robot->reserve(raw->size());
    const Eigen::Isometry3f tf = t_robot_sensor.cast<float>();
    Eigen::Vector3f elevation_axis = tf.linear() * Eigen::Vector3f::UnitX();
    Eigen::Vector3f boresight = tf.linear() * Eigen::Vector3f::UnitZ();
    elevation_axis.normalize();
    boresight.normalize();
    const Eigen::Vector3f origin = tf.translation();
    const std::uint32_t observation = ++m_observation_sequence;
    for (const auto& p : raw->points)
    {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
          !std::isfinite(p.intensity) || !std::isfinite(p.range) ||
          !std::isfinite(p.vertical_uncertainty) ||
          !std::isfinite(p.range_sigma) || !std::isfinite(p.prominence) ||
          !std::isfinite(p.echo_width) || !(p.range > 0.0F))
      {
        continue;
      }
      SonarReconstructionPoint out{};
      setPointPosition(out, tf * Eigen::Vector3f(p.x, p.y, p.z));
      out.intensity = p.intensity;
      out.range = p.range;
      out.azimuth = p.azimuth;
      out.elevation_half_angle = std::atan2(
        std::fabs(p.vertical_uncertainty), p.range);
      setPointOrigin(out, origin);
      setPointElevationAxis(out, elevation_axis);
      setPointBoresight(out, boresight);
      out.range_sigma = std::max(0.0F, p.range_sigma);
      out.return_prominence = std::max(0.0F, p.prominence);
      out.echo_width = std::max(0.0F, p.echo_width);
      out.observation = observation;
      in_robot->push_back(out);
    }
    in_robot->width = static_cast<std::uint32_t>(in_robot->size());
    in_robot->height = 1;
    in_robot->is_dense = false;
    if (in_robot->empty()) return;

    m_reconstruction_buffer.push_back({stamp, in_robot});
    while (!m_reconstruction_buffer.empty() &&
           m_reconstruction_buffer.back().stamp -
             m_reconstruction_buffer.front().stamp > m_buffer_seconds)
    {
      m_reconstruction_buffer.pop_front();
    }
  }

  const BufferedCloud* findNearest(const std::deque<BufferedCloud>& buffer,
                                   const double stamp) const
  {
    const BufferedCloud* best = nullptr;
    double best_dt            = m_stamp_tolerance;
    for (const auto& entry : buffer)
    {
      const double dt = std::fabs(entry.stamp - stamp);
      if (dt <= best_dt)
      {
        best_dt = dt;
        best    = &entry;
      }
    }
    return best;
  }

  template<typename BufferedT>
  static void discardAssociated(
    std::deque<BufferedT>& buffer, const double associated_through)
  {
    // Association is strictly one-way: the next keyframe ignores every
    // entry <= m_last_*_assoc_stamp. Keeping those clouds until the generic
    // time horizon expired retained several seconds of the full-resolution
    // tile stream even though they could never be consumed again.
    while (!buffer.empty() && buffer.front().stamp <= associated_through)
    {
      buffer.pop_front();
    }
  }

  // Each topic is tracked separately: a delayed cloud from one source must
  // not look like a seek merely because another source has advanced farther.
  // A true regression on any one stream starts a clean replay segment.
  bool observeInputStamp(const double stamp, double& last_stamp, const char* source)
  {
    if (m_reset_on_time_rewind &&
        stamp > 0.0 &&
        last_stamp > 0.0 &&
        stamp + m_time_rewind_tolerance < last_stamp)
    {
      RCLCPP_WARN(get_logger(),
                  "%s time moved backwards by %.3f s; starting a clean map "
                  "assembler replay segment",
                  source, last_stamp - stamp);
      resetReplaySession();
      last_stamp = stamp;
      return true;
    }
    last_stamp = std::max(last_stamp, stamp);
    return false;
  }

  void resetReplaySession()
  {
    {
      std::lock_guard<std::mutex> lock(m_navigation_fit_mutex);
      ++m_navigation_fit_epoch;
      m_pending_navigation_fit.reset();
    }
    ++m_replay_epoch;
    m_hits_buffer.clear();
    m_clear_buffer.clear();
    m_survey_buffer.clear();
    m_tile_buffer.clear();
    m_reconstruction_buffer.clear();
    m_keyframes.clear();

    m_keyframes_without_evidence = 0;
    m_dirty = false;
    m_have_new = false;
    // Render-owned VDB/derived state may currently be rebuilding on another
    // executor lane. Do not mutate it here. The active render carries the old
    // epoch and will refuse to publish after the rewind; the first render of
    // the new segment performs a complete reset from its keyframe snapshot.
    m_force_full_render = true;
    m_last_assoc_stamp = 0.0;
    m_last_tile_assoc_stamp = 0.0;
    m_last_reconstruction_assoc_stamp = 0.0;
    m_last_hits_stamp = 0.0;
    m_last_clear_stamp = 0.0;
    m_last_survey_stamp = 0.0;
    m_last_tile_stamp = 0.0;
    m_last_reconstruction_stamp = 0.0;
    m_last_traj_stamp = 0.0;
    m_observation_sequence = 0;
    m_spill_failures = 0;

    // Keep old segment files recoverable and write replayed evidence into a
    // fresh namespace so repeated keyframe indices never overwrite or splice.
    ++m_replay_segment;
    m_spill_dir.clear();
    setUpSpill();
  }

  bool replayEpochCurrent(const std::uint64_t epoch)
  {
    std::lock_guard<std::mutex> lock(m_ingest_mutex);
    return epoch == m_replay_epoch;
  }

  void onTrajectory(const sensor_msgs::msg::PointCloud2& msg)
  {
    // require all consumed fields: a PointCloud2ConstIterator throws
    // std::runtime_error on a missing field, which would escape this
    // subscription callback and terminate the node.
    auto has_field = [&msg](const char* name) {
      for (const auto& f : msg.fields)
        if (f.name == name) return true;
      return false;
    };
    if (!has_field("t") || !has_field("x") || !has_field("y") ||
        !has_field("z") || !has_field("roll") || !has_field("pitch") ||
        !has_field("yaw") || !has_field("i"))
    {
      RCLCPP_ERROR_ONCE(get_logger(),
                        "trajectory cloud missing an x/y/z/roll/pitch/yaw/i/t "
                        "field — slam node too old; assembler disabled");
      return;
    }

    const double msg_stamp = rclcpp::Time(msg.header.stamp).seconds();
    observeInputStamp(msg_stamp, m_last_traj_stamp, "trajectory");
    sensor_msgs::PointCloud2ConstIterator<float> ix(msg, "x"), iy(msg, "y"), iz(msg, "z"),
      iroll(msg, "roll"), ipitch(msg, "pitch"), iyaw(msg, "yaw"), ii(msg, "i"),
      it(msg, "t");

    for (; ix != ix.end(); ++ix, ++iy, ++iz, ++iroll, ++ipitch, ++iyaw, ++ii, ++it)
    {
      const size_t idx = static_cast<size_t>(*ii);
      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      pose.translate(Eigen::Vector3d(*ix, *iy, *iz));
      pose.rotate(Eigen::AngleAxisd(*iyaw, Eigen::Vector3d::UnitZ()) *
                  Eigen::AngleAxisd(*ipitch, Eigen::Vector3d::UnitY()) *
                  Eigen::AngleAxisd(*iroll, Eigen::Vector3d::UnitX()));

      if (idx < m_keyframes.size())
      {
        // existing keyframe: did the graph move it?
        auto& kf                    = m_keyframes[idx];
        const Eigen::Vector3d dxyz  = pose.translation() - kf.pose.translation();
        const Eigen::AngleAxisd rot(kf.pose.rotation().transpose() * pose.rotation());
        // z has its own epsilon: a depth-only SLAM correction moves the
        // whole elevation product, and the xy/yaw gate used to swallow it —
        // the map stayed anchored at unoptimized depths forever.
        if (dxyz.head<2>().norm() > m_pose_eps_xy ||
            std::fabs(dxyz.z()) > m_pose_eps_z ||
            std::fabs(rot.angle()) > m_pose_eps_yaw)
        {
          kf.pose = pose;
          m_dirty = true;
          // Render-owned accumulators are invalidated after the renderer has
          // copied this complete pose generation. Do not touch them from the
          // concurrent ingestion callback.
        }
        continue;
      }

      // new keyframe: snapshot evidence from the buffers
      KeyframeEvidence kf;
      kf.stamp = msg_stamp + static_cast<double>(*it);
      kf.pose  = pose;
      const BufferedCloud* hits  = findNearest(m_hits_buffer, kf.stamp);
      const BufferedCloud* clear = findNearest(m_clear_buffer, kf.stamp);
      if (hits != nullptr)
      {
        kf.hits          = hits->cloud;
        kf.hits_origin   = hits->sensor_origin;
        kf.has_evidence  = true;
      }
      if (clear != nullptr)
      {
        kf.clear         = clear->cloud;
        kf.clear_origin  = clear->sensor_origin;
        kf.has_evidence  = true;
      }
      if (!kf.has_evidence)
      {
        ++m_keyframes_without_evidence;
      }

      // dense survey aggregation: fold EVERY buffered survey cloud since the
      // previous keyframe into this keyframe's robot frame via odom deltas
      // (keyframes are ~1 s apart, map_points ~10 Hz — nearest-stamp
      // snapshotting would discard ~90% of the survey density)
      if (m_survey_pub)
      {
        const rclcpp::Time kf_time(static_cast<int64_t>(kf.stamp * 1e9));
        Eigen::Isometry3d t_odom_kf;
        if (lookupAtStamp(m_robot_frame, kf_time, t_odom_kf, m_odom_frame,
                          /*allow_fallback=*/false))
        {
          SurveyCloudPtrT agg(new SurveyCloudT);
          for (const auto& entry : m_survey_buffer)
          {
            if (entry.stamp <= m_last_assoc_stamp ||
                entry.stamp > kf.stamp + m_stamp_tolerance)
            {
              continue;
            }
            const rclcpp::Time e_time(static_cast<int64_t>(entry.stamp * 1e9));
            Eigen::Isometry3d t_odom_e;
            if (!lookupAtStamp(m_robot_frame, e_time, t_odom_e, m_odom_frame,
                               /*allow_fallback=*/false))
            {
              continue;
            }
            const Eigen::Isometry3d t_kf_e = t_odom_kf.inverse() * t_odom_e;
            SurveyCloudT moved;
            pcl::transformPointCloud(*entry.cloud, moved, t_kf_e.cast<float>());
            *agg += moved;
          }
          if (!agg->empty())
          {
            kf.survey = voxelSurvey(agg, static_cast<float>(m_survey_resolution));
          }
          // Advance past the acceptance window's UPPER edge: advancing only
          // to kf.stamp left (kf.stamp, kf.stamp + tolerance] eligible for
          // this keyframe AND the next — the same clouds re-anchored through
          // two poses, inflating support. And advance only on a successful
          // odom lookup: on failure the batch stays buffered for the next
          // keyframe instead of being silently unassociated forever.
          m_last_assoc_stamp = kf.stamp + m_stamp_tolerance;
          discardAssociated(m_survey_buffer, m_last_assoc_stamp);
        }
        else
        {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 10000,
            "keyframe odom lookup failed at %.3f; holding %zu survey clouds "
            "for the next keyframe",
            kf.stamp, m_survey_buffer.size());
        }
      }

      // Lossless tile aggregation mirrors the survey's odom-delta deskew and
      // supplies only Product A, the graph-corrected image mosaic.
      if (m_tile_pub)
      {
        const rclcpp::Time kf_time(static_cast<int64_t>(kf.stamp * 1e9));
        Eigen::Isometry3d t_odom_kf;
        if (lookupAtStamp(m_robot_frame, kf_time, t_odom_kf, m_odom_frame,
                          /*allow_fallback=*/false))
        {
          ReconstructionCloudPtrT agg(new ReconstructionCloudT);
          for (const auto& entry : m_tile_buffer)
          {
            if (entry.stamp <= m_last_tile_assoc_stamp ||
                entry.stamp > kf.stamp + m_stamp_tolerance)
            {
              continue;
            }
            const rclcpp::Time e_time(static_cast<int64_t>(entry.stamp * 1e9));
            Eigen::Isometry3d t_odom_e;
            if (!lookupAtStamp(m_robot_frame, e_time, t_odom_e, m_odom_frame,
                               /*allow_fallback=*/false))
            {
              continue;
            }
            const Eigen::Isometry3f t_kf_e =
              (t_odom_kf.inverse() * t_odom_e).cast<float>();
            ReconstructionCloudPtrT moved =
              transformReconstructionCloud(entry.cloud, t_kf_e);
            *agg += *moved;
          }
          if (!agg->empty())
          {
            kf.tile = agg;
          }
          m_last_tile_assoc_stamp = kf.stamp + m_stamp_tolerance;
          discardAssociated(m_tile_buffer, m_last_tile_assoc_stamp);
        }
        else
        {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 10000,
            "keyframe odom lookup failed at %.3f; holding %zu tile clouds "
            "for the next keyframe",
            kf.stamp, m_tile_buffer.size());
        }
      }

      // Surface-return aggregation is deliberately independent of the tile:
      // each record is one acoustic lobe with radial uncertainty, while its
      // origin/axis/boresight still move together under graph corrections.
      if (m_surface_pub)
      {
        const rclcpp::Time kf_time(static_cast<int64_t>(kf.stamp * 1e9));
        Eigen::Isometry3d t_odom_kf;
        if (lookupAtStamp(m_robot_frame, kf_time, t_odom_kf, m_odom_frame,
                          /*allow_fallback=*/false))
        {
          ReconstructionCloudPtrT agg(new ReconstructionCloudT);
          for (const auto& entry : m_reconstruction_buffer)
          {
            if (entry.stamp <= m_last_reconstruction_assoc_stamp ||
                entry.stamp > kf.stamp + m_stamp_tolerance)
            {
              continue;
            }
            const rclcpp::Time e_time(static_cast<int64_t>(entry.stamp * 1e9));
            Eigen::Isometry3d t_odom_e;
            if (!lookupAtStamp(m_robot_frame, e_time, t_odom_e, m_odom_frame,
                               /*allow_fallback=*/false))
            {
              continue;
            }
            const Eigen::Isometry3f t_kf_e =
              (t_odom_kf.inverse() * t_odom_e).cast<float>();
            ReconstructionCloudPtrT moved =
              transformReconstructionCloud(entry.cloud, t_kf_e);
            *agg += *moved;
          }
          if (!agg->empty()) kf.reconstruction = agg;
          m_last_reconstruction_assoc_stamp =
            kf.stamp + m_stamp_tolerance;
          discardAssociated(
            m_reconstruction_buffer, m_last_reconstruction_assoc_stamp);
        }
        else
        {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 10000,
            "keyframe odom lookup failed at %.3f; holding %zu reconstruction "
            "clouds for the next keyframe",
            kf.stamp, m_reconstruction_buffer.size());
        }
      }

      // Spill before the move so the write sees the clouds, and key the files
      // by the index this keyframe is about to occupy.
      spillKeyframe(kf, m_keyframes.size());
      m_keyframes.push_back(std::move(kf));
      m_have_new = true;
    }
  }

  static SurveyCloudPtrT voxelSurvey(const SurveyCloudPtrT& in, const float leaf)
  {
    // NOT pcl::VoxelGrid: its centroid accumulators are a closed set and
    // value-initialize every custom field to 0.0 — see survey_voxel.hpp.
    return voxelReduceSurvey(in, leaf);
  }

  // 21 bits per axis (signed, two's-complement low bits) -> +/-1M cells,
  // i.e. +/-52 km at the 0.05 m survey resolution
  static uint64_t voxelKey(const float x, const float y, const float z, const double res)
  {
    const auto q = [res](const float v) {
      return static_cast<uint64_t>(
        static_cast<int64_t>(std::floor(static_cast<double>(v) / res)) & 0x1FFFFF);
    };
    return (q(x) << 42) | (q(y) << 21) | q(z);
  }

  void integrateKeyframe(const KeyframeEvidence& kf, const size_t idx)
  {
    // Scoped: the loaded clouds are released when this returns, so a full
    // re-render holds one keyframe at a time rather than all of them.
    const LoadedEvidence e =
      loadEvidence(kf, idx, /*want_occupancy=*/true, /*want_survey=*/false);
    // Use each cloud's own sensor origin: raycastPointCloud does max-range
    // clipping relative to the origin, so hit endpoints must be judged against
    // the hits cloud's origin, not the clear cloud's (which used to overwrite it).
    if (e.hits && !e.hits->empty())
    {
      const Eigen::Vector3d origin = kf.pose * kf.hits_origin;
      CloudPtrT in_map(new CloudT);
      pcl::transformPointCloud(*e.hits, *in_map, kf.pose.cast<float>());
      m_map->insertPointCloud(in_map, origin, "hits");
    }
    if (e.clear && !e.clear->empty())
    {
      const Eigen::Vector3d origin = kf.pose * kf.clear_origin;
      CloudPtrT in_map(new CloudT);
      pcl::transformPointCloud(*e.clear, *in_map, kf.pose.cast<float>());
      m_map->insertPointCloud(in_map, origin, "clear");
    }
  }

  void accumulateSurvey(const KeyframeEvidence& kf, const size_t idx)
  {
    const LoadedEvidence e =
      loadEvidence(kf, idx, /*want_occupancy=*/false, /*want_survey=*/true);
    if (!e.survey || e.survey->empty())
    {
      return;
    }
    SurveyCloudT in_map;
    pcl::transformPointCloud(*e.survey, in_map, kf.pose.cast<float>());
    // Count a support cell at most once per keyframe. This preserves the
    // meaning of support when the fine survey contains many adjacent range
    // bins or several buffered pings from the same keyframe interval.
    std::unordered_set<uint64_t> support_this_keyframe;
    support_this_keyframe.reserve(in_map.size());
    for (const auto& p : in_map.points)
    {
      support_this_keyframe.insert(
        voxelKey(p.x, p.y, p.z, m_survey_support_resolution));
      VoxAcc& a = m_vox[voxelKey(p.x, p.y, p.z, m_survey_resolution)];
      a.x += p.x;
      a.y += p.y;
      a.z += p.z;
      a.i += p.intensity;
      a.r += p.range;
      if (p.incidence >= 0.0f)
      {
        a.ci += p.incidence;
        ++a.nci;
      }
      if (std::isfinite(p.texture) && std::isfinite(p.texture_squared))
      {
        a.texture += p.texture;
        a.texture_squared += p.texture_squared;
        ++a.ntexture;
      }
      if (std::isfinite(p.elevation_lo_offset) &&
          std::isfinite(p.elevation_hi_offset) &&
          std::isfinite(p.elevation_resolved))
      {
        // Store absolute bounds while accumulating so the final offsets are
        // relative to the reduced voxel centroid, not to whichever source
        // representative happened to arrive first.
        a.elevation_lo += p.z + p.elevation_lo_offset;
        a.elevation_hi += p.z + p.elevation_hi_offset;
        a.elevation_resolved +=
          std::clamp(static_cast<double>(p.elevation_resolved), 0.0, 1.0);
        ++a.nelevation;
      }
      ++a.n;
    }
    for (const uint64_t key : support_this_keyframe)
    {
      ++m_survey_support[key];
    }
  }

  // Walk the accumulated survey voxels, applying the occupancy mask, and hand
  // each reduced 13-field row to `fn`. Shared by the topic and the PCD export
  // so the published product and the file on disk cannot drift apart.
  template <typename F>
  void forEachSurveyRow(F&& fn)
  {
    auto grid = m_map->getGrid();  // handle first, then lock (locks internally)
    std::shared_lock mask_lock(*m_map->getMapMutex());
    auto acc = grid->getConstAccessor();
    for (const auto& [key, a] : m_vox)
    {
      (void)key;
      if (a.n <= 0)
      {
        continue;
      }
      const float cx = static_cast<float>(a.x / a.n);
      const float cy = static_cast<float>(a.y / a.n);
      const float cz = static_cast<float>(a.z / a.n);
      const auto support_it = m_survey_support.find(
        voxelKey(cx, cy, cz, m_survey_support_resolution));
      const int support = support_it == m_survey_support.end()
        ? 0
        : support_it->second;
      if (m_survey_occupancy_mask)
      {
        const openvdb::Coord c =
          openvdb::Coord::round(grid->worldToIndex(openvdb::Vec3d(cx, cy, cz)));
        if (!(acc.isValueOn(c) && acc.getValue(c) > 0.0f))
        {
          continue;
        }
      }
      const float texture =
        a.ntexture > 0 ? static_cast<float>(a.texture / a.ntexture) : 0.0f;
      const double texture_variance =
        a.ntexture > 0
          ? a.texture_squared / a.ntexture -
              static_cast<double>(texture) * texture
          : 0.0;
      const SurveyRow row = {
        cx,
        cy,
        cz,
        static_cast<float>(a.i / a.n),
        static_cast<float>(a.r / a.n),
        a.nci > 0 ? static_cast<float>(a.ci / a.nci) : -1.0f,
        static_cast<float>(support),
        0.0f,
        texture,
        static_cast<float>(std::max(0.0, texture_variance)),
        a.nelevation > 0
          ? static_cast<float>(a.elevation_lo / a.nelevation - cz)
          : 0.0f,
        a.nelevation > 0
          ? static_cast<float>(a.elevation_hi / a.nelevation - cz)
          : 0.0f,
        a.nelevation > 0
          ? static_cast<float>(a.elevation_resolved / a.nelevation)
          : 0.0f};
      fn(row);
    }
  }

  // Write the consolidated survey product. Returns false with `message` set on
  // any failure so the service reports it rather than claiming success.
  bool exportSurvey(std::string& message)
  {
    if (m_export_path.empty())
    {
      message = "export_path is unset";
      return false;
    }
    if (!m_survey_pub)
    {
      message = "survey stream is disabled (survey_topic unset)";
      return false;
    }
    // Never serialize a stale product: a trajectory update that landed
    // within render_min_period of this call (shutdown included) would
    // otherwise leave the occupancy mask and the accumulator at the old
    // poses, with nothing logged. Force the render path, then fold anything
    // the accumulator has not seen.
    m_last_product_render = -std::numeric_limits<double>::infinity();
    // renderIfNeeded snapshots the ingestion state under its mutex. Calling
    // it unconditionally avoids racing an unlocked dirty-flag probe with the
    // ingestion executor lane; it returns immediately when already current.
    renderIfNeeded(/*publish_outputs=*/false);
    syncSurveyAccumulator();

    pcl::PointCloud<SurveyExportPoint> out;
    out.reserve(m_vox.size());
    forEachSurveyRow([&out](const SurveyRow& row) {
      SurveyExportPoint p;
      p.x          = row[0];
      p.y          = row[1];
      p.z          = row[2];
      p.intensity  = row[3];
      p.range      = row[4];
      p.incidence  = row[5];
      p.support    = row[6];
      p.pose_sigma = row[7];
      p.texture = row[8];
      p.texture_variance = row[9];
      p.elevation_lo_offset = row[10];
      p.elevation_hi_offset = row[11];
      p.elevation_resolved_fraction = row[12];
      out.push_back(p);
    });
    if (out.empty())
    {
      message = "nothing to export (no survey voxels)";
      return false;
    }
    const std::filesystem::path path(m_export_path);
    if (path.has_parent_path())
    {
      std::error_code ec;
      std::filesystem::create_directories(path.parent_path(), ec);
    }
    // Write to a sibling temp and rename: an export interrupted midway would
    // otherwise leave a truncated file in place of the previous good one, and
    // rename within a directory is atomic.
    const std::string tmp = m_export_path + ".part";
    auto cloud = out.makeShared();
    if (!writeSpill(cloud, tmp))
    {
      message = "failed to write " + tmp;
      return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec)
    {
      message = "failed to move " + tmp + " into place: " + ec.message();
      return false;
    }
    message = "wrote " + std::to_string(out.size()) + " points to " + m_export_path;
    RCLCPP_INFO(get_logger(), "%s", message.c_str());
    return true;
  }

  bool isConfirmedSurfacePoint(const ReconstructionRow& p) const
  {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
      std::isfinite(p.intensity) && std::isfinite(p.confidence) &&
      p.intensity >= static_cast<float>(m_surface_min_return_intensity) &&
      p.confidence >= static_cast<float>(m_surface_min_confidence);
  }

  // A ribbon intersection is only a geometric hypothesis. Keep it as
  // conservative global-occupancy evidence when it passes the multi-view
  // gates AND the occupancy volume has not explicitly observed that 3-D voxel
  // as free. Unknown (background value 0) remains eligible: absence of a clear
  // ray is not evidence against a surface. OccupancyVDBMapping stores misses
  // as inactive negative-log-odds values, so isValueOn() cannot be used for
  // this test.
  std::vector<ReconstructionRow> selectConfirmedSurface(
    const std::vector<ReconstructionRow>& reconstructed,
    std::size_t* free_space_rejected = nullptr) const
  {
    std::vector<ReconstructionRow> navigation;
    navigation.reserve(reconstructed.size());
    std::size_t rejected = 0;

    // getGrid() locks internally; obtain the handle before taking the shared
    // map lock, matching every other read-side path in this node.
    auto grid = m_map->getGrid();
    std::shared_lock map_lock(*m_map->getMapMutex());
    auto acc = grid->getConstAccessor();
    for (const auto& row : reconstructed)
    {
      if (!isConfirmedSurfacePoint(row))
      {
        continue;
      }
      const openvdb::Coord coord = openvdb::Coord::round(
        grid->worldToIndex(openvdb::Vec3d(row.x, row.y, row.z)));
      if (acc.getValue(coord) < 0.0F)
      {
        ++rejected;
        continue;
      }
      navigation.push_back(row);
    }
    if (free_space_rejected != nullptr)
    {
      *free_space_rejected = rejected;
    }
    return navigation;
  }

  // Convert the structure-preserving supported survey into navigation
  // candidates without coarsening its centimetre representation. `support`
  // is distinct keyframes, not pings or neighboring image bins. Elevation-
  // resolved returns earn the full support confidence; unresolved returns are
  // allowed only after substantially more revisits. Explicit free-space
  // evidence still vetoes either kind, while unknown space remains eligible.
  std::vector<NavigationRow> selectSurveyNavigationSurface(
    const std::vector<SurveyRow>& supported,
    std::size_t* free_space_rejected = nullptr) const
  {
    std::vector<NavigationRow> navigation;
    navigation.reserve(supported.size());
    std::size_t rejected = 0;

    auto grid = m_map->getGrid();
    std::shared_lock map_lock(*m_map->getMapMutex());
    auto acc = grid->getConstAccessor();
    for (const auto& row : supported)
    {
      const float support = row[6];
      const float intensity = row[3];
      if (!std::isfinite(row[0]) || !std::isfinite(row[1]) ||
          !std::isfinite(row[2]) || !std::isfinite(intensity) ||
          !std::isfinite(support) ||
          support < static_cast<float>(m_survey_min_support) ||
          intensity < static_cast<float>(m_navigation_min_intensity))
      {
        continue;
      }
      const float resolved = std::isfinite(row[12])
        ? std::clamp(row[12], 0.0F, 1.0F) : 0.0F;
      const float support_score = 1.0F - std::exp(
        -support / static_cast<float>(m_survey_min_support));
      const float elevation_score = static_cast<float>(
        m_navigation_unresolved_confidence_scale) +
        (1.0F - static_cast<float>(
          m_navigation_unresolved_confidence_scale)) * resolved;
      const float confidence = support_score * elevation_score;
      if (confidence < static_cast<float>(m_navigation_min_confidence))
      {
        continue;
      }

      const openvdb::Coord coord = openvdb::Coord::round(
        grid->worldToIndex(openvdb::Vec3d(row[0], row[1], row[2])));
      if (acc.getValue(coord) < 0.0F)
      {
        ++rejected;
        continue;
      }
      navigation.push_back({
        row[0], row[1], row[2], intensity, row[4], row[5], support,
        confidence, row[7], row[8], row[9], row[10], row[11], resolved});
    }
    if (free_space_rejected != nullptr)
    {
      *free_space_rejected = rejected;
    }
    return navigation;
  }

  std::vector<SurfelRow> buildNavigationSurfels(
    std::vector<NavigationRow>& navigation) const
  {
    std::vector<ReconstructionRow> candidates;
    candidates.reserve(navigation.size());
    for (const auto& row : navigation)
    {
      candidates.push_back({
        row.x, row.y, row.z, row.intensity, row.support,
        0.0F, row.confidence,
        0.0F, 0.0F, 0.0F, 0.0F});
    }
    std::vector<SurfelRow> surfels = fitSurfaceElements(
      candidates, static_cast<float>(m_survey_resolution),
      static_cast<float>(m_surfel_radius_m), m_surfel_min_neighbors,
      static_cast<float>(m_surfel_max_surface_variation),
      static_cast<float>(m_surfel_max_projection));
    surfels.erase(
      std::remove_if(
        surfels.begin(), surfels.end(), [this, &navigation](const SurfelRow& p) {
          return !std::isfinite(p.x) || !std::isfinite(p.y) ||
            !std::isfinite(p.z) || !std::isfinite(p.confidence) ||
            p.confidence < static_cast<float>(m_navigation_min_confidence) ||
            p.source_index >= navigation.size();
        }),
      surfels.end());
    for (const auto& surfel : surfels)
    {
      auto& row = navigation[surfel.source_index];
      // Keep the dense navigation return at its measured survey position. The
      // projected coordinate belongs only to the strict surfel product; the
      // dense product receives the fitted orientation/quality metadata.
      row.normal_x = surfel.normal_x;
      row.normal_y = surfel.normal_y;
      row.normal_z = surfel.normal_z;
      row.curvature = surfel.curvature;
      row.residual = surfel.residual;
      row.normal_valid = 1.0F;
    }
    return surfels;
  }

  void queueNavigationFit(
    const rclcpp::Time& stamp, const RenderDemand& demand,
    const std::size_t free_space_rejected,
    std::vector<NavigationRow>&& navigation)
  {
    NavigationFitJob job;
    job.generation = ++m_navigation_fit_generation;
    job.stamp = stamp;
    job.publish_navigation = demand.navigation;
    job.publish_surfels = demand.navigation_surfels;
    job.free_space_rejected = free_space_rejected;
    job.navigation = std::move(navigation);
    job.queued_at = SteadyClock::now();
    {
      std::lock_guard<std::mutex> lock(m_navigation_fit_mutex);
      if (m_navigation_fit_stop)
      {
        return;
      }
      job.epoch = m_navigation_fit_epoch;
      if (m_pending_navigation_fit.has_value())
      {
        ++m_navigation_fits_coalesced;
      }
      // At most one future fit is retained. If the renderer advances several
      // generations while an exact fit is running, only the newest complete
      // navigation snapshot is useful after the active one finishes.
      m_pending_navigation_fit = std::move(job);
    }
    m_navigation_fit_cv.notify_one();
  }

  void navigationFitWorker()
  {
    while (true)
    {
      NavigationFitJob job;
      {
        std::unique_lock<std::mutex> lock(m_navigation_fit_mutex);
        m_navigation_fit_cv.wait(lock, [this] {
          return m_navigation_fit_stop || m_pending_navigation_fit.has_value();
        });
        if (m_navigation_fit_stop)
        {
          return;
        }
        job = std::move(*m_pending_navigation_fit);
        m_pending_navigation_fit.reset();
      }

      const double queue_ms = elapsedWallMs(job.queued_at);
      const auto fit_started = SteadyClock::now();
      std::vector<SurfelRow> surfels =
        buildNavigationSurfels(job.navigation);
      const double fit_ms = elapsedWallMs(fit_started);

      std::uint64_t coalesced = 0;
      bool newer_pending = false;
      bool stale = false;
      {
        std::lock_guard<std::mutex> lock(m_navigation_fit_mutex);
        coalesced = m_navigation_fits_coalesced;
        newer_pending = m_pending_navigation_fit.has_value();
        stale = m_navigation_fit_stop || job.epoch != m_navigation_fit_epoch;
      }
      ProductPublishMetrics publish_metrics;
      if (!stale)
      {
        publish_metrics = publishNavigationProducts(
          job.stamp, job.publish_navigation, job.publish_surfels,
          job.navigation, surfels);
      }
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 30000,
        "navigation fit: generation=%llu queue=%.1f ms fit=%.1f ms "
        "publish=%.1f/%.1f ms rows=%zu surfels=%zu rejected_free=%zu "
        "coalesced=%llu newer_pending=%d stale=%d",
        static_cast<unsigned long long>(job.generation), queue_ms, fit_ms,
        publish_metrics.navigation_ms, publish_metrics.surfel_ms,
        job.navigation.size(), surfels.size(), job.free_space_rejected,
        static_cast<unsigned long long>(coalesced),
        static_cast<int>(newer_pending), static_cast<int>(stale));
    }
  }

  void stopNavigationFitWorker()
  {
    {
      std::lock_guard<std::mutex> lock(m_navigation_fit_mutex);
      if (!m_navigation_fit_thread.joinable())
      {
        return;
      }
      m_navigation_fit_stop = true;
      m_pending_navigation_fit.reset();
    }
    m_navigation_fit_cv.notify_one();
    m_navigation_fit_thread.join();
  }

  void collectNavigationProducts(
    std::vector<NavigationRow>& navigation,
    std::vector<SurfelRow>& surfels)
  {
    // Force pending occupancy evidence/graph corrections through first: the
    // clean subset includes both the graph-corrected survey and the explicit
    // free-space contradiction mask.
    m_last_product_render = -std::numeric_limits<double>::infinity();
    renderIfNeeded(/*publish_outputs=*/false);
    syncSurveyAccumulator();
    std::vector<SurveyRow> supported;
    supported.reserve(m_vox.size());
    forEachSurveyRow([this, &supported](const SurveyRow& row) {
      if (row[6] >= static_cast<float>(m_survey_min_support))
      {
        supported.push_back(row);
      }
    });
    navigation = selectSurveyNavigationSurface(supported);
    surfels = buildNavigationSurfels(navigation);
  }

  template<typename PointT>
  bool writeNavigationExport(
    pcl::PointCloud<PointT>& out, const std::string& path,
    const char* product, std::string& message)
  {
    if (out.empty())
    {
      message = std::string("nothing to export (no ") + product +
        " passed the gates)";
      return false;
    }
    const std::filesystem::path destination(path);
    if (destination.has_parent_path())
    {
      std::error_code ec;
      std::filesystem::create_directories(destination.parent_path(), ec);
    }
    const std::string tmp = path + ".part";
    auto cloud = out.makeShared();
    if (!writeSpill(cloud, tmp))
    {
      message = "failed to write " + tmp;
      return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, destination, ec);
    if (ec)
    {
      message = "failed to move " + tmp + " into place: " + ec.message();
      return false;
    }
    message = "wrote " + std::to_string(out.size()) + " " + product +
      " to " + path;
    RCLCPP_INFO(get_logger(), "%s", message.c_str());
    return true;
  }

  // Export the exact dense return set carried by ~/navigation_pointcloud.
  bool exportNavigationSurface(std::string& message)
  {
    if (m_navigation_export_path.empty())
    {
      message = "navigation_export_path is unset";
      return false;
    }
    if (!m_navigation_pub)
    {
      message = "navigation returns are disabled (survey_topic unset)";
      return false;
    }

    std::vector<NavigationRow> navigation;
    std::vector<SurfelRow> surfels;
    collectNavigationProducts(navigation, surfels);
    pcl::PointCloud<NavigationExportPoint> out;
    out.reserve(navigation.size());
    for (const auto& row : navigation)
    {
      NavigationExportPoint p;
      p.x = row.x;
      p.y = row.y;
      p.z = row.z;
      p.intensity = row.intensity;
      p.range = row.range;
      p.incidence = row.incidence;
      p.support = row.support;
      p.confidence = row.confidence;
      p.pose_sigma = row.pose_sigma;
      p.texture = row.texture;
      p.texture_variance = row.texture_variance;
      p.elevation_lo_offset = row.elevation_lo_offset;
      p.elevation_hi_offset = row.elevation_hi_offset;
      p.elevation_resolved_fraction = row.elevation_resolved_fraction;
      p.normal_x = row.normal_x;
      p.normal_y = row.normal_y;
      p.normal_z = row.normal_z;
      p.curvature = row.curvature;
      p.residual = row.residual;
      p.normal_valid = row.normal_valid;
      out.push_back(p);
    }
    return writeNavigationExport(
      out, m_navigation_export_path, "navigation returns", message);
  }

  // Export the exact strict subset carried by
  // ~/navigation_surfel_pointcloud.
  bool exportNavigationSurfels(std::string& message)
  {
    if (m_navigation_surfel_export_path.empty())
    {
      message = "navigation_surfel_export_path is unset";
      return false;
    }
    if (!m_navigation_surfel_pub)
    {
      message = "navigation surfels are disabled (survey_topic unset)";
      return false;
    }

    std::vector<NavigationRow> navigation;
    std::vector<SurfelRow> surfels;
    collectNavigationProducts(navigation, surfels);
    pcl::PointCloud<SurfaceExportPoint> out;
    out.reserve(surfels.size());
    for (const auto& row : surfels)
    {
      SurfaceExportPoint p;
      p.x = row.x;
      p.y = row.y;
      p.z = row.z;
      p.intensity = row.intensity;
      p.support = row.support;
      p.view_span_deg = row.view_span_deg;
      p.confidence = row.confidence;
      p.normal_x = row.normal_x;
      p.normal_y = row.normal_y;
      p.normal_z = row.normal_z;
      p.curvature = row.curvature;
      p.residual = row.residual;
      p.range_sigma = row.range_sigma;
      p.echo_width = row.echo_width;
      p.echo_prominence = row.echo_prominence;
      p.peak_prominence = row.peak_prominence;
      out.push_back(p);
    }
    return writeNavigationExport(
      out, m_navigation_surfel_export_path, "navigation surfels", message);
  }

  // Bring the survey accumulator up to date with the keyframe list: rebuild
  // from scratch when the graph moved (the old contributions are wrong),
  // then fold any keyframes it has not seen. Shared by the render path and
  // the export path so neither can read a lagging accumulator — survey-only
  // keyframes used to wait for the next OCCUPANCY change to be folded in.
  void syncSurveyAccumulator()
  {
    if (!m_survey_pub)
    {
      return;
    }
    if (m_survey_stale)
    {
      m_vox.clear();
      m_survey_support.clear();
      m_survey_upto  = 0;
      m_survey_stale = false;
    }
    for (size_t i = m_survey_upto; i < m_render_keyframes.size(); ++i)
    {
      accumulateSurvey(m_render_keyframes[i], i);
    }
    m_survey_upto = m_render_keyframes.size();
  }

  void accumulateTile(const KeyframeEvidence& kf, const size_t idx)
  {
    const LoadedEvidence e = loadEvidence(
      kf, idx, /*want_occupancy=*/false, /*want_survey=*/false,
      /*want_reconstruction=*/false, /*want_tile=*/true);
    if (!e.tile || e.tile->empty())
    {
      return;
    }
    const Eigen::Isometry3f pose = kf.pose.cast<float>();
    for (const auto& local : e.tile->points)
    {
      const SonarReconstructionPoint p =
        transformReconstructionPoint(local, pose);
      TileVoxAcc& a = m_tile_vox[voxelKey(
        p.x, p.y, p.z, m_tile_resolution)];
      a.x += p.x;
      a.y += p.y;
      a.z += p.z;
      a.intensity += p.intensity;
      a.range += p.range;
      a.half_angle += p.elevation_half_angle;
      ++a.n;
    }
  }

  void accumulateSurface(const KeyframeEvidence& kf, const size_t idx)
  {
    const LoadedEvidence e = loadEvidence(
      kf, idx, /*want_occupancy=*/false, /*want_survey=*/false,
      /*want_reconstruction=*/true);
    if (!e.reconstruction || e.reconstruction->empty()) return;
    const Eigen::Isometry3f pose = kf.pose.cast<float>();
    for (const auto& local : e.reconstruction->points)
    {
      m_surface_accumulator->add(
        transformReconstructionPoint(local, pose));
    }
  }

  void syncSurfaceAccumulator()
  {
    if (!m_surface_pub)
    {
      return;
    }
    if (m_surface_stale)
    {
      m_surface_accumulator->clear();
      m_surface_upto = 0;
      m_surface_stale = false;
    }
    for (size_t i = m_surface_upto; i < m_render_keyframes.size(); ++i)
    {
      accumulateSurface(m_render_keyframes[i], i);
    }
    m_surface_upto = m_render_keyframes.size();
  }

  void syncTileAccumulator()
  {
    if (!m_tile_pub)
    {
      return;
    }
    if (m_tile_stale)
    {
      std::unordered_map<uint64_t, TileVoxAcc> empty;
      m_tile_vox.swap(empty);
      m_tile_upto = 0;
      m_tile_stale = false;
    }
    for (size_t i = m_tile_upto; i < m_render_keyframes.size(); ++i)
    {
      accumulateTile(m_render_keyframes[i], i);
    }
    m_tile_upto = m_render_keyframes.size();
  }

  void releaseTileAccumulator()
  {
    // The immutable full-resolution tile evidence remains on disk.
    // Releasing this derived hash table therefore loses no product: a later
    // subscriber rebuilds the exact graph-corrected tile mosaic on demand.
    // swap(), unlike clear(), also returns the multi-million-bucket allocation.
    if (!m_tile_vox.empty())
    {
      std::unordered_map<uint64_t, TileVoxAcc> empty;
      m_tile_vox.swap(empty);
    }
    m_tile_upto = 0;
    m_tile_stale = true;
  }

  template<std::size_t N, typename EmitRows>
  void publishFloatRows(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher,
    const std::array<const char*, N>& names,
    const std::size_t maximum_rows,
    EmitRows&& emit_rows,
    const rclcpp::Time& stamp)
  {
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = m_map_frame;
    msg.height = 1;
    msg.is_bigendian = false;
    msg.is_dense = true;
    msg.point_step = static_cast<std::uint32_t>(N * sizeof(float));
    msg.fields.reserve(N);
    for (std::size_t f = 0; f < N; ++f)
    {
      sensor_msgs::msg::PointField field;
      field.name = names[f];
      field.offset = static_cast<std::uint32_t>(f * sizeof(float));
      field.datatype = sensor_msgs::msg::PointField::FLOAT32;
      field.count = 1;
      msg.fields.push_back(field);
    }
    msg.data.resize(maximum_rows * msg.point_step);
    std::size_t emitted = 0;
    const auto append = [&msg, &emitted, maximum_rows](
      const std::array<float, N>& row) {
        if (emitted >= maximum_rows)
        {
          return;
        }
        std::memcpy(
          msg.data.data() + emitted * msg.point_step,
          row.data(), msg.point_step);
        ++emitted;
      };
    emit_rows(append);
    msg.data.resize(emitted * msg.point_step);
    msg.width = static_cast<std::uint32_t>(emitted);
    msg.row_step = msg.point_step * msg.width;
    publisher->publish(msg);
  }

  static bool hasSubscribers(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher)
  {
    return publisher &&
      (publisher->get_subscription_count() +
       publisher->get_intra_process_subscription_count()) > 0;
  }

  ProductPublishMetrics publishReconstructionProducts(
    const rclcpp::Time& stamp,
    const RenderDemand& demand,
    const std::vector<ReconstructionRow>& reconstructed,
    const std::size_t confirmed_count,
    const std::size_t surface_free_space_rejected,
    const std::size_t survey_buffer_size,
    const std::size_t tile_buffer_size,
    const std::size_t reconstruction_buffer_size)
  {
    ProductPublishMetrics metrics;
    if (!m_tile_pub && !m_surface_pub)
    {
      return metrics;
    }
    auto stage_started = SteadyClock::now();
    if (demand.tile)
    {
      syncTileAccumulator();

      // Product A: graph-corrected centre-plane intensity tiles. Aperture is
      // metadata here, never painted as thickness. The immutable source
      // evidence remains on disk, so this large derived cache is resident and
      // serialized only while somebody is actually inspecting the tile.
      constexpr std::array<const char*, 7> tile_names = {
        "x", "y", "z", "intensity", "range", "support", "aperture_half_deg"};
      publishFloatRows(
        m_tile_pub, tile_names, m_tile_vox.size(),
        [this](const auto& append) {
          for (const auto& [key, a] : m_tile_vox)
          {
            (void)key;
            if (a.n <= 0)
            {
              continue;
            }
            append(std::array<float, 7>{
              static_cast<float>(a.x / a.n),
              static_cast<float>(a.y / a.n),
              static_cast<float>(a.z / a.n),
              static_cast<float>(a.intensity / a.n),
              static_cast<float>(a.range / a.n),
              static_cast<float>(a.n),
              static_cast<float>(a.half_angle / a.n * 180.0 / M_PI)});
          }
        }, stamp);
    }
    else
    {
      releaseTileAccumulator();
    }
    metrics.tile_ms = elapsedWallMs(stage_started);

    // Product B: only ribbon intersections with independent angular support.
    constexpr std::array<const char*, 11> surface_names = {
      "x", "y", "z", "intensity", "support", "view_span_deg", "confidence",
      "range_sigma", "echo_width", "echo_prominence", "peak_prominence"};
    stage_started = SteadyClock::now();
    if (demand.surface)
    {
      publishFloatRows(
        m_surface_pub, surface_names, reconstructed.size(),
        [&reconstructed](const auto& append) {
          for (const auto& p : reconstructed)
          {
            append(std::array<float, 11>{
              p.x, p.y, p.z, p.intensity, p.support,
              p.view_span_deg, p.confidence, p.range_sigma, p.echo_width,
              p.echo_prominence, p.peak_prominence});
          }
        }, stamp);
    }
    metrics.surface_ms = elapsedWallMs(stage_started);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 30000,
      "reconstruction products: tile %s (%zu voxels), "
      "%zu/%zu confirmed ribbon voxels (%zu rejected by observed free "
      "space), %zu survey + %zu tile + %zu reconstruction pings awaiting "
      "keyframe association",
      demand.tile ? "resident" : "on disk", m_tile_vox.size(),
      confirmed_count, reconstructed.size(), surface_free_space_rejected,
      survey_buffer_size, tile_buffer_size, reconstruction_buffer_size);
    return metrics;
  }

  ProductPublishMetrics publishNavigationProducts(
    const rclcpp::Time& stamp,
    const bool publish_navigation,
    const bool publish_surfels,
    const std::vector<NavigationRow>& navigation,
    const std::vector<SurfelRow>& surfels)
  {
    ProductPublishMetrics metrics;
    // These are retained global snapshots. A zero stamp asks cross-frame
    // consumers for the latest transform instead of binding the durable
    // sample to an expired TF entry.
    const rclcpp::Time timeless_stamp(0, 0, stamp.get_clock_type());
    constexpr std::array<const char*, 20> navigation_names = {
      "x", "y", "z", "intensity", "range", "incidence", "support",
      "confidence", "pose_sigma", "texture", "texture_variance",
      "elevation_lo_offset", "elevation_hi_offset",
      "elevation_resolved_fraction", "normal_x", "normal_y", "normal_z",
      "curvature", "residual", "normal_valid"};
    auto stage_started = SteadyClock::now();
    if (publish_navigation)
    {
      publishFloatRows(
        m_navigation_pub, navigation_names, navigation.size(),
        [&navigation](const auto& append) {
          for (const auto& p : navigation)
          {
            append(std::array<float, 20>{
              p.x, p.y, p.z, p.intensity, p.range, p.incidence, p.support,
              p.confidence, p.pose_sigma, p.texture, p.texture_variance,
              p.elevation_lo_offset, p.elevation_hi_offset,
              p.elevation_resolved_fraction, p.normal_x, p.normal_y,
              p.normal_z, p.curvature, p.residual, p.normal_valid});
          }
        }, timeless_stamp);
    }
    metrics.navigation_ms = elapsedWallMs(stage_started);

    constexpr std::array<const char*, 16> surfel_names = {
      "x", "y", "z", "intensity", "support", "view_span_deg", "confidence",
      "normal_x", "normal_y", "normal_z", "curvature", "residual",
      "range_sigma", "echo_width", "echo_prominence", "peak_prominence"};
    stage_started = SteadyClock::now();
    if (publish_surfels)
    {
      publishFloatRows(
        m_navigation_surfel_pub, surfel_names, surfels.size(),
        [&surfels](const auto& append) {
          for (const auto& p : surfels)
          {
            append(std::array<float, 16>{
              p.x, p.y, p.z, p.intensity, p.support,
              p.view_span_deg, p.confidence,
              p.normal_x, p.normal_y, p.normal_z, p.curvature, p.residual,
              p.range_sigma, p.echo_width, p.echo_prominence,
              p.peak_prominence});
          }
        }, timeless_stamp);
    }
    metrics.surfel_ms = elapsedWallMs(stage_started);
    return metrics;
  }

  // Merge the quality-controlled multi-view surface into the graph-corrected
  // planning grid. The VDB projection still supplies observed/free/unknown
  // state from echo-bounded clearing rays. In ConfirmedSurface mode its
  // provisional capped-curtain hits are demoted to unknown before confirmed
  // surface columns are marked lethal; lack of multi-view confirmation is not
  // evidence of free space. Local STVL retains those immediate conservative
  // hits independently.
  void applyNavigationSurfaceToOccupancy(
    nav_msgs::msg::OccupancyGrid& grid,
    const std::vector<ReconstructionRow>& navigation) const
  {
    if (m_global_occupancy_mode == GlobalOccupancyMode::Hits)
    {
      return;
    }

    const double resolution = grid.info.resolution > 0.0
      ? static_cast<double>(grid.info.resolution) : m_resolution;
    if (!(resolution > 0.0) || !std::isfinite(resolution))
    {
      return;
    }

    const bool old_valid = grid.info.width > 0 && grid.info.height > 0 &&
      grid.data.size() ==
        static_cast<std::size_t>(grid.info.width) * grid.info.height;
    int old_min_x = 0;
    int old_min_y = 0;
    int old_max_x = -1;
    int old_max_y = -1;
    if (old_valid)
    {
      // createMappingOutput places the grid corner half a voxel below the
      // integer VDB cell centre. Recover that absolute cell index exactly.
      old_min_x = static_cast<int>(std::llround(
        grid.info.origin.position.x / resolution + 0.5));
      old_min_y = static_cast<int>(std::llround(
        grid.info.origin.position.y / resolution + 0.5));
      old_max_x = old_min_x + static_cast<int>(grid.info.width) - 1;
      old_max_y = old_min_y + static_cast<int>(grid.info.height) - 1;
    }

    int min_x = old_min_x;
    int min_y = old_min_y;
    int max_x = old_max_x;
    int max_y = old_max_y;
    bool have_bounds = old_valid;
    for (const auto& p : navigation)
    {
      const int x = static_cast<int>(std::llround(p.x / resolution));
      const int y = static_cast<int>(std::llround(p.y / resolution));
      if (!have_bounds)
      {
        min_x = max_x = x;
        min_y = max_y = y;
        have_bounds = true;
      }
      else
      {
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
      }
    }

    if (!have_bounds)
    {
      return;
    }

    if (!old_valid || min_x != old_min_x || min_y != old_min_y ||
        max_x != old_max_x || max_y != old_max_y)
    {
      const std::uint32_t new_width =
        static_cast<std::uint32_t>(max_x - min_x + 1);
      const std::uint32_t new_height =
        static_cast<std::uint32_t>(max_y - min_y + 1);
      std::vector<std::int8_t> expanded(
        static_cast<std::size_t>(new_width) * new_height, -1);
      if (old_valid)
      {
        const int x_offset = old_min_x - min_x;
        const int y_offset = old_min_y - min_y;
        for (std::uint32_t y = 0; y < grid.info.height; ++y)
        {
          const auto old_offset = static_cast<std::size_t>(y) * grid.info.width;
          const auto new_offset =
            static_cast<std::size_t>(y + y_offset) * new_width + x_offset;
          std::copy_n(
            grid.data.begin() + old_offset, grid.info.width,
            expanded.begin() + new_offset);
        }
      }
      grid.info.width = new_width;
      grid.info.height = new_height;
      grid.info.resolution = static_cast<float>(resolution);
      grid.info.origin.position.x = (min_x - 0.5) * resolution;
      grid.info.origin.position.y = (min_y - 0.5) * resolution;
      grid.info.origin.orientation.w = 1.0;
      grid.data.swap(expanded);
    }

    if (m_global_occupancy_mode == GlobalOccupancyMode::ConfirmedSurface)
    {
      for (auto& cell : grid.data)
      {
        if (cell == 100)
        {
          cell = -1;
        }
      }
    }

    for (const auto& p : navigation)
    {
      const int x = static_cast<int>(std::llround(p.x / resolution)) - min_x;
      const int y = static_cast<int>(std::llround(p.y / resolution)) - min_y;
      if (x < 0 || y < 0 || x >= static_cast<int>(grid.info.width) ||
          y >= static_cast<int>(grid.info.height))
      {
        continue;
      }
      grid.data[static_cast<std::size_t>(y) * grid.info.width + x] = 100;
    }
  }

  void replaceLatchedTileWithEmpty(const rclcpp::Time& stamp)
  {
    constexpr std::array<const char*, 7> tile_names = {
      "x", "y", "z", "intensity", "range", "support", "aperture_half_deg"};
    publishFloatRows(
      m_tile_pub, tile_names, 0,
      [](const auto&) {}, stamp);
  }

  void renderIfNeeded(const bool publish_outputs = true)
  {
    const auto render_started = SteadyClock::now();
    const double now = get_clock()->now().seconds();
    const RenderDemand demand{
      hasSubscribers(m_tile_pub),
      hasSubscribers(m_surface_pub),
      hasSubscribers(m_survey_pub),
      hasSubscribers(m_supported_survey_pub),
      hasSubscribers(m_navigation_pub),
      hasSubscribers(m_navigation_surfel_pub),
      hasSubscribers(m_cloud_pub)};
    const bool consumer_started =
      (demand.tile && !m_tile_requested_previous) ||
      (demand.surface && !m_surface_requested_previous) ||
      (demand.full_survey && !m_full_survey_requested_previous) ||
      (demand.supported_survey &&
       !m_supported_survey_requested_previous) ||
      (demand.navigation && !m_navigation_requested_previous) ||
      (demand.navigation_surfels &&
       !m_navigation_surfel_requested_previous) ||
      (demand.vdb_cloud && !m_cloud_requested_previous);

    // A durable full tile sample can itself hold more than 100 MB in DDS.
    // Once the inspector disconnects, replace it with an empty schema-bearing
    // sample and return the derived hash-table allocation. A later subscriber
    // is detected above and triggers an exact rebuild from the disk evidence.
    if (!demand.tile)
    {
      if (m_tile_requested_previous)
      {
        replaceLatchedTileWithEmpty(get_clock()->now());
      }
      releaseTileAccumulator();
    }
    m_tile_requested_previous = demand.tile;
    m_surface_requested_previous = demand.surface;
    m_full_survey_requested_previous = demand.full_survey;
    m_supported_survey_requested_previous = demand.supported_survey;
    m_navigation_requested_previous = demand.navigation;
    m_navigation_surfel_requested_previous = demand.navigation_surfels;
    m_cloud_requested_previous = demand.vdb_cloud;

    // Snapshot only compact keyframe metadata and immutable cloud handles.
    // Disk reads, VDB integration, complete accumulator rebuilds, message
    // construction, and publication all happen after releasing the ingestion
    // lock, so source callbacks continue draining their DDS queues.
    bool changed = false;
    bool evidence_rendered = false;
    const char* render_kind = consumer_started ? "consumer" : "none";
    std::uint64_t render_epoch = m_render_epoch;
    std::size_t keyframes_without_evidence = 0;
    std::size_t hits_buffer_size = 0;
    std::size_t clear_buffer_size = 0;
    std::size_t survey_buffer_size = 0;
    std::size_t tile_buffer_size = 0;
    std::size_t reconstruction_buffer_size = 0;
    double survey_lag = 0.0;
    double tile_lag = 0.0;
    double reconstruction_lag = 0.0;
    bool full_render = false;
    bool append_render = false;
    {
      std::lock_guard<std::mutex> lock(m_ingest_mutex);
      const bool due =
        now - m_last_product_render >= m_render_min_period;
      const bool have_keyframes = !m_keyframes.empty();
      if ((m_force_full_render || m_dirty) && due && have_keyframes)
      {
        full_render = true;
        changed = true;
        evidence_rendered = true;
        render_kind = "full";
        m_force_full_render = false;
        m_dirty = false;
        m_have_new = false;
      }
      else if (m_have_new && due)
      {
        append_render = true;
        changed = true;
        evidence_rendered = true;
        render_kind = "append";
        m_have_new = false;
      }
      else if (consumer_started && have_keyframes &&
               !m_force_full_render && !m_dirty && !m_have_new &&
               !m_render_keyframes.empty())
      {
        // Republish the last complete internally consistent generation. If
        // ingestion has newer unrendered work, wait for its normal render
        // instead of mixing new keyframes with the previous occupancy state.
        changed = true;
      }

      if (!changed)
      {
        return;
      }
      if (full_render || append_render)
      {
        m_render_keyframes = m_keyframes;
        render_epoch = m_replay_epoch;
      }
      keyframes_without_evidence = m_keyframes_without_evidence;
      hits_buffer_size = m_hits_buffer.size();
      clear_buffer_size = m_clear_buffer.size();
      survey_buffer_size = m_survey_buffer.size();
      tile_buffer_size = m_tile_buffer.size();
      reconstruction_buffer_size = m_reconstruction_buffer.size();
      survey_lag = std::max(0.0, m_last_survey_stamp - m_last_assoc_stamp);
      tile_lag = std::max(0.0, m_last_tile_stamp - m_last_tile_assoc_stamp);
      reconstruction_lag = std::max(
        0.0, m_last_reconstruction_stamp -
        m_last_reconstruction_assoc_stamp);
    }

    auto stage_started = SteadyClock::now();

    if (full_render)
    {
      // the graph moved: re-render everything at the current poses
      m_map->resetMap();
      m_occupancy_upto = 0;
      m_survey_stale = true;
      m_surface_stale = true;
      m_tile_stale = true;
      for (size_t i = 0; i < m_render_keyframes.size(); ++i)
      {
        if (m_render_keyframes[i].has_evidence)
        {
          integrateKeyframe(m_render_keyframes[i], i);
        }
      }
      m_occupancy_upto = m_render_keyframes.size();
      m_render_epoch = render_epoch;
      ++m_full_renders;
    }
    else if (append_render)
    {
      // append-only: integrate keyframes that haven't been rendered yet
      if (m_occupancy_upto > m_render_keyframes.size())
      {
        // Defensive recovery for an unexpected segment-size regression.
        m_map->resetMap();
        m_occupancy_upto = 0;
        m_survey_stale = true;
        m_surface_stale = true;
        m_tile_stale = true;
      }
      for (size_t i = m_occupancy_upto;
           i < m_render_keyframes.size(); ++i)
      {
        if (m_render_keyframes[i].has_evidence)
        {
          integrateKeyframe(m_render_keyframes[i], i);
        }
      }
      m_occupancy_upto = m_render_keyframes.size();
      m_render_epoch = render_epoch;
    }

    // An input-time rewind can occur while this independent executor lane is
    // rebuilding. Never publish the old segment after the rewind; the new
    // segment's force-full flag remains set for the next timer invocation.
    {
      std::lock_guard<std::mutex> lock(m_ingest_mutex);
      if (render_epoch != m_replay_epoch)
      {
        return;
      }
    }
    const double integration_ms = elapsedWallMs(stage_started);
    if (evidence_rendered)
    {
      m_last_product_render = now;
    }
    if (!publish_outputs)
    {
      return;
    }

    stage_started = SteadyClock::now();
    std::vector<ReconstructionRow> reconstructed;
    const bool reconstruction_needed = demand.surface ||
      m_global_occupancy_mode != GlobalOccupancyMode::Hits;
    if (reconstruction_needed)
    {
      syncSurfaceAccumulator();
      reconstructed = m_surface_accumulator->rows();
    }
    const double surface_build_ms = elapsedWallMs(stage_started);

    stage_started = SteadyClock::now();
    visualization_msgs::msg::Marker marker_msg;
    sensor_msgs::msg::PointCloud2 cloud_msg;
    nav_msgs::msg::OccupancyGrid grid_msg;
    // handle first, then lock (getGrid() locks internally)
    auto grid = m_map->getGrid();
    std::shared_lock map_lock(*m_map->getMapMutex());
    VDBMappingTools<VDBMapT>::createMappingOutput(grid,
                                                  m_map_frame,
                                                  marker_msg,
                                                  cloud_msg,
                                                  grid_msg,
                                                  /*create_marker=*/false,
                                                  /*create_pointcloud=*/demand.vdb_cloud,
                                                  /*create_occupancy_grid=*/true,
                                                  /*lower_z_limit=*/0.0,
                                                  /*upper_z_limit=*/0.0,
                                                  static_cast<float>(m_resolution),
                                                  m_two_dim_projection_threshold);
    map_lock.unlock();
    std::size_t surface_free_space_rejected = 0;
    std::vector<ReconstructionRow> confirmed;
    if (m_global_occupancy_mode != GlobalOccupancyMode::Hits)
    {
      confirmed =
        selectConfirmedSurface(reconstructed, &surface_free_space_rejected);
    }
    // Planning remains conservative: every confirmed, non-contradicted voxel
    // marks occupancy. Local planarity only controls the operator/registration
    // surfel cloud, so a thin obstacle cannot disappear merely because it has
    // too few neighbors for a stable normal.
    applyNavigationSurfaceToOccupancy(grid_msg, confirmed);
    if (!replayEpochCurrent(render_epoch))
    {
      return;
    }
    const auto stamp      = get_clock()->now();
    cloud_msg.header.stamp = stamp;
    grid_msg.header.stamp  = stamp;
    if (demand.vdb_cloud)
    {
      m_cloud_pub->publish(cloud_msg);
    }
    m_grid_pub->publish(grid_msg);
    const double occupancy_output_ms = elapsedWallMs(stage_started);

    // Graph-anchored dense SURVEY render: keyframe-local rich survey clouds at
    // the CURRENT optimized poses, aggregated by a support-counting voxel
    // pass (pcl::VoxelGrid cannot emit counts) and optionally masked to
    // occupied voxels so the clearing evidence scrubs transients out of the
    // survey product too. Output layout (13 float32 fields, 52-byte stride):
    //   x y z intensity range incidence support pose_sigma texture
    //   texture_variance elevation_lo_offset elevation_hi_offset
    //   elevation_resolved_fraction
    // `support` = distinct keyframes occupying the point's independent 5 cm
    // correspondence cell (repeatability, not neighboring bin density);
    // `pose_sigma` is RESERVED (0) until the trajectory topic carries
    // per-keyframe marginals. `incidence` is averaged over the points that
    // know it (>= 0), -1 when none do — no sentinel dilution.
    stage_started = SteadyClock::now();
    std::vector<SurveyRow> supported_survey_rows;
    if (demand.surveyProducts())
    {
      // Fold in only what is new. Re-accumulating every keyframe on every
      // render was affordable while the clouds were resident; with them on
      // disk it would re-read the whole survey history every
      // render_min_period. The accumulator is therefore kept across renders
      // and discarded only when the graph moves -- which is precisely when
      // the old contributions became wrong -- so the published product still
      // always reflects the current optimized poses.
      syncSurveyAccumulator();

      static const char* names[kSurveyOutputFields] = {
        "x",
        "y",
        "z",
        "intensity",
        "range",
        "incidence",
        "support",
        "pose_sigma",
        "texture",
        "texture_variance",
        "elevation_lo_offset",
        "elevation_hi_offset",
        "elevation_resolved_fraction"};
      const auto initialize_survey_message = [&](
        sensor_msgs::msg::PointCloud2& msg)
      {
        msg.header.stamp    = stamp;
        msg.header.frame_id = m_map_frame;
        msg.height          = 1;
        msg.is_bigendian    = false;
        msg.is_dense        = true;
        for (std::size_t f = 0; f < kSurveyOutputFields; ++f)
        {
          sensor_msgs::msg::PointField pf;
          pf.name     = names[f];
          pf.offset   = static_cast<uint32_t>(f * 4);
          pf.datatype = sensor_msgs::msg::PointField::FLOAT32;
          pf.count    = 1;
          msg.fields.push_back(pf);
        }
        msg.point_step =
          static_cast<uint32_t>(kSurveyOutputFields * sizeof(float));
        msg.data.reserve(m_vox.size() * msg.point_step);
      };
      sensor_msgs::msg::PointCloud2 survey_msg;
      sensor_msgs::msg::PointCloud2 supported_survey_msg;
      supported_survey_rows.reserve(m_vox.size());
      if (demand.full_survey)
      {
        initialize_survey_message(survey_msg);
      }
      if (demand.supported_survey)
      {
        initialize_survey_message(supported_survey_msg);
      }

      const auto append_row = [](sensor_msgs::msg::PointCloud2& msg,
                                 const SurveyRow& row)
      {
        const auto* bytes =
          reinterpret_cast<const uint8_t*>(row.data());
        msg.data.insert(
          msg.data.end(), bytes,
          bytes + kSurveyOutputFields * sizeof(float));
      };
      forEachSurveyRow([&](const SurveyRow& row) {
        if (demand.full_survey)
        {
          append_row(survey_msg, row);
        }
        if (demand.supported_survey &&
            row[6] >= static_cast<float>(m_survey_min_support))
        {
          append_row(supported_survey_msg, row);
        }
        if (demand.navigationProducts() &&
            row[6] >= static_cast<float>(m_survey_min_support))
        {
          supported_survey_rows.push_back(row);
        }
      });
      const auto finish_and_publish = [](
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
        sensor_msgs::msg::PointCloud2& msg)
      {
        msg.width = static_cast<uint32_t>(
          msg.data.size() / msg.point_step);
        msg.row_step = msg.point_step * msg.width;
        pub->publish(msg);
      };
      if (demand.full_survey)
      {
        if (!replayEpochCurrent(render_epoch))
        {
          return;
        }
        finish_and_publish(m_survey_pub, survey_msg);
      }
      if (demand.supported_survey)
      {
        if (!replayEpochCurrent(render_epoch))
        {
          return;
        }
        finish_and_publish(m_supported_survey_pub, supported_survey_msg);
      }
    }
    const double survey_build_ms = elapsedWallMs(stage_started);

    stage_started = SteadyClock::now();
    std::size_t navigation_free_space_rejected = 0;
    std::vector<NavigationRow> navigation;
    if (demand.navigationProducts())
    {
      navigation = selectSurveyNavigationSurface(
        supported_survey_rows, &navigation_free_space_rejected);
    }
    const double navigation_select_ms = elapsedWallMs(stage_started);

    if (!replayEpochCurrent(render_epoch))
    {
      return;
    }
    const ProductPublishMetrics publish_metrics =
      publishReconstructionProducts(
        stamp, demand, reconstructed, confirmed.size(),
        surface_free_space_rejected, survey_buffer_size, tile_buffer_size,
        reconstruction_buffer_size);

    stage_started = SteadyClock::now();
    const std::size_t navigation_count = navigation.size();
    // Preserve the exact 1 cm, normal-bearing cloud while removing its
    // all-point fit from the subscription executor. The worker owns at most
    // one active and one newest pending generation; it never publishes a
    // reduced-fidelity intermediate cloud.
    if (demand.navigationProducts())
    {
      queueNavigationFit(
        stamp, demand, navigation_free_space_rejected,
        std::move(navigation));
    }
    const double navigation_dispatch_ms = elapsedWallMs(stage_started);

    const double total_ms = elapsedWallMs(render_started);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 30000,
      "render profile: kind=%s total=%.1f ms integrate=%.1f occupancy=%.1f "
      "surface=%.1f survey=%.1f navigation=%.1f dispatch=%.1f "
      "publish[tile=%.1f surface=%.1f]; "
      "demand[t=%d s=%d fs=%d ss=%d n=%d ns=%d vdb=%d]; "
      "state[kf=%zu survey_vox=%zu surface_cells=%zu nav_queued=%zu "
      "buffers=%zu/%zu/%zu/%zu/%zu lag=%.2f/%.2f/%.2f s]",
      render_kind, total_ms, integration_ms, occupancy_output_ms,
      surface_build_ms, survey_build_ms, navigation_select_ms,
      navigation_dispatch_ms, publish_metrics.tile_ms,
      publish_metrics.surface_ms,
      static_cast<int>(demand.tile), static_cast<int>(demand.surface),
      static_cast<int>(demand.full_survey),
      static_cast<int>(demand.supported_survey),
      static_cast<int>(demand.navigation),
      static_cast<int>(demand.navigation_surfels),
      static_cast<int>(demand.vdb_cloud), m_render_keyframes.size(), m_vox.size(),
      m_surface_accumulator->candidateCellCount(), navigation_count,
      hits_buffer_size, clear_buffer_size, survey_buffer_size,
      tile_buffer_size, reconstruction_buffer_size,
      survey_lag, tile_lag, reconstruction_lag);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 30000,
                         "assembled map: %zu keyframes (%zu without evidence), "
                         "%zu full re-renders",
                         m_render_keyframes.size(), keyframes_without_evidence,
                         m_full_renders);
  }

  double m_resolution = 0.1;
  std::string m_map_frame;
  std::string m_robot_frame;
  double m_buffer_seconds   = 6.0;
  double m_stamp_tolerance  = 0.06;
  int m_input_queue_depth   = 5;
  bool m_input_reliable = false;
  bool m_allow_latest_tf_fallback = true;
  double m_tf_buffer_duration = 10.0;
  bool m_reset_on_time_rewind = true;
  double m_time_rewind_tolerance = 0.5;
  double m_render_min_period = 2.0;
  double m_pose_eps_xy      = 0.05;
  double m_pose_eps_yaw     = 0.02;
  double m_pose_eps_z       = 0.05;
  int m_two_dim_projection_threshold = 3;

  double m_survey_resolution   = 0.05;
  double m_survey_support_resolution = 0.05;
  int m_survey_min_support = 2;
  bool m_survey_occupancy_mask = true;
  std::string m_odom_frame     = "odom";
  double m_last_assoc_stamp    = 0.0;
  double m_tile_resolution = 0.05;
  double m_surface_resolution = 0.10;
  int m_surface_min_observations = 3;
  double m_surface_min_view_span_deg = 6.0;
  int m_surface_max_samples_per_return = 31;
  int m_surface_peak_radius_voxels = 0;
  double m_surface_min_return_intensity = 0.0;
  double m_surface_min_confidence = 0.45;
  double m_navigation_min_confidence = 0.45;
  double m_navigation_min_intensity = 0.10;
  double m_navigation_unresolved_confidence_scale = 0.50;
  double m_surfel_radius_m = 0.10;
  int m_surfel_min_neighbors = 5;
  double m_surfel_max_surface_variation = 0.12;
  double m_surfel_max_projection = 0.01;
  GlobalOccupancyMode m_global_occupancy_mode = GlobalOccupancyMode::Hits;
  double m_last_tile_assoc_stamp = 0.0;
  double m_last_reconstruction_assoc_stamp = 0.0;

  std::unique_ptr<VDBMapT> m_map;
  std::unique_ptr<tf2_ros::Buffer> m_tf_buffer;
  std::shared_ptr<tf2_ros::TransformListener> m_tf_listener;

  std::deque<BufferedCloud> m_hits_buffer;
  std::deque<BufferedCloud> m_clear_buffer;
  std::deque<BufferedSurvey> m_survey_buffer;
  std::deque<BufferedReconstruction> m_tile_buffer;
  std::deque<BufferedReconstruction> m_reconstruction_buffer;
  // Every field above through the authoritative keyframe vector below is
  // owned by the ingestion callback group. The render group only holds this
  // mutex long enough to copy compact keyframe metadata and diagnostics.
  std::mutex m_ingest_mutex;
  std::vector<KeyframeEvidence> m_keyframes;
  size_t m_keyframes_without_evidence = 0;
  bool m_dirty    = false;
  bool m_have_new = false;
  bool m_force_full_render = false;
  std::uint64_t m_replay_epoch = 0;

  // Render-group-owned state. Immutable evidence handles in this snapshot
  // keep a complete generation alive while ingestion appends or corrects the
  // authoritative vector concurrently.
  std::vector<KeyframeEvidence> m_render_keyframes;
  size_t m_occupancy_upto = 0;
  size_t m_full_renders = 0;
  std::uint64_t m_render_epoch = 0;
  double m_last_product_render = 0.0;

  // Evidence spill / export. Empty spill dir = disabled (clouds stay
  // resident); m_spill_failures counts keyframes that had to stay in RAM
  // because their write failed.
  std::string m_spill_dir;
  size_t m_spill_failures = 0;
  size_t m_replay_segment = 0;
  std::string m_export_path;
  bool m_export_on_shutdown = true;
  std::string m_navigation_export_path;
  bool m_navigation_export_on_shutdown = true;
  std::string m_navigation_surfel_export_path;
  bool m_navigation_surfel_export_on_shutdown = true;

  // Survey accumulator, persistent across renders. Unlike the keyframe
  // evidence this is bounded by the surveyed VOLUME rather than by elapsed
  // time -- revisiting ground merges into existing voxels instead of adding
  // new ones -- so it saturates and is safe to keep resident. m_survey_upto
  // is how many keyframes are already folded in; m_survey_stale forces a
  // rebuild after the graph moves.
  std::unordered_map<uint64_t, VoxAcc> m_vox;
  std::unordered_map<uint64_t, int> m_survey_support;
  std::unordered_map<uint64_t, TileVoxAcc> m_tile_vox;
  std::unique_ptr<MultiViewSurfaceAccumulator> m_surface_accumulator;
  size_t m_survey_upto = 0;
  bool m_survey_stale  = false;
  size_t m_surface_upto = 0;
  bool m_surface_stale = false;
  size_t m_tile_upto = 0;
  bool m_tile_stale = true;
  bool m_tile_requested_previous = false;
  bool m_surface_requested_previous = false;
  bool m_full_survey_requested_previous = false;
  bool m_supported_survey_requested_previous = false;
  bool m_navigation_requested_previous = false;
  bool m_navigation_surfel_requested_previous = false;
  bool m_cloud_requested_previous = false;
  std::mutex m_navigation_fit_mutex;
  std::condition_variable m_navigation_fit_cv;
  std::optional<NavigationFitJob> m_pending_navigation_fit;
  std::thread m_navigation_fit_thread;
  bool m_navigation_fit_stop = false;
  std::uint64_t m_navigation_fit_generation = 0;
  std::uint64_t m_navigation_fit_epoch = 0;
  std::uint64_t m_navigation_fits_coalesced = 0;
  double m_last_hits_stamp = 0.0;
  double m_last_clear_stamp = 0.0;
  double m_last_survey_stamp = 0.0;
  double m_last_tile_stamp = 0.0;
  double m_last_reconstruction_stamp = 0.0;
  double m_last_traj_stamp = 0.0;
  std::uint32_t m_observation_sequence = 0;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_hits_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_clear_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_traj_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_survey_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_tile_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
    m_reconstruction_sub;
  rclcpp::CallbackGroup::SharedPtr m_ingest_callback_group;
  rclcpp::CallbackGroup::SharedPtr m_render_callback_group;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_cloud_pub;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr m_grid_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_survey_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
    m_supported_survey_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_tile_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_surface_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_navigation_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
    m_navigation_surfel_pub;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr m_export_srv;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr m_navigation_export_srv;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
    m_navigation_surfel_export_srv;
  rclcpp::TimerBase::SharedPtr m_render_timer;

public:
  // Called from main() after spin() returns, while the node and its logger are
  // still alive -- doing this from the destructor would run after
  // rclcpp::shutdown() has torn the context down.
  void exportOnShutdown()
  {
    // No publisher may outlive the ROS context, and the exact export path
    // below performs its own synchronous fit from the latest authoritative
    // state. Finish the active worker and discard any superseded pending fit
    // before writing final products.
    stopNavigationFitWorker();
    if (m_export_on_shutdown && !m_export_path.empty())
    {
      std::string message;
      if (!exportSurvey(message))
      {
        RCLCPP_WARN(
          get_logger(), "shutdown survey export skipped: %s", message.c_str());
      }
    }
    if (m_navigation_export_on_shutdown && !m_navigation_export_path.empty())
    {
      std::string message;
      if (!exportNavigationSurface(message))
      {
        RCLCPP_WARN(
          get_logger(), "shutdown navigation export skipped: %s",
          message.c_str());
      }
    }
    if (m_navigation_surfel_export_on_shutdown &&
        !m_navigation_surfel_export_path.empty())
    {
      std::string message;
      if (!exportNavigationSurfels(message))
      {
        RCLCPP_WARN(
          get_logger(), "shutdown navigation surfel export skipped: %s",
          message.c_str());
      }
    }
  }
};

}  // namespace vdb_mapping_ros2

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vdb_mapping_ros2::VDBMapAssembler>();
  // One lane drains serialized sensor/trajectory callbacks while another
  // performs complete-map renders. A third keeps parameter/control services
  // responsive; exact normal fitting has its own bounded worker above.
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 3);
  executor.add_node(node);
  executor.spin();
  node->exportOnShutdown();
  rclcpp::shutdown();
  return 0;
}
