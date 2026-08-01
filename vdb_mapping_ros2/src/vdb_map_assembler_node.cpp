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
#include <cmath>
#include <ctime>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
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
// SurveyPoint/SurveyExportPoint, so this TU must supply its own.
#include <pcl/io/pcd_io.h>
#include <pcl/io/impl/pcd_io.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
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

namespace vdb_mapping_ros2 {

constexpr std::size_t kSurveyOutputFields = 13;
using SurveyRow = std::array<float, kSurveyOutputFields>;

class VDBMapAssembler : public rclcpp::Node
{
public:
  using VDBMapT   = vdb_mapping::OccupancyVDBMapping;
  using PointT    = pcl::PointXYZ;
  using CloudT    = pcl::PointCloud<PointT>;
  using CloudPtrT = CloudT::Ptr;
  using SurveyCloudT    = pcl::PointCloud<SurveyPoint>;
  using SurveyCloudPtrT = SurveyCloudT::Ptr;

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
    // publish only survey points whose occupancy voxel is occupied — the
    // clearing evidence then scrubs transients out of the survey product too
    declare_parameter<bool>("survey_occupancy_mask", true);
    declare_parameter<std::string>("odom_frame", "odom");
    // evidence association
    declare_parameter<double>("buffer_seconds", 6.0);
    declare_parameter<double>("stamp_tolerance", 0.06);
    declare_parameter<int>("input_queue_depth", 5);
    declare_parameter<bool>("input_reliable", false);
    declare_parameter<bool>("allow_latest_tf_fallback", true);
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

    get_parameter("resolution", m_resolution);
    get_parameter("map_frame", m_map_frame);
    get_parameter("robot_frame", m_robot_frame);
    get_parameter("buffer_seconds", m_buffer_seconds);
    get_parameter("stamp_tolerance", m_stamp_tolerance);
    get_parameter("input_queue_depth", m_input_queue_depth);
    get_parameter("input_reliable", m_input_reliable);
    get_parameter("allow_latest_tf_fallback", m_allow_latest_tf_fallback);
    get_parameter("reset_on_time_rewind", m_reset_on_time_rewind);
    get_parameter("time_rewind_tolerance", m_time_rewind_tolerance);
    m_input_queue_depth = std::max(1, m_input_queue_depth);
    m_time_rewind_tolerance = std::max(0.0, m_time_rewind_tolerance);
    get_parameter("render_min_period", m_render_min_period);
    get_parameter("pose_epsilon_xy", m_pose_eps_xy);
    get_parameter("pose_epsilon_yaw", m_pose_eps_yaw);
    get_parameter("pose_epsilon_z", m_pose_eps_z);
    get_parameter("two_dim_projection_threshold", m_two_dim_projection_threshold);
    get_parameter("survey_resolution", m_survey_resolution);
    get_parameter("survey_occupancy_mask", m_survey_occupancy_mask);
    get_parameter("odom_frame", m_odom_frame);
    get_parameter("export_path", m_export_path);
    get_parameter("export_on_shutdown", m_export_on_shutdown);
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
    m_map->setConfig(cfg);
    const auto ros_clock = get_clock();
    m_map->setTimeCallback([ros_clock]() -> uint64_t {
      return static_cast<uint64_t>(
        std::max<int64_t>(0, ros_clock->now().nanoseconds()));
    });
    // fill evidence: endpoints paint, rays never carve
    m_map->addInputSource("hits", 0.0, 0.0, /*ray_clearing=*/false, /*endpoint_hits=*/true);
    // clearing evidence: rays carve their full length, endpoints never paint
    m_map->addInputSource("clear", 0.0, 0.0, /*ray_clearing=*/true, /*endpoint_hits=*/false);

    m_tf_buffer   = std::make_unique<tf2_ros::Buffer>(get_clock());
    m_tf_listener = std::make_shared<tf2_ros::TransformListener>(*m_tf_buffer);

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
        bufferCloud(*msg, m_hits_buffer, true);
      });
    m_clear_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
      clear_topic, qos, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        bufferCloud(*msg, m_clear_buffer, false);
    });
    // trajectory is latched by the SLAM node
    m_traj_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
      traj_topic,
      rclcpp::QoS(rclcpp::KeepLast(static_cast<std::size_t>(m_input_queue_depth)))
        .reliable().transient_local(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { onTrajectory(*msg); });

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
          bufferSurvey(*msg);
        });
      m_survey_pub = create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/survey_pointcloud", map_qos);
    }

    m_export_srv = create_service<std_srvs::srv::Trigger>(
      "~/export_survey",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        res->success = exportSurvey(res->message);
      });

    m_render_timer = create_timer(std::chrono::milliseconds(500),
                                  [this] { renderIfNeeded(); });

    RCLCPP_INFO(get_logger(),
                "Map assembler up: hits=%s clear=%s traj=%s res=%.2f",
                hits_topic.c_str(), clear_topic.c_str(), traj_topic.c_str(), m_resolution);
  }

private:
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
    Eigen::Vector3d hits_origin  = Eigen::Vector3d::Zero();
    Eigen::Vector3d clear_origin = Eigen::Vector3d::Zero();
    bool integrated = false;
    bool has_evidence = false;
    bool spilled = false;
    size_t n_hits = 0;
    size_t n_clear = 0;
    size_t n_survey = 0;
  };

  // Evidence resolved for one keyframe: either the resident handles or clouds
  // just read back from disk. Held only for the duration of one keyframe's
  // integration, which is what bounds render-time RAM to a single keyframe.
  struct LoadedEvidence
  {
    CloudPtrT hits;
    CloudPtrT clear;
    SurveyCloudPtrT survey;
  };

  struct BufferedSurvey
  {
    double stamp;
    SurveyCloudPtrT cloud;  // in robot frame at `stamp`
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

  std::string spillPath(const size_t idx, const char* kind) const
  {
    std::ostringstream p;
    p << m_spill_dir << "/kf_" << std::setw(6) << std::setfill('0') << idx
      << '_' << kind << ".pcd";
    return p.str();
  }

  template <typename CloudPtr>
  bool writeSpill(const CloudPtr& cloud, const std::string& path)
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
      if (writer.writeBinary(path, *cloud) == 0)
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
    kf.hits.reset();
    kf.clear.reset();
    kf.survey.reset();
    kf.spilled = true;
  }

  // Resolve one keyframe's evidence, reading it back only if it was spilled
  // and only the parts the caller will actually use.
  LoadedEvidence loadEvidence(const KeyframeEvidence& kf, const size_t idx,
                              const bool want_occupancy, const bool want_survey)
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
      return e;
    }
    if (want_occupancy)
    {
      if (kf.n_hits > 0)
      {
        e.hits = readSpill<CloudT>(spillPath(idx, "hits"));
      }
      if (kf.n_clear > 0)
      {
        e.clear = readSpill<CloudT>(spillPath(idx, "clear"));
      }
    }
    if (want_survey && kf.n_survey > 0)
    {
      e.survey = readSpill<SurveyCloudT>(spillPath(idx, "survey"));
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
  // frame is DYNAMIC — pivot head — so no caching)
  bool lookupAtStamp(const std::string& frame,
                     const builtin_interfaces::msg::Time& stamp,
                     Eigen::Isometry3d& out,
                     const std::string& target = "")
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
      if (!m_allow_latest_tf_fallback)
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
    m_hits_buffer.clear();
    m_clear_buffer.clear();
    m_survey_buffer.clear();
    m_keyframes.clear();
    m_vox.clear();
    m_map->resetMap();

    m_keyframes_without_evidence = 0;
    m_full_renders = 0;
    m_dirty = false;
    m_have_new = false;
    m_last_full_render = 0.0;
    m_last_assoc_stamp = 0.0;
    m_survey_upto = 0;
    m_survey_stale = false;
    m_last_hits_stamp = 0.0;
    m_last_clear_stamp = 0.0;
    m_last_survey_stamp = 0.0;
    m_last_traj_stamp = 0.0;
    m_spill_failures = 0;

    // Keep old segment files recoverable and write replayed evidence into a
    // fresh namespace so repeated keyframe indices never overwrite or splice.
    ++m_replay_segment;
    m_spill_dir.clear();
    setUpSpill();
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
          // This keyframe's points were folded into the survey accumulator at
          // the OLD pose and there is no way to subtract one keyframe's
          // contribution back out of a voxel mean, so the accumulator has to
          // be rebuilt from scratch on the next render.
          m_survey_stale = true;
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
        if (lookupAtStamp(m_robot_frame, kf_time, t_odom_kf, m_odom_frame))
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
            if (!lookupAtStamp(m_robot_frame, e_time, t_odom_e, m_odom_frame))
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

  void integrateKeyframe(KeyframeEvidence& kf, const size_t idx)
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
    kf.integrated = true;
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
    for (const auto& p : in_map.points)
    {
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
  }

  // Walk the accumulated survey voxels, applying the occupancy mask, and hand
  // each reduced 8-field row to `fn`. Shared by the topic and the PCD export
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
        static_cast<float>(a.n),
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
    if (m_dirty || m_have_new)
    {
      m_last_full_render = -std::numeric_limits<double>::infinity();
      renderIfNeeded();
    }
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
      m_survey_upto  = 0;
      m_survey_stale = false;
    }
    for (size_t i = m_survey_upto; i < m_keyframes.size(); ++i)
    {
      accumulateSurvey(m_keyframes[i], i);
    }
    m_survey_upto = m_keyframes.size();
  }

  void renderIfNeeded()
  {
    const double now = get_clock()->now().seconds();
    bool changed     = false;

    if (m_dirty && now - m_last_full_render >= m_render_min_period)
    {
      // the graph moved: re-render everything at the current poses
      m_map->resetMap();
      for (auto& kf : m_keyframes)
      {
        kf.integrated = false;
      }
      for (size_t i = 0; i < m_keyframes.size(); ++i)
      {
        if (m_keyframes[i].has_evidence)
        {
          integrateKeyframe(m_keyframes[i], i);
        }
      }
      m_dirty            = false;
      m_have_new         = false;
      m_last_full_render = now;
      ++m_full_renders;
      changed = true;
    }
    else if (m_have_new)
    {
      // append-only: integrate keyframes that haven't been rendered yet
      for (size_t i = 0; i < m_keyframes.size(); ++i)
      {
        if (m_keyframes[i].has_evidence && !m_keyframes[i].integrated)
        {
          integrateKeyframe(m_keyframes[i], i);
          changed = true;
        }
      }
      m_have_new = false;
    }

    if (!changed)
    {
      return;
    }

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
                                                  /*create_pointcloud=*/true,
                                                  /*create_occupancy_grid=*/true,
                                                  /*lower_z_limit=*/0.0,
                                                  /*upper_z_limit=*/0.0,
                                                  static_cast<float>(m_resolution),
                                                  m_two_dim_projection_threshold);
    map_lock.unlock();
    const auto stamp      = get_clock()->now();
    cloud_msg.header.stamp = stamp;
    grid_msg.header.stamp  = stamp;
    m_cloud_pub->publish(cloud_msg);
    m_grid_pub->publish(grid_msg);

    // Graph-anchored dense SURVEY render: keyframe-local rich survey clouds at
    // the CURRENT optimized poses, aggregated by a support-counting voxel
    // pass (pcl::VoxelGrid cannot emit counts) and optionally masked to
    // occupied voxels so the clearing evidence scrubs transients out of the
    // survey product too. Output layout (13 float32 fields, 52-byte stride):
    //   x y z intensity range incidence support pose_sigma texture
    //   texture_variance elevation_lo_offset elevation_hi_offset
    //   elevation_resolved_fraction
    // `support` = points merged into the voxel (observation density —
    // per-voxel confidence and the coverage measure in one field);
    // `pose_sigma` is RESERVED (0) until the trajectory topic carries
    // per-keyframe marginals. `incidence` is averaged over the points that
    // know it (>= 0), -1 when none do — no sentinel dilution.
    if (m_survey_pub)
    {
      // Fold in only what is new. Re-accumulating every keyframe on every
      // render was affordable while the clouds were resident; with them on
      // disk it would re-read the whole survey history every
      // render_min_period. The accumulator is therefore kept across renders
      // and discarded only when the graph moves -- which is precisely when
      // the old contributions became wrong -- so the published product still
      // always reflects the current optimized poses.
      syncSurveyAccumulator();

      sensor_msgs::msg::PointCloud2 survey_msg;
      survey_msg.header.stamp    = stamp;
      survey_msg.header.frame_id = m_map_frame;
      survey_msg.height          = 1;
      survey_msg.is_bigendian    = false;
      survey_msg.is_dense        = true;
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
      for (std::size_t f = 0; f < kSurveyOutputFields; ++f)
      {
        sensor_msgs::msg::PointField pf;
        pf.name     = names[f];
        pf.offset   = static_cast<uint32_t>(f * 4);
        pf.datatype = sensor_msgs::msg::PointField::FLOAT32;
        pf.count    = 1;
        survey_msg.fields.push_back(pf);
      }
      survey_msg.point_step =
        static_cast<uint32_t>(kSurveyOutputFields * sizeof(float));
      survey_msg.data.reserve(m_vox.size() * survey_msg.point_step);

      forEachSurveyRow([&survey_msg](const SurveyRow& row) {
        const auto* bytes =
          reinterpret_cast<const uint8_t*>(row.data());
        survey_msg.data.insert(
          survey_msg.data.end(), bytes,
          bytes + kSurveyOutputFields * sizeof(float));
      });
      survey_msg.width    = static_cast<uint32_t>(survey_msg.data.size() /
                                                  survey_msg.point_step);
      survey_msg.row_step = survey_msg.point_step * survey_msg.width;
      m_survey_pub->publish(survey_msg);
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 30000,
                         "assembled map: %zu keyframes (%zu without evidence), "
                         "%zu full re-renders",
                         m_keyframes.size(), m_keyframes_without_evidence,
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
  bool m_reset_on_time_rewind = true;
  double m_time_rewind_tolerance = 0.5;
  double m_render_min_period = 2.0;
  double m_pose_eps_xy      = 0.05;
  double m_pose_eps_yaw     = 0.02;
  double m_pose_eps_z       = 0.05;
  int m_two_dim_projection_threshold = 3;

  double m_survey_resolution   = 0.05;
  bool m_survey_occupancy_mask = true;
  std::string m_odom_frame     = "odom";
  double m_last_assoc_stamp    = 0.0;

  std::unique_ptr<VDBMapT> m_map;
  std::unique_ptr<tf2_ros::Buffer> m_tf_buffer;
  std::shared_ptr<tf2_ros::TransformListener> m_tf_listener;

  std::deque<BufferedCloud> m_hits_buffer;
  std::deque<BufferedCloud> m_clear_buffer;
  std::deque<BufferedSurvey> m_survey_buffer;
  std::vector<KeyframeEvidence> m_keyframes;
  size_t m_keyframes_without_evidence = 0;
  size_t m_full_renders               = 0;
  bool m_dirty    = false;
  bool m_have_new = false;
  double m_last_full_render = 0.0;

  // Evidence spill / export. Empty spill dir = disabled (clouds stay
  // resident); m_spill_failures counts keyframes that had to stay in RAM
  // because their write failed.
  std::string m_spill_dir;
  size_t m_spill_failures = 0;
  size_t m_replay_segment = 0;
  std::string m_export_path;
  bool m_export_on_shutdown = true;

  // Survey accumulator, persistent across renders. Unlike the keyframe
  // evidence this is bounded by the surveyed VOLUME rather than by elapsed
  // time -- revisiting ground merges into existing voxels instead of adding
  // new ones -- so it saturates and is safe to keep resident. m_survey_upto
  // is how many keyframes are already folded in; m_survey_stale forces a
  // rebuild after the graph moves.
  std::unordered_map<uint64_t, VoxAcc> m_vox;
  size_t m_survey_upto = 0;
  bool m_survey_stale  = false;
  double m_last_hits_stamp = 0.0;
  double m_last_clear_stamp = 0.0;
  double m_last_survey_stamp = 0.0;
  double m_last_traj_stamp = 0.0;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_hits_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_clear_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_traj_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_survey_sub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_cloud_pub;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr m_grid_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_survey_pub;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr m_export_srv;
  rclcpp::TimerBase::SharedPtr m_render_timer;

public:
  // Called from main() after spin() returns, while the node and its logger are
  // still alive -- doing this from the destructor would run after
  // rclcpp::shutdown() has torn the context down.
  void exportOnShutdown()
  {
    if (!m_export_on_shutdown || m_export_path.empty())
    {
      return;
    }
    std::string message;
    if (!exportSurvey(message))
    {
      RCLCPP_WARN(get_logger(), "shutdown export skipped: %s", message.c_str());
    }
  }
};

}  // namespace vdb_mapping_ros2

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vdb_mapping_ros2::VDBMapAssembler>();
  rclcpp::spin(node);
  node->exportOnShutdown();
  rclcpp::shutdown();
  return 0;
}
