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

#include <cmath>
#include <deque>
#include <memory>
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
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

#include <vdb_mapping/OccupancyVDBMapping.hpp>
#include <vdb_mapping_ros2/VDBMappingTools.hpp>

// Survey map stream point: wire layout matches
// the map_points union exactly (x@0 y@4 z@8 intensity@16 range@20
// incidence@24, 32-byte stride).
struct SurveyPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float range;
  float incidence;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
  SurveyPoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    float, range, range)(float, incidence, incidence))

namespace vdb_mapping_ros2 {

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
    // Dense graph-anchored SURVEY product: subscribe the 6-field map_points
    // union, aggregate ALL inter-keyframe clouds into each keyframe's frame
    // (odom-delta transforms), and re-render the intensity cloud at the
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
    // rendering policy
    declare_parameter<double>("render_min_period", 2.0);
    declare_parameter<double>("pose_epsilon_xy", 0.05);
    declare_parameter<double>("pose_epsilon_yaw", 0.02);
    declare_parameter<int>("two_dim_projection_threshold", 3);

    get_parameter("resolution", m_resolution);
    get_parameter("map_frame", m_map_frame);
    get_parameter("robot_frame", m_robot_frame);
    get_parameter("buffer_seconds", m_buffer_seconds);
    get_parameter("stamp_tolerance", m_stamp_tolerance);
    get_parameter("render_min_period", m_render_min_period);
    get_parameter("pose_epsilon_xy", m_pose_eps_xy);
    get_parameter("pose_epsilon_yaw", m_pose_eps_yaw);
    get_parameter("two_dim_projection_threshold", m_two_dim_projection_threshold);
    get_parameter("survey_resolution", m_survey_resolution);
    get_parameter("survey_occupancy_mask", m_survey_occupancy_mask);
    get_parameter("odom_frame", m_odom_frame);

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

    auto qos = rclcpp::QoS(5).best_effort();
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
      traj_topic, rclcpp::QoS(1).reliable().transient_local(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { onTrajectory(*msg); });

    m_cloud_pub = create_publisher<sensor_msgs::msg::PointCloud2>("~/vdb_map_pointcloud", 1);
    m_grid_pub =
      create_publisher<nav_msgs::msg::OccupancyGrid>("~/vdb_map_occupancy", 1);

    std::string survey_topic;
    get_parameter("survey_topic", survey_topic);
    if (!survey_topic.empty())
    {
      m_survey_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
        survey_topic, qos, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          bufferSurvey(*msg);
        });
      m_survey_pub =
        create_publisher<sensor_msgs::msg::PointCloud2>("~/survey_pointcloud", 1);
    }

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
    CloudPtrT hits;                  // robot frame
    CloudPtrT clear;
    // dense 6-field survey aggregate (ALL inter-keyframe map_points clouds,
    // odom-delta transformed into THIS keyframe's robot frame)
    SurveyCloudPtrT survey;
    Eigen::Vector3d hits_origin  = Eigen::Vector3d::Zero();
    Eigen::Vector3d clear_origin = Eigen::Vector3d::Zero();
    bool integrated = false;
    bool has_evidence = false;
  };

  struct BufferedSurvey
  {
    double stamp;
    SurveyCloudPtrT cloud;  // in robot frame at `stamp`
  };

  // transform an incoming cloud into the robot frame AT THE CLOUD'S STAMP —
  // the sonar frame rides the live pivot_head (cameraHead.tilt, +/-54 deg),
  // so the old static-TF cache froze the head angle into every snapshot
  void bufferCloud(const sensor_msgs::msg::PointCloud2& msg,
                   std::deque<BufferedCloud>& buffer,
                   const bool downsample)
  {
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
    entry.stamp         = rclcpp::Time(msg.header.stamp).seconds();
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

  // buffer a survey (map_points union) cloud in the robot frame at its stamp
  void bufferSurvey(const sensor_msgs::msg::PointCloud2& msg)
  {
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
    entry.stamp = rclcpp::Time(msg.header.stamp).seconds();
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
        if (dxyz.head<2>().norm() > m_pose_eps_xy ||
            std::fabs(rot.angle()) > m_pose_eps_yaw)
        {
          kf.pose = pose;
          m_dirty = true;
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
        }
        m_last_assoc_stamp = kf.stamp;
      }

      m_keyframes.push_back(std::move(kf));
      m_have_new = true;
    }
  }

  static SurveyCloudPtrT voxelSurvey(const SurveyCloudPtrT& in, const float leaf)
  {
    if (!in || in->empty() || leaf <= 0.0f)
    {
      return in;
    }
    pcl::VoxelGrid<SurveyPoint> vg;
    vg.setLeafSize(leaf, leaf, leaf);
    vg.setInputCloud(in);
    SurveyCloudPtrT out(new SurveyCloudT);
    vg.filter(*out);
    return out;
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

  void integrateKeyframe(KeyframeEvidence& kf)
  {
    // Use each cloud's own sensor origin: raycastPointCloud does max-range
    // clipping relative to the origin, so hit endpoints must be judged against
    // the hits cloud's origin, not the clear cloud's (which used to overwrite it).
    if (kf.hits && !kf.hits->empty())
    {
      const Eigen::Vector3d origin = kf.pose * kf.hits_origin;
      CloudPtrT in_map(new CloudT);
      pcl::transformPointCloud(*kf.hits, *in_map, kf.pose.cast<float>());
      m_map->insertPointCloud(in_map, origin, "hits");
    }
    if (kf.clear && !kf.clear->empty())
    {
      const Eigen::Vector3d origin = kf.pose * kf.clear_origin;
      CloudPtrT in_map(new CloudT);
      pcl::transformPointCloud(*kf.clear, *in_map, kf.pose.cast<float>());
      m_map->insertPointCloud(in_map, origin, "clear");
    }
    kf.integrated = true;
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
      for (auto& kf : m_keyframes)
      {
        if (kf.has_evidence)
        {
          integrateKeyframe(kf);
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
      for (auto& kf : m_keyframes)
      {
        if (kf.has_evidence && !kf.integrated)
        {
          integrateKeyframe(kf);
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

    // Graph-anchored dense SURVEY render: keyframe-local 6-field clouds at
    // the CURRENT optimized poses, aggregated by a support-counting voxel
    // pass (pcl::VoxelGrid cannot emit counts) and optionally masked to
    // occupied voxels so the clearing evidence scrubs transients out of the
    // survey product too. Output layout (8 float32 fields, 32-byte stride):
    //   x y z intensity range incidence support pose_sigma
    // `support` = points merged into the voxel (observation density —
    // per-voxel confidence and the coverage measure in one field);
    // `pose_sigma` is RESERVED (0) until the trajectory topic carries
    // per-keyframe marginals. `incidence` is averaged over the points that
    // know it (>= 0), -1 when none do — no sentinel dilution.
    if (m_survey_pub)
    {
      struct VoxAcc
      {
        double x = 0, y = 0, z = 0, i = 0, r = 0, ci = 0;
        int n = 0, nci = 0;
      };
      std::unordered_map<uint64_t, VoxAcc> vox;
      for (const auto& kf : m_keyframes)
      {
        if (!kf.survey || kf.survey->empty())
        {
          continue;
        }
        SurveyCloudT in_map;
        pcl::transformPointCloud(*kf.survey, in_map, kf.pose.cast<float>());
        for (const auto& p : in_map.points)
        {
          VoxAcc& a = vox[voxelKey(p.x, p.y, p.z, m_survey_resolution)];
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
          ++a.n;
        }
      }

      sensor_msgs::msg::PointCloud2 survey_msg;
      survey_msg.header.stamp    = stamp;
      survey_msg.header.frame_id = m_map_frame;
      survey_msg.height          = 1;
      survey_msg.is_bigendian    = false;
      survey_msg.is_dense        = true;
      static const char* names[8] = {"x",     "y",         "z",
                                     "intensity", "range", "incidence",
                                     "support",   "pose_sigma"};
      for (int f = 0; f < 8; ++f)
      {
        sensor_msgs::msg::PointField pf;
        pf.name     = names[f];
        pf.offset   = static_cast<uint32_t>(f * 4);
        pf.datatype = sensor_msgs::msg::PointField::FLOAT32;
        pf.count    = 1;
        survey_msg.fields.push_back(pf);
      }
      survey_msg.point_step = 32;
      survey_msg.data.reserve(vox.size() * survey_msg.point_step);

      {
        std::shared_lock mask_lock(*m_map->getMapMutex());
        auto acc = grid->getConstAccessor();
        for (const auto& [key, a] : vox)
        {
          const float cx = static_cast<float>(a.x / a.n);
          const float cy = static_cast<float>(a.y / a.n);
          const float cz = static_cast<float>(a.z / a.n);
          if (m_survey_occupancy_mask)
          {
            const openvdb::Coord c = openvdb::Coord::round(
              grid->worldToIndex(openvdb::Vec3d(cx, cy, cz)));
            if (!(acc.isValueOn(c) && acc.getValue(c) > 0.0f))
            {
              continue;
            }
          }
          const float row[8] = {
            cx,
            cy,
            cz,
            static_cast<float>(a.i / a.n),
            static_cast<float>(a.r / a.n),
            a.nci > 0 ? static_cast<float>(a.ci / a.nci) : -1.0f,
            static_cast<float>(a.n),
            0.0f};
          const auto* bytes = reinterpret_cast<const uint8_t*>(row);
          survey_msg.data.insert(survey_msg.data.end(), bytes, bytes + 32);
        }
      }
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
  double m_render_min_period = 2.0;
  double m_pose_eps_xy      = 0.05;
  double m_pose_eps_yaw     = 0.02;
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

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_hits_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_clear_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_traj_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_survey_sub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_cloud_pub;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr m_grid_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_survey_pub;
  rclcpp::TimerBase::SharedPtr m_render_timer;
};

}  // namespace vdb_mapping_ros2

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vdb_mapping_ros2::VDBMapAssembler>());
  rclcpp::shutdown();
  return 0;
}
