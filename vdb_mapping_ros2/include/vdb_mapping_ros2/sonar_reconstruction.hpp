#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

// Minimal named-field view of sonar_proc's SonarPoint wire product. PCL reads
// these by field name and ignores the remaining diagnostic fields.
struct SonarTilePoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float range;
  float azimuth;
  float vertical_uncertainty;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// A tile return after the exact ping-time sensor pose has been attached. Both
// the centre point and sensor origin are points; elevation_axis and boresight
// are vectors. Keeping all four in keyframe-local coordinates is what lets a
// later graph correction move the measurement without losing the pivot-head
// geometry needed to reconstruct its vertical-aperture ribbon.
struct SonarReconstructionPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float range;
  float azimuth;
  float elevation_half_angle;
  float origin_x;
  float origin_y;
  float origin_z;
  float elevation_axis_x;
  float elevation_axis_y;
  float elevation_axis_z;
  float boresight_x;
  float boresight_y;
  float boresight_z;
  std::uint32_t observation;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
  SonarTilePoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    float, range, range)(float, azimuth, azimuth)(
    float, vertical_uncertainty, vertical_uncertainty))

POINT_CLOUD_REGISTER_POINT_STRUCT(
  SonarReconstructionPoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    float, range, range)(float, azimuth, azimuth)(
    float, elevation_half_angle, elevation_half_angle)(
    float, origin_x, origin_x)(float, origin_y, origin_y)(float, origin_z, origin_z)(
    float, elevation_axis_x, elevation_axis_x)(
    float, elevation_axis_y, elevation_axis_y)(
    float, elevation_axis_z, elevation_axis_z)(
    float, boresight_x, boresight_x)(float, boresight_y, boresight_y)(
    float, boresight_z, boresight_z)(std::uint32_t, observation, observation))

namespace vdb_mapping_ros2
{

using ReconstructionCloud = pcl::PointCloud<SonarReconstructionPoint>;
using ReconstructionCloudPtr = ReconstructionCloud::Ptr;

inline Eigen::Vector3f pointPosition(const SonarReconstructionPoint& p)
{
  return {p.x, p.y, p.z};
}

inline Eigen::Vector3f pointOrigin(const SonarReconstructionPoint& p)
{
  return {p.origin_x, p.origin_y, p.origin_z};
}

inline Eigen::Vector3f pointElevationAxis(const SonarReconstructionPoint& p)
{
  return {p.elevation_axis_x, p.elevation_axis_y, p.elevation_axis_z};
}

inline Eigen::Vector3f pointBoresight(const SonarReconstructionPoint& p)
{
  return {p.boresight_x, p.boresight_y, p.boresight_z};
}

inline void setPointPosition(SonarReconstructionPoint& p, const Eigen::Vector3f& v)
{
  p.x = v.x();
  p.y = v.y();
  p.z = v.z();
}

inline void setPointOrigin(SonarReconstructionPoint& p, const Eigen::Vector3f& v)
{
  p.origin_x = v.x();
  p.origin_y = v.y();
  p.origin_z = v.z();
}

inline void setPointElevationAxis(
  SonarReconstructionPoint& p, const Eigen::Vector3f& v)
{
  p.elevation_axis_x = v.x();
  p.elevation_axis_y = v.y();
  p.elevation_axis_z = v.z();
}

inline void setPointBoresight(SonarReconstructionPoint& p, const Eigen::Vector3f& v)
{
  p.boresight_x = v.x();
  p.boresight_y = v.y();
  p.boresight_z = v.z();
}

inline SonarReconstructionPoint transformReconstructionPoint(
  const SonarReconstructionPoint& in, const Eigen::Isometry3f& transform)
{
  SonarReconstructionPoint out = in;
  setPointPosition(out, transform * pointPosition(in));
  setPointOrigin(out, transform * pointOrigin(in));
  setPointElevationAxis(out, transform.linear() * pointElevationAxis(in));
  setPointBoresight(out, transform.linear() * pointBoresight(in));
  return out;
}

inline ReconstructionCloudPtr transformReconstructionCloud(
  const ReconstructionCloudPtr& in, const Eigen::Isometry3f& transform)
{
  ReconstructionCloudPtr out(new ReconstructionCloud);
  if (!in)
  {
    return out;
  }
  out->reserve(in->size());
  for (const auto& p : in->points)
  {
    out->push_back(transformReconstructionPoint(p, transform));
  }
  out->width = static_cast<std::uint32_t>(out->size());
  out->height = 1;
  out->is_dense = false;
  return out;
}

// Number of samples required to rasterize the spherical elevation arc with no
// gap larger than resolution. Keep it odd so the measured centre plane is
// always represented, and retain both aperture edges. A finite cap makes the
// cost explicit at long range without changing the angular support.
inline int elevationRibbonSampleCount(
  const float range, const float half_angle, const float resolution,
  const int max_samples)
{
  if (!(range > 0.0F) || !(half_angle > 0.0F) || !(resolution > 0.0F))
  {
    return 1;
  }
  int count = static_cast<int>(
    std::ceil(2.0F * range * half_angle / resolution)) + 1;
  count = std::max(3, count);
  if ((count & 1) == 0)
  {
    ++count;
  }
  if (max_samples > 0 && count > max_samples)
  {
    count = std::max(3, max_samples);
    if ((count & 1) == 0)
    {
      --count;
    }
  }
  return count;
}

template<typename F>
inline void forEachElevationRibbonSample(
  const SonarReconstructionPoint& p, const float resolution,
  const int max_samples, F&& fn)
{
  const Eigen::Vector3f origin = pointOrigin(p);
  const Eigen::Vector3f centre_vector = pointPosition(p) - origin;
  Eigen::Vector3f axis = pointElevationAxis(p);
  if (!centre_vector.allFinite() || !axis.allFinite() ||
      !(p.range > 0.0F) || !(p.elevation_half_angle >= 0.0F))
  {
    return;
  }
  const float axis_norm = axis.norm();
  if (!(axis_norm > 1e-6F))
  {
    return;
  }
  axis /= axis_norm;
  const int count = elevationRibbonSampleCount(
    p.range, p.elevation_half_angle, resolution, max_samples);
  if (count == 1)
  {
    fn(pointPosition(p));
    return;
  }
  for (int i = 0; i < count; ++i)
  {
    const float u = static_cast<float>(i) / static_cast<float>(count - 1);
    const float delta = -p.elevation_half_angle +
      2.0F * p.elevation_half_angle * u;
    // Same slant-range sphere as sonar_proc::placeAtElevationOffset:
    // shrink the in-fan centre ray by cos(delta) and move along optical -x.
    const Eigen::Vector3f q = origin + std::cos(delta) * centre_vector -
      p.range * std::sin(delta) * axis;
    fn(q);
  }
}

inline float boresightElevation(const SonarReconstructionPoint& p)
{
  const Eigen::Vector3f b = pointBoresight(p);
  const float norm = b.norm();
  if (!(norm > 1e-6F))
  {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::asin(std::clamp(b.z() / norm, -1.0F, 1.0F));
}

inline std::uint64_t reconstructionVoxelKey(
  const float x, const float y, const float z, const float resolution)
{
  const auto q = [resolution](const float v) {
    return static_cast<std::uint64_t>(
      static_cast<std::int64_t>(std::llround(v / resolution)) & 0x1FFFFF);
  };
  return (q(x) << 42) | (q(y) << 21) | q(z);
}

struct ReconstructionRow
{
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
  float intensity = 0.0F;
  float support = 0.0F;
  float view_span_deg = 0.0F;
  float confidence = 0.0F;
};

// Per-voxel, per-ping maximum evidence. Hundreds of neighbouring pixels from
// one sonar image may cross one voxel, but that is still ONE view, not hundreds
// of independent observations. Because reconstruction clouds are appended in
// observation order, one pending maximum per cell provides exact de-duplication
// without a cell x observation hash table.
class MultiViewSurfaceAccumulator
{
public:
  MultiViewSurfaceAccumulator(
    const float resolution, const int min_observations,
    const float min_view_span_rad, const int max_samples)
    : resolution_(resolution), min_observations_(min_observations),
      min_view_span_rad_(min_view_span_rad), max_samples_(max_samples)
  {}

  void clear()
  {
    cells_.clear();
  }

  void add(const SonarReconstructionPoint& p)
  {
    const float aspect = boresightElevation(p);
    if (!std::isfinite(aspect) || !std::isfinite(p.intensity))
    {
      return;
    }
    forEachElevationRibbonSample(
      p, resolution_, max_samples_,
      [this, &p, aspect](const Eigen::Vector3f& q) {
        if (!q.allFinite())
        {
          return;
        }
        Cell& c = cells_[reconstructionVoxelKey(
          q.x(), q.y(), q.z(), resolution_)];
        if (!c.pending || c.pending_observation != p.observation)
        {
          commit(c);
          c.pending = true;
          c.pending_observation = p.observation;
          c.pending_intensity = p.intensity;
          c.pending_position = q;
          c.pending_aspect = aspect;
        }
        else if (p.intensity > c.pending_intensity)
        {
          c.pending_intensity = p.intensity;
          c.pending_position = q;
          c.pending_aspect = aspect;
        }
      });
  }

  std::vector<ReconstructionRow> rows()
  {
    std::vector<ReconstructionRow> out;
    out.reserve(cells_.size());
    for (auto& [key, c] : cells_)
    {
      (void)key;
      commit(c);
      if (c.support < min_observations_)
      {
        continue;
      }
      const float span = c.max_aspect - c.min_aspect;
      if (span + 1e-6F < min_view_span_rad_)
      {
        continue;
      }
      const float support_score = std::min(
        1.0F, static_cast<float>(c.support) /
          static_cast<float>(std::max(1, min_observations_)));
      const float span_score = min_view_span_rad_ > 0.0F
        ? std::min(1.0F, span / min_view_span_rad_) : 1.0F;
      const Eigen::Vector3f centre = c.position_sum /
        static_cast<float>(c.support);
      out.push_back({
        centre.x(), centre.y(), centre.z(),
        c.intensity_sum / static_cast<float>(c.support),
        static_cast<float>(c.support),
        span * 180.0F / static_cast<float>(M_PI),
        support_score * span_score});
    }
    return out;
  }

  std::size_t candidateCellCount() const
  {
    return cells_.size();
  }

private:
  struct Cell
  {
    Eigen::Vector3f position_sum = Eigen::Vector3f::Zero();
    float intensity_sum = 0.0F;
    int support = 0;
    float min_aspect = std::numeric_limits<float>::infinity();
    float max_aspect = -std::numeric_limits<float>::infinity();
    bool pending = false;
    std::uint32_t pending_observation = 0;
    float pending_intensity = 0.0F;
    Eigen::Vector3f pending_position = Eigen::Vector3f::Zero();
    float pending_aspect = 0.0F;
  };

  static void commit(Cell& c)
  {
    if (!c.pending)
    {
      return;
    }
    c.position_sum += c.pending_position;
    c.intensity_sum += c.pending_intensity;
    c.min_aspect = std::min(c.min_aspect, c.pending_aspect);
    c.max_aspect = std::max(c.max_aspect, c.pending_aspect);
    ++c.support;
    c.pending = false;
  }

  float resolution_;
  int min_observations_;
  float min_view_span_rad_;
  int max_samples_;
  std::unordered_map<std::uint64_t, Cell> cells_;
};

}  // namespace vdb_mapping_ros2
