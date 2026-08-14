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
    const float min_view_span_rad, const int max_samples,
    const int peak_radius_voxels = 0,
    const float min_return_intensity = 0.0F)
    : resolution_(resolution), min_observations_(min_observations),
      min_view_span_rad_(min_view_span_rad), max_samples_(max_samples),
      peak_radius_voxels_(std::max(0, peak_radius_voxels)),
      min_return_intensity_(std::clamp(min_return_intensity, 0.0F, 1.0F))
  {}

  void clear()
  {
    cells_.clear();
  }

  void add(const SonarReconstructionPoint& p)
  {
    const float aspect = boresightElevation(p);
    Eigen::Vector3f elevation_axis = pointElevationAxis(p);
    const float axis_norm = elevation_axis.norm();
    if (!std::isfinite(aspect) || !std::isfinite(p.intensity) ||
        p.intensity < min_return_intensity_ ||
        !elevation_axis.allFinite() || !(axis_norm > 1e-6F))
    {
      return;
    }
    elevation_axis /= axis_norm;
    const Eigen::Vector3f measured_position = pointPosition(p);
    forEachElevationRibbonSample(
      p, resolution_, max_samples_,
      [this, &p, aspect, &elevation_axis, &measured_position](
        const Eigen::Vector3f& q) {
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
          c.pending_axis = elevation_axis;
          c.pending_centre_distance = (q - measured_position).squaredNorm();
        }
        else if (p.intensity > c.pending_intensity)
        {
          c.pending_intensity = p.intensity;
          c.pending_position = q;
          c.pending_aspect = aspect;
          c.pending_axis = elevation_axis;
          c.pending_centre_distance = (q - measured_position).squaredNorm();
        }
      });
  }

  std::vector<ReconstructionRow> rows()
  {
    std::vector<ReconstructionRow> out;
    // Most ribbon candidates never earn the independent support/view-span
    // gates. Reserving for every candidate made a sparse output allocate as
    // though all uncertainty volume were publishable (hundreds of MB on the
    // 08-12 survey). Grow from a modest seed instead.
    out.reserve(std::min<std::size_t>(cells_.size(), 65536));
    for (auto& [key, c] : cells_)
    {
      (void)key;
      commit(c);
    }
    for (const auto& [key, c] : cells_)
    {
      if (!qualifies(c)) continue;
      const float span = c.max_aspect - c.min_aspect;
      float strongest_neighbor = 0.0F;
      if (!isElevationPeak(key, c, strongest_neighbor))
      {
        continue;
      }
      // Unlike the former value/threshold clamp, these scores retain useful
      // dynamic range after thresholding (the old formula made confidence
      // mathematically equal to 1 for every emitted point).
      const float support_score = 1.0F - std::exp(
        -static_cast<float>(c.support) /
        static_cast<float>(std::max(1, min_observations_)));
      const float span_score = min_view_span_rad_ > 0.0F
        ? 1.0F - std::exp(-span / min_view_span_rad_) : 1.0F;
      const float score = evidenceScore(c);
      const float prominence = score > 1e-6F
        ? std::clamp((score - strongest_neighbor) / score, 0.0F, 1.0F)
        : 0.0F;
      const Eigen::Vector3f centre = c.position_sum /
        static_cast<float>(c.support);
      out.push_back({
        centre.x(), centre.y(), centre.z(),
        c.intensity_sum / static_cast<float>(c.support),
        static_cast<float>(c.support),
        span * 180.0F / static_cast<float>(M_PI),
        std::sqrt(support_score * span_score) *
          (0.5F + 0.5F * prominence)});
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
    Eigen::Vector3f axis_sum = Eigen::Vector3f::Zero();
    float centre_distance_sum = 0.0F;
    bool pending = false;
    std::uint32_t pending_observation = 0;
    float pending_intensity = 0.0F;
    Eigen::Vector3f pending_position = Eigen::Vector3f::Zero();
    float pending_aspect = 0.0F;
    Eigen::Vector3f pending_axis = Eigen::Vector3f::Zero();
    float pending_centre_distance = 0.0F;
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
    // Only the uncertainty-axis orientation matters for peak suppression;
    // opposite vehicle headings must not cancel the accumulated direction.
    c.axis_sum += c.pending_axis.cwiseAbs();
    c.centre_distance_sum += c.pending_centre_distance;
    ++c.support;
    c.pending = false;
  }

  bool qualifies(const Cell& c) const
  {
    return c.support >= min_observations_ &&
      c.max_aspect - c.min_aspect + 1e-6F >= min_view_span_rad_;
  }

  static float evidenceScore(const Cell& c)
  {
    // Support is primary, while return strength breaks the broad integer
    // plateaus created when many aperture ribbons cross adjacent voxels.
    return 0.25F * static_cast<float>(c.support) +
      0.75F * c.intensity_sum;
  }

  static std::uint64_t offsetKey(
    const std::uint64_t key, const int axis, const int offset)
  {
    constexpr std::uint64_t mask = 0x1FFFFFULL;
    std::uint64_t x = (key >> 42) & mask;
    std::uint64_t y = (key >> 21) & mask;
    std::uint64_t z = key & mask;
    const auto shifted = [offset](const std::uint64_t value) {
      return static_cast<std::uint64_t>(
        (static_cast<std::int64_t>(value) + offset) & 0x1FFFFFLL);
    };
    if (axis == 0) x = shifted(x);
    else if (axis == 1) y = shifted(y);
    else z = shifted(z);
    return (x << 42) | (y << 21) | z;
  }

  bool isElevationPeak(
    const std::uint64_t key, const Cell& c,
    float& strongest_neighbor) const
  {
    strongest_neighbor = 0.0F;
    if (peak_radius_voxels_ <= 0) return true;
    const Eigen::Vector3f axis = c.axis_sum.cwiseAbs();
    int dominant_axis = 0;
    if (axis.y() > axis.x()) dominant_axis = 1;
    if (axis.z() > axis[dominant_axis]) dominant_axis = 2;
    if (!(axis[dominant_axis] > 1e-6F)) return true;

    const float score = evidenceScore(c);
    const float centre_distance = c.centre_distance_sum /
      static_cast<float>(c.support);
    for (int distance = 1; distance <= peak_radius_voxels_; ++distance)
    {
      for (const int direction : {-1, 1})
      {
        const auto found = cells_.find(offsetKey(
          key, dominant_axis, direction * distance));
        if (found == cells_.end() || !qualifies(found->second)) continue;
        const Cell& neighbor = found->second;
        const float neighbor_score = evidenceScore(neighbor);
        strongest_neighbor = std::max(strongest_neighbor, neighbor_score);
        if (neighbor_score > score + 1e-6F) return false;
        if (std::fabs(neighbor_score - score) <= 1e-6F &&
            neighbor.centre_distance_sum /
              static_cast<float>(neighbor.support) + 1e-6F < centre_distance)
          return false;
      }
    }
    return true;
  }

  float resolution_;
  int min_observations_;
  float min_view_span_rad_;
  int max_samples_;
  int peak_radius_voxels_;
  float min_return_intensity_;
  std::unordered_map<std::uint64_t, Cell> cells_;
};

}  // namespace vdb_mapping_ros2
