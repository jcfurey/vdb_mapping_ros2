#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
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

// sonar_proc's range-consolidated surface input. It shares the physical fan
// geometry with SonarTilePoint but represents one echo lobe rather than every
// bright image bin, and carries the measured radial uncertainty/contrast.
struct SonarReconstructionReturnPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float range;
  float azimuth;
  float vertical_uncertainty;
  float range_sigma;
  float prominence;
  float echo_width;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// A tile or lobe-consolidated return after the exact ping-time sensor pose has
// been attached. Both the centre point and sensor origin are points;
// elevation_axis and boresight are vectors. Keeping all four in keyframe-local
// coordinates is what lets a later graph correction move the measurement
// without losing the pivot-head geometry needed for its aperture ribbon.
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
  float range_sigma;
  float return_prominence;
  float echo_width;
  std::uint32_t observation;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
  SonarTilePoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    float, range, range)(float, azimuth, azimuth)(
    float, vertical_uncertainty, vertical_uncertainty))

POINT_CLOUD_REGISTER_POINT_STRUCT(
  SonarReconstructionReturnPoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    float, range, range)(float, azimuth, azimuth)(
    float, vertical_uncertainty, vertical_uncertainty)(
    float, range_sigma, range_sigma)(float, prominence, prominence)(
    float, echo_width, echo_width))

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
    float, boresight_z, boresight_z)(float, range_sigma, range_sigma)(
    float, return_prominence, return_prominence)(float, echo_width, echo_width)(
    std::uint32_t, observation, observation))

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
  float range_sigma = 0.0F;
  float echo_width = 0.0F;
  float echo_prominence = 0.0F;
  float peak_prominence = 0.0F;
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
          c.pending_range_sigma = std::max(0.0F, p.range_sigma);
          c.pending_echo_width = std::max(0.0F, p.echo_width);
          c.pending_return_prominence = std::max(0.0F, p.return_prominence);
        }
        else if (p.intensity > c.pending_intensity)
        {
          c.pending_intensity = p.intensity;
          c.pending_position = q;
          c.pending_aspect = aspect;
          c.pending_axis = elevation_axis;
          c.pending_centre_distance = (q - measured_position).squaredNorm();
          c.pending_range_sigma = std::max(0.0F, p.range_sigma);
          c.pending_echo_width = std::max(0.0F, p.echo_width);
          c.pending_return_prominence = std::max(0.0F, p.return_prominence);
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
      // A flat ribbon-overlap plateau is still unresolved elevation, even if
      // it has enormous support. Give zero-prominence cells zero confidence
      // instead of the historical 0.5 floor that admitted nearly every voxel
      // into the navigation product. The exponential keeps modest but real
      // peaks useful without letting support overwhelm ambiguity.
      const float prominence_score = 1.0F - std::exp(-4.0F * prominence);
      const Eigen::Vector3f centre = c.position_sum /
        static_cast<float>(c.support);
      out.push_back({
        centre.x(), centre.y(), centre.z(),
        c.intensity_sum / static_cast<float>(c.support),
        static_cast<float>(c.support),
        span * 180.0F / static_cast<float>(M_PI),
        std::sqrt(support_score * span_score * prominence_score),
        c.range_sigma_sum / static_cast<float>(c.support),
        c.echo_width_sum / static_cast<float>(c.support),
        c.return_prominence_sum / static_cast<float>(c.support),
        prominence});
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
    float range_sigma_sum = 0.0F;
    float echo_width_sum = 0.0F;
    float return_prominence_sum = 0.0F;
    bool pending = false;
    std::uint32_t pending_observation = 0;
    float pending_intensity = 0.0F;
    Eigen::Vector3f pending_position = Eigen::Vector3f::Zero();
    float pending_aspect = 0.0F;
    Eigen::Vector3f pending_axis = Eigen::Vector3f::Zero();
    float pending_centre_distance = 0.0F;
    float pending_range_sigma = 0.0F;
    float pending_echo_width = 0.0F;
    float pending_return_prominence = 0.0F;
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
    // Only axis orientation matters, not sign. Align every observation to the
    // accumulated hemisphere so opposite vehicle headings reinforce the same
    // physical uncertainty line without destroying its diagonal direction.
    Eigen::Vector3f aligned_axis = c.pending_axis;
    if (c.axis_sum.squaredNorm() > 1e-12F &&
        c.axis_sum.dot(aligned_axis) < 0.0F)
    {
      aligned_axis = -aligned_axis;
    }
    c.axis_sum += aligned_axis;
    c.centre_distance_sum += c.pending_centre_distance;
    c.range_sigma_sum += c.pending_range_sigma;
    c.echo_width_sum += c.pending_echo_width;
    c.return_prominence_sum += c.pending_return_prominence;
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
    const std::uint64_t key, const int dx, const int dy, const int dz)
  {
    constexpr std::uint64_t mask = 0x1FFFFFULL;
    std::uint64_t x = (key >> 42) & mask;
    std::uint64_t y = (key >> 21) & mask;
    std::uint64_t z = key & mask;
    const auto shifted = [](const std::uint64_t value, const int offset) {
      return static_cast<std::uint64_t>(
        (static_cast<std::int64_t>(value) + offset) & 0x1FFFFFLL);
    };
    x = shifted(x, dx);
    y = shifted(y, dy);
    z = shifted(z, dz);
    return (x << 42) | (y << 21) | z;
  }

  bool isElevationPeak(
    const std::uint64_t key, const Cell& c,
    float& strongest_neighbor) const
  {
    strongest_neighbor = 0.0F;
    if (peak_radius_voxels_ <= 0) return true;
    Eigen::Vector3f axis = c.axis_sum;
    const float axis_norm = axis.norm();
    if (!(axis_norm > 1e-6F)) return true;
    axis /= axis_norm;

    const float score = evidenceScore(c);
    const float centre_distance = c.centre_distance_sum /
      static_cast<float>(c.support);
    for (int distance = 1; distance <= peak_radius_voxels_; ++distance)
    {
      for (const int direction : {-1, 1})
      {
        const Eigen::Vector3f delta =
          axis * static_cast<float>(direction * distance);
        const int dx = static_cast<int>(std::lround(delta.x()));
        const int dy = static_cast<int>(std::lround(delta.y()));
        const int dz = static_cast<int>(std::lround(delta.z()));
        if (dx == 0 && dy == 0 && dz == 0) continue;
        const auto found = cells_.find(offsetKey(
          key, dx, dy, dz));
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

// Lidar-style surface element derived from a connected neighborhood of
// graph-corrected survey returns. The unrefined ReconstructionRow remains the
// measured evidence product; this record is the thin operator/registration
// surface and therefore carries an estimated normal and an explicit fit
// residual.
struct SurfelRow
{
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
  float intensity = 0.0F;
  float support = 0.0F;
  float view_span_deg = 0.0F;
  float confidence = 0.0F;
  float normal_x = 0.0F;
  float normal_y = 0.0F;
  float normal_z = 1.0F;
  float curvature = 0.0F;
  float residual = 0.0F;
  float range_sigma = 0.0F;
  float echo_width = 0.0F;
  float echo_prominence = 0.0F;
  float peak_prominence = 0.0F;
};

// Fit one weighted local plane per confirmed voxel and project only that
// voxel's representative onto the plane. This removes voxel stair steps and
// isolated flashlight speckle without moving points tangentially, blurring
// edges, or imposing a global floor/wall model. The projection is explicitly
// capped because a pretty surface must never outrun the measured resolution.
inline std::vector<SurfelRow> fitSurfaceElements(
  const std::vector<ReconstructionRow>& input,
  const float resolution,
  const float radius_m,
  const int minimum_neighbors,
  const float maximum_surface_variation,
  const float maximum_projection)
{
  std::vector<SurfelRow> out;
  if (input.empty() || !(resolution > 0.0F) || !(radius_m > 0.0F) ||
      minimum_neighbors < 1 || !(maximum_surface_variation > 0.0F) ||
      !(maximum_projection >= 0.0F))
  {
    return out;
  }

  // The representation and the geometric neighborhood are deliberately
  // independent. A 1 cm survey can retain distinct wall detail while a
  // 10 cm metric neighborhood still supplies enough samples for a stable
  // normal. Bucket at the metric radius so each query visits at most the 27
  // adjacent buckets instead of walking (2r/resolution + 1)^3 empty 1 cm
  // voxels for every point.
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> lookup;
  lookup.reserve(std::max<std::size_t>(1, input.size() / 8));
  for (std::size_t i = 0; i < input.size(); ++i)
  {
    const auto& p = input[i];
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
    {
      lookup[reconstructionVoxelKey(p.x, p.y, p.z, radius_m)].push_back(i);
    }
  }

  const float radius_squared = radius_m * radius_m;
  const float gaussian_sigma = std::max(0.5F * radius_m, resolution);
  const float gaussian_variance = gaussian_sigma * gaussian_sigma;
  const float radius_in_cells = radius_m / resolution;
  std::vector<std::pair<const ReconstructionRow*, float>> local;
  local.reserve(std::min<std::size_t>(
    input.size(), static_cast<std::size_t>(std::max(
      64.0F, 2.0F * static_cast<float>(M_PI) *
        radius_in_cells * radius_in_cells))));
  out.reserve(input.size());
  for (const auto& p : input)
  {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
      continue;

    Eigen::Vector3f weighted_sum = Eigen::Vector3f::Zero();
    float weight_sum = 0.0F;
    int neighbors = 0;
    local.clear();
    const Eigen::Vector3f centre(p.x, p.y, p.z);
    for (int dx = -1; dx <= 1; ++dx)
    {
      for (int dy = -1; dy <= 1; ++dy)
      {
        for (int dz = -1; dz <= 1; ++dz)
        {
          const auto found = lookup.find(reconstructionVoxelKey(
            p.x + static_cast<float>(dx) * radius_m,
            p.y + static_cast<float>(dy) * radius_m,
            p.z + static_cast<float>(dz) * radius_m,
            radius_m));
          if (found == lookup.end()) continue;
          for (const std::size_t index : found->second)
          {
            const auto& q = input[index];
            const Eigen::Vector3f position(q.x, q.y, q.z);
            const float distance_squared = (position - centre).squaredNorm();
            if (distance_squared > radius_squared + 1e-9F) continue;
            const float spatial = std::exp(
              -0.5F * distance_squared /
              std::max(gaussian_variance, 1e-6F));
            const float evidence = std::sqrt(std::max(q.support, 1.0F)) *
              std::max(q.confidence, 0.05F);
            const float weight = spatial * evidence;
            weighted_sum += weight * position;
            weight_sum += weight;
            local.emplace_back(&q, weight);
            ++neighbors;
          }
        }
      }
    }
    if (neighbors < minimum_neighbors || !(weight_sum > 1e-6F)) continue;

    const Eigen::Vector3f centroid = weighted_sum / weight_sum;
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto& [q, weight] : local)
    {
      const Eigen::Vector3f delta(q->x - centroid.x(), q->y - centroid.y(),
                                  q->z - centroid.z());
      covariance.noalias() += weight * delta * delta.transpose();
    }
    covariance /= weight_sum;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) continue;
    const Eigen::Vector3f eigenvalues = solver.eigenvalues().cwiseMax(0.0F);
    const float total = eigenvalues.sum();
    if (!(total > 1e-9F)) continue;
    const float variation = eigenvalues.x() / total;
    if (!std::isfinite(variation) ||
        variation > maximum_surface_variation) continue;

    Eigen::Vector3f normal = solver.eigenvectors().col(0).normalized();
    Eigen::Index dominant = 0;
    normal.cwiseAbs().maxCoeff(&dominant);
    if (normal[dominant] < 0.0F) normal = -normal;
    const Eigen::Vector3f position(p.x, p.y, p.z);
    const float signed_distance = normal.dot(position - centroid);
    const float correction = std::clamp(
      signed_distance, -maximum_projection, maximum_projection);
    const Eigen::Vector3f refined = position - correction * normal;
    out.push_back({
      refined.x(), refined.y(), refined.z(), p.intensity, p.support,
      // Confidence describes independent-view/elevation evidence. Curvature
      // and residual already describe local fit quality, and variation above
      // the configured ceiling was rejected just above. Multiplying the two
      // here and then applying navigation_min_confidence again silently made
      // the evidence gate much stricter and removed most valid wall surfels.
      p.view_span_deg, p.confidence,
      normal.x(), normal.y(), normal.z(), variation,
      std::sqrt(eigenvalues.x()), p.range_sigma, p.echo_width,
      p.echo_prominence, p.peak_prominence});
  }
  return out;
}

}  // namespace vdb_mapping_ros2
