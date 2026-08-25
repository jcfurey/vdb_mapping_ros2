// Survey stream point type and its all-fields voxel reduction.
//
// pcl::VoxelGrid cannot reduce this type: its centroid machinery is a CLOSED
// accumulator set (xyz / normal / curvature / rgba / intensity / label), so
// every custom field of a SurveyPoint came out of filter() value-initialized
// to 0.0 — silently zeroing texture moments, elevation bounds, range and
// incidence before they were ever aggregated, while the zeros passed every
// downstream validity guard. The reduction here averages every field with
// the same sentinel rules the map-frame accumulator applies:
//   * incidence contributes only when >= 0; a voxel with no contributor
//     emits the -1 sentinel;
//   * texture moments and elevation bounds contribute only when finite; a
//     voxel with no contributor emits NaN so downstream isfinite() guards
//     skip it instead of averaging in a fake 0;
//   * elevation bounds accumulate as ABSOLUTE z and re-relativize against
//     the reduced centroid, so offsets stay offsets whatever the source
//     representative's z was.
#pragma once

#define PCL_NO_PRECOMPILE
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

// Survey stream point. The PointXYZI-compatible prefix is followed by
// radiometric texture moments and elevation uncertainty. PCL addresses fields
// by name, so the in-memory padding need not match sonar_proc's packed
// PointCloud2 stride.
struct SurveyPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float range;
  float incidence;
  float texture;
  float texture_squared;
  float elevation_lo_offset;
  float elevation_hi_offset;
  float elevation_resolved;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
  SurveyPoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    float, range, range)(float, incidence, incidence)(float, texture, texture)(
    float, texture_squared, texture_squared)(
    float, elevation_lo_offset, elevation_lo_offset)(
    float, elevation_hi_offset, elevation_hi_offset)(
    float, elevation_resolved, elevation_resolved))

namespace vdb_mapping_ros2 {

// 21 bits per axis (signed, two's-complement low bits) -> +/-1M cells.
// Same packing as the assembler's map-frame voxel key.
inline uint64_t surveyVoxelKey(const float x, const float y, const float z, const float leaf)
{
  const auto ix = static_cast<int64_t>(std::floor(x / leaf));
  const auto iy = static_cast<int64_t>(std::floor(y / leaf));
  const auto iz = static_cast<int64_t>(std::floor(z / leaf));
  const uint64_t mask = (1ULL << 21U) - 1ULL;
  return ((static_cast<uint64_t>(ix) & mask) << 42U) |
         ((static_cast<uint64_t>(iy) & mask) << 21U) |
         (static_cast<uint64_t>(iz) & mask);
}

inline pcl::PointCloud<SurveyPoint>::Ptr
voxelReduceSurvey(const pcl::PointCloud<SurveyPoint>::Ptr& in, const float leaf)
{
  if (!in || in->empty() || !std::isfinite(leaf) || leaf <= 0.0F) {
    return in;
  }
  struct Acc
  {
    double x = 0, y = 0, z = 0, intensity = 0, range = 0;
    double ci = 0, texture = 0, texture_squared = 0;
    double elevation_lo = 0, elevation_hi = 0, elevation_resolved = 0;
    uint32_t n = 0, nci = 0, ntexture = 0, nelevation = 0;
  };
  std::unordered_map<uint64_t, Acc> cells;
  cells.reserve(in->size());
  for (const auto& p : in->points)
  {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        !std::isfinite(p.intensity) || !std::isfinite(p.range)) {
      continue;
    }
    Acc& a = cells[surveyVoxelKey(p.x, p.y, p.z, leaf)];
    a.x += p.x;
    a.y += p.y;
    a.z += p.z;
    a.intensity += p.intensity;
    a.range += p.range;
    ++a.n;
    if (p.incidence >= 0.0F)
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
    if (std::isfinite(p.elevation_lo_offset) && std::isfinite(p.elevation_hi_offset) &&
        std::isfinite(p.elevation_resolved))
    {
      a.elevation_lo += p.z + p.elevation_lo_offset;
      a.elevation_hi += p.z + p.elevation_hi_offset;
      a.elevation_resolved +=
        std::clamp(static_cast<double>(p.elevation_resolved), 0.0, 1.0);
      ++a.nelevation;
    }
  }
  pcl::PointCloud<SurveyPoint>::Ptr out(new pcl::PointCloud<SurveyPoint>);
  out->reserve(cells.size());
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (const auto& [key, a] : cells)
  {
    (void)key;
    if (a.n == 0)
    {
      continue;
    }
    SurveyPoint p{};
    p.x         = static_cast<float>(a.x / a.n);
    p.y         = static_cast<float>(a.y / a.n);
    p.z         = static_cast<float>(a.z / a.n);
    p.intensity = static_cast<float>(a.intensity / a.n);
    p.range     = static_cast<float>(a.range / a.n);
    p.incidence = a.nci > 0 ? static_cast<float>(a.ci / a.nci) : -1.0F;
    if (a.ntexture > 0)
    {
      p.texture         = static_cast<float>(a.texture / a.ntexture);
      p.texture_squared = static_cast<float>(a.texture_squared / a.ntexture);
    }
    else
    {
      p.texture         = nan;
      p.texture_squared = nan;
    }
    if (a.nelevation > 0)
    {
      p.elevation_lo_offset = static_cast<float>(a.elevation_lo / a.nelevation - p.z);
      p.elevation_hi_offset = static_cast<float>(a.elevation_hi / a.nelevation - p.z);
      p.elevation_resolved  = static_cast<float>(a.elevation_resolved / a.nelevation);
    }
    else
    {
      p.elevation_lo_offset = nan;
      p.elevation_hi_offset = nan;
      p.elevation_resolved  = nan;
    }
    out->push_back(p);
  }
  out->width    = static_cast<uint32_t>(out->size());
  out->height   = 1;
  out->is_dense = false;
  return out;
}

}  // namespace vdb_mapping_ros2
