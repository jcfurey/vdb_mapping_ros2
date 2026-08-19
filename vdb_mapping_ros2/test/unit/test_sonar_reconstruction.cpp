#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include <vdb_mapping_ros2/sonar_reconstruction.hpp>

namespace
{

SonarReconstructionPoint observation(const float degrees, const std::uint32_t id)
{
  const float a = degrees * static_cast<float>(M_PI) / 180.0F;
  SonarReconstructionPoint p{};
  p.range = 5.0F;
  p.intensity = 0.6F;
  p.elevation_half_angle = 10.0F * static_cast<float>(M_PI) / 180.0F;
  p.origin_x = p.origin_y = p.origin_z = 0.0F;
  // Centre ray b(a); elevation axis a(a) makes delta=a land at +X.
  p.x = p.range * std::cos(a);
  p.y = 0.0F;
  p.z = p.range * std::sin(a);
  p.elevation_axis_x = -std::sin(a);
  p.elevation_axis_y = 0.0F;
  p.elevation_axis_z = std::cos(a);
  p.boresight_x = std::cos(a);
  p.boresight_y = 0.0F;
  p.boresight_z = std::sin(a);
  p.observation = id;
  return p;
}

}  // namespace

TEST(SonarReconstruction, RibbonStaysOnMeasuredSlantRange)
{
  const auto p = observation(0.0F, 1);
  int count = 0;
  bool saw_centre = false;
  vdb_mapping_ros2::forEachElevationRibbonSample(
    p, 0.1F, 31, [&](const Eigen::Vector3f& q) {
      EXPECT_NEAR(q.norm(), p.range, 1e-5F);
      saw_centre = saw_centre || (q - Eigen::Vector3f(5, 0, 0)).norm() < 1e-5F;
      ++count;
    });
  EXPECT_GT(count, 3);
  EXPECT_TRUE(saw_centre);
}

TEST(SonarReconstruction, UncappedRibbonHonoursResolutionAtHfAndLfRange)
{
  const float half_angle = 10.0F * static_cast<float>(M_PI) / 180.0F;
  EXPECT_EQ(
    vdb_mapping_ros2::elevationRibbonSampleCount(
      5.0F, half_angle, 0.1F, 0),
    19);
  EXPECT_EQ(
    vdb_mapping_ros2::elevationRibbonSampleCount(
      30.0F, half_angle, 0.1F, 0),
    107);
  EXPECT_EQ(
    vdb_mapping_ros2::elevationRibbonSampleCount(
      30.0F, half_angle, 0.1F, 31),
    31);

  auto lf = observation(0.0F, 1);
  lf.range = 30.0F;
  lf.x = 30.0F;
  std::vector<Eigen::Vector3f> samples;
  vdb_mapping_ros2::forEachElevationRibbonSample(
    lf, 0.1F, 0,
    [&samples](const Eigen::Vector3f& q) { samples.push_back(q); });
  ASSERT_EQ(samples.size(), 107U);
  for (std::size_t i = 1; i < samples.size(); ++i)
  {
    EXPECT_LE((samples[i] - samples[i - 1]).norm(), 0.1F + 1e-5F);
  }
}

TEST(SonarReconstruction, TransformMovesPointsAndOriginsButOnlyRotatesAxes)
{
  const auto p = observation(0.0F, 7);
  Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
  tf.translate(Eigen::Vector3f(2.0F, 3.0F, 4.0F));
  tf.rotate(Eigen::AngleAxisf(static_cast<float>(M_PI) / 2.0F,
                             Eigen::Vector3f::UnitY()));
  const auto out = vdb_mapping_ros2::transformReconstructionPoint(p, tf);
  EXPECT_TRUE(vdb_mapping_ros2::pointPosition(out).isApprox(
    tf * vdb_mapping_ros2::pointPosition(p), 1e-5F));
  EXPECT_TRUE(vdb_mapping_ros2::pointOrigin(out).isApprox(tf.translation(), 1e-5F));
  EXPECT_TRUE(vdb_mapping_ros2::pointElevationAxis(out).isApprox(
    tf.linear() * vdb_mapping_ros2::pointElevationAxis(p), 1e-5F));
  EXPECT_EQ(out.observation, 7U);
}

TEST(SonarReconstruction, OnePingCountsOnceAndViewDiversityIsRequired)
{
  vdb_mapping_ros2::MultiViewSurfaceAccumulator acc(
    0.20F, 3, 6.0F * static_cast<float>(M_PI) / 180.0F, 5);

  // Repeating one image's pixel cannot manufacture independent support.
  const auto same = observation(0.0F, 1);
  acc.add(same);
  acc.add(same);
  acc.add(same);
  EXPECT_TRUE(acc.rows().empty());

  acc.clear();
  acc.add(observation(-10.0F, 1));
  acc.add(observation(0.0F, 2));
  acc.add(observation(10.0F, 3));
  const auto rows = acc.rows();
  const auto target = std::find_if(rows.begin(), rows.end(), [](const auto& r) {
    return std::fabs(r.x - 5.0F) < 0.15F &&
      std::hypot(r.y, r.z) < 0.15F;
  });
  ASSERT_NE(target, rows.end());
  EXPECT_GE(target->support, 3.0F);
  EXPECT_GE(target->view_span_deg, 19.0F);
  EXPECT_NEAR(target->intensity, 0.6F, 1e-5F);
  EXPECT_GT(target->confidence, 0.0F);
  EXPECT_LT(target->confidence, 1.0F);
}

TEST(SonarReconstruction, ElevationPeakFilterRejectsApertureVolume)
{
  vdb_mapping_ros2::MultiViewSurfaceAccumulator volume(
    0.20F, 3, 6.0F * static_cast<float>(M_PI) / 180.0F, 21, 0);
  vdb_mapping_ros2::MultiViewSurfaceAccumulator surface(
    0.20F, 3, 6.0F * static_cast<float>(M_PI) / 180.0F, 21, 2);
  for (const float angle : {-5.0F, 0.0F, 5.0F})
  {
    const auto p = observation(
      angle, static_cast<std::uint32_t>(angle + 6.0F));
    volume.add(p);
    surface.add(p);
  }
  const auto volume_rows = volume.rows();
  const auto surface_rows = surface.rows();
  ASSERT_FALSE(volume_rows.empty());
  ASSERT_FALSE(surface_rows.empty());
  EXPECT_LT(surface_rows.size(), volume_rows.size());
  EXPECT_TRUE(std::any_of(
    surface_rows.begin(), surface_rows.end(), [](const auto& row) {
      return std::fabs(row.x - 5.0F) < 0.15F &&
        std::hypot(row.y, row.z) < 0.15F;
    }));
}

TEST(SonarReconstruction, ElevationPeakFilterFollowsDiagonalUncertaintyAxis)
{
  vdb_mapping_ros2::MultiViewSurfaceAccumulator unfiltered(
    0.10F, 3, 6.0F * static_cast<float>(M_PI) / 180.0F, 0, 0);
  vdb_mapping_ros2::MultiViewSurfaceAccumulator filtered(
    0.10F, 3, 6.0F * static_cast<float>(M_PI) / 180.0F, 0, 2);
  const float diagonal = std::sqrt(0.5F);
  for (const float angle : {-10.0F, 0.0F, 10.0F})
  {
    auto p = observation(
      angle, static_cast<std::uint32_t>(angle + 11.0F));
    // Give all three observations the same diagonal aperture ribbon while
    // retaining distinct boresight aspects. This is an intentionally
    // unresolved volume: NMS must walk the actual diagonal uncertainty axis,
    // not whichever Cartesian component happens to be marginally largest.
    p.x = 5.0F;
    p.y = 0.0F;
    p.z = 0.0F;
    p.elevation_axis_x = 0.0F;
    p.elevation_axis_y = diagonal;
    p.elevation_axis_z = diagonal;
    unfiltered.add(p);
    filtered.add(p);
  }

  const auto volume = unfiltered.rows();
  const auto peaks = filtered.rows();
  ASSERT_GT(volume.size(), 10U);
  EXPECT_LT(peaks.size(), volume.size() / 2U);
  EXPECT_TRUE(std::any_of(peaks.begin(), peaks.end(), [](const auto& row) {
    return std::fabs(row.x - 5.0F) < 0.11F &&
      std::hypot(row.y, row.z) < 0.11F;
  }));
}

TEST(SonarReconstruction, WeakReturnsCannotVoteForStructure)
{
  vdb_mapping_ros2::MultiViewSurfaceAccumulator surface(
    0.20F, 3, 6.0F * static_cast<float>(M_PI) / 180.0F, 21, 2, 0.20F);
  for (const float angle : {-5.0F, 0.0F, 5.0F})
  {
    auto p = observation(angle, static_cast<std::uint32_t>(angle + 6.0F));
    p.intensity = 0.19F;
    surface.add(p);
  }
  EXPECT_TRUE(surface.rows().empty());

  surface.clear();
  for (const float angle : {-5.0F, 0.0F, 5.0F})
  {
    auto p = observation(angle, static_cast<std::uint32_t>(angle + 6.0F));
    p.intensity = 0.20F;
    surface.add(p);
  }
  EXPECT_FALSE(surface.rows().empty());
}

TEST(SonarReconstruction, LocalPlaneFitProducesBoundedLidarStyleSurfels)
{
  std::vector<vdb_mapping_ros2::ReconstructionRow> patch;
  for (int iy = -2; iy <= 2; ++iy)
  {
    for (int iz = -2; iz <= 2; ++iz)
    {
      vdb_mapping_ros2::ReconstructionRow p;
      // A lightly voxel-stepped wall. Refinement may remove only the normal
      // error; it must not smear the regular y/z sample locations.
      p.x = 5.0F + 0.004F * static_cast<float>((iy + iz) % 3 - 1);
      p.y = 0.04F * static_cast<float>(iy);
      p.z = 0.04F * static_cast<float>(iz);
      p.intensity = 0.7F;
      p.support = 4.0F;
      p.view_span_deg = 20.0F;
      p.confidence = 0.8F;
      p.range_sigma = 0.03F;
      p.echo_width = 0.08F;
      p.echo_prominence = 0.4F;
      p.peak_prominence = 0.5F;
      patch.push_back(p);
    }
  }
  vdb_mapping_ros2::ReconstructionRow isolated = patch.front();
  isolated.x = 9.0F;
  isolated.y = 9.0F;
  isolated.z = 9.0F;
  patch.push_back(isolated);

  const auto surfels = vdb_mapping_ros2::fitSurfaceElements(
    patch, 0.01F, 0.10F, 5, 0.12F, 0.01F);
  ASSERT_GE(surfels.size(), 20U);
  EXPECT_TRUE(std::none_of(surfels.begin(), surfels.end(), [](const auto& p) {
    return p.x > 8.0F;
  }));
  const auto centre = std::min_element(
    surfels.begin(), surfels.end(), [](const auto& a, const auto& b) {
      return std::hypot(a.y, a.z) < std::hypot(b.y, b.z);
    });
  ASSERT_NE(centre, surfels.end());
  EXPECT_NEAR(std::fabs(centre->normal_x), 1.0F, 0.02F);
  EXPECT_NEAR(centre->normal_y, 0.0F, 0.10F);
  EXPECT_NEAR(centre->normal_z, 0.0F, 0.10F);
  EXPECT_LT(std::fabs(centre->x - 5.0F), 0.02F);
  EXPECT_NEAR(centre->y, 0.0F, 1e-6F);
  EXPECT_NEAR(centre->z, 0.0F, 1e-6F);
  EXPECT_LT(centre->curvature, 0.02F);
  EXPECT_LT(centre->residual, 0.02F);
  EXPECT_NEAR(centre->confidence, 0.8F, 1e-6F);
  EXPECT_NEAR(centre->range_sigma, 0.03F, 1e-6F);
  EXPECT_NEAR(centre->echo_prominence, 0.4F, 1e-6F);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
