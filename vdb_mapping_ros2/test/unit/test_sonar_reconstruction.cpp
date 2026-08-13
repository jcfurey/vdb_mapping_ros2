#include <gtest/gtest.h>

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
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
