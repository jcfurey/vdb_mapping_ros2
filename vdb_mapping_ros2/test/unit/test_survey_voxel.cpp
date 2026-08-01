// Pins the property that motivated replacing pcl::VoxelGrid in the
// assembler: EVERY survey field must survive voxel reduction. VoxelGrid's
// closed accumulator set value-initialized all custom fields to 0.0, and the
// zeros passed every downstream validity guard — the exported survey carried
// fictitious texture, incidence and elevation columns.
#include <gtest/gtest.h>

#include <cmath>

#include <vdb_mapping_ros2/survey_voxel.hpp>

namespace {

SurveyPoint make(float x, float y, float z, float intensity, float range,
                 float incidence, float texture, float texture_squared,
                 float lo, float hi, float resolved)
{
  SurveyPoint p{};
  p.x                   = x;
  p.y                   = y;
  p.z                   = z;
  p.intensity           = intensity;
  p.range               = range;
  p.incidence           = incidence;
  p.texture             = texture;
  p.texture_squared     = texture_squared;
  p.elevation_lo_offset = lo;
  p.elevation_hi_offset = hi;
  p.elevation_resolved  = resolved;
  return p;
}

}  // namespace

TEST(SurveyVoxel, AllFieldsSurviveReduction)
{
  pcl::PointCloud<SurveyPoint>::Ptr in(new pcl::PointCloud<SurveyPoint>);
  // two points in the same 0.5 m voxel with distinct metadata everywhere
  in->push_back(make(0.10F, 0.10F, 1.00F, 0.2F, 5.0F, 0.4F, 0.10F, 0.02F,
                     -0.30F, 0.10F, 1.0F));
  in->push_back(make(0.20F, 0.20F, 1.20F, 0.6F, 7.0F, 0.8F, 0.30F, 0.10F,
                     -0.10F, 0.30F, 0.0F));
  const auto out = vdb_mapping_ros2::voxelReduceSurvey(in, 0.5F);
  ASSERT_EQ(out->size(), 1U);
  const SurveyPoint& p = out->points[0];
  EXPECT_NEAR(p.x, 0.15F, 1e-6);
  EXPECT_NEAR(p.z, 1.10F, 1e-6);
  EXPECT_NEAR(p.intensity, 0.4F, 1e-6);
  EXPECT_NEAR(p.range, 6.0F, 1e-6);
  EXPECT_NEAR(p.incidence, 0.6F, 1e-6);
  EXPECT_NEAR(p.texture, 0.2F, 1e-6);
  EXPECT_NEAR(p.texture_squared, 0.06F, 1e-6);
  // absolute bounds: (1.0-0.3 + 1.2-0.1)/2 = 0.9; (1.0+0.1 + 1.2+0.3)/2 = 1.3
  // re-relativized against the centroid z 1.1
  EXPECT_NEAR(p.elevation_lo_offset, 0.9F - 1.1F, 1e-5);
  EXPECT_NEAR(p.elevation_hi_offset, 1.3F - 1.1F, 1e-5);
  EXPECT_NEAR(p.elevation_resolved, 0.5F, 1e-6);
}

TEST(SurveyVoxel, SentinelsNeverDiluteAndNeverFabricate)
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  pcl::PointCloud<SurveyPoint>::Ptr in(new pcl::PointCloud<SurveyPoint>);
  // voxel A: one point knows incidence, the other declines (-1); texture
  // known by one only
  in->push_back(make(0.1F, 0.1F, 0.1F, 0.5F, 5.0F, -1.0F, nan, nan, nan, nan, nan));
  in->push_back(make(0.2F, 0.2F, 0.2F, 0.5F, 5.0F, 0.6F, 0.4F, 0.16F, -0.2F,
                     0.2F, 1.0F));
  // voxel B (far away): nobody knows anything optional
  in->push_back(make(9.0F, 9.0F, 9.0F, 0.1F, 3.0F, -1.0F, nan, nan, nan, nan, nan));
  const auto out = vdb_mapping_ros2::voxelReduceSurvey(in, 0.5F);
  ASSERT_EQ(out->size(), 2U);
  const bool a_first  = out->points[0].z < 1.0F;
  const SurveyPoint& a = out->points[a_first ? 0 : 1];
  const SurveyPoint& b = out->points[a_first ? 1 : 0];
  // the knowing contributor defines the mean; the declining one is excluded
  EXPECT_NEAR(a.incidence, 0.6F, 1e-6);
  EXPECT_NEAR(a.texture, 0.4F, 1e-6);
  EXPECT_NEAR(a.elevation_resolved, 1.0F, 1e-6);
  // and a voxel with no contributor emits the sentinel, not a fake zero
  EXPECT_FLOAT_EQ(b.incidence, -1.0F);
  EXPECT_TRUE(std::isnan(b.texture));
  EXPECT_TRUE(std::isnan(b.elevation_lo_offset));
  EXPECT_TRUE(std::isnan(b.elevation_resolved));
}

TEST(SurveyVoxel, DegenerateLeafPassesThrough)
{
  pcl::PointCloud<SurveyPoint>::Ptr in(new pcl::PointCloud<SurveyPoint>);
  in->push_back(make(1, 2, 3, 0.5F, 5.0F, 0.5F, 0.1F, 0.01F, -0.1F, 0.1F, 1.0F));
  EXPECT_EQ(vdb_mapping_ros2::voxelReduceSurvey(in, 0.0F), in);
  EXPECT_EQ(vdb_mapping_ros2::voxelReduceSurvey(nullptr, 0.5F), nullptr);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
