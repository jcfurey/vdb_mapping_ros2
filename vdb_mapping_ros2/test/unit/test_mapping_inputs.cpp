#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <tf2_ros/static_transform_broadcaster.hpp>
#include <vdb_mapping_ros2/VDBMappingROS2.hpp>

namespace {
using Mapper = vdb_mapping_ros2::VDBMappingROS2;
using Cloud = sensor_msgs::msg::PointCloud2;
using Field = sensor_msgs::msg::PointField;
using Add = vdb_mapping_interfaces::srv::AddPointsToGrid;
using Remove = vdb_mapping_interfaces::srv::RemovePointsFromGrid;

// Match the sonar map_points prefix, including its unused fields and padding.
Cloud makeCloud(const std::vector<std::array<float, 3>> &points,
                const std::string &frame = "map", int32_t seconds = 10) {
  Cloud cloud;
  cloud.header.frame_id = frame;
  cloud.header.stamp.sec = seconds;
  cloud.width = points.size();
  cloud.height = 1;
  cloud.point_step = 32;
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.is_dense = true;
  const uint16_t endian_probe = 1;
  cloud.is_bigendian =
      *reinterpret_cast<const unsigned char *>(&endian_probe) == 0;
  for (const auto &entry :
       std::vector<std::pair<std::string, uint32_t>>{{"x", 0},
                                                     {"y", 4},
                                                     {"z", 8},
                                                     {"intensity", 16},
                                                     {"range", 20},
                                                     {"incidence", 24}}) {
    Field field;
    field.name = entry.first;
    field.offset = entry.second;
    field.datatype = Field::FLOAT32;
    field.count = 1;
    cloud.fields.push_back(field);
  }
  cloud.data.resize(cloud.row_step);
  for (size_t i = 0; i < points.size(); ++i) {
    std::memcpy(cloud.data.data() + i * cloud.point_step, points[i].data(),
                3 * sizeof(float));
  }
  return cloud;
}

class MappingInputs : public ::testing::Test {
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void SetUp() override {
    rclcpp::NodeOptions options;
    options.parameter_overrides(
        {{"map_frame", "map"},
         {"robot_frame", "base_link"},
         {"resolution", 1.0},
         {"sources", std::vector<std::string>{"hits", "clear"}},
         {"hits.topic", "/test/hits"},
         {"hits.ray_clearing", false},
         {"clear.topic", "/test/clear"},
         {"clear.endpoint_hits", false},
         {"accumulate_updates", false},
         {"deterministic_input", true},
         {"publish_pointcloud", false},
         {"publish_vis_marker", false},
         {"publish_occupancy_grid", false},
         {"visualization_rate", 0.0},
         {"tf_lookup_timeout", 0.05}});
    mapper = std::make_shared<Mapper>(options);
    decoder = std::make_unique<Mapper::VDBMapT>(1.0);
    decoder->stop();
  }

  Mapper::VDBMapT::GridT::Ptr snapshot() {
    using Section = vdb_mapping_interfaces::srv::GetMapSection;
    auto request = std::make_shared<Section::Request>();
    auto response = std::make_shared<Section::Response>();
    request->header.frame_id = "map";
    request->bounding_box.min_corner.x = -10.0;
    request->bounding_box.min_corner.y = -10.0;
    request->bounding_box.min_corner.z = -10.0;
    request->bounding_box.max_corner.x = 10.0;
    request->bounding_box.max_corner.y = 10.0;
    request->bounding_box.max_corner.z = 10.0;
    mapper->getMapFullSectionCallback(request, response);
    EXPECT_TRUE(response->success);
    return decoder->byteArrayToGrid<Mapper::VDBMapT::GridT>(
        response->section.map);
  }

  template <typename Service> bool edit(const Cloud &cloud) {
    auto request = std::make_shared<typename Service::Request>();
    auto response = std::make_shared<typename Service::Response>();
    request->points = cloud;
    if constexpr (std::is_same_v<Service, Add>)
      mapper->addPointsToGridCallback(request, response);
    else
      mapper->removePointsFromGridCallback(request, response);
    return response->success;
  }

  void feed(const Cloud &cloud, const std::string &source = "hits",
            const std::string &origin = "") {
    vdb_mapping_ros2::SensorSource sensor{};
    sensor.source_id = source;
    sensor.sensor_origin_frame = origin;
    mapper->cloudCallback(std::make_shared<Cloud>(cloud), sensor);
  }

  std::shared_ptr<Mapper> mapper;
  std::unique_ptr<Mapper::VDBMapT> decoder;
};

TEST_F(MappingInputs, MalformedEditCloudsAreRejected) {
  const auto valid = makeCloud({{2.0F, 0.0F, 0.0F}});
  std::vector<Cloud> invalid;
  auto cloud = valid;
  cloud.width = 0;
  cloud.height = 2;
  cloud.row_step = 0;
  cloud.data.clear();
  invalid.push_back(cloud); // previously divided by zero
  cloud = valid;
  cloud.height = 0;
  invalid.push_back(cloud);
  cloud = valid;
  cloud.fields[0].count = 2; // PCL will not map a vector field to scalar x
  invalid.push_back(cloud);
  cloud = valid;
  cloud.fields.push_back(cloud.fields[0]);
  invalid.push_back(cloud);
  cloud = valid;
  cloud.fields[0].offset = cloud.point_step;
  invalid.push_back(cloud);
  cloud = valid;
  cloud.fields[0].datatype = Field::FLOAT64;
  invalid.push_back(cloud);
  cloud = valid;
  cloud.data.resize(8);
  invalid.push_back(cloud);
  cloud = valid;
  cloud.is_bigendian = !cloud.is_bigendian;
  invalid.push_back(cloud);
  cloud = valid;
  cloud.header.stamp.sec = -1;
  invalid.push_back(cloud);
  cloud = valid;
  cloud.header.stamp.nanosec = 1000000000U;
  invalid.push_back(cloud);
  cloud = valid;
  cloud.header.frame_id.clear();
  invalid.push_back(cloud);
  for (size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_FALSE(edit<Add>(invalid[i]));
    EXPECT_FALSE(edit<Remove>(invalid[i]));
  }
  EXPECT_TRUE(snapshot()->empty());
}

TEST_F(MappingInputs, PaddedOrganizedCloudPreservesCoordinates) {
  auto cloud = makeCloud({{1.0F, 0.0F, 0.0F},
                          {2.0F, 0.0F, 0.0F},
                          {3.0F, 0.0F, 0.0F},
                          {4.0F, 0.0F, 0.0F}});
  cloud.width = 2;
  cloud.height = 2;
  cloud.row_step = 2 * cloud.point_step + 8;
  cloud.data.insert(cloud.data.begin() + 2 * cloud.point_step, 8, 0);
  cloud.data.resize(cloud.height * cloud.row_step);
  ASSERT_TRUE(edit<Add>(cloud));
  const auto grid = snapshot();
  EXPECT_EQ(grid->activeVoxelCount(), 4U);
  for (int x = 1; x <= 4; ++x)
    EXPECT_TRUE(grid->tree().isValueOn(openvdb::Coord(x, 0, 0)));
}

TEST_F(MappingInputs, GridEditsUseCloudFrame) {
  auto broadcaster_node = std::make_shared<rclcpp::Node>("mapping_input_tf");
  tf2_ros::StaticTransformBroadcaster broadcaster(broadcaster_node);
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.child_frame_id = "sensor";
  transform.transform.translation.x = 3.0;
  transform.transform.rotation.w = std::sqrt(0.5);
  transform.transform.rotation.z = std::sqrt(0.5);
  broadcaster.sendTransform(transform);

  const auto cloud = makeCloud({{1.0F, 0.0F, 0.0F}}, "sensor");
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool added = false;
  while (!added && std::chrono::steady_clock::now() < deadline) {
    added = edit<Add>(cloud);
    if (!added)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ASSERT_TRUE(added);
  auto grid = snapshot();
  EXPECT_TRUE(grid->tree().isValueOn(openvdb::Coord(3, 1, 0)));
  EXPECT_FALSE(grid->tree().isValueOn(openvdb::Coord(1, 0, 0)));
  ASSERT_TRUE(edit<Remove>(cloud));
  grid = snapshot();
  EXPECT_EQ(grid->activeVoxelCount(), 0U);
  EXPECT_LT(grid->tree().getValue(openvdb::Coord(3, 1, 0)), 0.0F);
  EXPECT_FALSE(edit<Add>(makeCloud({{2.0F, 0.0F, 0.0F}}, "missing_tf")));
}

TEST_F(MappingInputs, RejectedCloudDoesNotResetReplay) {
  feed(makeCloud({{1.0F, 0.0F, 0.0F}}, "map", 10));
  feed(makeCloud({{2.0F, 0.0F, 0.0F}}, "missing_tf", 100));
  feed(makeCloud({{3.0F, 0.0F, 0.0F}}, "map", 11));
  auto grid = snapshot();
  EXPECT_TRUE(grid->tree().isValueOn(openvdb::Coord(1, 0, 0)));
  EXPECT_TRUE(grid->tree().isValueOn(openvdb::Coord(3, 0, 0)));
  // A real rewind still starts a new map session.
  feed(makeCloud({{4.0F, 0.0F, 0.0F}}, "map", 1));
  grid = snapshot();
  EXPECT_EQ(grid->activeVoxelCount(), 1U);
  EXPECT_TRUE(grid->tree().isValueOn(openvdb::Coord(4, 0, 0)));
}

TEST_F(MappingInputs, MalformedSensorCloudsDoNotModifyMap) {
  auto cloud = makeCloud({{1.0F, 0.0F, 0.0F}});
  cloud.header.stamp.sec = -1;
  EXPECT_NO_THROW(feed(cloud));
  cloud = makeCloud({{1.0F, 0.0F, 0.0F}});
  cloud.fields[0].count = 2;
  feed(cloud);
  cloud = makeCloud({{1.0F, 0.0F, 0.0F}});
  cloud.is_bigendian = !cloud.is_bigendian;
  feed(cloud);
  EXPECT_TRUE(snapshot()->empty());
}

TEST_F(MappingInputs, HitAndClearingSourcesKeepTheirRoles) {
  feed(makeCloud({{2.0F, 0.0F, 0.0F}}));
  auto grid = snapshot();
  EXPECT_TRUE(grid->tree().isValueOn(openvdb::Coord(2, 0, 0)));
  EXPECT_FLOAT_EQ(grid->tree().getValue(openvdb::Coord(1, 0, 0)), 0.0F);
  feed(makeCloud({{3.0F, 0.0F, 0.0F}}), "clear");
  grid = snapshot();
  EXPECT_FALSE(grid->tree().isValueOn(openvdb::Coord(3, 0, 0)));
  EXPECT_LT(grid->tree().getValue(openvdb::Coord(1, 0, 0)), 0.0F);
  EXPECT_LT(grid->tree().getValue(openvdb::Coord(2, 0, 0)),
            std::log(0.7 / 0.3));

  // Free-space confidence must survive full-section transport and application.
  auto section = std::make_shared<vdb_mapping_interfaces::msg::UpdateGrid>();
  section->header.frame_id = "map";
  section->map = decoder->gridToByteArray<Mapper::VDBMapT::GridT>(grid);
  auto remote = std::make_shared<vdb_mapping_ros2::RemoteSource>();
  remote->active = true;
  mapper->resetMap();
  mapper->mapFullSectionCallback(section, remote);
  EXPECT_FLOAT_EQ(snapshot()->tree().getValue(openvdb::Coord(1, 0, 0)),
                  grid->tree().getValue(openvdb::Coord(1, 0, 0)));
}

TEST_F(MappingInputs, PretransformedHitsUseSensorOriginForRange) {
  auto broadcaster_node = std::make_shared<rclcpp::Node>("mapping_range_tf");
  tf2_ros::StaticTransformBroadcaster broadcaster(broadcaster_node);
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.child_frame_id = "range_sensor";
  transform.transform.translation.x = 6.0;
  transform.transform.rotation.w = 1.0;
  broadcaster.sendTransform(transform);
  const auto probe = makeCloud({{0.0F, 0.0F, 0.0F}}, "range_sensor");
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool ready = false;
  while (!ready && std::chrono::steady_clock::now() < deadline) {
    ready = edit<Remove>(probe);
    if (!ready)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ASSERT_TRUE(ready);
  mapper->resetMap();

  // These coordinates already use map_frame. The negative endpoint is 12 m
  // from the sensor and must be dropped despite being only 6 m from map zero.
  feed(makeCloud({{-6.0F, 0.0F, 0.0F}, {9.0F, 0.0F, 0.0F}}), "hits",
       "range_sensor");
  const auto grid = snapshot();
  EXPECT_EQ(grid->activeVoxelCount(), 1U);
  EXPECT_TRUE(grid->tree().isValueOn(openvdb::Coord(9, 0, 0)));
  EXPECT_FALSE(grid->tree().isValueOn(openvdb::Coord(-6, 0, 0)));
}
} // namespace
