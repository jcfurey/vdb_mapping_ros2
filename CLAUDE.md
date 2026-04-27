# CLAUDE.md - AI Assistant Guide for vdb_mapping_ros2

## Project Overview

ROS 2 wrapper for the [vdb_mapping](https://github.com/fzi-forschungszentrum-informatik/vdb_mapping) library, providing fast volumetric 3D mapping using OpenVDB. Developed at FZI Forschungszentrum Informatik. Licensed under Apache 2.0.

The system integrates LiDAR point clouds into an occupancy grid backed by OpenVDB's sparse voxel data structure, with support for distributed/remote mapping across multiple robots.

## Repository Structure

```
vdb_mapping_ros2/                     # Repository root
├── CLAUDE.md                         # This file
├── README.md                         # User-facing documentation
├── .gitlab-ci.yml                    # CI config (Humble, Jazzy, Rolling)
├── vdb_mapping_ros2/                 # Main ROS 2 package
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── include/vdb_mapping_ros2/
│   │   ├── VDBMappingROS2.hpp        # Main node class declarations
│   │   └── VDBMappingTools.hpp       # Static helpers for visualization output
│   ├── src/
│   │   ├── VDBMappingROS2.cpp              # Main node implementation + composable registration
│   │   └── vdb_mapping_ros_node.cpp        # Standalone node entry point
│   ├── launch/
│   │   ├── vdb_mapping_ros2.py             # Main mapping (multi-threaded)
│   │   ├── vdb_map_server_ros2.py          # Map server (single-threaded)
│   │   └── vdb_remote_mapping_ros2.py      # Remote mapping instance
│   └── config/
│       ├── vdb_params.yaml                 # Main mapping parameters
│       ├── vdb_map_server_params.yaml      # Map server parameters
│       └── vdb_remote_params.yaml          # Remote instance parameters
└── vdb_mapping_interfaces/           # Interface definitions package
    ├── CMakeLists.txt
    ├── package.xml
    ├── msg/                          # 3 custom messages
    │   ├── BoundingBox.msg
    │   ├── UpdateGrid.msg
    │   └── Ray.msg
    └── srv/                          # 10 custom services
        ├── LoadMap.srv, LoadMapFromPCD.srv
        ├── Raytrace.srv, BatchRaytrace.srv
        ├── GetMapSection.srv
        ├── AddPointsToGrid.srv, RemovePointsFromGrid.srv
        ├── AddArtificialAreas.srv
        ├── TriggerMapSectionUpdate.srv
        └── ToggleRemoteSource.srv
```

## Architecture

### Two ROS 2 Packages

1. **vdb_mapping_interfaces** - Message and service definitions only. Built with `rosidl_default_generators`. No C++ logic. Dependencies: `geometry_msgs`, `nav_msgs`, `sensor_msgs`, `std_msgs`.

2. **vdb_mapping_ros2** - The actual mapping node. Header-only template library + composable node component. Depends on `vdb_mapping` (core library), `vdb_mapping_interfaces`, and standard ROS 2 packages.

### Key Classes

- **`VDBMappingROS2`** (`VDBMappingROS2.hpp` / `.cpp`) - Main node class. Extends `rclcpp::Node`. Handles all ROS integration: subscriptions, publishers, services, TF2 transforms, timers. The mapping backend is fixed to `vdb_mapping::OccupancyVDBMapping` via the `VDBMapT` type alias.

- **`VDBMappingTools<VDBMappingT>`** (`VDBMappingTools.hpp`) - Static utility class for converting VDB grids into ROS visualization formats (Marker, PointCloud2, OccupancyGrid). Includes height-based color coding and occupancy grid smoothing.

- **`RemoteSource`** / **`SensorSource`** (structs in `VDBMappingROS2.hpp`) - Configuration structs for remote mapping sources and local sensor inputs.

### Node Execution Model

The node runs as a **composable ROS 2 component** (`rclcpp_components`). Launch files use `ComposableNodeContainer` to load it. The main launch uses `component_container_mt` (multi-threaded executor). Three callback groups provide thread isolation:
- `m_accumulation_cb_group` - Sensor data accumulation
- `m_visualization_cb_group` - Map visualization publishing
- `m_remote_cb_group` - Remote mapping operations

### Core External Dependency

The `vdb_mapping` library (not in this repo) provides the actual mapping algorithms. This package is purely the ROS 2 wrapper. OpenVDB 5.0+ is required.

## Build & Development

### Build Commands

```bash
# Standard colcon build (from workspace root, not this repo)
colcon build --packages-select vdb_mapping_interfaces vdb_mapping_ros2

# Build with specific packages up-to (includes dependencies)
colcon build --packages-up-to vdb_mapping_ros2

# Install dependencies first
rosdep install --from-paths src --ignore-src -r -y
```

### Build Configuration

- **C++ Standard**: C++17 (`cxx_std_17`)
- **Build type**: Defaults to Release if not specified
- **Compiler flags**: `-Wall -Wextra -Wpedantic` (GCC/Clang)
- **Build system**: ament_cmake for both packages
- `vdb_mapping_ros2` is the single SHARED library target. The composable node is registered from inside its sources.

### Running

```bash
ros2 launch vdb_mapping_ros2 vdb_mapping_ros2.py        # Main mapping
ros2 launch vdb_mapping_ros2 vdb_map_server_ros2.py     # Map server
ros2 launch vdb_mapping_ros2 vdb_remote_mapping_ros2.py # Remote mapping
```

## CI/CD

GitLab CI pipeline testing three ROS 2 distributions:
- **Humble** (Ubuntu 22.04)
- **Jazzy** (Ubuntu 24.04, Clang Format 18)
- **Rolling** (Ubuntu 24.04, Clang Format 18)

Pipeline is inherited from an external `continuous_integration/ci_scripts` project. No local test suite exists in the repository.

## Code Conventions

### Naming
- **Classes**: PascalCase (`VDBMappingROS2`, `VDBMappingTools`)
- **Member variables**: `m_` prefix with snake_case (`m_map_frame`, `m_vdb_map`, `m_tf_buffer`)
- **Methods**: camelCase (`publishMap`, `cloudCallback`, `setUpVDBMap`)
- **Service callbacks**: camelCase ending in `Callback` (`resetMapCallback`, `raytraceCallback`)
- **Setup methods**: `setUp` prefix (`setUpVDBMap`, `setUpLocalSources`, `setUpRemoteSources`)
- **Parameters**: snake_case in YAML and `declare_parameter` calls (`map_frame`, `max_range`, `prob_hit`)
- **ROS topics**: snake_case with `~/` node-relative prefix (`~/vdb_map_visualization`, `~/vdb_map_pointcloud`)

### File Naming
- Headers: PascalCase `.hpp` files (`VDBMappingROS2.hpp`)
- Sources: PascalCase `.cpp` for the main implementation (`VDBMappingROS2.cpp`); snake_case for entry points (`vdb_mapping_ros_node.cpp`)
- Launch files: snake_case Python files (`vdb_mapping_ros2.py`)
- Config files: snake_case YAML files (`vdb_params.yaml`)

### Code Style
- All source files begin with an Emacs mode line: `// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-`
- Apache 2.0 license block at the top of every source file (between `-- BEGIN LICENSE BLOCK --` and `-- END LICENSE BLOCK --` markers)
- Doxygen-style comments with `\brief`, `\param`, `\returns` for public methods
- Include guard format: `#ifndef VDB_MAPPING_ROS2_FILENAME_H_INCLUDED`
- Alignment: values aligned with spaces in assignments (e.g., `m_config.fast_mode = ...`)
- Clang Format is enforced in CI (version 18 on Jazzy/Rolling)
- Indentation: 2 spaces (no tabs)

### File Layout
The main class is split into `VDBMappingROS2.hpp` (declarations) and `VDBMappingROS2.cpp` (definitions + composable-node registration). When adding a new method, declare it in the header and implement it in the .cpp.

### Parameter Declaration Pattern
Parameters follow a consistent `declare_parameter` / `get_parameter` two-step pattern:
```cpp
this->declare_parameter<type>("param_name", default_value);
this->get_parameter("param_name", m_member_variable);
```

## Adding New Features

### Adding a new ROS service
1. Define the `.srv` file in `vdb_mapping_interfaces/srv/`
2. Add it to `rosidl_generate_interfaces()` in `vdb_mapping_interfaces/CMakeLists.txt`
3. Add the include in `VDBMappingROS2.hpp`
4. Declare the callback in `VDBMappingROS2.hpp` and implement it in `VDBMappingROS2.cpp`
5. Register the service in `setUpServices()`

### Adding a new ROS message
1. Define the `.msg` file in `vdb_mapping_interfaces/msg/`
2. Add it to `rosidl_generate_interfaces()` in `vdb_mapping_interfaces/CMakeLists.txt`
3. If it depends on new message packages, add them to both `CMakeLists.txt` and `package.xml`

### Adding a new parameter
1. Add the parameter to the relevant config YAML file(s)
2. Declare and read it in the appropriate `setUp*()` method in `VDBMappingROS2.cpp`
3. Add a member variable with `m_` prefix
4. Document it in `README.md`

### Adding a new dependency
1. Add `find_package()` in `CMakeLists.txt`
2. Add to `THIS_PACKAGE_INCLUDE_DEPENDS` list in `CMakeLists.txt`
3. Add `<depend>` entry in `package.xml`
