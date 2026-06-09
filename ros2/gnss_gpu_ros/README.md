# gnss_gpu_ros — robust GNSS fix filtering for ROS 2

A small `ament_python` package that brings the trajectory post-processing
ideas validated on the [GSDC2023 Kaggle challenge](../../docs/gsdc2023_solution.md)
to outdoor robots, in causal (streaming) form:

- **Hampel spike gate** — trailing-window median/MAD outlier rejection.
  Offline, the same idea cut the worst frame-to-frame jump by **93%** with
  zero false positives across 41 driving trips.
- **Constant-velocity Kalman filter** — per-axis forward smoothing, the
  causal half of the offline RTS smoother layer (**39/41 trips improved**).

GNSS spikes from multipath and NLOS are a classic failure mode for outdoor
robot localization — a single 50 m jump can wreck an EKF fusion or a
`robot_localization` setup. This node gates those spikes *before* they reach
your fusion stack.

```text
        NavSatFix              NavSatFix (gated + smoothed)
fix ──▶ robust_navsat_filter ──▶ fix_filtered
                              └▶ path_filtered   (nav_msgs/Path, for RViz)
```

## Build

Tested on ROS 2 Jazzy. Only `rclpy`, `sensor_msgs`, `nav_msgs`,
`geometry_msgs`, and NumPy are required — nothing from the CUDA side of the
repository.

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/rsasaki0109/gnss_gpu.git
cd ~/ros2_ws
colcon build --packages-select gnss_gpu_ros
source install/setup.bash
```

## Run

```bash
ros2 launch gnss_gpu_ros robust_navsat_filter.launch.py
# or, remapping your driver's fix topic:
ros2 run gnss_gpu_ros robust_navsat_filter --ros-args -r fix:=/your_gnss_driver/fix
```

## Topics

| Topic | Type | Direction |
|---|---|---|
| `fix` | `sensor_msgs/NavSatFix` | subscribe |
| `fix_filtered` | `sensor_msgs/NavSatFix` | publish |
| `path_filtered` | `nav_msgs/Path` | publish (local East/North plane anchored at the first fix) |

## Parameters

| Parameter | Default | Meaning |
|---|--:|---|
| `hampel_window` | 21 | trailing window length [fixes] for the spike gate |
| `hampel_k` | 2.5 | gate threshold in MAD-scaled sigmas |
| `kalman_sigma_a` | 1.0 | CV process noise (acceleration) [m/s²] |
| `kalman_sigma_z` | 1.0 | measurement noise [m] |
| `use_hampel` | true | enable the spike gate |
| `use_kalman` | true | enable the Kalman stage |
| `path_frame_id` | `map` | frame for `path_filtered` |
| `path_max_poses` | 2000 | ring-buffer cap for the path |

Tuning notes from the offline A/Bs: raising `kalman_sigma_z` to 3 starts to
over-smooth; lowering `kalman_sigma_a` to 0.5 over-trusts the motion model on
aggressive driving. The defaults (1.0/1.0) were the sweet spot.

## Tests

The filter math is plain NumPy (`gnss_gpu_ros/filters.py`) and unit-tested
without ROS:

```bash
cd ros2/gnss_gpu_ros && PYTHONPATH=. python3 -m pytest test/ -q
```

## See also

- [GSDC2023 solution write-up](../../docs/gsdc2023_solution.md) — where these
  layers come from and what each was worth offline
- [gnss_gpu](https://github.com/rsasaki0109/gnss_gpu) — the parent project:
  GPU particle filters, ray-traced NLOS, robust SPP
