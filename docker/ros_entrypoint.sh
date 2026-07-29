#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
source /opt/gnss_gpu_ws/install/setup.bash
exec "$@"
