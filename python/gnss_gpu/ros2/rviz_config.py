"""Helper to generate RViz2 configuration for GNSS GPU visualisation."""

import os

_RVIZ_CONFIG_YAML = """\
Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/PointCloud2
      Name: Particles
      Enabled: true
      Topic:
        Value: /gnss/particles
        Depth: 5
        Reliability Policy: Reliable
      Size (Pixels): 2
      Color Transformer: FlatColor
      Color: 0; 255; 0
      Style: Points
      Alpha: 0.5
      Decay Time: 0
      Min Color: 0; 0; 0
      Max Color: 255; 255; 255
      Use Fixed Frame: true

    - Class: rviz_default_plugins/PointCloud2
      Name: PF Fix
      Enabled: true
      Topic:
        Value: /gnss/fix_pf
        Depth: 5
        Reliability Policy: Reliable
      Size (Pixels): 10
      Color Transformer: FlatColor
      Color: 255; 0; 0
      Style: Points
      Alpha: 1.0

    - Class: rviz_default_plugins/MarkerArray
      Name: Skyplot
      Enabled: true
      Topic:
        Value: /gnss/skyplot
        Depth: 5
        Reliability Policy: Reliable

  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: map
    Frame Rate: 30

  Tools:
    - Class: rviz_default_plugins/MoveCamera

  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 200
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Name: Current View
      Near Clip Distance: 0.01
      Pitch: 0.7854
      Yaw: 0.7854
"""


def generate_rviz_config(output_path="gnss_gpu.rviz"):
    """Write an RViz2 configuration file for GNSS GPU visualisation.

    Parameters
    ----------
    output_path : str
        Path where the .rviz file will be written.

    Returns
    -------
    str
        The absolute path of the written file.
    """
    abs_path = os.path.abspath(output_path)
    with open(abs_path, "w") as f:
        f.write(_RVIZ_CONFIG_YAML)
    return abs_path


def get_rviz_config_string():
    """Return the RViz2 configuration as a YAML string.

    Returns
    -------
    str
        YAML configuration text.
    """
    return _RVIZ_CONFIG_YAML


if __name__ == "__main__":
    path = generate_rviz_config()
    print(f"RViz2 config written to {path}")
