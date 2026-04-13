"""Export LOS/NLOS results to KML for Google Earth visualization.

Generates a KML file with:
  - Receiver trajectory (yellow line)
  - Per-epoch satellite rays (green=LOS, red=NLOS)
  - Satellite markers with elevation/status info
  - Receiver position markers at each epoch
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from gnss_gpu.urban_signal_sim import ecef_to_lla


def _ecef_to_lla_deg(x, y, z):
    """ECEF to (lat_deg, lon_deg, alt_m)."""
    lat, lon, alt = ecef_to_lla(x, y, z)
    return math.degrees(lat), math.degrees(lon), alt


def _add_style(doc, style_id, color, line_width=2, icon_scale=0.6,
               icon_href="http://maps.google.com/mapfiles/kml/paddle/wht-blank.png"):
    """Add a KML Style element."""
    style = ET.SubElement(doc, "Style", id=style_id)

    line = ET.SubElement(style, "LineStyle")
    ET.SubElement(line, "color").text = color
    ET.SubElement(line, "width").text = str(line_width)

    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "scale").text = str(icon_scale)
    icon = ET.SubElement(icon_style, "Icon")
    ET.SubElement(icon, "href").text = icon_href

    label = ET.SubElement(style, "LabelStyle")
    ET.SubElement(label, "scale").text = "0.7"
    ET.SubElement(label, "color").text = color


def export_kml(results, output_path, name="GNSS LOS/NLOS Visualization"):
    """Export LOS/NLOS results to KML.

    Args:
        results: list of dicts, each with:
            - rx_ecef: [3] receiver ECEF
            - sat_ecef: [n_sat, 3] satellite ECEF
            - prn_list: list of PRN numbers
            - is_los: [n_sat] bool array
            - visible: [n_sat] bool array
            - elevations: [n_sat] float array (rad)
            - excess_delays: [n_sat] float array (m)
            - epoch_label: str (optional)
        output_path: output .kml file path
        name: KML document name
    """
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = name

    # Styles
    _add_style(doc, "los_ray", "ff00d400", line_width=2)     # green (AABBGGRR)
    _add_style(doc, "nlos_ray", "ff6b6bff", line_width=3)    # red
    _add_style(doc, "mp_ray", "ff3dd9ff", line_width=1)      # yellow
    _add_style(doc, "rx_marker", "ffffffff", icon_scale=0.8,
               icon_href="http://maps.google.com/mapfiles/kml/paddle/wht-circle.png")
    _add_style(doc, "los_sat", "ff00d400", icon_scale=0.5,
               icon_href="http://maps.google.com/mapfiles/kml/paddle/grn-circle.png")
    _add_style(doc, "nlos_sat", "ff6b6bff", icon_scale=0.5,
               icon_href="http://maps.google.com/mapfiles/kml/paddle/red-circle.png")
    _add_style(doc, "trajectory", "ff00ffff", line_width=3)   # yellow

    # Trajectory folder
    if len(results) > 1:
        traj_folder = ET.SubElement(doc, "Folder")
        ET.SubElement(traj_folder, "name").text = "Trajectory"
        traj_pm = ET.SubElement(traj_folder, "Placemark")
        ET.SubElement(traj_pm, "name").text = "Receiver Path"
        ET.SubElement(traj_pm, "styleUrl").text = "#trajectory"
        traj_line = ET.SubElement(traj_pm, "LineString")
        ET.SubElement(traj_line, "altitudeMode").text = "clampToGround"
        coords = []
        for r in results:
            lat, lon, alt = _ecef_to_lla_deg(*r["rx_ecef"])
            coords.append(f"{lon},{lat},{alt}")
        ET.SubElement(traj_line, "coordinates").text = " ".join(coords)

    # Per-epoch folders
    for epoch_idx, r in enumerate(results):
        rx_ecef = r["rx_ecef"]
        sat_ecef = r["sat_ecef"]
        prn_list = r["prn_list"]
        is_los = r["is_los"]
        visible = r["visible"]
        elevations = r["elevations"]
        excess_delays = r.get("excess_delays", np.zeros(len(prn_list)))
        label = r.get("epoch_label", f"Epoch {epoch_idx}")

        rx_lat, rx_lon, rx_alt = _ecef_to_lla_deg(*rx_ecef)

        folder = ET.SubElement(doc, "Folder")
        ET.SubElement(folder, "name").text = label
        # Collapse by default if many epochs
        if len(results) > 5:
            ET.SubElement(folder, "visibility").text = "0"

        # Receiver marker
        rx_pm = ET.SubElement(folder, "Placemark")
        ET.SubElement(rx_pm, "name").text = f"RX {label}"
        ET.SubElement(rx_pm, "styleUrl").text = "#rx_marker"
        desc = f"lat: {rx_lat:.6f}\nlon: {rx_lon:.6f}\nalt: {rx_alt:.1f}m"
        ET.SubElement(rx_pm, "description").text = desc
        pt = ET.SubElement(rx_pm, "Point")
        ET.SubElement(pt, "altitudeMode").text = "clampToGround"
        ET.SubElement(pt, "coordinates").text = f"{rx_lon},{rx_lat},{rx_alt}"

        n_sat = len(prn_list)
        for i in range(n_sat):
            if not visible[i]:
                continue

            sat = sat_ecef[i]
            sat_lat, sat_lon, sat_alt = _ecef_to_lla_deg(*sat)
            el_deg = math.degrees(elevations[i])
            los = bool(is_los[i])
            mp = float(excess_delays[i])

            status = "LOS" if los else "NLOS"
            style = "#los_ray" if los else "#nlos_ray"
            # Ray line (receiver -> intermediate point ~50km along LOS)
            # Don't draw all the way to satellite orbit
            direction = np.array(sat) - np.array(rx_ecef)
            dist = np.linalg.norm(direction)
            # Show ray up to 200m altitude above receiver for visibility
            ray_length = min(50000, dist)  # 50km max
            ray_end = np.array(rx_ecef) + direction / dist * ray_length
            ray_lat, ray_lon, ray_alt = _ecef_to_lla_deg(*ray_end)

            ray_pm = ET.SubElement(folder, "Placemark")
            ray_name = f"PRN{prn_list[i]} {status} el={el_deg:.0f}°"
            if mp > 0.1:
                ray_name += f" MP={mp:.1f}m"
            ET.SubElement(ray_pm, "name").text = ray_name
            ET.SubElement(ray_pm, "styleUrl").text = style
            desc_text = f"PRN: {prn_list[i]}\nStatus: {status}\nElevation: {el_deg:.1f}°"
            if mp > 0.1:
                desc_text += f"\nMultipath delay: {mp:.1f}m"
            ET.SubElement(ray_pm, "description").text = desc_text
            line = ET.SubElement(ray_pm, "LineString")
            ET.SubElement(line, "altitudeMode").text = "absolute"
            ET.SubElement(line, "coordinates").text = (
                f"{rx_lon},{rx_lat},{rx_alt + 2} "
                f"{ray_lon},{ray_lat},{ray_alt}"
            )

    # Write KML
    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)


def export_epoch_kml(rx_ecef, sat_ecef, prn_list, result, output_path,
                     name="LOS/NLOS Single Epoch"):
    """Convenience: export a single epoch result to KML."""
    epoch_data = {
        "rx_ecef": rx_ecef,
        "sat_ecef": sat_ecef,
        "prn_list": prn_list,
        "is_los": result["is_los"],
        "visible": result["visible"],
        "elevations": result["elevations"],
        "excess_delays": result.get("excess_delays", np.zeros(len(prn_list))),
        "epoch_label": "Epoch 0",
    }
    export_kml([epoch_data], output_path, name=name)
