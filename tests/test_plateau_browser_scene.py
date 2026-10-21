"""Independent geometry and datum checks for the static browser data exporter."""

from collections import Counter
import csv
import gzip
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "plateau_browser_export", ROOT / "tools/build_plateau_browser_scene.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
try:
    HAS_SHAPELY = tuple(
        map(int, importlib.metadata.version("shapely").split(".")[:2])
    ) >= (2, 1)
except importlib.metadata.PackageNotFoundError:
    HAS_SHAPELY = False


class DatumTests(unittest.TestCase):
    def test_bilinear_geoid_and_missing_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "grid.asc"
            path.write_text(
                "35 139 .016667 .025 2 2 1 test\n30 34 38 42\n", encoding="ascii"
            )
            grid = builder.GeoidGrid(path)
            self.assertAlmostEqual(grid(35 + 1 / 120, 139 + 0.0125), 36, places=9)
            self.assertAlmostEqual(grid.dy, 1 / 60, places=15)
            with self.assertRaisesRegex(ValueError, "outside"):
                grid(0, 0)
            path.write_text(
                "35 139 .016667 .025 2 2 1 test\n30 999 38 42\n", encoding="ascii"
            )
            with self.assertRaisesRegex(ValueError, "Missing"):
                builder.GeoidGrid(path)(35 + 1 / 120, 139 + 0.0125)

    def test_wgs84_known_axes_and_orthogonal_local_frame(self):
        self.assertEqual(builder.ecef(0, 0, 0), (6378137, 0, 0))
        point = builder.ecef(35.62931853, 139.78712595, 44.6995)
        for actual, expected in zip(point, (-3963426.7981, 3350882.1576, 3694865.5458)):
            self.assertAlmostEqual(actual, expected, delta=0.0002)
        axes = builder.frame(35.6, 139.8)
        for i, a in enumerate(axes):
            for j, b in enumerate(axes):
                self.assertAlmostEqual(
                    sum(x * y for x, y in zip(a, b)), int(i == j), places=14
                )

    def test_excluded_path_is_rejected_before_opening(self):
        with self.assertRaises(ValueError):
            builder.allowed(ROOT / ".claude" / "do-not-read")


@unittest.skipUnless(HAS_SHAPELY, "Exporter geometry tests require Shapely >= 2.1")
class GeometryTests(unittest.TestCase):
    def test_concavity_and_hole_survive_constrained_triangulation(self):
        from shapely import Polygon, Point, union_all

        outer = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (6, 10, 0), (6, 6, 0), (0, 6, 0)]
        hole = [(1, 1, 0), (3, 1, 0), (3, 3, 0), (1, 3, 0)]
        triangles = builder.triangulate([outer, hole])
        merged = union_all([Polygon([(p[0], p[1]) for p in t]) for t in triangles])
        expected = Polygon([p[:2] for p in outer], [[p[:2] for p in hole]])
        self.assertAlmostEqual(expected.area, 72)
        self.assertLess(merged.symmetric_difference(expected).area, 1e-12)
        self.assertFalse(merged.covers(Point(2, 2)))
        self.assertFalse(merged.covers(Point(2, 8)))
        self.assertTrue(set(outer + hole).issubset({p for t in triangles for p in t}))

    def test_vertical_face_and_self_touching_source_repair(self):
        from shapely import Polygon, make_valid, union_all

        ring = [(0, 0), (4, 0), (4, 4), (2, 4), (2, 2), (0, 4), (2, 2), (0, 0)]
        points = [(7, y, z) for y, z in ring]
        counts = Counter()
        triangles = builder.triangulate([points], counts)
        self.assertTrue(triangles)
        self.assertEqual(counts["topology_repaired_faces"], 1)
        self.assertTrue(all(p[0] == 7 for t in triangles for p in t))
        actual = union_all([Polygon([(p[1], p[2]) for p in t]) for t in triangles])
        expected = make_valid(Polygon(ring), method="linework")
        self.assertLess(actual.symmetric_difference(expected).area, 1e-12)

    def test_highest_lod_is_selected_per_component_without_duplicate_low_lod(self):
        def polygon(height):
            coordinates = f"35.6 139.77 {height} 35.60001 139.77 {height} 35.60001 139.77001 {height} 35.6 139.77001 {height} 35.6 139.77 {height}"
            return f"<gml:Polygon><gml:exterior><gml:LinearRing><gml:posList>{coordinates}</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon>"

        source = f"""<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0" xmlns:gml="http://www.opengis.net/gml" xmlns:bldg="http://www.opengis.net/citygml/building/2.0">
          <gml:boundedBy><gml:Envelope srsName="http://www.opengis.net/def/crs/EPSG/0/6697"/></gml:boundedBy>
          <core:cityObjectMember><bldg:Building gml:id="parent">
          <bldg:lod1Solid>{polygon(1)}</bldg:lod1Solid><bldg:lod2MultiSurface>{polygon(2)}</bldg:lod2MultiSurface>
          <bldg:consistsOfBuildingPart><bldg:BuildingPart gml:id="part"><bldg:lod1Solid>{polygon(3)}</bldg:lod1Solid></bldg:BuildingPart></bldg:consistsOfBuildingPart>
          </bldg:Building></core:cityObjectMember></core:CityModel>"""
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "test.gml").write_text(source)
            reference = folder / "reference.csv"
            with reference.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "GPS TOW (s)",
                        "GPS Week",
                        "Latitude (deg)",
                        "Longitude (deg)",
                        "ECEF X (m)",
                        "ECEF Y (m)",
                        "ECEF Z (m)",
                    ]
                )
                writer.writerow(
                    [1000, 2032, 35.6, 139.77, *builder.ecef(35.6, 139.77, 0)]
                )
            geoid = folder / "geoid.asc"
            geoid.write_text("35 139 1 1 2 2 1 test\n0 0 0 0\n")
            nav = folder / "base.nav"
            nav.write_text("independent LoD fixture")
            with patch.object(builder, "read_navigation", return_value=[]):
                manifest = builder.build(folder, reference, nav, geoid, folder / "out")
            scene = json.loads(
                gzip.decompress((folder / "out/odaiba-scene.json.gz").read_bytes())
            )
            self.assertEqual(manifest["stats"]["triangles"], 4)
            self.assertEqual(manifest["stats"]["polygons"], 2)
            self.assertEqual(
                [(o["id"], o["lod"]) for o in scene["objects"]],
                [("parent", 2), ("part", 1)],
            )


if __name__ == "__main__":
    unittest.main()
