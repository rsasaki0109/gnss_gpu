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



class NavigationTests(unittest.TestCase):
    @staticmethod
    def nav_file(path, gpga=True, health=0):
        header = ""
        if gpga:
            header = f"GPGA {2.6e-9:17.10E}{-1.5e-14:16.9E}{259200:7d}{2032:5d}          TIME SYSTEM CORR\n"
        text = header + "                                                            END OF HEADER\n"
        for system in "GECJ":
            fields = [0.0] * 28
            fields[5], fields[7], fields[8] = .005, 5153.7, 273600
            fields[17], fields[18] = (517 if system == "E" else 0), (676 if system == "C" else 2032)
            fields[20], fields[21], fields[24] = 1, (health if system == "E" else 0), 273500
            text += f"{system}01 2018 12 19 04 00 00" + "".join(f"{v:19.12E}" for v in [1e-4, 0, 0]) + "\n"
            for i in range(0, 28, 4):
                text += "    " + "".join(f"{v:19.12E}" for v in fields[i:i+4]) + "\n"
        path.write_text(text, encoding="ascii")

    def test_four_system_records_keep_native_weeks_and_time_corrections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.nav"
            self.nav_file(path)
            records = builder.read_navigation(path, 273600, 274000, 2032)
            self.assertEqual([r["id"] for r in records], ["G01", "E01", "C01", "J01"])
            self.assertEqual(records[2]["parameters"][18], 676)
            self.assertEqual(records[1]["gpga"], [2.6e-9, -1.5e-14, 259200, 2032])
            self.assertEqual(len(builder.read_navigation(path, 273600, 274000, 2032, systems="G")), 1)
            self.nav_file(path, health=1)
            self.assertNotIn("E01", [r["id"] for r in builder.read_navigation(path, 273600, 274000, 2032)])
            self.nav_file(path, gpga=False)
            with self.assertRaisesRegex(ValueError, "GPGA"):
                builder.read_navigation(path, 273600, 274000, 2032)

    def test_navigation_refresh_is_reproducible_and_leaves_mesh_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            nav = folder / "mixed.nav"
            self.nav_file(nav)
            mesh = folder / "mesh.bin.gz"
            mesh.write_bytes(b"unchanged geometry")
            scene = {"week": 2032, "track": [[273600], [274000]], "navigation": [], "provenance": {}}
            info = builder.compress(folder / "scene.json.gz", json.dumps(scene).encode())
            manifest = {"scene": info, "mesh": {"file": mesh.name}, "provenance": {}}
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            builder.update_navigation(nav, folder)
            first = (folder / "scene.json.gz").read_bytes()
            builder.update_navigation(nav, folder)
            self.assertEqual((folder / "scene.json.gz").read_bytes(), first)
            self.assertEqual(mesh.read_bytes(), b"unchanged geometry")
            (folder / "scene.json.gz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                builder.update_navigation(nav, folder)



    def test_glonass_state_units_clock_sign_and_utc_frame_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "glonass.nav"
            def write(health=0, frame=273600, year=2018, leap=None):
                text = (f"{leap:6d}                                                      LEAP SECONDS\n" if leap is not None else "")
                text += "                                                            END OF HEADER\n"
                text += f"R06 {year} 12 19 04 15 00" + "".join(f"{v:19.12E}" for v in [1e-4, 2e-11, frame]) + "\n"
                for row in [[19000, 1.2, 1e-9, health], [11000, -2.3, -2e-9, -4], [12000, .8, 3e-9, 0]]:
                    text += "    " + "".join(f"{v:19.12E}" for v in row) + "\n"
                path.write_text(text, encoding="ascii")
            write()
            r = builder.read_navigation(path, 273600, 274000, 2032)[0]
            self.assertEqual(r["position"], [19000000, 11000000, 12000000])
            self.assertEqual(r["velocity"], [1200, -2300, 800])
            for actual, expected in zip(r["acceleration"], [1e-6, -2e-6, 3e-6]):
                self.assertAlmostEqual(actual, expected, places=15)
            self.assertEqual(r["clock"][0], 1e-4)
            self.assertEqual(r["utcOffset"], 18)
            self.assertEqual(r["frequencyChannel"], -4)
            self.assertEqual(r["transmission"], 2032 * 604800 + 273600)
            self.assertNotIn("parameters", r)
            for kwargs in ({"health": 1}, {"frame": 0.9999e9}):
                write(**kwargs)
                with self.assertRaisesRegex(ValueError, "No matching"):
                    builder.read_navigation(path, 273600, 274000, 2032)
            write(leap=17)
            with self.assertRaisesRegex(ValueError, "UTC offset"):
                builder.read_navigation(path, 273600, 274000, 2032)
            write(year=2019)
            with self.assertRaisesRegex(ValueError, "LEAP SECONDS"):
                builder.read_navigation(path, 273600, 274000, 2032)


if __name__ == "__main__":
    unittest.main()
