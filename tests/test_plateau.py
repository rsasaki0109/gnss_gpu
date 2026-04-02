"""Tests for PLATEAU CityGML loader."""

import os
import sys
import tempfile
import textwrap

import numpy as np
import pytest

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from gnss_gpu.io.citygml import parse_citygml, Building
from gnss_gpu.io.plateau import PlateauLoader, load_plateau
from gnss_gpu.raytrace import BuildingModel

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MINIMAL_CITYGML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <core:CityModel
      xmlns:core="http://www.opengis.net/citygml/2.0"
      xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
      xmlns:gml="http://www.opengis.net/gml">
      <core:cityObjectMember>
        <bldg:Building gml:id="test_bldg_1">
          <bldg:lod1Solid>
            <gml:Solid>
              <gml:exterior>
                <gml:CompositeSurface>
                  <gml:surfaceMember>
                    <gml:Polygon>
                      <gml:exterior>
                        <gml:LinearRing>
                          <gml:posList>
                            0.0 0.0 0.0
                            0.0 10.0 0.0
                            10.0 10.0 0.0
                            10.0 0.0 0.0
                            0.0 0.0 0.0
                          </gml:posList>
                        </gml:LinearRing>
                      </gml:exterior>
                    </gml:Polygon>
                  </gml:surfaceMember>
                  <gml:surfaceMember>
                    <gml:Polygon>
                      <gml:exterior>
                        <gml:LinearRing>
                          <gml:posList>
                            0.0 0.0 20.0
                            10.0 0.0 20.0
                            10.0 10.0 20.0
                            0.0 10.0 20.0
                            0.0 0.0 20.0
                          </gml:posList>
                        </gml:LinearRing>
                      </gml:exterior>
                    </gml:Polygon>
                  </gml:surfaceMember>
                </gml:CompositeSurface>
              </gml:exterior>
            </gml:Solid>
          </bldg:lod1Solid>
        </bldg:Building>
      </core:cityObjectMember>
    </core:CityModel>
""")


GEOGRAPHIC_CITYGML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <core:CityModel
      xmlns:core="http://www.opengis.net/citygml/2.0"
      xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
      xmlns:gml="http://www.opengis.net/gml">
      <core:cityObjectMember>
        <bldg:Building gml:id="geo_bldg_1">
          <bldg:lod1Solid>
            <gml:Solid>
              <gml:exterior>
                <gml:CompositeSurface>
                  <gml:surfaceMember>
                    <gml:Polygon>
                      <gml:exterior>
                        <gml:LinearRing>
                          <gml:posList>
                            35.68120 139.76710 5.0
                            35.68120 139.76720 5.0
                            35.68130 139.76720 5.0
                            35.68130 139.76710 5.0
                            35.68120 139.76710 5.0
                          </gml:posList>
                        </gml:LinearRing>
                      </gml:exterior>
                    </gml:Polygon>
                  </gml:surfaceMember>
                  <gml:surfaceMember>
                    <gml:Polygon>
                      <gml:exterior>
                        <gml:LinearRing>
                          <gml:posList>
                            35.68120 139.76710 25.0
                            35.68130 139.76710 25.0
                            35.68130 139.76720 25.0
                            35.68120 139.76720 25.0
                            35.68120 139.76710 25.0
                          </gml:posList>
                        </gml:LinearRing>
                      </gml:exterior>
                    </gml:Polygon>
                  </gml:surfaceMember>
                </gml:CompositeSurface>
              </gml:exterior>
            </gml:Solid>
          </bldg:lod1Solid>
        </bldg:Building>
      </core:cityObjectMember>
    </core:CityModel>
""")


@pytest.fixture
def citygml_file(tmp_path):
    """Write the minimal CityGML to a temporary file."""
    p = tmp_path / "test.gml"
    p.write_text(MINIMAL_CITYGML, encoding="utf-8")
    return p


@pytest.fixture
def geographic_citygml_file(tmp_path):
    """Write a minimal CityGML that stores coordinates in lat/lon/height."""
    p = tmp_path / "geographic.gml"
    p.write_text(GEOGRAPHIC_CITYGML, encoding="utf-8")
    return p


@pytest.fixture
def sample_gml_path():
    """Path to the sample PLATEAU GML shipped in the data/ directory."""
    return os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_plateau.gml"
    )


# ---------------------------------------------------------------------------
# CityGML parser tests
# ---------------------------------------------------------------------------

class TestCityGMLParser:

    def test_parse_buildings(self, citygml_file):
        buildings = parse_citygml(citygml_file)
        assert len(buildings) == 1
        b = buildings[0]
        assert b.id == "test_bldg_1"
        assert b.lod == 1
        assert len(b.polygons) == 2

    def test_polygon_shape(self, citygml_file):
        buildings = parse_citygml(citygml_file)
        for poly in buildings[0].polygons:
            assert poly.ndim == 2
            assert poly.shape[1] == 3
            # 5 points (closed ring)
            assert poly.shape[0] == 5

    def test_parse_sample_plateau(self, sample_gml_path):
        if not os.path.exists(sample_gml_path):
            pytest.skip("sample_plateau.gml not found")
        buildings = parse_citygml(sample_gml_path)
        assert len(buildings) == 3
        # Each building should have 6 polygons (6 faces of a box)
        for b in buildings:
            assert len(b.polygons) == 6


# ---------------------------------------------------------------------------
# Coordinate conversion tests
# ---------------------------------------------------------------------------

class TestCoordinateConversion:

    def test_gauss_kruger_roundtrip_origin(self):
        """At the zone origin, y=0 and x=0 should map back to the origin."""
        loader = PlateauLoader(zone=9)
        lat, lon = loader._gauss_kruger_inverse(
            0.0, 0.0, loader._lat0, loader._lon0
        )
        assert abs(np.degrees(lat) - 36.0) < 1e-8
        assert abs(np.degrees(lon) - 139.83333) < 1e-4

    def test_tokyo_station_approximate(self):
        """Verify that coordinates near Tokyo Station produce sensible ECEF.

        Tokyo Station is approximately at:
          lat ~35.6812, lon ~139.7671, alt ~5 m
        In zone 9, this corresponds to roughly:
          Y ~ -35295 m (southing from lat 36)
          X ~ -2835 m (westing from lon 139.8333)
        """
        loader = PlateauLoader(zone=9)
        # Use approximate plane-rect values for Tokyo Station
        y_north = -35295.0
        x_east = -2835.0
        z_up = 5.0

        lat, lon = loader._gauss_kruger_inverse(
            y_north, x_east, loader._lat0, loader._lon0
        )
        lat_deg = np.degrees(lat)
        lon_deg = np.degrees(lon)

        # Should be close to Tokyo Station
        assert abs(lat_deg - 35.68) < 0.05, f"lat={lat_deg}"
        assert abs(lon_deg - 139.77) < 0.05, f"lon={lon_deg}"

    def test_lla_to_ecef_known_point(self):
        """Check ECEF conversion for a known point.

        Tokyo Station (~35.6812N, 139.7671E, 5m) should produce ECEF
        approximately:
          X ~ -3959000 m
          Y ~ +3352000 m
          Z ~ +3697000 m
        """
        loader = PlateauLoader(zone=9)
        lat = np.radians(35.6812)
        lon = np.radians(139.7671)
        alt = 5.0

        ecef = loader._lla_to_ecef(lat, lon, alt)

        assert abs(ecef[0] - (-3959000)) < 5000, f"X={ecef[0]}"
        assert abs(ecef[1] - 3352000) < 5000, f"Y={ecef[1]}"
        assert abs(ecef[2] - 3697000) < 5000, f"Z={ecef[2]}"

    def test_invalid_zone(self):
        with pytest.raises(ValueError, match="zone must be 1-19"):
            PlateauLoader(zone=0)
        with pytest.raises(ValueError, match="zone must be 1-19"):
            PlateauLoader(zone=20)


# ---------------------------------------------------------------------------
# Triangulation tests
# ---------------------------------------------------------------------------

class TestTriangulation:

    def test_triangle_from_quad(self):
        """A quad (4 unique vertices + closing) should produce 2 triangles."""
        coords = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0],  # closing vertex
        ], dtype=np.float64)

        tris = PlateauLoader._polygon_to_triangles(coords)
        assert tris is not None
        assert tris.shape == (2, 3, 3)

    def test_triangle_from_triangle(self):
        """A triangle (3 unique vertices + closing) should produce 1 triangle."""
        coords = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=np.float64)

        tris = PlateauLoader._polygon_to_triangles(coords)
        assert tris is not None
        assert tris.shape == (1, 3, 3)

    def test_degenerate_polygon(self):
        """A polygon with fewer than 3 unique vertices should return None."""
        coords = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=np.float64)

        tris = PlateauLoader._polygon_to_triangles(coords)
        assert tris is None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestPlateauLoader:

    def test_load_inline_citygml(self, citygml_file):
        """Load minimal CityGML and produce a BuildingModel."""
        loader = PlateauLoader(zone=9)
        model = loader.load_citygml(citygml_file)
        assert isinstance(model, BuildingModel)
        # 2 polygons, each a quad => 2 * 2 = 4 triangles
        assert model.triangles.shape == (4, 3, 3)

    def test_load_geographic_citygml(self, geographic_citygml_file):
        """Geographic lat/lon PLATEAU exports should map near Tokyo Station."""
        loader = PlateauLoader(zone=9)
        model = loader.load_citygml(geographic_citygml_file)
        assert isinstance(model, BuildingModel)
        assert model.triangles.shape == (4, 3, 3)

        expected = loader._lla_to_ecef(np.radians(35.68120), np.radians(139.76710), 5.0)
        first_vertex = model.triangles[0, 0]
        assert np.linalg.norm(first_vertex - expected) < 5.0

    def test_load_sample_plateau_file(self, sample_gml_path):
        """Load the shipped sample PLATEAU GML file."""
        if not os.path.exists(sample_gml_path):
            pytest.skip("sample_plateau.gml not found")
        loader = PlateauLoader(zone=9)
        model = loader.load_citygml(sample_gml_path)
        assert isinstance(model, BuildingModel)
        # 3 buildings * 6 faces * 2 triangles/quad = 36
        assert model.triangles.shape[0] == 36
        assert model.triangles.shape[1:] == (3, 3)

    def test_load_directory(self, tmp_path):
        """Load from a directory containing one GML file."""
        gml = tmp_path / "building.gml"
        gml.write_text(MINIMAL_CITYGML, encoding="utf-8")

        loader = PlateauLoader(zone=9)
        model = loader.load_directory(tmp_path)
        assert isinstance(model, BuildingModel)
        assert model.triangles.shape[0] > 0

    def test_load_directory_no_files(self, tmp_path):
        loader = PlateauLoader(zone=9)
        with pytest.raises(FileNotFoundError):
            loader.load_directory(tmp_path)

    def test_convenience_function_file(self, citygml_file):
        model = load_plateau(citygml_file, zone=9)
        assert isinstance(model, BuildingModel)
        assert model.triangles.shape[0] > 0

    def test_convenience_function_directory(self, tmp_path):
        gml = tmp_path / "a.gml"
        gml.write_text(MINIMAL_CITYGML, encoding="utf-8")
        model = load_plateau(tmp_path, zone=9)
        assert isinstance(model, BuildingModel)

    def test_ecef_output_reasonable(self, sample_gml_path):
        """ECEF coordinates near Tokyo should have magnitude ~6.37e6 m."""
        if not os.path.exists(sample_gml_path):
            pytest.skip("sample_plateau.gml not found")
        model = load_plateau(sample_gml_path, zone=9)
        # Check that vertices are near the Earth's surface
        for tri in model.triangles:
            for vertex in tri:
                r = np.linalg.norm(vertex)
                assert 6.3e6 < r < 6.4e6, f"vertex radius {r} out of range"
