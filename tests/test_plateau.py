"""Tests for PLATEAU CityGML loader."""

import os
import sys
import tempfile
import textwrap

import numpy as np
import pytest

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from gnss_gpu.io.citygml import (
    Building,
    CityFeature,
    SUPPORTED_KINDS,
    parse_citygml,
)
from gnss_gpu.io.plateau import PlateauLoader, load_plateau
from gnss_gpu.raytrace import BuildingModel

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_KIND_TO_NS = {
    "bldg": ("http://www.opengis.net/citygml/building/2.0", "Building"),
    "brid": ("http://www.opengis.net/citygml/bridge/2.0", "Bridge"),
}


def _make_minimal_citygml(
    kind: str,
    gml_id: str,
    lod: int,
    polygons: list[list[tuple[float, float, float]]],
) -> str:
    """Build a minimal CityGML 2.0 document for one feature.

    ``polygons`` is a list of closed rings.  The function emits one
    ``surfaceMember`` per ring inside a ``lod<N>Solid`` of the given kind.
    """
    ns_uri, root_tag = _KIND_TO_NS[kind]

    def fmt_pos(p):
        return " ".join(f"{v:.5f}".rstrip("0").rstrip(".") or "0" for v in p)

    surface_members = "\n".join(
        f"""                  <gml:surfaceMember>
                    <gml:Polygon>
                      <gml:exterior>
                        <gml:LinearRing>
                          <gml:posList>
                            {' '.join(fmt_pos(p) for p in ring)}
                          </gml:posList>
                        </gml:LinearRing>
                      </gml:exterior>
                    </gml:Polygon>
                  </gml:surfaceMember>"""
        for ring in polygons
    )

    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <core:CityModel
          xmlns:core="http://www.opengis.net/citygml/2.0"
          xmlns:{kind}="{ns_uri}"
          xmlns:gml="http://www.opengis.net/gml">
          <core:cityObjectMember>
            <{kind}:{root_tag} gml:id="{gml_id}">
              <{kind}:lod{lod}Solid>
                <gml:Solid>
                  <gml:exterior>
                    <gml:CompositeSurface>
{surface_members}
                    </gml:CompositeSurface>
                  </gml:exterior>
                </gml:Solid>
              </{kind}:lod{lod}Solid>
            </{kind}:{root_tag}>
          </core:cityObjectMember>
        </core:CityModel>
    """)


_BOX_LOCAL = [
    [(0.0, 0.0, 0.0), (0.0, 10.0, 0.0), (10.0, 10.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    [(0.0, 0.0, 20.0), (10.0, 0.0, 20.0), (10.0, 10.0, 20.0), (0.0, 10.0, 20.0), (0.0, 0.0, 20.0)],
]
_BOX_GEO = [
    [(35.68120, 139.76710, 5.0), (35.68120, 139.76720, 5.0),
     (35.68130, 139.76720, 5.0), (35.68130, 139.76710, 5.0),
     (35.68120, 139.76710, 5.0)],
    [(35.68120, 139.76710, 25.0), (35.68130, 139.76710, 25.0),
     (35.68130, 139.76720, 25.0), (35.68120, 139.76720, 25.0),
     (35.68120, 139.76710, 25.0)],
]
_BRIDGE_DECK = [
    [(0.0, 0.0, 10.0), (0.0, 5.0, 10.0), (20.0, 5.0, 10.0), (20.0, 0.0, 10.0), (0.0, 0.0, 10.0)],
    [(0.0, 0.0, 12.0), (20.0, 0.0, 12.0), (20.0, 5.0, 12.0), (0.0, 5.0, 12.0), (0.0, 0.0, 12.0)],
]

MINIMAL_CITYGML = _make_minimal_citygml("bldg", "test_bldg_1", lod=1, polygons=_BOX_LOCAL)
GEOGRAPHIC_CITYGML = _make_minimal_citygml("bldg", "geo_bldg_1", lod=1, polygons=_BOX_GEO)
MINIMAL_BRIDGE_CITYGML = _make_minimal_citygml("brid", "test_brid_1", lod=2, polygons=_BRIDGE_DECK)


# Hand-rolled fixture with multiple ``cityObjectMember`` entries and
# non-canonical namespace prefix names / declaration order.  Exercises
# parser paths the factory above does not (factory always uses
# canonical prefixes ``core/bldg/brid/gml`` and a single member).
_MULTI_MEMBER_NONCANONICAL_CITYGML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <c:CityModel
      xmlns:g="http://www.opengis.net/gml"
      xmlns:b="http://www.opengis.net/citygml/building/2.0"
      xmlns:c="http://www.opengis.net/citygml/2.0">
      <c:cityObjectMember>
        <b:Building g:id="b_one">
          <b:lod1Solid>
            <g:Solid>
              <g:exterior>
                <g:CompositeSurface>
                  <g:surfaceMember>
                    <g:Polygon>
                      <g:exterior>
                        <g:LinearRing>
                          <g:posList>0 0 0 1 0 0 1 1 0 0 1 0 0 0 0</g:posList>
                        </g:LinearRing>
                      </g:exterior>
                    </g:Polygon>
                  </g:surfaceMember>
                </g:CompositeSurface>
              </g:exterior>
            </g:Solid>
          </b:lod1Solid>
        </b:Building>
      </c:cityObjectMember>
      <c:cityObjectMember>
        <b:Building g:id="b_two">
          <b:lod2Solid>
            <g:Solid>
              <g:exterior>
                <g:CompositeSurface>
                  <g:surfaceMember>
                    <g:Polygon>
                      <g:exterior>
                        <g:LinearRing>
                          <g:posList>10 10 0 11 10 0 11 11 0 10 11 0 10 10 0</g:posList>
                        </g:LinearRing>
                      </g:exterior>
                    </g:Polygon>
                  </g:surfaceMember>
                  <g:surfaceMember>
                    <g:Polygon>
                      <g:exterior>
                        <g:LinearRing>
                          <g:posList>10 10 5 10 11 5 11 11 5 11 10 5 10 10 5</g:posList>
                        </g:LinearRing>
                      </g:exterior>
                    </g:Polygon>
                  </g:surfaceMember>
                </g:CompositeSurface>
              </g:exterior>
            </g:Solid>
          </b:lod2Solid>
        </b:Building>
      </c:cityObjectMember>
      <c:cityObjectMember>
        <b:Building g:id="b_three">
          <b:lod1Solid>
            <g:Solid>
              <g:exterior>
                <g:CompositeSurface>
                  <g:surfaceMember>
                    <g:Polygon>
                      <g:exterior>
                        <g:LinearRing>
                          <g:posList>20 20 0 21 20 0 21 21 0 20 21 0 20 20 0</g:posList>
                        </g:LinearRing>
                      </g:exterior>
                    </g:Polygon>
                  </g:surfaceMember>
                </g:CompositeSurface>
              </g:exterior>
            </g:Solid>
          </b:lod1Solid>
        </b:Building>
      </c:cityObjectMember>
    </c:CityModel>
""")


@pytest.fixture
def citygml_file(tmp_path):
    """Write the minimal CityGML to a temporary file."""
    p = tmp_path / "test.gml"
    p.write_text(MINIMAL_CITYGML, encoding="utf-8")
    return p


@pytest.fixture
def bridge_citygml_file(tmp_path):
    """Write the minimal bridge CityGML to a temporary file.

    PLATEAU naming convention: bridge GMLs contain ``_brid_`` in the
    filename, which the directory loader uses to dispatch to the bridge
    parser.
    """
    p = tmp_path / "53393683_brid_6697_op.gml"
    p.write_text(MINIMAL_BRIDGE_CITYGML, encoding="utf-8")
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

    def test_parse_bridge_kind(self, bridge_citygml_file):
        # Default kind="bldg" must not pick up bridges.
        assert parse_citygml(bridge_citygml_file) == []
        bridges = parse_citygml(bridge_citygml_file, kind="brid")
        assert len(bridges) == 1
        b = bridges[0]
        assert b.id == "test_brid_1"
        assert b.kind == "brid"
        assert b.lod == 2
        assert len(b.polygons) == 2
        # Building remains as a back-compat alias for CityFeature.
        assert isinstance(b, CityFeature) and b.__class__ is Building

    def test_parse_unsupported_kind_raises(self, bridge_citygml_file):
        with pytest.raises(ValueError):
            parse_citygml(bridge_citygml_file, kind="tran")

    def test_supported_kinds_constant(self):
        assert SUPPORTED_KINDS == frozenset({"bldg", "brid"})

    def test_parse_multi_member_noncanonical_prefixes(self, tmp_path):
        """Parser must handle multiple ``cityObjectMember`` entries and
        non-canonical namespace prefix names (``c:``/``b:``/``g:``
        instead of ``core:``/``bldg:``/``gml:``) and a non-canonical
        declaration order on the root element."""
        p = tmp_path / "multi.gml"
        p.write_text(_MULTI_MEMBER_NONCANONICAL_CITYGML, encoding="utf-8")
        buildings = parse_citygml(p)
        ids = sorted(b.id for b in buildings)
        assert ids == ["b_one", "b_three", "b_two"]
        # b_two has 2 polygons (lod2), the other two have 1 each (lod1)
        polys_per_id = {b.id: len(b.polygons) for b in buildings}
        assert polys_per_id == {"b_one": 1, "b_two": 2, "b_three": 1}
        lod_per_id = {b.id: b.lod for b in buildings}
        assert lod_per_id == {"b_one": 1, "b_two": 2, "b_three": 1}


class TestBridgeLoaderIntegration:

    def _stage(self, citygml_file, bridge_citygml_file):
        # The bldg fixture writes test.gml; rename it to match PLATEAU
        # ``_bldg_`` naming so the directory loader infers kind correctly.
        directory = bridge_citygml_file.parent
        renamed = directory / "53393683_bldg_6697_op.gml"
        os.rename(citygml_file, renamed)
        return directory

    def test_directory_loader_skips_bridges_by_default(
        self, citygml_file, bridge_citygml_file
    ):
        directory = self._stage(citygml_file, bridge_citygml_file)
        bldg_only = load_plateau(directory, zone=9)
        bldg_plus_brid = load_plateau(directory, zone=9, kinds=("bldg", "brid"))
        assert bldg_plus_brid.triangles.shape[0] > bldg_only.triangles.shape[0]
        # Single-file load infers the bridge kind from the filename.
        bridges_only = load_plateau(bridge_citygml_file, zone=9)
        assert bridges_only.triangles.shape[0] >= 2

    def test_include_bridges_alias_matches_kinds(
        self, citygml_file, bridge_citygml_file
    ):
        directory = self._stage(citygml_file, bridge_citygml_file)
        via_kinds = load_plateau(directory, zone=9, kinds=("bldg", "brid"))
        via_alias = load_plateau(directory, zone=9, include_bridges=True)
        assert via_kinds.triangles.shape == via_alias.triangles.shape

    def test_unsupported_kind_in_loader_raises(self, citygml_file, bridge_citygml_file):
        directory = self._stage(citygml_file, bridge_citygml_file)
        with pytest.raises(ValueError):
            load_plateau(directory, zone=9, kinds=("bldg", "tran"))

    def test_include_bridges_false_with_brid_in_kinds_warns(
        self, citygml_file, bridge_citygml_file
    ):
        """``include_bridges=False`` is additive-only; it must not silently
        suppress 'brid' from ``kinds``.  A UserWarning is emitted to flag
        the misuse, and bridges remain loaded."""
        directory = self._stage(citygml_file, bridge_citygml_file)
        with pytest.warns(UserWarning, match="additive-only"):
            model = load_plateau(
                directory, zone=9, kinds=("bldg", "brid"), include_bridges=False
            )
        # Bridges still loaded -> tris > bldg-only.
        bldg_only = load_plateau(directory, zone=9, kinds=("bldg",))
        assert model.triangles.shape[0] > bldg_only.triangles.shape[0]


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

    def test_geoid_correction_constant_shifts_alt(self):
        """A constant geoid_correction must add N to the alt before
        the ellipsoidal-to-ECEF conversion.  At 5 m TP + N=36.7 m the
        ECEF radial offset along the local up vector should equal the
        difference vs the no-correction reference, to within numerics."""
        loader_off = PlateauLoader(zone=9)
        loader_on = PlateauLoader(zone=9, geoid_correction=36.7)
        lat = np.radians(35.6812)
        lon = np.radians(139.7671)
        ecef_off = loader_off._lla_to_ecef(lat, lon, 5.0)
        ecef_on = loader_on._lla_to_ecef(lat, lon, 5.0)
        delta = np.linalg.norm(ecef_on - ecef_off)
        assert abs(delta - 36.7) < 1e-3, f"|delta|={delta:.6f}"
        assert np.linalg.norm(ecef_on) > np.linalg.norm(ecef_off)

    def test_geoid_correction_no_warning_when_supplied(self):
        """Supplying any geoid_correction must suppress the
        PLATEAU-orthometric UserWarning (the user has acknowledged
        the datum)."""
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            PlateauLoader(zone=9, geoid_correction=36.7)
            PlateauLoader(zone=9, geoid_correction=lambda lat, lon: 36.7)

    def test_geoid_correction_callable(self):
        """A callable geoid model must be invoked with (lat_deg, lon_deg)."""
        seen = []

        def _model(lat_deg, lon_deg):
            seen.append((lat_deg, lon_deg))
            return 10.0

        loader = PlateauLoader(zone=9, geoid_correction=_model)
        loader._lla_to_ecef(np.radians(35.65), np.radians(139.78), 0.0)
        assert seen and abs(seen[0][0] - 35.65) < 1e-9
        assert abs(seen[0][1] - 139.78) < 1e-9

    def test_geoid_correction_egm96_tokyo(self):
        """Regression: with EGM96, a PLATEAU mesh ground-level vertex
        in Tokyo Hamamatsucho (TP ~ 0 m) must end up within ~3 m of
        the rover's ellipsoidal ground level (~40 m).  Without the
        correction it sat 36-37 m below."""
        try:
            import pyproj  # noqa: F401
        except ImportError:
            pytest.skip("pyproj not available")
        loader = PlateauLoader(zone=9, geoid_correction="egm96")
        lat = np.radians(35.65); lon = np.radians(139.78)
        ecef_ground = loader._lla_to_ecef(lat, lon, 0.0)
        # Reference rover at the same lat/lon but with ellipsoidal
        # alt = 40 m (matches reference.csv at this trajectory).
        loader_ref = PlateauLoader(zone=9)  # no correction
        ecef_rover = loader_ref._lla_to_ecef(lat, lon, 40.0)
        delta_radial = np.linalg.norm(ecef_ground) - np.linalg.norm(ecef_rover)
        # |TP_0 - rover_40| should be ~|N - 40| = ~|36.5 - 40| = ~3.5m
        assert abs(delta_radial) < 5.0, (
            f"corrected ground vs rover_40 radial diff = {delta_radial:.2f} m; "
            "expected within 5 m"
        )

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
