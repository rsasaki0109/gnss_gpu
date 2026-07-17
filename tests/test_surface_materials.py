"""Tests for per-triangle reflection material classification.

Covers:
  * normal-based classification on a synthetic box mesh (wall/roof/ground)
  * per-triangle Fresnel differing between glass and concrete for identical
    incidence geometry
  * the legacy single-material `UrbanSignalSimulator` path staying
    numerically unchanged (regression)
  * an optional integration test against the bundled PLATEAU Odaiba sample
    data, skipped unless that data actually carries LOD2 boundary-surface
    tags (it does not today -- see module docstring in surface_materials.py)
"""

import glob
import os
import warnings

import numpy as np
import pytest

from gnss_gpu.fresnel import reflection_coefficient
from gnss_gpu.io.citygml import parse_citygml
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.surface_materials import (
    DEFAULT_SURFACE_MATERIALS,
    classify_surface_materials,
)
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator

_ODAIBA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "data", "plateau_odaiba"
)


# ---------------------------------------------------------------------------
# Normal-based classification on a synthetic box mesh
# ---------------------------------------------------------------------------


def _synthetic_box_triangles():
    """Return (triangles, expected_categories) for a hand-built 10x10x10 box.

    Unlike `BuildingModel.create_box` (whose bottom cap has a *downward*
    outward normal, as a real closed solid would), this constructs an
    explicit "ground" triangle with an *upward* normal near the base --
    matching the `classify_surface_materials` fallback heuristic's
    definition of "ground" (upward-facing + near the lowest point of the
    mesh), which is the simplified convention this module documents.
    """
    tris = []
    categories = []

    # Wall: vertical face at x=0, spanning z=[0, 10]. Normal ~ (+/-1, 0, 0).
    tris.append([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 10.0, 10.0]])
    categories.append("wall")

    # Wall: vertical face at y=0, spanning z=[0, 10]. Normal ~ (0, +/-1, 0).
    tris.append([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 10.0]])
    categories.append("wall")

    # Roof: horizontal cap at z=10 (top of the box), upward normal.
    tris.append([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [0.0, 10.0, 10.0]])
    categories.append("roof")

    # Ground: horizontal patch at z=0 (base of the box), upward normal --
    # e.g. a terrain/footprint slab, distinguished from the roof by height.
    tris.append([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    categories.append("ground")

    return np.array(tris, dtype=np.float64), categories


def test_classify_by_normal_labels_wall_roof_ground():
    triangles, expected = _synthetic_box_triangles()
    mapping = {"wall": "glass", "roof": "brick", "ground": "wet_ground"}

    materials = classify_surface_materials(triangles, mapping=mapping)

    assert materials.shape == (4,)
    expected_materials = [mapping[c] for c in expected]
    assert materials.tolist() == expected_materials


def test_classify_by_normal_uses_default_materials_when_mapping_omitted():
    triangles, expected = _synthetic_box_triangles()

    materials = classify_surface_materials(triangles)

    expected_materials = [DEFAULT_SURFACE_MATERIALS[c] for c in expected]
    assert materials.tolist() == expected_materials


def test_classify_empty_mesh_returns_empty_array():
    triangles = np.empty((0, 3, 3), dtype=np.float64)
    materials = classify_surface_materials(triangles)
    assert materials.shape == (0,)


def test_classify_rejects_unknown_mapping_material():
    triangles, _ = _synthetic_box_triangles()
    with pytest.raises(ValueError):
        classify_surface_materials(triangles, mapping={"wall": "not_a_material"})


def test_classify_rejects_unknown_mapping_category():
    triangles, _ = _synthetic_box_triangles()
    with pytest.raises(ValueError):
        classify_surface_materials(triangles, mapping={"window": "glass"})


def test_classify_geocentric_ecef_up_uses_earth_normal():
    """When mesh coordinates are ECEF-scale, "up" is auto-detected as the
    geocentric normal rather than a fixed z-axis. Rotate the synthetic box
    so its local z-axis aligns with an ECEF geocentric-up direction near
    Tokyo, then translate it out to that ECEF position: the classification
    must be unchanged (same wall/roof/ground labels) because the auto-up
    logic tracks the rotation, not a fixed world axis."""
    triangles, expected = _synthetic_box_triangles()
    mapping = {"wall": "glass", "roof": "brick", "ground": "wet_ground"}

    ecef_position = np.array([-3.96e6, 3.35e6, 3.70e6])
    up_dir = ecef_position / np.linalg.norm(ecef_position)

    # Build an orthonormal basis (e1, e2, up_dir) and rotate the box's local
    # (x, y, z) axes into (e1, e2, up_dir) before translating to ECEF scale.
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, up_dir)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up_dir, arbitrary)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up_dir, e1)
    rotation = np.stack([e1, e2, up_dir], axis=1)  # columns = local x, y, z

    rotated = triangles @ rotation.T
    shifted = rotated + ecef_position

    materials = classify_surface_materials(shifted, mapping=mapping)
    expected_materials = [mapping[c] for c in expected]
    assert materials.tolist() == expected_materials


def test_classify_explicit_up_vector():
    triangles, expected = _synthetic_box_triangles()
    mapping = {"wall": "glass", "roof": "brick", "ground": "wet_ground"}
    materials = classify_surface_materials(
        triangles, mapping=mapping, up=np.array([0.0, 0.0, 1.0])
    )
    expected_materials = [mapping[c] for c in expected]
    assert materials.tolist() == expected_materials


# ---------------------------------------------------------------------------
# Tag-based classification (surface_kinds override)
# ---------------------------------------------------------------------------


def test_surface_kinds_tag_wins_over_normal_heuristic():
    """A CityGML-tag-derived "roof" on a triangle whose normal looks like a
    wall (e.g. a sloped LOD2 roof face) must be honored over the fallback
    normal heuristic."""
    # A near-vertical triangle that the normal heuristic would call "wall".
    triangles = np.array(
        [[[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 10.0, 10.0]]],
        dtype=np.float64,
    )
    mapping = {"wall": "glass", "roof": "brick", "ground": "wet_ground"}

    # Without a tag: normal heuristic says "wall".
    materials_untagged = classify_surface_materials(triangles, mapping=mapping)
    assert materials_untagged.tolist() == ["glass"]

    # With an explicit "roof" tag: tag wins.
    materials_tagged = classify_surface_materials(
        triangles, mapping=mapping, surface_kinds=["roof"]
    )
    assert materials_tagged.tolist() == ["brick"]


def test_surface_kinds_unknown_falls_back_to_normal_heuristic():
    triangles, expected = _synthetic_box_triangles()
    mapping = {"wall": "glass", "roof": "brick", "ground": "wet_ground"}
    materials = classify_surface_materials(
        triangles, mapping=mapping, surface_kinds=["unknown"] * 4
    )
    expected_materials = [mapping[c] for c in expected]
    assert materials.tolist() == expected_materials


def test_surface_kinds_length_mismatch_raises():
    triangles, _ = _synthetic_box_triangles()
    with pytest.raises(ValueError):
        classify_surface_materials(triangles, surface_kinds=["wall"])


# ---------------------------------------------------------------------------
# Fresnel amplitude differs per material for identical geometry
# ---------------------------------------------------------------------------


def test_per_triangle_fresnel_differs_glass_wall_vs_concrete_roof():
    """Same incidence angle, different material -> different reflection
    amplitude. This is the core physical claim of the feature: reflection
    strength should be surface-dependent, not a single global constant."""
    incidence_angle = np.radians(35.0)

    wall_material = DEFAULT_SURFACE_MATERIALS["wall"]  # concrete by default
    glass_amp = reflection_coefficient(incidence_angle, "glass")
    wall_amp = reflection_coefficient(incidence_angle, wall_material)

    assert glass_amp != pytest.approx(wall_amp, rel=1e-6)
    assert abs(glass_amp - wall_amp) / wall_amp > 1e-3


def test_classify_surface_materials_then_fresnel_differs_per_triangle():
    """End-to-end: classify two identically-shaped triangles into different
    categories, then confirm their Fresnel amplitudes differ at the same
    incidence angle."""
    # Two identical wall-shaped triangles (translated apart so they are
    # distinct triangles), tagged wall vs roof via surface_kinds so the
    # *only* difference driving the Fresnel amplitude is the material.
    wall_tri = [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 10.0, 10.0]]
    other_tri = [[20.0, 0.0, 0.0], [20.0, 10.0, 0.0], [20.0, 10.0, 10.0]]
    triangles = np.array([wall_tri, other_tri], dtype=np.float64)
    mapping = {"wall": "glass", "roof": "concrete", "ground": "wet_ground"}

    materials = classify_surface_materials(
        triangles, mapping=mapping, surface_kinds=["wall", "roof"]
    )
    assert materials.tolist() == ["glass", "concrete"]

    incidence_angle = np.radians(40.0)
    amp0 = reflection_coefficient(incidence_angle, materials[0])
    amp1 = reflection_coefficient(incidence_angle, materials[1])
    assert amp0 != pytest.approx(amp1, rel=1e-6)


# ---------------------------------------------------------------------------
# UrbanSignalSimulator wiring: per_triangle opt-in vs. legacy single-material
# ---------------------------------------------------------------------------


def _wall_mesh():
    # Same geometry used in test_urban_reflection_paths.py: a thin vertical
    # wall that yields exactly one first-order reflection for this rx/sat.
    return BuildingModel.create_box(
        np.array([0.05, 0.0, 0.0], dtype=np.float64), 0.1, 20.0, 20.0
    )


def _reflection_geometry():
    rx = np.array([-10.0, -3.0, 0.0], dtype=np.float64)
    sats = np.array([[-20.0, 9.0, 0.0]], dtype=np.float64)
    return rx, sats


class _StubBuildingModel:
    """Pure-Python building model: LOS stubbed, reflections delegated."""

    def __init__(self, mesh_model, los_flags):
        self._mesh = mesh_model
        self._los_flags = list(los_flags)
        self.triangles = mesh_model.triangles

    def check_los(self, rx_ecef, sat_ecef):
        sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        return np.asarray(self._los_flags[: sats.shape[0]], dtype=bool)

    def compute_reflection_paths(self, rx_ecef, sat_ecef, max_paths=4):
        return self._mesh.compute_reflection_paths(rx_ecef, sat_ecef, max_paths=max_paths)


class _CaptureSignalGenerator:
    def __init__(self, sampling_freq=2.6e6):
        self.sampling_freq = sampling_freq
        self.channels = None

    def generate_epoch(self, channels, n_samples=None):
        self.channels = [dict(ch) for ch in channels]
        count = int(n_samples) if n_samples is not None else int(self.sampling_freq * 1e-3)
        return np.zeros(2 * count, dtype=np.float32)


def test_single_material_default_path_unchanged_regression():
    """The pre-existing single-`reflector_material` code path must produce
    the exact same amplitude it always has: `fresnel.reflection_coefficient`
    evaluated at the path's incidence angle for the fixed material -- this
    guards against the per-triangle plumbing accidentally touching the
    default (non-"per_triangle") branch."""
    rx, sats = _reflection_geometry()
    mesh = _wall_mesh()
    building = _StubBuildingModel(mesh, los_flags=[False])

    sim = UrbanSignalSimulator(
        building_model=building,
        max_reflection_paths=2,
        reflector_material="concrete",
        nlos_attenuation_db=6.0,
    )
    sim.sim = _CaptureSignalGenerator(sim.sim.sampling_freq)

    result = sim.compute_epoch(rx_ecef=rx, sat_ecef=sats, prn_list=[5])

    paths = result["reflection_paths"][0]
    assert len(paths) == 1
    path = paths[0]

    los_amplitude = 10.0 ** (-6.0 / 20.0)  # NLOS direct-path amplitude
    expected_coeff = reflection_coefficient(path.incidence_angle, "concrete")
    expected_amplitude = los_amplitude * float(expected_coeff)

    mp_channel = sim.sim.channels[1]
    assert mp_channel["amplitude"] == pytest.approx(expected_amplitude, rel=1e-9)


def test_per_triangle_mode_requires_triangle_materials():
    with pytest.raises(ValueError):
        UrbanSignalSimulator(reflector_material="per_triangle")


def test_per_triangle_mode_uses_per_triangle_material():
    """With reflector_material="per_triangle", the reflection amplitude must
    use the *reflecting triangle's* material (via `triangle_materials`)
    instead of a single scene-wide material."""
    rx, sats = _reflection_geometry()
    mesh = _wall_mesh()
    building = _StubBuildingModel(mesh, los_flags=[False])

    n_tri = mesh.triangles.shape[0]
    triangle_materials = np.full(n_tri, "concrete", dtype=object)

    sim_concrete = UrbanSignalSimulator(
        building_model=building,
        max_reflection_paths=2,
        reflector_material="per_triangle",
        triangle_materials=triangle_materials,
        nlos_attenuation_db=6.0,
    )
    sim_concrete.sim = _CaptureSignalGenerator(sim_concrete.sim.sampling_freq)
    result_concrete = sim_concrete.compute_epoch(rx_ecef=rx, sat_ecef=sats, prn_list=[5])
    path = result_concrete["reflection_paths"][0][0]

    triangle_materials_glass = np.full(n_tri, "concrete", dtype=object)
    triangle_materials_glass[path.triangle_id] = "glass"

    sim_glass = UrbanSignalSimulator(
        building_model=building,
        max_reflection_paths=2,
        reflector_material="per_triangle",
        triangle_materials=triangle_materials_glass,
        nlos_attenuation_db=6.0,
    )
    sim_glass.sim = _CaptureSignalGenerator(sim_glass.sim.sampling_freq)
    result_glass = sim_glass.compute_epoch(rx_ecef=rx, sat_ecef=sats, prn_list=[5])

    amp_concrete = sim_concrete.sim.channels[1]["amplitude"]
    amp_glass = sim_glass.sim.channels[1]["amplitude"]

    assert amp_concrete != pytest.approx(amp_glass, rel=1e-6)

    los_amplitude = 10.0 ** (-6.0 / 20.0)
    expected_glass = los_amplitude * float(
        reflection_coefficient(path.incidence_angle, "glass")
    )
    assert amp_glass == pytest.approx(expected_glass, rel=1e-9)


# ---------------------------------------------------------------------------
# Optional integration test against the bundled PLATEAU Odaiba sample data
# ---------------------------------------------------------------------------


def _odaiba_has_lod2_surface_tags():
    if not os.path.isdir(_ODAIBA_DIR):
        return False
    for path in sorted(glob.glob(os.path.join(_ODAIBA_DIR, "*.gml")))[:3]:
        try:
            features = parse_citygml(path, kind="bldg")
        except Exception:
            continue
        for feat in features:
            for kind in feat.surface_kinds:
                if kind in ("wall", "roof", "ground"):
                    return True
    return False


@pytest.mark.skipif(
    not _odaiba_has_lod2_surface_tags(),
    reason=(
        "plateau_odaiba sample data has no LOD2 boundary-surface tags "
        "(current data is LOD1 lod1Solid extrusions only)"
    ),
)
def test_plateau_odaiba_lod2_materials_use_citygml_tags():  # pragma: no cover
    from gnss_gpu.io.plateau import load_plateau

    sample = sorted(glob.glob(os.path.join(_ODAIBA_DIR, "*.gml")))[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _model, materials = load_plateau(
            sample, zone=9, geoid_correction=None, return_materials=True
        )
    assert materials.shape[0] > 0
    assert set(materials.tolist()) <= set(DEFAULT_SURFACE_MATERIALS.values())


def test_plateau_odaiba_data_is_lod1_without_surface_tags():
    """Documents the actual state of the bundled sample data at the time
    this feature was written: PLATEAU Odaiba ships LOD1 `lod1Solid`
    extrusions with no `bldg:WallSurface`/`RoofSurface`/`GroundSurface`
    tags, so material classification for this dataset goes through the
    triangle-normal fallback, not the CityGML-tag path. If PLATEAU data is
    ever refreshed to LOD2, this test will start failing loudly rather than
    silently -- update it (and enable the skipif test above) at that point.
    """
    if not os.path.isdir(_ODAIBA_DIR):
        pytest.skip("plateau_odaiba sample data not present")
    sample_files = sorted(glob.glob(os.path.join(_ODAIBA_DIR, "*.gml")))
    if not sample_files:
        pytest.skip("plateau_odaiba sample data not present")

    features = parse_citygml(sample_files[0], kind="bldg")
    assert features, "expected at least one building feature"
    all_kinds = {k for feat in features for k in feat.surface_kinds}
    assert all_kinds == {"unknown"}
    assert all(feat.lod == 1 for feat in features)
