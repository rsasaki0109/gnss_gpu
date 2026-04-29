"""Low-level CityGML parser.

Parses CityGML files and extracts building geometry as polygon coordinate
arrays.  This module is intentionally independent of any coordinate system
conversion so that it can be reused with non-PLATEAU CityGML data.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np

CityGmlKind = Literal["bldg", "brid"]

# Common CityGML / GML namespaces (order of preference when auto-detecting)
_NS_CANDIDATES = {
    "core": [
        "http://www.opengis.net/citygml/2.0",
        "http://www.opengis.net/citygml/1.0",
    ],
    "bldg": [
        "http://www.opengis.net/citygml/building/2.0",
        "http://www.opengis.net/citygml/building/1.0",
    ],
    "brid": [
        "http://www.opengis.net/citygml/bridge/2.0",
        "http://www.opengis.net/citygml/bridge/1.0",
    ],
    "gml": [
        "http://www.opengis.net/gml",
    ],
    "gen": [
        "http://www.opengis.net/citygml/generics/2.0",
        "http://www.opengis.net/citygml/generics/1.0",
    ],
}

# Mapping from feature kind to (namespace family, root element name).
_FEATURE_KINDS: dict[str, tuple[str, str]] = {
    "bldg": ("bldg", "Building"),
    "brid": ("brid", "Bridge"),
}

SUPPORTED_KINDS: frozenset[str] = frozenset(_FEATURE_KINDS)


@dataclass
class CityFeature:
    """A parsed CityGML feature (building or bridge).

    Attributes:
        id: ``gml:id`` attribute of the feature element (may be *None*).
        kind: ``"bldg"`` or ``"brid"`` — see :data:`SUPPORTED_KINDS`.
        lod: Level of detail (1, 2, 3, or 4).  ``0`` if unknown.
        polygons: List of polygon coordinate arrays, each with shape ``(N, 3)``.
    """

    id: Optional[str] = None
    kind: str = "bldg"
    lod: int = 0
    polygons: List[np.ndarray] = field(default_factory=list)


# Back-compat alias: callers historically imported ``Building``.
Building = CityFeature


def _detect_namespaces(root):
    """Build a namespace dict from the document root."""
    ns = {}
    # Collect all declared namespaces from the root element
    declared = {}
    for attr, val in root.attrib.items():
        if attr.startswith("{"):
            continue
        if attr.startswith("xmlns:"):
            prefix = attr.split(":", 1)[1]
            declared[prefix] = val
        elif attr == "xmlns":
            declared[""] = val

    # Also try iterparse-style detection from tag URIs already in tree
    for elem in root.iter():
        tag = elem.tag
        if tag.startswith("{"):
            uri = tag[1:].split("}", 1)[0]
            declared.setdefault("_auto_" + uri, uri)

    # Match known namespace families
    for family, candidates in _NS_CANDIDATES.items():
        for candidate in candidates:
            # Check declared values
            for _prefix, uri in declared.items():
                if uri == candidate:
                    ns[family] = candidate
                    break
            if family in ns:
                break

    # Fallback: if gml namespace not found, try the one with 'gml' in its URI
    if "gml" not in ns:
        for _prefix, uri in declared.items():
            if "opengis.net/gml" in uri:
                ns["gml"] = uri
                break

    return ns


def _parse_poslist(elem_text):
    """Parse a ``gml:posList`` string into an ``(N, 3)`` float64 array."""
    values = elem_text.strip().split()
    coords = np.array([float(v) for v in values], dtype=np.float64)
    if coords.size % 3 != 0:
        raise ValueError(
            f"posList length {coords.size} is not a multiple of 3"
        )
    return coords.reshape(-1, 3)


def _extract_polygons(element, ns):
    """Recursively extract all ``gml:Polygon`` coordinate arrays."""
    gml = ns.get("gml", "http://www.opengis.net/gml")

    polygons = []

    for poslist in element.iter(f"{{{gml}}}posList"):
        if poslist.text:
            polygons.append(_parse_poslist(poslist.text))

    # Also handle gml:pos sequences inside gml:LinearRing
    for ring in element.iter(f"{{{gml}}}LinearRing"):
        pos_elems = ring.findall(f"{{{gml}}}pos")
        if pos_elems:
            coords = []
            for p in pos_elems:
                vals = p.text.strip().split()
                coords.extend(float(v) for v in vals)
            arr = np.array(coords, dtype=np.float64).reshape(-1, 3)
            polygons.append(arr)

    return polygons


def _resolve_family_uri(ns: dict, ns_family: str) -> str:
    """Return the namespace URI for a CityGML family, with a 2.0 fallback."""
    return ns.get(
        ns_family,
        f"http://www.opengis.net/citygml/{ns_family}/2.0",
    )


def _determine_lod(elem, ns, ns_family: str = "bldg") -> int:
    """Determine the LoD of a CityGML feature element."""
    family_uri = _resolve_family_uri(ns, ns_family)
    for lod in (4, 3, 2, 1):
        for suffix in ("Solid", "MultiSurface"):
            tag = f"{{{family_uri}}}lod{lod}{suffix}"
            if elem.find(f".//{tag}") is not None:
                return lod
    return 0


def parse_citygml(filepath, kind: CityGmlKind = "bldg") -> List[CityFeature]:
    """Parse a CityGML file and return a list of :class:`CityFeature` objects.

    Parameters
    ----------
    filepath : str or Path
        Path to a CityGML ``.gml`` file.
    kind : str
        Feature kind to extract.  ``"bldg"`` (default) extracts
        ``bldg:Building`` elements; ``"brid"`` extracts ``brid:Bridge``.
        Both share the same LoD-tagged polygon-set serialisation, so they
        are returned through one :class:`CityFeature` dataclass with the
        ``kind`` field set accordingly.

    Returns
    -------
    list[CityFeature]
        Each item carries its ``id``, ``kind``, ``lod``, and polygons
        (each ``(N, 3)``).
    """
    if kind not in _FEATURE_KINDS:
        raise ValueError(
            f"unsupported CityGML kind: {kind!r}; supported: {sorted(SUPPORTED_KINDS)}"
        )
    ns_family, root_tag = _FEATURE_KINDS[kind]

    filepath = Path(filepath)
    tree = ET.parse(filepath)
    root = tree.getroot()

    ns = _detect_namespaces(root)
    feature_uri = _resolve_family_uri(ns, ns_family)
    gml_uri = ns.get("gml", "http://www.opengis.net/gml")

    features: List[CityFeature] = []
    for feat_elem in root.iter(f"{{{feature_uri}}}{root_tag}"):
        gml_id = feat_elem.get(f"{{{gml_uri}}}id") or feat_elem.get("id")
        features.append(
            CityFeature(
                id=gml_id,
                kind=kind,
                lod=_determine_lod(feat_elem, ns, ns_family=ns_family),
                polygons=_extract_polygons(feat_elem, ns),
            )
        )
    return features
