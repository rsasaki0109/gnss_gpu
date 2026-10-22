"""Independent Python complex-arithmetic reference for the browser RF model."""

import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "reference_fresnel", root / "python/gnss_gpu/fresnel.py"
    )
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rows = []
    for material, angle, frequency, polarization in itertools.product(
        ["concrete", "glass", "brick", "wood"],
        [0, 0.1, 0.4, 0.8, 1.2, math.pi / 2 - 1e-6, math.pi / 2],
        [1561.098e6, 1575.42e6, 1598.0625e6, 1605.375e6],
        ["rhcp", "rhcp_cross", "unpolarized"],
    ):
        if polarization == "unpolarized":
            a, b = reference.fresnel_coefficients(
                angle, reference.complex_permittivity(material, frequency)
            )
            power = (abs(a) ** 2 + abs(b) ** 2) / 2
        else:
            power = reference.reflection_coefficient(
                angle, material, frequency, polarization
            ) ** 2
        rows.append(
            dict(material=material, angle=angle, frequency=frequency,
                 polarization=polarization, power=float(power))
        )
    target = root / "tests/site/fixtures/fresnel-power.json"
    content = json.dumps(rows, separators=(",", ":")) + "\n"
    if args.check:
        if target.read_text(encoding="utf-8") != content:
            raise SystemExit("Browser RF reference fixture is stale")
    else:
        target.write_text(content, encoding="utf-8", newline="\n")
    print(f"Verified {len(rows)} independent Python Fresnel power references")


if __name__ == "__main__":
    main()
