import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("strength_export", ROOT / "tools/build_browser_recorded_strength.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def header(content, label):
    return f"{content:<60}{label}\n"


def fixture(*, unit="", producer="RTKCONV 2.4.3 b31", system="GPS", extra="", value=42.125):
    lines = [header("     3.02           OBSERVATION DATA    M: Mixed", "RINEX VERSION / TYPE"), header(producer, "PGM / RUN BY / DATE"), header("G    4 C1C L1C D1C S1C", "SYS / # / OBS TYPES"), header("  2018    12    19     3    56   15.0000000     " + f"{system:>3}", "TIME OF FIRST OBS")]
    if unit:
        lines.append(header(unit, "SIGNAL STRENGTH UNIT"))
    lines.extend([extra, header("", "END OF HEADER")])
    for second in ("15.1000000", "15.2010000"):
        lines.append(f"> 2018 12 19 3 56 {second} 0 1\n")
        lines.append("G 1" + " " * 48 + f"{value:14.3f}19\n")
    return "".join(lines)


class RecordedStrengthExportTests(unittest.TestCase):
    def read(self, text):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rover.obs"
            source.write_text(text, encoding="ascii")
            return MODULE.read_strength(source, 2032, 273375, 273376, "trimble")

    def test_original_numeric_fields_ids_and_millisecond_times(self):
        data = self.read(fixture())
        self.assertEqual(data["signals"]["G01"]["samples"], [[273375.1, 42.125], [273375.201, 42.125]])
        self.assertEqual(data["signals"]["G01"]["code"], "S1C")
        self.assertEqual(data["unitStatus"], "inferred-rtkconv")

    def test_declared_units_and_unknown_units_are_distinct(self):
        self.assertEqual(self.read(fixture(unit="DBHZ", producer="Other"))["unitStatus"], "declared")
        for options in ({"producer": "Other"}, {"unit": "arbitrary"}, {"system": "UTC"}):
            with self.assertRaises(ValueError):
                self.read(fixture(**options))

    def test_zero_and_blank_strength_are_missing_not_one_digit_ssi(self):
        data = self.read(fixture(value=0))
        self.assertEqual(data["signals"], {})
        self.assertEqual(data["counts"]["blankOrZero"], 2)
        data = self.read(fixture().replace(f"{42.125:14.3f}19", " " * 14 + "19"))
        self.assertEqual(data["signals"], {})

    def test_unsupported_scaling_clock_offsets_and_header_updates_fail(self):
        bad_scale = header("G   10   0", "SYS / SCALE FACTOR")
        with self.assertRaises(ValueError):
            self.read(fixture(extra=bad_scale))
        with self.assertRaises(ValueError):
            self.read(fixture().replace("0 1\n", "0 1 0.01\n"))
        with self.assertRaises(ValueError):
            self.read(fixture().replace("0 1\n", "4 1\n"))

    def test_out_of_order_epochs_and_non_finite_values_fail(self):
        for source in (fixture().replace("15.2010000", "15.0010000"), fixture(value=float("nan"))):
            with self.assertRaises(ValueError):
                self.read(source)


if __name__ == "__main__":
    unittest.main()
