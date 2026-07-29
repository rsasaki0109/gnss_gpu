# Python exception-handling audit — 2026-07-29

Scope: `python/gnss_gpu/**/*.py`, with emphasis on `except ...: pass` and
`except Exception`.

## Findings and decisions

- Native core import: previously swallowed `ImportError` and left names in
  `__all__` undefined. It now distinguishes an absent optional extension from
  a broken installed extension. CPU-only installs receive defined placeholders
  and `NativeBackendUnavailableError`; dependent-DLL/import failures propagate.
- Windows DLL registration: `OSError` is still non-fatal, but is recorded in
  `_DLL_DIR_ERRORS` and emits a warning instead of disappearing.
- Doppler ephemeris differentiation: no longer catches every exception.
  Invalid `dt` fails validation, expected input/numeric failures return `None`,
  and unexpected programming/runtime failures propagate.
- Scenario and urban simulation fallbacks: native/third-party geometry calls
  can raise several implementation-specific exception types, so the broad
  boundary catch is intentional. Previously silent fallback sites now emit a
  deduplicated or standard warning.
- RINEX/NMEA record parsing: narrow `ValueError`/`IndexError` catches are kept.
  Skipping an individual malformed record is the parser's documented tolerant
  behavior.
- Empty class bodies, no-op destructors, shape-validation branches, and
  `KeyboardInterrupt` handlers are not exception suppression and are retained.

## Rule

Do not add silent `except Exception` or `except ...: pass`. At an optional
integration boundary, catch broadly only when fallback is part of the public
contract, report the failure, and preserve deterministic fallback behavior.
Catch expected validation/parser errors narrowly. Never translate a broken
installed native extension into “backend unavailable.”
