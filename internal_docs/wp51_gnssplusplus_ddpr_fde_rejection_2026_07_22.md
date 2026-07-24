# WP51 latest GNSS++ DDPR FDE transfer (2026-07-22)

The latest `gnssplusplus-library` `develop` commit was inspected at
`7085fb81379018e2211b03664634acf74962aeae`. Its DD-pseudorange anchor uses
leave-one-out fault detection and exclusion: when a large post-fit residual
remains, every one-row exclusion is solved and the lowest-RMS hypothesis is
retained. This behavior was ported behind default-off `DDWLSConfig` fields, so
existing callers do not change.

On Nagoya run1 epochs 1436--1656, the same 44 accepted evidence epochs improve
from 4.65 m to 1.65 m median anchor error and from 24/44 to 44/44 within 5 m.
The gate used no reference truth; errors are post-selection audit fields.

The median FDE block translation was then passed through the existing carrier
arc, GSI height, and 128-candidate LAMBDA refinement. It improved the proposal
to 1.496 m median with 0.069 m block spread, but generated zero sub-50 cm
epochs and retained 12.43 m DDPR RMS. It therefore fails closed. The FDE
supplier remains available for later partial-AR work, while WP45 production
and M4 remain unchanged.

