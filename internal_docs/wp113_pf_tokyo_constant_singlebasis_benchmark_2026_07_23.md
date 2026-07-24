# WP113 Tokyo constant single-basis promotion

WP113 promotes Tokyo run1 epochs 11385--11440 using carrier-reference rank 1.
Candidate 1 ranks first in calibrated OSM road-band, carrier RMS, and CP/PR
evidence and has a 133.3% runner margin. Both independent holdouts fail closed.

The new fixed-epoch multi-sample GSI gate supplies an eight-point 1 m laser
height cluster without runtime network access. Its Up-correction spread is
0.486 m under the frozen 0.5 m bound.

The full-denominator audit gains 16 epochs with zero loss. Tokyo advances from
3,268/11,924 (27.4069%) to 3,284/11,924 (27.5411%). FIX and false FIX remain
zero; runtime FGO and production truth input remain disabled; M4 is unchanged;
shadow and production trajectories are byte-identical.
