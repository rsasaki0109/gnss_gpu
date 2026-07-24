# WP95 constant single-basis promotion

WP95 promotes Nagoya run1 epochs 6710--6765 using carrier-reference rank 0.
Candidate 7 ranks 2/7/5 in calibrated OSM road-band, carrier RMS, and CP/PR
evidence, and its 28.57% runner margin exceeds the frozen 20% gate. Both
independent holdouts fail closed under the unchanged selector.

The full-denominator audit gains 42 epochs with zero loss. Nagoya moves from
5,177/7,583 (68.2711%) to 5,219/7,583 (68.8250%). FIX and false FIX remain
zero; runtime FGO and production truth input remain disabled; M4 is unchanged;
shadow and production trajectories are byte-identical.
