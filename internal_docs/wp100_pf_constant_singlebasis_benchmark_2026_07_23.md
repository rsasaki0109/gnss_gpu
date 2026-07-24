# WP100 constant single-basis promotion

WP100 promotes Nagoya run1 epochs 5170--5225 using carrier-reference rank 2.
Candidate 50 ranks 7/2/1 for calibrated OSM road-band, carrier RMS, and CP/PR
evidence. Its 20% runner margin exactly satisfies the frozen gate, and both
independent holdouts fail closed under recomputation.

The full-denominator audit gains 55 epochs with zero loss. Nagoya moves from
5,219/7,583 (68.8250%) to 5,274/7,583 (69.5503%). FIX and false FIX remain zero;
runtime FGO and production truth input remain disabled; M4 is unchanged; the
shadow and production trajectories are byte-identical.
