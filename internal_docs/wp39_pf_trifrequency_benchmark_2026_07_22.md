# WP39 PF trifrequency benchmark

WP38 promotes a production anchor at Nagoya 6073--6539 using an independent
three-family DD pseudorange rank consensus. Candidate 59 ranks 2/3/2 across
primary/secondary/tertiary families, beats the runner by 85.7%, and audits at
0.473 m. Two accepted positive holdouts pass the same fixed gate and two
unsafe supply holdouts fail closed.

Adding only this anchor to the common WP37 fragmentation-gated smoother raises
Nagoya from 4,120/7,583 (54.3321%) to 4,790/7,583 (63.1676%): +670 epochs and
+8.8356 percentage points. The gain contains 670 newly sub-50 cm epochs and
zero losses; +33 occur before the new anchor and +637 from 6073 through the
next production anchor. Declared FIX and false FIX remain zero, runtime FGO is
disabled, no development anchor is used, and the full denominator is
unchanged. Tokyo remains 3,265/11,924 (27.3818%).

The exact validation, production anchor, benchmark artifacts, hashes, and M4
baseline are locked in
`internal_docs/wp39_pf_trifrequency_benchmark_2026_07_22.json`.

