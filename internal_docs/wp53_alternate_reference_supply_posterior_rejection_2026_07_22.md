# WP53 alternate-reference supply / posterior rejection (2026-07-22)

WP53 adds a default-off elevation-ranked carrier reference. Rank 0 preserves
the historical highest-elevation reference; ranks 1 and 2 use the second and
third reference candidates independently for every constellation/frequency
family. The selection is truth-free and changes reference-specific gaps,
cycle-slip boundaries, and integer arcs.

This is the first successful candidate-supply result in Nagoya 1436--1656.
Rank 1 supplies a basin with 0.520 m audit median and 91/220 sub-50 cm epochs;
ranks 0 and 2 supply none. Re-fitting the complete rank-1 pool under ranks 0
and 2 preserves a useful candidate. Its linear bootstrap profile would move
the locked full trajectory from 4,856 to 4,940 sub-50 cm epochs, an 84-epoch
gain with zero loss.

The posterior does not yet identify the candidate safely. A three-basis
geometric-consistency selector passes its 20% margin but selects candidate 8,
whose audit error is 5.40 m. Summed carrier RMS selects the useful candidate 2,
but separates it from the runner by only 1.71%, below the frozen 20% margin.
Changing the rule after reading these audits would leak truth into production,
so WP53 is retained as a supply success and a posterior rejection. WP45 and M4
remain unchanged.

