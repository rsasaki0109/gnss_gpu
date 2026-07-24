# WP131 PF-only Tokyo cross-basis/CPPR benchmark

WP131 promotes the constant correction for Tokyo run1 epochs 11660--11715.
The truth-free selector fuses three-reference convergence, maximum carrier RMS,
and CP/PR ranks. It selects candidate 24 with family ranks 2/3/1 and a 2.0
runner margin. Nagoya WP53 and Tokyo WP129 fail closed under the same selector.

The complete 11,924-epoch audit improves `<50cm_full` from 3,394 to 3,411
epochs (28.606172425360615%), gaining 17 and losing none. The production and
shadow trajectories are byte-identical. Declared FIX and false FIX remain zero,
runtime FGO remains disabled, and the M4 locked artifacts are unchanged.
