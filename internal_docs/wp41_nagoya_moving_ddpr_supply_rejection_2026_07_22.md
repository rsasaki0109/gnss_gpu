# WP41 Nagoya moving DDPR supply rejection (2026-07-22)

The first fixed moving block, 2327--2382, supplies 11 evidence epochs and 188
carrier rows. Carrier RMS and four-way bootstrap spread pass for every tested
hypothesis, so carrier alone is strongly multimodal.

Raw absolute DDPR fails even at the non-eligible truth-seeded ceiling
(82.73 m RMS). Biases learned from the preceding accepted static anchor reduce
that ceiling to 20.68 m with satellite bias propagation and 23.06 m with exact
DD-pair medians, still far above the frozen 4 m gate. No hypothesis is selected
and WP39 remains unchanged at 4,790/7,583, FIX=0, false FIX=0.

The result rules out static-to-moving code-bias carryover for this block. The
next measurement family must eliminate block-local DD pair bias as a nuisance
parameter and prove cross-block separation before promotion.
