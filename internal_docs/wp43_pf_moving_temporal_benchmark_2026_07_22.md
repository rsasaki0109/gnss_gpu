# WP43 PF moving temporal benchmark (2026-07-22)

Nagoya run1 improves from 4,790/7,583 (63.1676%) to 4,814/7,583
(63.4841%). The promoted 2327--2382 block uses a continuous interpolation of
four carrier/DD bootstrap offsets, gains 24 epochs, and loses none. FIX and
false FIX remain zero.

The result is PF-only, uses no runtime FGO, and preserves the complete epoch
denominator. WP42 selection and promotion are truth-free; reference positions
are loaded only for the frozen full-run audit. The 86% target remains open with
1,708 additional sub-50 cm epochs required.
