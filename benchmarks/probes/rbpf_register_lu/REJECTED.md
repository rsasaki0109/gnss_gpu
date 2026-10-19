# Initial prototype rejected

Do not adopt this initial DLL. Its whole-score gain missed the10percent gate,
and additional finite-input backsolve-overflow tests reported9 Racecheck warning
groups for shared status read/write ordering despite matching output bits.
The original files and timing are frozen evidence.

Use the separately validated corrected source in
[rbpf_register_lu_synced](../rbpf_register_lu_synced/README.md).
Initial timings do not measure that corrected DLL. See
[D253](../../../internal_docs/rbpf_register_lu_2026_10_17.md).
