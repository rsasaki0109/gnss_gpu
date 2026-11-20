#pragma once

#include <cstdint>

namespace gnss_gpu {

// Systematic resampling offset u ~ U[0, 1) derived from a 64-bit seed.
// The resampling kernels place the i-th pointer at (u + i) / N, so u must
// span the whole unit interval; it must not be pre-divided by N.
// splitmix64 decorrelates consecutive seeds (callers advance the seed per
// resampling call), and the top 53 bits give an exact double in [0, 1).
inline double systematic_offset_from_seed(std::uint64_t seed) {
    std::uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

}  // namespace gnss_gpu
