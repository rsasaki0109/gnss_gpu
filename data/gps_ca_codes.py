#!/usr/bin/env python3
"""GPS C/A code reference data and generation.

This script contains the G2 tap delay table for GPS C/A code generation
and can regenerate all 32 PRN codes for verification.

LFSR polynomials:
    G1: x^10 + x^3 + 1          feedback taps: 3, 10
    G2: x^10 + x^9 + x^8 + x^6 + x^3 + x^2 + 1   feedback taps: 2, 3, 6, 8, 9, 10

Output: CA[i] = G1[i] XOR G2_delayed[i], mapped to +1/-1
"""

# G2 tap pairs for PRN 1-32 (1-indexed register positions)
G2_TAP_TABLE = {
    1:  (2, 6),
    2:  (3, 7),
    3:  (4, 8),
    4:  (5, 9),
    5:  (1, 9),
    6:  (2, 10),
    7:  (1, 8),
    8:  (2, 9),
    9:  (3, 10),
    10: (2, 3),
    11: (3, 4),
    12: (5, 6),
    13: (6, 7),
    14: (7, 8),
    15: (8, 9),
    16: (9, 10),
    17: (1, 4),
    18: (2, 5),
    19: (3, 6),
    20: (4, 7),
    21: (5, 8),
    22: (6, 9),
    23: (1, 3),
    24: (4, 6),
    25: (5, 7),
    26: (6, 8),
    27: (7, 9),
    28: (8, 10),
    29: (1, 6),
    30: (2, 7),
    31: (3, 8),
    32: (4, 9),
}


def generate_ca_code(prn):
    """Generate 1023-chip GPS C/A code for given PRN.

    Args:
        prn: Satellite PRN number (1-32).

    Returns:
        List of 1023 integers, each +1 or -1.
    """
    if prn < 1 or prn > 32:
        raise ValueError(f"PRN must be 1-32, got {prn}")

    g1 = [1] * 10
    g2 = [1] * 10

    tap1, tap2 = G2_TAP_TABLE[prn]
    tap1 -= 1  # convert to 0-indexed
    tap2 -= 1

    code = []
    for _ in range(1023):
        # Output
        g1_out = g1[9]
        g2_delayed = g2[tap1] ^ g2[tap2]
        ca_bit = g1_out ^ g2_delayed
        code.append(2 * ca_bit - 1)

        # G1 feedback: taps 3, 10 (0-indexed: 2, 9)
        g1_fb = g1[2] ^ g1[9]
        # G2 feedback: taps 2, 3, 6, 8, 9, 10 (0-indexed: 1, 2, 5, 7, 8, 9)
        g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]

        # Shift
        g1 = [g1_fb] + g1[:9]
        g2 = [g2_fb] + g2[:9]

    return code


def print_first_chips(n=10):
    """Print first n chips of each PRN code."""
    for prn in range(1, 33):
        code = generate_ca_code(prn)
        chips_str = ",".join(f"{c:+d}" for c in code[:n])
        n_pos = sum(1 for c in code if c == 1)
        print(f"PRN {prn:2d}: [{chips_str}]  (+1 count: {n_pos})")


if __name__ == "__main__":
    print("GPS C/A Code Reference (first 10 chips per PRN)")
    print("=" * 60)
    print_first_chips()
