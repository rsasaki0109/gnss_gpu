#!/usr/bin/env python3
"""WP18: emit the SHIPPED .pos (Q=4 only at the nb>=9 report floor; the
per-pair gamma>0.99 gate is already the PF's fix condition) from a raw
runner .pos + its npz. Usage:

  relabel_nb9_pos.py <run.pos> <run.npz> <out.pos> [nb_floor]
"""
import sys
import numpy as np


def main():
    pos_in, npz_in, pos_out = sys.argv[1:4]
    floor = int(sys.argv[4]) if len(sys.argv) > 4 else 9
    d = np.load(npz_in)
    nb_by_tow = {round(float(t), 1): int(nb)
                 for t, nb in zip(d["tow"], d["rb_nb"])}
    n_demoted = 0
    out_lines = []
    with open(pos_in) as fh:
        for line in fh:
            if line.startswith("%"):
                out_lines.append(line)
                continue
            parts = line.split()
            tow = round(float(parts[1]), 1)
            if parts[8] == "4" and nb_by_tow.get(tow, 0) < floor:
                # demote to float: rebuild the row (the scorer splits on
                # whitespace and reads Q at token index 8)
                parts[8] = "5"
                line = " ".join(parts) + "\n"
                n_demoted += 1
            out_lines.append(line)
    with open(pos_out, "w") as fh:
        fh.writelines(out_lines)
    print(f"wrote {pos_out}: demoted {n_demoted} fixes below nb>={floor}")


if __name__ == "__main__":
    main()
