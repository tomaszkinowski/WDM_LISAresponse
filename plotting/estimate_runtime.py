"""How long make_figures.py would take on this machine, and whether it fits in memory.

    python estimate_runtime.py

Runs a calibration of about a minute (CALIBRATION, short records of the same
pipeline the figures use), compares its time with the same calibration in the
reference run (REFERENCE, written by ``make_figures.py --log``), and scales every
script's reference time by that ratio. Each script's reference peak memory is shown
against this machine's, flagged where it passes 80 %. One speed ratio stands in for
everything, so read the estimates as good to a factor of about two.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
REFERENCE = HERE / "numbers" / "make_figures_reference.txt"
CALIBRATION = [("numbers_interpolation.py", ["--short"]),
               ("fig12_emri_link_comparison.py", ["--short"])]


def calibrate():
    """Wall time of the calibration runs, in seconds."""
    tick = time.perf_counter()
    for script, args in CALIBRATION:
        subprocess.run([sys.executable, script, *args], cwd=HERE, check=True,
                       stdout=subprocess.DEVNULL)
    return time.perf_counter() - tick


def reference():
    """The reference run's header lines, its rows (name, seconds, peak GiB) and its
    calibration time."""
    header, rows, calibration = [], [], None
    for line in REFERENCE.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if cells[0] == "calibration":
            calibration = float(cells[1])
        elif len(cells) >= 3 and cells[0] != "script":
            rows.append((cells[0], float(cells[1]), float(cells[2])))
        elif len(cells) == 1 and cells[0] and not rows:
            header.append(cells[0])
    return header, rows, calibration


def duration(seconds):
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


if __name__ == "__main__":
    print("calibrating (about a minute) ...", flush=True)
    here = calibrate()
    if not REFERENCE.exists():
        sys.exit(f"calibration {here:.1f} s; no reference run to compare with "
                 f"({REFERENCE.name} is written by make_figures.py --log)")
    header, rows, there = reference()
    ratio = here / there
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    print("reference run:\n  " + "\n  ".join(header))
    print(f"calibration here {here:.1f} s, there {there:.1f} s: this machine is "
          f"{1 / ratio:.2f}x the reference's speed; memory here {memory:.1f} GiB\n")
    width = max(len(name) for name, *_ in rows)
    for name, seconds, peak in rows:
        flag = "  <- more than 80 % of this machine's memory" if peak > 0.8 * memory else ""
        print(f"{name:<{width}}  ~{duration(seconds * ratio):>8}   peak {peak:5.2f} GiB{flag}")
    print(f"{'total':<{width}}  ~{duration(sum(s for _, s, _ in rows) * ratio):>8}")
