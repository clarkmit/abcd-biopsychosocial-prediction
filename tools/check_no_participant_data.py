#!/usr/bin/env python
"""Guard against committing files that could contain ABCD participant data.

Used as a pre-commit hook and in CI (``git ls-files -z | xargs -0 python tools/check_no_participant_data.py``).
A file is refused when

1. its type is a data, statistics-package, array or archive format, unless it is on the allow-list;
2. it is larger than ``MAX_BYTES`` and not on the allow-list (large files are almost always data); or
3. it contains an ABCD/NDA participant identifier (GUIDs of the form ``NDAR_INV...``).

ABCD data may only be shared under an approved Data Use Certification and must never reach
this repository, including its history. Keep these rules in sync with ``.gitignore``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DATA_SUFFIXES = {
    # tabular and statistics-package formats
    ".csv", ".tsv", ".xls", ".xlsx", ".parquet", ".feather", ".arrow", ".sav", ".dta", ".sas7bdat", ".xpt",
    ".rdata", ".rda", ".rds",
    # serialized Python / array / scientific formats
    ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".h5", ".hdf5", ".mat", ".nii", ".db", ".sqlite",
    # archives (their contents cannot be scanned)
    ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".tar", ".rar",
}  # fmt: skip
ALLOWED = {
    "data/simulated/CLEAN_ABCD_5_1_panel_simulated.csv",  # fully simulated, no participant data
    "data/variable_map.csv",  # variable names and source tables only
}
MAX_BYTES = 5_000_000
GUID = re.compile(rb"NDAR_?INV[A-Z0-9]{6,}")


def check(paths: list[str]) -> list[str]:
    problems = []
    for name in paths:
        p = Path(name)
        rel = p.as_posix()
        allowed = rel in ALLOWED
        if not allowed and any(suffix.lower() in DATA_SUFFIXES for suffix in p.suffixes):
            problems.append(f"{rel}: data or archive file type is not allowed in this repository")
            continue
        try:
            blob = p.read_bytes()
        except OSError:
            continue
        if not allowed and len(blob) > MAX_BYTES:
            problems.append(f"{rel}: larger than {MAX_BYTES // 1_000_000} MB; data files are not allowed")
        elif GUID.search(blob):
            problems.append(f"{rel}: contains an ABCD participant identifier (NDAR_INV...)")
    return problems


def main(argv: list[str]) -> int:
    problems = check(argv)
    for msg in problems:
        print(f"BLOCKED  {msg}", file=sys.stderr)
    if problems:
        print("\nABCD data must never be committed (Data Use Certification). See CONTRIBUTING.md.", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
