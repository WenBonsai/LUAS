"""Quick sanity checks for MultiWOZ assets expected by LUAS converters.

Run from repo root:
  python scripts/check_multiwoz_paths.py

This does not download anything. It only reports what's missing.
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    expected = repo_root / "multiwoz" / "data" / "MultiWOZ_2.2"
    required_files = [
        expected / "dialog_acts.json",
        expected / "train" / "dialogues_001.json",
        expected / "val" / "dialogues_001.json",
        expected / "test" / "dialogues_001.json",
    ]

    print(f"Repo root: {repo_root}")
    print(f"Expected MultiWOZ path: {expected}")
    print("")

    exists = expected.exists()
    print(f"Exists: {exists}")

    missing = [p for p in required_files if not p.exists()]
    if missing:
        print("\nMissing required files (examples):")
        for p in missing[:10]:
            print(f"  - {p.relative_to(repo_root)}")

        print("\nFix:")
        print(f"  Place MultiWOZ 2.2 data so that this folder exists:")
        print(f"    {expected}")
        print("  (It should contain dialog_acts.json and train/val/test subfolders.)")
        return 2

    print("\nOK: MultiWOZ 2.2 files appear present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
