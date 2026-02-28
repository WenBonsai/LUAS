"""Convert LUAS-provided MultiWOZ JSONL to a simple instruction-tuning JSONL.

Input (JSONL): generation/multiwoz/datas/multiwoz.json
Each line is a dict with keys: services, turns, status, preference

Output (JSONL): data_local/{train,dev}.jsonl
Each line is:
  {"instruction": str, "input": str, "output": str}

This is intentionally lightweight so you can finetune small models (e.g., Qwen2-0.5B)
without needing the official MultiWOZ 2.2 raw release.

Run:
  .\.venv\Scripts\python.exe scripts\convert_multiwoz_jsonl_to_sft.py \
    --in_file generation\multiwoz\datas\multiwoz.json \
    --out_dir data_local \
    --dev_ratio 0.02 \
    --seed 42

Notes:
- We treat the "output" as the final turn's status JSON (dialogue state).
- "input" is a concatenation of the dialogue turns.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _normalize_slot_key(k: str) -> str:
    # Keep original keys but normalize whitespace/casing a bit.
    return str(k).strip()


def status_from_references(turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best-effort gold-state reconstruction.

    The LUAS-provided MultiWOZ JSONL often has `status: false`.
    However, SYSTEM turns include a `reference` list that contains
    `slot_values` for the active intent.

    We take the last turn that contains reference slot_values and convert it to:
      { service: { slot: value_or_list } }
    """

    last_ref_with_slots: dict[str, Any] | None = None
    for t in turns:
        ref = t.get("reference")
        if not isinstance(ref, list) or not ref:
            continue
        # reference can be a list of api results OR intent state dicts.
        for r in ref:
            if isinstance(r, dict) and isinstance(r.get("slot_values"), dict) and r.get("service"):
                last_ref_with_slots = r

    if not last_ref_with_slots:
        return None

    service = str(last_ref_with_slots.get("service"))
    slot_values = last_ref_with_slots.get("slot_values") or {}
    if not isinstance(slot_values, dict):
        return None

    state: dict[str, Any] = {}
    for k, v in slot_values.items():
        nk = _normalize_slot_key(k)
        # Keep lists as-is; if a singleton list appears, keep it (DST metrics can handle lists).
        state[nk] = v

    return {service: state}


def format_dialogue(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for t in turns:
        speaker = t.get("speaker") or t.get("role") or t.get("from") or "unknown"
        text = t.get("text") or t.get("utterance") or t.get("content") or ""
        speaker = str(speaker).strip().lower()
        if speaker in {"user", "usr", "customer", "human"}:
            prefix = "User"
        elif speaker in {"system", "sys", "assistant", "agent", "bot"}:
            prefix = "Assistant"
        else:
            prefix = speaker.title() if speaker else "Turn"
        text = str(text).strip().replace("\r\n", " ").replace("\n", " ")
        if text:
            lines.append(f"{prefix}: {text}")
    return "\n".join(lines).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dev_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_examples", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    in_file = Path(args.in_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(args.seed)

    examples: list[dict[str, str]] = []

    with in_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            turns = obj.get("turns") or []
            status = obj.get("status")
            services = obj.get("services") or []

            if not isinstance(turns, list) or not turns:
                continue

            dialogue = format_dialogue(turns)
            if not dialogue:
                continue

            if status is None:
                continue

            # Many lines have `"status": false` (boolean), which is not useful
            # as a gold dialogue state. Try to reconstruct from turn references.
            if isinstance(status, bool):
                recovered = status_from_references(turns)
                if recovered is None:
                    # Skip examples where we can't recover any state.
                    continue
                status = recovered

            if not isinstance(status, (dict, list)):
                continue

            instruction = (
                "Given the dialogue, extract the final dialogue state as JSON. "
                "Only output JSON."
            )
            if services:
                instruction += f"\nDomains: {', '.join(map(str, services))}"

            ex = {
                "instruction": instruction,
                "input": dialogue,
                "output": json.dumps(status, ensure_ascii=False),
            }
            examples.append(ex)

            if args.max_examples and len(examples) >= args.max_examples:
                break

    if not examples:
        raise SystemExit("No examples produced. Check input schema.")

    rnd.shuffle(examples)
    n_dev = max(1, int(len(examples) * args.dev_ratio))
    dev = examples[:n_dev]
    train = examples[n_dev:]

    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with dev_path.open("w", encoding="utf-8") as f:
        for ex in dev:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote train: {train_path} ({len(train)} examples)")
    print(f"Wrote dev  : {dev_path} ({len(dev)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
