"""Evaluate a base Qwen model + LoRA adapter on an SFT JSONL dataset.

Metrics ("accuracy-like"):
- exact_match: prediction text == gold output text (strict)
- json_parse_ok: predicted text contains a valid JSON object (best-effort)
- avg_loss: average cross-entropy loss on the output tokens (same masking idea as training)
- ppl: exp(avg_loss)

This isn't the official LUAS metric/JGA; it's a lightweight check that runs locally.

Example (PowerShell):
  .\.venv\Scripts\python.exe scripts\eval_qwen_lora.py `
    --model_name "Qwen/Qwen2-0.5B-Instruct" `
    --adapter_dir runs\qwen_lora_full\lora_adapter `
    --data_file data_full\dev.jsonl `
    --max_examples 200
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception as e:  # pragma: no cover
    raise RuntimeError("peft is required. Install requirements.smoke.txt") from e


def build_chat_prompt(tokenizer: Any, instruction: str, user_input: str) -> str:
    instruction = instruction.strip()
    user_input = user_input.strip()

    system = "You are a helpful assistant for dialogue state tracking. Output JSON only.".strip()
    user = (
        f"{instruction}\n\nDialogue:\n{user_input}\n\n"
        "Return the final dialogue state as a JSON object. Output { ... } only."
    ).strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_json_snippet(text: str) -> str | None:
    """Extract the first top-level JSON object from text."""
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def json_parse_ok(text: str) -> bool:
    snippet = extract_json_snippet(text)
    if snippet is None:
        return False
    try:
        json.loads(snippet)
        return True
    except Exception:
        return False


def _norm_text(x: Any) -> str:
    # conservative normalization for DST-ish values
    s = str(x)
    s = s.strip().lower()
    s = " ".join(s.split())
    s = s.replace("’", "'")
    return s


def _to_slot_pairs(obj: Any, prefix: str = "") -> set[tuple[str, str]]:
    """Flatten nested JSON into (slot, value) pairs.

    - dict -> recurse with key path
    - list -> treat as multiple values for the same slot path
    - scalar -> a value for the slot path

    We skip null/empty values.
    """
    pairs: set[tuple[str, str]] = set()

    if obj is None:
        return pairs

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            pairs |= _to_slot_pairs(v, key)
        return pairs

    if isinstance(obj, list):
        if not obj:
            return pairs
        for v in obj:
            pairs |= _to_slot_pairs(v, prefix)
        return pairs

    # scalar
    val = _norm_text(obj)
    if val in ("", "none", "null"):
        return pairs
    slot = _norm_text(prefix) if prefix else "value"
    pairs.add((slot, val))
    return pairs


def parse_json_best_effort(text: str) -> Any | None:
    """Parse a JSON object from text, best-effort.

    Returns parsed object or None.
    """
    snippet = extract_json_snippet(text)
    if snippet is None:
        return None
    try:
        return json.loads(snippet)
    except Exception:
        return None


def prf(pred_pairs: set[tuple[str, str]], gold_pairs: set[tuple[str, str]]) -> dict[str, float]:
    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"tp": float(tp), "fp": float(fp), "fn": float(fn), "precision": prec, "recall": rec, "f1": f1}


class SftEvalDataset(Dataset):
    def __init__(self, path: str, tokenizer: Any, max_seq_len: int, max_examples: int | None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rows: list[dict[str, str]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if max_examples is not None and max_examples > 0 and len(self.rows) >= max_examples:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not all(k in obj for k in ("instruction", "input", "output")):
                    continue
                self.rows.append({
                    "instruction": str(obj["instruction"]),
                    "input": str(obj["input"]),
                    "output": str(obj["output"]),
                })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        prompt = build_chat_prompt(self.tokenizer, r["instruction"], r["input"])
        gold = r["output"].strip() + "\n"
        full = prompt + gold

        # Prompt-only encode for generation.
        tok_prompt = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors=None,
        )

        # Full encode for loss masking.
        tok_full = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors=None,
        )

        prompt_ids = tok_prompt["input_ids"]
        full_ids = tok_full["input_ids"]
        prompt_len = min(len(prompt_ids), len(full_ids))

        labels_np = list(full_ids)
        for i in range(min(prompt_len, len(labels_np))):
            labels_np[i] = -100

        # Pad full tensors.
        input_ids = torch.full((self.max_seq_len,), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((self.max_seq_len,), dtype=torch.long)
        labels = torch.full((self.max_seq_len,), -100, dtype=torch.long)

        n = min(len(full_ids), self.max_seq_len)
        input_ids[:n] = torch.tensor(full_ids[:n], dtype=torch.long)
        attention_mask[:n] = 1
        labels[:n] = torch.tensor(labels_np[:n], dtype=torch.long)

        # Pad prompt-only tensors for generation.
        gen_input_ids = torch.full((self.max_seq_len,), self.tokenizer.pad_token_id, dtype=torch.long)
        gen_attention_mask = torch.zeros((self.max_seq_len,), dtype=torch.long)
        pn = min(len(prompt_ids), self.max_seq_len)
        gen_input_ids[:pn] = torch.tensor(prompt_ids[:pn], dtype=torch.long)
        gen_attention_mask[:pn] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "gen_input_ids": gen_input_ids,
            "gen_attention_mask": gen_attention_mask,
            "prompt": prompt,
            "gold": gold.strip(),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("input_ids", "attention_mask", "labels"):
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    for k in ("gen_input_ids", "gen_attention_mask"):
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["prompt"] = [b["prompt"] for b in batch]
    out["gold"] = [b["gold"] for b in batch]
    return out


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--data_file", type=str, required=True)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument(
        "--max_examples",
        type=int,
        default=200,
        help="Max examples to evaluate. Use 0 for no limit.",
    )
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--output_json", type=str, default="")
    ap.add_argument(
        "--no_generate",
        action="store_true",
        help="Loss-only evaluation (skip slow text generation).",
    )
    ap.add_argument(
        "--print_every",
        type=int,
        default=50,
        help="Print progress every N batches (loss-only mode especially).",
    )
    ap.add_argument(
        "--dst_metrics",
        action="store_true",
        help="Compute DST-style metrics (slot P/R/F1 + JGA-like exact-state match). Requires generation unless --no_generate.",
    )
    ap.add_argument(
        "--force_json_prefix",
        action="store_true",
        default=True,
        help="Prepend '{' to generation input so the model is forced to start with a JSON object (default: True).",
    )

    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.to(device)
    model.eval()

    max_ex: int | None = None if args.max_examples == 0 else args.max_examples
    ds = SftEvalDataset(args.data_file, tokenizer, args.max_seq_len, max_ex)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    total_loss = 0.0
    n_loss = 0
    n = 0
    exact = 0
    parse_ok = 0

    # DST-style aggregates
    jga = 0
    sum_tp = 0.0
    sum_fp = 0.0
    sum_fn = 0.0

    import time
    t0 = time.time()

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if out.loss is not None and torch.isfinite(out.loss):
            total_loss += float(out.loss.detach().cpu().item())
            n_loss += 1

        if args.no_generate:
            # In loss-only mode we don't compute exact match / json parse metrics.
            n += int(input_ids.shape[0])
            if args.print_every > 0 and (n % max(1, args.print_every) == 0):
                elapsed = max(1e-6, time.time() - t0)
                avg_so_far = total_loss / max(1, n_loss)
                ex_per_s = n / elapsed
                print(f"[eval] processed={n} avg_loss={avg_so_far:.4f} ex/s={ex_per_s:.2f}")
            continue

        # Generate and compare (batch_size usually 1)
        gen_input_ids = batch["gen_input_ids"].to(device)
        gen_attention_mask = batch["gen_attention_mask"].to(device)

        # Optionally prepend "{" so the model is forced to complete a JSON object.
        if args.force_json_prefix:
            brace_ids = tokenizer("{", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            # Append brace token(s) to each item in the batch.
            gen_input_ids = torch.cat(
                [gen_input_ids, brace_ids.expand(gen_input_ids.shape[0], -1)], dim=1
            )
            brace_mask = torch.ones(gen_attention_mask.shape[0], brace_ids.shape[1], dtype=torch.long, device=device)
            gen_attention_mask = torch.cat([gen_attention_mask, brace_mask], dim=1)

        prompt_tok_len_after_prefix = int(gen_input_ids.shape[1])
        gen = model.generate(
            input_ids=gen_input_ids,
            attention_mask=gen_attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_tok_len = prompt_tok_len_after_prefix
        for i in range(gen.shape[0]):
            # Decode only the *newly generated* tokens — never the prompt.
            new_tokens = gen[i][prompt_tok_len:]
            pred_full = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            # If we force-prefixed "{", prepend it to the decoded prediction.
            if args.force_json_prefix:
                pred_full = "{" + pred_full
            # Extract the first well-formed JSON object if present.
            pred_json = extract_json_snippet(pred_full)
            pred = pred_json if pred_json is not None else pred_full
            gold = batch["gold"][i].strip()

            if pred == gold:
                exact += 1
            if json_parse_ok(pred):
                parse_ok += 1

            if args.dst_metrics:
                pred_obj = parse_json_best_effort(pred)
                gold_obj = parse_json_best_effort(gold)
                if pred_obj is not None and gold_obj is not None:
                    pred_pairs = _to_slot_pairs(pred_obj)
                    gold_pairs = _to_slot_pairs(gold_obj)
                    stats = prf(pred_pairs, gold_pairs)
                    sum_tp += stats["tp"]
                    sum_fp += stats["fp"]
                    sum_fn += stats["fn"]
                    if pred_pairs == gold_pairs:
                        jga += 1
            n += 1

    avg_loss = total_loss / max(1, n_loss)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")

    slot_prec = None
    slot_rec = None
    slot_f1 = None
    jga_rate = None
    if args.dst_metrics and (not args.no_generate):
        slot_prec = (sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) > 0 else 0.0
        slot_rec = (sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) > 0 else 0.0
        slot_f1 = (2 * slot_prec * slot_rec / (slot_prec + slot_rec)) if (slot_prec + slot_rec) > 0 else 0.0
        jga_rate = jga / max(1, n)

    metrics = {
        "examples": n,
        "avg_loss": avg_loss,
        "ppl": ppl,
    "exact_match": None if args.no_generate else (exact / max(1, n)),
    "json_parse_ok": None if args.no_generate else (parse_ok / max(1, n)),
    "slot_precision": slot_prec,
    "slot_recall": slot_rec,
    "slot_f1": slot_f1,
    "jga": jga_rate,
    "no_generate": bool(args.no_generate),
    "dst_metrics": bool(args.dst_metrics),
        "adapter_dir": str(adapter_dir),
        "model_name": args.model_name,
        "data_file": args.data_file,
    }

    print("[eval] " + json.dumps(metrics, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[eval] wrote: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
