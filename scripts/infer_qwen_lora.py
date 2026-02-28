"""Run inference with a base Qwen model + a trained LoRA adapter.

This script is for quick sanity checks after training `scripts/train_qwen_lora.py`.
It will:
- load the base model (e.g., Qwen/Qwen2-0.5B-Instruct)
- load the PEFT LoRA adapter from a local folder (runs/.../lora_adapter)
- run generation on a few JSONL examples and print:
  - the prompt
  - the model prediction
  - the gold output

Example (PowerShell):
  .\.venv\Scripts\python.exe scripts\infer_qwen_lora.py `
    --model_name "Qwen/Qwen2-0.5B-Instruct" `
    --adapter_dir runs\qwen_lora_full\lora_adapter `
    --data_file data_full\dev.jsonl `
    --num_samples 5

Notes:
- CPUs work fine but are slow.
- If you see HF unauth warning, set HF_TOKEN (optional).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
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

def extract_first_json_object(text: str) -> str | None:
    """Extract the first top-level JSON object substring from generated text."""
    if not text:
        return None
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


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not all(k in obj for k in ("instruction", "input", "output")):
                continue
            rows.append(obj)
    return rows


def try_parse_json(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    # Try to find a JSON object in the text.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return False
    snippet = text[start : end + 1]
    try:
        json.loads(snippet)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--data_file", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)

    args = ap.parse_args()

    random.seed(args.seed)

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device={device}")

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

    rows = load_jsonl(args.data_file)
    if not rows:
        raise RuntimeError(f"No valid rows found in: {args.data_file}")

    n = min(args.num_samples, len(rows))
    samples = random.sample(rows, n)

    json_ok = 0
    exact = 0

    for i, r in enumerate(samples, start=1):
        prompt = build_chat_prompt(tokenizer, str(r["instruction"]), str(r["input"]))
        gold = str(r["output"]).strip()

        tok = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )
        tok = {k: v.to(device) for k, v in tok.items()}

        # Append "{" to force the model to begin a JSON object.
        brace_ids = tokenizer("{", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        input_ids = torch.cat([tok["input_ids"], brace_ids], dim=1)
        attn_mask = torch.cat([tok["attention_mask"], torch.ones(1, brace_ids.shape[1], dtype=torch.long, device=device)], dim=1)
        prompt_tok_len = int(input_ids.shape[1])

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens, then prepend the forced "{".
        new_tokens = out[0][prompt_tok_len:]
        pred_full = "{" + tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_json = extract_first_json_object(pred_full)
        pred = pred_json if pred_json is not None else pred_full.strip()

        print("=" * 100)
        print(f"[sample {i}/{n}]")
        print("--- PROMPT ---")
        print(prompt)
        print("--- PRED ---")
        print(pred)
        print("--- GOLD ---")
        print(gold)

        if try_parse_json(pred):
            json_ok += 1
        if pred.strip() == gold.strip():
            exact += 1

    print("=" * 100)
    print(f"[summary] samples={n} json_parse_ok={json_ok}/{n} exact_match={exact}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
