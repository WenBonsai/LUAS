"""Minimal LoRA finetune for small Qwen models on CPU or GPU.

Dataset format: JSONL lines with {instruction, input, output}
Created by: scripts/convert_multiwoz_jsonl_to_sft.py

This is *not* the full LUAS training pipeline. It's a pragmatic, low-RAM path to
"replicate the repo" training on a local machine.

Run (Linux/UCloud):
  python3 scripts/train_qwen_lora.py
    --train_file data_full/train.jsonl
    --dev_file data_full/dev.jsonl
    --model_name Qwen/Qwen2-0.5B-Instruct
    --max_steps 1200
    --max_seq_len 256
    --output_dir runs/qwen_lora_ucloud

Run (PowerShell):
  .venv/Scripts/python.exe scripts/train_qwen_lora.py
    --train_file data_local/train.jsonl
    --dev_file data_local/dev.jsonl
    --model_name Qwen/Qwen2-0.5B-Instruct
    --max_steps 40
    --max_seq_len 256
    --output_dir runs/qwen_lora

Outputs:
- runs/qwen_lora/lora_adapter/    (PEFT adapter)
- runs/qwen_lora/eval_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

try:
    from peft import LoraConfig, get_peft_model
except Exception as e:  # pragma: no cover
    raise RuntimeError("peft is required. Install requirements.smoke.txt") from e


def build_chat_prompt(tokenizer: Any, instruction: str, user_input: str) -> str:
    """Prefer Qwen's chat template to avoid prompt-echoing.

    We explicitly tell the assistant to output JSON only.
    """

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

    # add_generation_prompt=True makes the template end right before assistant content.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class JsonlSftDataset(Dataset):
    def __init__(self, path: str, tokenizer: Any, max_seq_len: int):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.rows: list[dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.rows[idx]
        prompt = build_chat_prompt(self.tokenizer, r["instruction"], r["input"])
        full = prompt + r["output"].strip() + "\n"

        # Tokenize WITHOUT padding first so we can do correct, length-aware masking.
        tok_full_np = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors=None,
        )
        tok_prompt_np = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors=None,
        )

        full_ids = tok_full_np["input_ids"]
        prompt_ids = tok_prompt_np["input_ids"]

        prompt_len = min(len(prompt_ids), len(full_ids))
        if prompt_len >= len(full_ids):
            min_out_tokens = 8
            if len(full_ids) > min_out_tokens:
                keep = len(full_ids) - min_out_tokens
                prompt_len = min(prompt_len, keep)

        labels_np = list(full_ids)
        for i in range(min(prompt_len, len(labels_np))):
            labels_np[i] = -100

        input_ids = torch.full((self.max_seq_len,), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((self.max_seq_len,), dtype=torch.long)
        labels = torch.full((self.max_seq_len,), -100, dtype=torch.long)

        n = min(len(full_ids), self.max_seq_len)
        input_ids[:n] = torch.tensor(full_ids[:n], dtype=torch.long)
        attention_mask[:n] = 1
        labels[:n] = torch.tensor(labels_np[:n], dtype=torch.long)

        if torch.all(labels[:n] == -100):
            labels[:n] = input_ids[:n]
            labels[: min(prompt_len, n)] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def eval_loss(model, dl, device: torch.device, max_batches: int = 10) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss.detach().cpu().item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--dev_file", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--output_dir", type=str, default="runs/qwen_lora")

    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_steps", type=int, default=5)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
    dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # Make sure padding is consistent everywhere (prevents some loss edge cases).
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Conservative LoRA targets; works for Llama-ish and Qwen-ish models
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora)
    model.to(device)

    train_ds = JsonlSftDataset(args.train_file, tokenizer, args.max_seq_len)
    dev_ds = JsonlSftDataset(args.dev_file, tokenizer, args.max_seq_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_updates = max(1, math.ceil(args.max_steps / max(1, args.grad_accum)))
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=min(args.warmup_steps, total_updates),
        num_training_steps=total_updates,
    )

    step = 0
    optim.zero_grad(set_to_none=True)

    skipped = 0
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        if out.loss is None:
            skipped += 1
            continue
        if not torch.isfinite(out.loss):
            skipped += 1
            if step % 10 == 0:
                print(f"[warn] non-finite loss at step={step}: {out.loss}")
            optim.zero_grad(set_to_none=True)
            step += 1
            if step >= args.max_steps:
                break
            continue

        loss = out.loss / args.grad_accum
        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

        if step % 10 == 0:
            print(f"[train] step={step} loss={float(out.loss.detach().cpu().item()):.4f}")

        if step % 20 == 0 and step > 0:
            dloss = eval_loss(model, dev_dl, device, max_batches=10)
            print(f"[eval] step={step} dev_loss={dloss:.4f}")

        step += 1
        if step >= args.max_steps:
            break

    # Save adapter
    adapter_dir = out_dir / "lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    final_dev_loss = eval_loss(model, dev_dl, device, max_batches=50)
    metrics = {"dev_loss": final_dev_loss, "model": args.model_name, "skipped_nonfinite": skipped}
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[done] saved adapter to: {adapter_dir}")
    print(f"[done] dev_loss={final_dev_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
