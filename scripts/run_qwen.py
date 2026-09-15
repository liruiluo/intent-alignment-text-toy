#!/usr/bin/env python3
"""Evaluate a local Qwen3.5 checkpoint in the text-navigation agent loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from intent_alignment_toy.dataset import load_tasks
from intent_alignment_toy.evaluation import (
    SYSTEM_PROMPT,
    build_agent_messages,
    build_policy_prompt,
    evaluate_with_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Local Qwen3.5-4B checkpoint path"
    )
    parser.add_argument("--dataset", default="data/intent_alignment_test_100.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--prompt-contract",
        choices=("v1.1-full-state", "v2-agent-loop"),
        default="v2-agent-loop",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def model_weight_manifest(model_path: Path) -> dict:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index["weight_map"].values()))
        files = [model_path / name for name in names]
    else:
        files = sorted(model_path.glob("*.safetensors"))
    if not files or any(not path.is_file() for path in files):
        raise ValueError(f"no complete safetensors weights found in {model_path}")

    def shard_id(path: Path) -> str:
        match = re.search(r"(\d+)-of-(\d+)\.safetensors$", path.name)
        return f"{match.group(1)}-of-{match.group(2)}" if match else path.name

    return {
        "files": [
            {
                "name": path.name,
                "shard_id": shard_id(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
    }


def model_config_fingerprint(config) -> dict:
    volatile_keys = {"_name_or_path", "name_or_path", "transformers_version"}

    def normalize(value):
        if isinstance(value, dict):
            return {
                key: normalize(item)
                for key, item in sorted(value.items())
                if key not in volatile_keys
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    payload = json.dumps(
        normalize(config.to_dict()),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "class": type(config).__name__,
        "model_type": config.model_type,
        "architectures": list(config.architectures or ()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def tokenizer_fingerprint(tokenizer) -> dict:
    """Return a path-independent fingerprint of evaluation tokenization semantics."""
    vocab_payload = json.dumps(
        sorted(tokenizer.get_vocab().items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    template_payload = json.dumps(
        tokenizer.chat_template,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "vocab_sha256": hashlib.sha256(vocab_payload).hexdigest(),
        "chat_template_sha256": hashlib.sha256(template_payload).hexdigest(),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model_path = Path(args.model).resolve()
    dataset_path = Path(args.dataset).resolve()
    tasks = load_tasks(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_path)
    architectures = set(config.architectures or ())
    model_loader = (
        AutoModelForImageTextToText
        if any(name.endswith("ForConditionalGeneration") for name in architectures)
        else AutoModelForCausalLM
    )
    model = model_loader.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()

    def policy(requests):
        decoded: list[str] = []
        for start in range(0, len(requests), args.batch_size):
            prompts = []
            for task, env, actions, raw_outputs in requests[
                start : start + args.batch_size
            ]:
                if args.prompt_contract == "v2-agent-loop":
                    messages = build_agent_messages(task, actions, raw_outputs)
                else:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_policy_prompt(task, env, actions),
                        },
                    ]
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
                model.device
            )
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = outputs[:, inputs["input_ids"].shape[1] :]
            decoded.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return decoded

    evaluation = evaluate_with_policy(tasks, policy)
    threshold_successes = math.ceil(0.95 * len(tasks))
    payload = {
        "protocol": {
            "name": (
                "intent-alignment-text-toy-v2-agent-loop"
                if args.prompt_contract == "v2-agent-loop"
                else "intent-alignment-text-toy-v1.1"
            ),
            "dataset": dataset_path.name,
            "dataset_sha256": sha256_file(dataset_path),
            "prompt_contract": args.prompt_contract,
            "decoding": {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "thinking": False,
            },
            "evaluation_batch_size": args.batch_size,
            "metric": "target reached within max_steps",
            "threshold": (
                f"sample success rate >= 95% (at least {threshold_successes}/{len(tasks)})"
            ),
        },
        "model": {
            "path": str(model_path),
            "config_sha256": sha256_file(model_path / "config.json"),
            "config_fingerprint": model_config_fingerprint(config),
            "tokenizer_config_sha256": optional_sha256(
                model_path / "tokenizer_config.json"
            ),
            "model_index_sha256": optional_sha256(
                model_path / "model.safetensors.index.json"
            ),
            "export_manifest_sha256": optional_sha256(
                model_path / "export_manifest.json"
            ),
            "weights": model_weight_manifest(model_path),
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        },
        "runtime": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
            "model_class": type(model).__name__,
        },
        **evaluation,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
