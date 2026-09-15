#!/usr/bin/env python3
"""Refresh the standalone HTML with the frozen published evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "qwen3.5-4b-verl-one-step-v2"
DEFAULT_HTML = ROOT / "showcase" / "intent-alignment-showcase.html"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    html_path = args.html.resolve()
    before = load(result_dir / "before-test100.json")
    after = load(result_dir / "after-test100.json")
    paired = load(result_dir / "paired-comparison.json")
    training = load(result_dir / "training-evidence.json")
    parameter = load(result_dir / "parameter-evidence.json")
    manifest = load(result_dir / "artifact-manifest.json")

    source_hashes = manifest["source_artifact_sha256"]
    published_hashes = manifest["published_artifact_sha256"]
    evidence = {
        "protocol": {
            "name": after["protocol"]["name"],
            "dataset_sha256": after["protocol"]["dataset_sha256"],
            "source_before_result_sha256": source_hashes["before-test100.json"],
            "published_before_result_sha256": published_hashes[
                "before-test100.json"
            ],
            "source_after_result_sha256": source_hashes["after-test100.json"],
            "published_after_result_sha256": published_hashes["after-test100.json"],
            "source_paired_comparison_sha256": source_hashes[
                "paired-comparison.json"
            ],
            "published_paired_comparison_sha256": published_hashes[
                "paired-comparison.json"
            ],
            "source_commit": manifest["source_commit"],
            "model": "Qwen3.5-4B",
            "device": "1 × NVIDIA B300 SXM6 AC",
            "decoding": after["protocol"]["decoding"],
            "metric": after["protocol"]["metric"],
            "threshold": after["protocol"]["threshold"],
        },
        "dataset": {"train": 400, "test": 100, "max_steps": 5},
        "training": {
            "framework": "veRL v0.9.0",
            "algorithm": "GRPO",
            "optimizer_updates": training["optimizer_updates_actual"],
            "global_step": training["global_step"],
            "trajectories": training["trajectory_count"],
            "groups": training["grpo_group_count"],
            "rollouts_per_group": training["rollouts_per_group"],
            "mixed_reward_groups": training["mixed_reward_group_count"],
            "actor_grad_norm": training["positive_finite_actor_grad_norm"][0],
        },
        "before": before["summary"],
        "after": after["summary"],
        "paired": {key: len(value) for key, value in paired["paired"].items()},
        "parameter": {
            "policy_core_tensor_count": parameter["policy_core_tensor_count"],
            "changed_probe_count": parameter["changed_probe_count"],
            "effective_bfloat16_changed_value_count": parameter[
                "effective_bfloat16_changed_value_count"
            ],
            "weights_changed": parameter["weights_changed"],
            "effective_bfloat16_weights_changed": parameter[
                "effective_bfloat16_weights_changed"
            ],
        },
        "tasks": after["tasks"],
    }
    payload = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    if "</script>" in payload.lower():
        raise ValueError("embedded evidence contains a closing script tag")

    html = html_path.read_text(encoding="utf-8")
    html, replacements = re.subn(
        r'(<script id="evidenceData" type="application/json">).*?(</script>)',
        lambda match: match.group(1) + payload + match.group(2),
        html,
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise ValueError("could not locate unique embedded evidence block")

    footer = (
        "冻结协议 <code>intent-alignment-text-toy-v2-agent-loop</code> · "
        f"训练源码 <code>{manifest['source_commit'][:8]}</code> · "
        "训练后公开文件 SHA "
        f"<code>{published_hashes['after-test100.json'][:8]}</code> · "
        "原始文件 SHA "
        f"<code>{source_hashes['after-test100.json'][:8]}</code>"
    )
    html, replacements = re.subn(
        r"冻结协议 <code>.*?</code> · 训练源码 <code>.*?</code> · .*?</span>",
        footer + "</span>",
        html,
        count=1,
    )
    if replacements != 1:
        raise ValueError("could not locate unique footer evidence label")
    html_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
