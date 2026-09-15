import importlib.util
from pathlib import Path

from intent_alignment_toy.dataset import load_records

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_verl_data", ROOT / "scripts" / "prepare_verl_data.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_verl_row_routes_to_custom_agent_loop_with_complete_task_identity():
    record = load_records(ROOT / "data" / "intent_alignment_train_400.jsonl")[0]
    row = MODULE.build_verl_row(record, 0)

    assert row["agent_name"] == "intent_alignment_agent"
    assert row["data_source"] == "intent_alignment_text_toy_v2"
    assert row["extra_info"]["task"]["id"] == record["id"]
    assert row["extra_info"]["task"]["target"] == record["target"]
    assert row["prompt"][0]["role"] == "system"
    assert "当前位置：(0, 0)" in row["prompt"][1]["content"]
