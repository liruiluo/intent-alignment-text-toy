# Frozen veRL parquet inputs

These two small parquet files are the exact inputs consumed by the reported
Qwen3.5-4B one-update GRPO/SPHERE runs. They are committed so the reproduction
entry point does not depend on a `datasets`/PyArrow serialization version.

| File | Rows | SHA256 |
|---|---:|---|
| `train.parquet` | 400 | `4ab5bfa4f55f073617795c403a10a1b45310d136691ea5dff0377a58cb991ae1` |
| `test.parquet` | 100 | `455a4c17c370bf40802a282215f69c71a5d21326ad185375211099bd6fe68730` |

Their source JSONL files are `../intent_alignment_train_400.jsonl` and
`../intent_alignment_test_100.jsonl`. The top-level reproduction script checks
all four hashes before training.
