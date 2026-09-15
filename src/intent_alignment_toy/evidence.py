"""Shared evidence contracts for the Qwen3.5-4B one-step experiment."""

MISSING = "__MISSING__"

EXPECTED_CONFIG_NORMALIZATION = {
    "/bos_token_id": {"before": MISSING, "after": None},
    "/dtype": {"before": MISSING, "after": "bfloat16"},
    "/eos_token_id": {"before": MISSING, "after": 248046},
    "/pad_token_id": {"before": MISSING, "after": 248044},
    "/text_config/bos_token_id": {"before": MISSING, "after": None},
    "/text_config/mtp_num_hidden_layers": {"before": 1, "after": 0},
    "/text_config/pad_token_id": {"before": MISSING, "after": None},
    "/text_config/partial_rotary_factor": {"before": MISSING, "after": 0.25},
    "/transformers_version": {"before": "4.57.0.dev0", "after": "5.5.3"},
    "/vision_config/dtype": {"before": MISSING, "after": "bfloat16"},
}
