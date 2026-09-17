import json
import math
from pathlib import Path

from intent_alignment_toy.sphere_contract import (
    build_packed_response_predictor_mask,
    load_sphere_variant,
    select_target_layer_names,
    sphere_penalty_and_feature_gradient_reference,
    sphere_penalty_reference,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "sphere_variants.json"


def test_isotropic_features_have_lower_penalty_than_rank_collapse():
    isotropic, alpha = sphere_penalty_reference([[math.sqrt(2), 0], [0, math.sqrt(2)]])
    collapsed, _ = sphere_penalty_reference([[math.sqrt(2), 0], [math.sqrt(2), 0]])

    assert math.isclose(alpha, 1.0, rel_tol=1e-12)
    assert math.isclose(isotropic, 0.0, abs_tol=1e-12)
    assert collapsed > isotropic


def test_adaptive_alpha_is_trace_over_feature_dimension():
    penalty, alpha = sphere_penalty_reference([[1, 0, 0], [0, 2, 0]])

    assert math.isclose(alpha, 5 / 6, rel_tol=1e-12)
    assert penalty > 0


def test_fixed_alpha_is_constant_and_duplicate_batch_invariant():
    rows = [[1, 2], [3, 4]]
    penalty, alpha = sphere_penalty_reference(rows, fixed_alpha=1.0)
    duplicated, duplicate_alpha = sphere_penalty_reference(rows + rows, fixed_alpha=1.0)

    assert alpha == duplicate_alpha == 1.0
    assert math.isclose(penalty, duplicated, rel_tol=1e-12)


def test_logical_minibatch_feature_gradient_matches_finite_difference():
    rows = [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0], [2.0, 1.0, 4.0]]
    penalty, alpha, gradient = sphere_penalty_and_feature_gradient_reference(rows)

    epsilon = 1e-5
    perturbed_plus = [row.copy() for row in rows]
    perturbed_minus = [row.copy() for row in rows]
    perturbed_plus[1][2] += epsilon
    perturbed_minus[1][2] -= epsilon
    plus, _ = sphere_penalty_reference(perturbed_plus)
    minus, _ = sphere_penalty_reference(perturbed_minus)
    finite_difference = (plus - minus) / (2 * epsilon)

    assert penalty > 0
    assert alpha > 0
    assert math.isclose(gradient[1][2], finite_difference, rel_tol=1e-6)


def test_logical_minibatch_penalty_is_not_sum_of_physical_partitions():
    first = [[1.0, 0.0], [1.0, 0.0]]
    second = [[0.0, 1.0], [0.0, 1.0]]
    logical, _, _ = sphere_penalty_and_feature_gradient_reference(first + second)
    first_local, _ = sphere_penalty_reference(first)
    second_local, _ = sphere_penalty_reference(second)

    assert math.isclose(logical, 0.0, abs_tol=1e-12)
    assert first_local + second_local > 0


def test_fixed_alpha_feature_gradient_uses_the_fixed_target():
    rows = [[1.0, 2.0], [3.0, 4.0]]
    penalty, alpha, gradient = sphere_penalty_and_feature_gradient_reference(
        rows, fixed_alpha=1.0
    )

    assert penalty > 0
    assert alpha == 1.0
    assert len(gradient) == 2
    assert all(len(row) == 2 for row in gradient)


def test_all_hidden_is_layerwise_sum_not_joint_concatenation():
    first, _ = sphere_penalty_reference([[1, 0], [1, 0]])
    second, _ = sphere_penalty_reference([[0, 1], [0, 1]])
    joint, _ = sphere_penalty_reference([[1, 0, 0, 1], [1, 0, 0, 1]])

    assert first + second != joint


def test_packed_response_predictor_mask_aligns_each_sequence_independently():
    # Sequence 0: p=3, r=3.  Response token j is predicted by hidden
    # position p-1+j; the middle response token is environment feedback and
    # therefore masked out.  Sequence 1 starts at packed offset 6 and must
    # not inherit the previous sequence's terminal shift.
    mask = build_packed_response_predictor_mask(
        sequence_lengths=[6, 4],
        prompt_lengths=[3, 2],
        response_lengths=[3, 2],
        response_loss_masks=[
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ],
    )

    assert mask == [
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        True,
        False,
    ]


def test_packed_response_predictor_mask_places_first_response_on_last_prompt():
    mask = build_packed_response_predictor_mask(
        sequence_lengths=[3],
        prompt_lengths=[2],
        response_lengths=[1],
        response_loss_masks=[[1, 0]],
    )

    assert mask == [False, True, False]


def test_packed_response_predictor_mask_rejects_true_padding_tokens():
    try:
        build_packed_response_predictor_mask(
            sequence_lengths=[4],
            prompt_lengths=[2],
            response_lengths=[2],
            response_loss_masks=[[1, 1, 1]],
        )
    except ValueError as error:
        assert "masked token beyond response length" in str(error)
    else:
        raise AssertionError("a true response-padding mask must fail closed")


def test_packed_response_predictor_mask_rejects_length_drift():
    try:
        build_packed_response_predictor_mask(
            sequence_lengths=[5],
            prompt_lengths=[2],
            response_lengths=[2],
            response_loss_masks=[[1, 1]],
        )
    except ValueError as error:
        assert "prompt + response" in str(error)
    else:
        raise AssertionError("packed sequence length drift must fail closed")


def test_layer_selection_covers_exact_qwen_inventory():
    names = [
        f"model.language_model.layers.{index}.mlp.down_proj" for index in range(32)
    ]

    assert select_target_layer_names(
        names, layer_mode="last", expected_hidden_layers=32
    ) == [names[-1]]
    assert (
        select_target_layer_names(
            names, layer_mode="all_hidden", expected_hidden_layers=32
        )
        == names
    )


def test_layer_selection_normalizes_observed_verl_fsdp_segments_only():
    names = [
        (
            "_fsdp_wrapped_module.model.visual.blocks.0."
            "_fsdp_wrapped_module.mlp.down_proj"
        ),
        *[
            "_fsdp_wrapped_module.model.language_model.layers."
            f"{index}._fsdp_wrapped_module.mlp.down_proj"
            for index in range(32)
        ],
    ]

    assert select_target_layer_names(
        names, layer_mode="last", expected_hidden_layers=32
    ) == [names[-1]]
    assert (
        select_target_layer_names(
            names, layer_mode="all_hidden", expected_hidden_layers=32
        )
        == names[1:]
    )


def test_layer_selection_rejects_noncanonical_language_model_prefix():
    names = [f"other.layers.{index}.mlp.down_proj" for index in range(32)]

    try:
        select_target_layer_names(names, layer_mode="last", expected_hidden_layers=32)
    except ValueError as error:
        assert "actual=[]" in str(error)
    else:
        raise AssertionError("noncanonical layer path must not be accepted")


def test_three_frozen_variants_have_only_the_intended_differences():
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert set(payload["variants"]) == {
        "sphere_dense_last_adaptive_v1",
        "sphere_dense_all_hidden_adaptive_v1",
        "sphere_dense_last_fixed_alpha_v1",
    }

    last = load_sphere_variant(REGISTRY, "sphere_dense_last_adaptive_v1")
    all_hidden = load_sphere_variant(REGISTRY, "sphere_dense_all_hidden_adaptive_v1")
    fixed = load_sphere_variant(REGISTRY, "sphere_dense_last_fixed_alpha_v1")

    assert last.layer_mode == fixed.layer_mode == "last"
    assert all_hidden.layer_mode == "all_hidden"
    assert last.target_scale_mode == all_hidden.target_scale_mode == "adaptive_trace"
    assert fixed.target_scale_mode == "fixed"
    assert fixed.fixed_alpha == 1.0
    assert last.rho == all_hidden.rho == fixed.rho == 1e-3
