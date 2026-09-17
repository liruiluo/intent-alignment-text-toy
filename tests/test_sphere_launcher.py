import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "run_verl_one_step.sh"
SPHERE = ROOT / "scripts" / "run_verl_sphere_one_step.sh"
WORKER = ROOT / "src" / "intent_alignment_toy" / "sphere_verl.py"
MAIN = ROOT / "src" / "intent_alignment_toy" / "sphere_main.py"


def test_base_launcher_has_a_narrow_custom_entrypoint_switch():
    script = BASE.read_text(encoding="utf-8")

    assert "TRAIN_MODULE=${TRAIN_MODULE:-verl.trainer.main_ppo}" in script
    assert "verl.trainer.main_ppo|intent_alignment_toy.sphere_main" in script
    assert '"$PYTHON_BIN" -m "$TRAIN_MODULE"' in script


def test_sphere_launcher_disables_recompute_and_compile_for_hooks():
    script = SPHERE.read_text(encoding="utf-8")

    assert "ACTOR_GRADIENT_CHECKPOINTING=false" in script
    assert "ACTOR_USE_TORCH_COMPILE=false" in script
    assert "IATOY_SPHERE_EVIDENCE_PATH" in script
    assert 'IATOY_SPHERE_RUN_ID="$(basename "$RUN_DIR")"' in script
    assert "IATOY_SPHERE_RUN_NONCE" in script
    assert "refusing to reuse an existing SPHERE worker evidence target" in script
    assert 'IATOY_SPHERE_EVIDENCE_PATH="${SPHERE_EVIDENCE_PATH:-' not in script
    assert "intent_alignment_toy.sphere_main" in script


def test_sphere_changes_the_actor_worker_not_the_agent_loop():
    worker = WORKER.read_text(encoding="utf-8")
    main = MAIN.read_text(encoding="utf-8")

    assert "actor_worker_cls = SphereTrainingWorker" in worker
    assert "SphereActorRolloutRefWorker" in worker
    assert "AgentLoopManagerTQ" in main
    assert "class SphereTaskRunnerV1" in main
    assert "generate_sequences" not in worker


def test_adaptive_gradient_ratio_has_two_backward_passes_explicit_sum_and_one_step():
    worker = WORKER.read_text(encoding="utf-8")

    assert worker.count("self.engine.forward_backward_batch(") == 2
    assert worker.count("self.engine.optimizer_step()") == 1
    assert "self._assemble_optimizer_gradient(base_gradients)" in worker
    assert '"full_tensor_residual_l2_norm"' in worker
    assert '"postcast_values_different_from_ppo"' in worker
    assert '"explicit_gradient_combination": 1' in worker
    assert '"combined_forward_backward": 0' in worker
    assert "self.sphere_variant.rho" in worker
    assert '"optimizer_step_exactly_one"' in worker

    train_batch = worker[worker.index("    def train_batch(") :]
    ppo_backward = train_batch.index("loss_function=self._ppo_collect_loss")
    clone_ppo = train_batch.index("self._clone_parameter_gradients()")
    zero_before_sphere = train_batch.index(
        "self.engine.optimizer_zero_grad()", clone_ppo
    )
    sphere_backward = train_batch.index("loss_function=self._sphere_only_loss")
    assemble = train_batch.index("self._assemble_optimizer_gradient(base_gradients)")
    optimizer_step = train_batch.index("self.engine.optimizer_step()")
    assert (
        ppo_backward
        < clone_ppo
        < zero_before_sphere
        < sphere_backward
        < assemble
        < optimizer_step
    )


def test_sphere_uses_one_exact_logical_minibatch_objective():
    worker = WORKER.read_text(encoding="utf-8")

    assert "_torch_sphere_penalty_and_feature_gradient" in worker
    assert "_finalize_logical_objective" in worker
    assert "_logical_feature_chunks" in worker
    assert "_logical_gradient_chunks" in worker
    assert '"logical_minibatch_exact_penalty": True' in worker
    assert '"physical_microbatch_penalty": False' in worker
    assert "_microbatch_identity" in worker


def test_sphere_only_backward_keeps_an_fsdp_output_anchor():
    worker = WORKER.read_text(encoding="utf-8")

    assert "anchored_surrogate = surrogate + base_loss * 0.0" in worker


def test_replay_surrogate_uses_stop_gradient_value_anchor():
    worker = WORKER.read_text(encoding="utf-8")

    assert "surrogate = raw_surrogate - raw_surrogate.detach() + value_anchor" in worker
    assert '"worker-evidence.failed.json"' in worker


def test_sphere_replay_restores_the_collection_torch_rng_state():
    worker = WORKER.read_text(encoding="utf-8")

    assert "replay_rng_state = self._capture_torch_rng_state()" in worker
    assert worker.count("self._restore_torch_rng_state(replay_rng_state)") == 1
    assert '"feature_relative_l2_difference"' in worker
    assert '"feature_replay_within_tolerance"' in worker


def test_gradient_projection_avoids_cuda_linspace_index_rounding():
    worker = WORKER.read_text(encoding="utf-8")

    assert "torch.linspace(" not in worker
    assert "flat.narrow(0, 0, count)" in worker


def test_sphere_packed_mask_uses_per_sequence_causal_alignment():
    worker = WORKER.read_text(encoding="utf-8")

    assert "build_packed_response_predictor_mask(" in worker
    assert "torch.roll(mask" not in worker
    assert "input_ids.offsets().diff()" in worker
    assert "attention_mask[:, :prompt_width].sum(dim=1)" in worker


def test_global_sphere_metrics_follow_verl_singleton_metric_contract():
    worker = WORKER.read_text(encoding="utf-8")

    assert 'output["metrics"][name] = Metric(' in worker
    assert 'output["metrics"][name] = [' not in worker


def test_sphere_loss_callbacks_match_verl_keyword_contract():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for name in ("_sphere_only_loss",):
        assert [argument.arg for argument in functions[name].args.args] == [
            "self",
            "model_output",
            "data",
            "dp_group",
        ]
    assert "_combined_loss" not in functions
