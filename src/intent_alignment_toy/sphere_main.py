"""Hydra-compatible veRL v0.9.0 entrypoint for dense SPHERE training."""

from __future__ import annotations

from pprint import pprint

import ray
from omegaconf import OmegaConf

from intent_alignment_toy.sphere_verl import SpherePPOTrainerSync


@ray.remote
class SphereTaskRunnerV1:
    """Minimal v0.9.0 TaskRunnerV1 specialization with a SPHERE trainer."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ
        from verl.utils.import_utils import load_class_from_fqn

        manager_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
            "agent_loop_manager_class"
        )
        manager_cls = (
            load_class_from_fqn(manager_fqn, "AgentLoopManager")
            if manager_fqn
            else AgentLoopManagerTQ
        )
        self.agent_loop_manager = manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config):
        import transfer_queue as tq
        from verl.utils.logging_utils import configure_verl_logging

        configure_verl_logging()
        if config.trainer.v1.trainer_mode != "sync":
            raise ValueError("dense SPHERE toy supports only veRL V1 sync mode")
        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = SpherePPOTrainerSync(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


def main() -> None:
    """Reuse the upstream Hydra CLI while replacing only its V1 task runner."""
    import verl.trainer.main_ppo as upstream_main

    upstream_main.TaskRunnerV1 = SphereTaskRunnerV1
    upstream_main.main()


if __name__ == "__main__":
    main()
