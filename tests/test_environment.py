import pytest

from intent_alignment_toy.environment import Task, TextNavigationEnv


def test_reaches_target_before_budget():
    env = TextNavigationEnv(Task("example", "五步以内向右走三米", (3, 0), 5))
    assert not env.step("RIGHT").terminated
    assert not env.step("RIGHT").terminated
    result = env.step("RIGHT")
    assert result.success
    assert result.position == (3, 0)
    assert result.steps_used == 3
    assert result.remaining_steps == 2
    assert result.reason == "target_reached"
    assert result.trajectory == ((0, 0), (1, 0), (2, 0), (3, 0))


def test_can_recover_from_detour_when_budget_allows():
    env = TextNavigationEnv(Task("detour", "五步以内向右走三米", (3, 0), 5))
    for action in ["LEFT", "RIGHT", "RIGHT", "RIGHT"]:
        result = env.step(action)
    assert not result.success
    assert not result.terminated
    assert env.step("RIGHT").success


def test_budget_exhaustion_fails():
    env = TextNavigationEnv(Task("failure", "三步以内向右走三米", (3, 0), 3))
    for action in ["UP", "RIGHT", "RIGHT"]:
        result = env.step(action)
    assert result.terminated
    assert not result.success
    assert result.reason == "budget_exhausted"


def test_invalid_action_fails_immediately():
    env = TextNavigationEnv(Task("invalid", "向右走一米", (1, 0), 1))
    result = env.step("GO_RIGHT")
    assert result.terminated
    assert not result.success
    assert result.reason == "invalid_action"


def test_cannot_step_after_termination():
    env = TextNavigationEnv(Task("done", "向右走一米", (1, 0), 1))
    env.step("RIGHT")
    with pytest.raises(RuntimeError):
        env.step("RIGHT")
