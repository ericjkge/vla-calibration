"""Exercise control-loop invariants without a model download or simulator install."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rollout = load_file("rollout", "sim/rollout.py")
patcher = load_file("patch_transformers", "scripts/patch_transformers.py")
launcher = load_file("run_smoke", "scripts/run_smoke.py")


def observation():
    return {
        "agentview_image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "robot0_eye_in_hand_image": np.arange(18, 36, dtype=np.uint8).reshape(2, 3, 3),
        "robot0_eef_pos": np.array([1, 2, 3]),
        "robot0_eef_quat": np.array([0, 0, 0, 1.0000001]),
        "robot0_gripper_qpos": np.array([0.1, 0.2]),
    }


def test_preprocessing_rotates_both_cameras_and_preserves_input():
    obs = observation()
    result = rollout.make_observation(obs, "pick up", lambda x: x)
    np.testing.assert_array_equal(result["observation/image"], obs["agentview_image"][::-1, ::-1])
    np.testing.assert_array_equal(
        result["observation/wrist_image"], obs["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    np.testing.assert_allclose(result["observation/state"], [1, 2, 3, 0, 0, 0, 0.1, 0.2])
    assert obs["robot0_eef_quat"][3] == 1.0000001


class Environment:
    def __init__(self, success_at=None):
        self.actions = []
        self.success_at = success_at

    def reset(self):
        self.actions = []

    def set_init_state(self, state):
        assert state == "initial"
        return observation()

    def step(self, action):
        self.actions.append(action)
        return observation(), 0, len(self.actions) == self.success_at, {}


class Client:
    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        return {"actions": np.full((10, 7), self.calls, dtype=float)}


def test_one_episode_settles_replans_and_stops_on_success():
    env, client = Environment(success_at=17), Client()
    result = rollout.run_episode(env, client, "initial", "test", lambda x: x)
    assert result == {"success": True, "steps": 7, "policy_calls": 2, "settle_steps": 10}
    assert env.actions[:10] == [rollout.DUMMY_ACTION] * 10
    assert env.actions[10:15] == [[1.0] * 7] * 5
    assert env.actions[15:] == [[2.0] * 7] * 2


def test_horizon_excludes_settling_steps():
    env = Environment()
    result = rollout.run_episode(env, Client(), "initial", "test", lambda x: x, max_steps=3)
    assert result["success"] is False
    assert result["steps"] == 3
    assert len(env.actions) == 13


@pytest.mark.parametrize(
    "actions", [np.zeros((4, 7)), np.zeros((10, 32)), np.full((10, 7), np.nan)]
)
def test_bad_actions_fail_instead_of_becoming_task_failures(actions):
    with pytest.raises(ValueError):
        rollout.validate_actions(actions, 5)


def test_runtime_error_propagates():
    class BrokenClient(Client):
        def infer(self, obs):
            raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        rollout.run_episode(Environment(), BrokenClient(), "initial", "test", lambda x: x)


def test_patching_breaks_hardlinks(tmp_path):
    source, installed = tmp_path / "source", tmp_path / "installed"
    source.mkdir()
    installed.mkdir()
    cache = tmp_path / "cached.py"
    cache.write_text("original")
    os.link(cache, installed / "module.py")
    (source / "module.py").write_text("patched")
    assert patcher.patch_tree(source, installed) == 1
    assert (installed / "module.py").read_text() == "patched"
    assert cache.read_text() == "original"


def test_checkpoint_requires_weights_and_copies_assets_from_checkpoint_root(tmp_path):
    from vlacal.policy.checkpoint import copy_normalization_assets, validate_checkpoint

    source, target = tmp_path / "source", tmp_path / "target"
    asset = source / "assets/physical-intelligence/libero/norm_stats.json"
    asset.parent.mkdir(parents=True)
    asset.write_text(json.dumps({"norm_stats": {}}))
    target.mkdir()
    with pytest.raises(FileNotFoundError, match="PyTorch"):
        validate_checkpoint(target)
    (target / "model.safetensors").write_bytes(b"test")
    with pytest.raises(FileNotFoundError, match="normalization"):
        validate_checkpoint(target)
    copy_normalization_assets(source, target)
    validate_checkpoint(target)
    assert (
        target / "assets/physical-intelligence/libero/norm_stats.json"
    ).read_bytes() == asset.read_bytes()


def test_dead_server_fails_readiness_immediately():
    class DeadProcess:
        returncode = 7

        def poll(self):
            return self.returncode

    with pytest.raises(RuntimeError, match="code 7"):
        launcher.wait_for_server(DeadProcess(), 1, 30)
