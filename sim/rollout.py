"""Single-episode control loop; dependency-light so preprocessing can be tested locally."""

import math
from collections import deque

import numpy as np

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def quaternion_to_axis_angle(quaternion):
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("Expected a finite XYZW quaternion")
    w = float(np.clip(quat[3], -1, 1))
    denominator = math.sqrt(1 - w * w)
    if math.isclose(denominator, 0):
        return np.zeros(3)
    return quat[:3] * (2 * math.acos(w)) / denominator


def make_observation(obs, prompt, resize):
    # Match upstream LIBERO evaluation: rotate BOTH camera images by 180 degrees.
    def image(key):
        return resize(np.ascontiguousarray(obs[key][::-1, ::-1]))

    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quaternion_to_axis_angle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("LIBERO observation must contain a finite 8D robot state")
    return {
        "observation/image": image("agentview_image"),
        "observation/wrist_image": image("robot0_eye_in_hand_image"),
        "observation/state": state,
        "prompt": str(prompt),
    }


def validate_actions(actions, replan_steps):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < replan_steps:
        raise ValueError("Policy must return an action chunk of shape (H, 7), H >= replan_steps")
    if not np.isfinite(actions).all():
        raise ValueError("Policy returned non-finite actions")
    return actions


def run_episode(
    env,
    client,
    initial_state,
    prompt,
    resize,
    *,
    max_steps=220,
    replan_steps=5,
    settle_steps=10,
    on_frame=None,
):
    if max_steps < 1 or replan_steps < 1 or settle_steps < 0:
        raise ValueError("Invalid episode length, replanning interval, or settling steps")
    env.reset()
    obs = env.set_init_state(initial_state)
    client.reset()
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    plan = deque()
    calls = 0
    success = False
    for step in range(max_steps):  # noqa: B007 -- used to report executed steps after the loop
        observation = make_observation(obs, prompt, resize)
        if on_frame is not None:
            on_frame(observation["observation/image"])
        if not plan:
            actions = validate_actions(client.infer(observation)["actions"], replan_steps)
            plan.extend(actions[:replan_steps])
            calls += 1
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            success = True
            break
    return {
        "success": success,
        "steps": step + 1,
        "policy_calls": calls,
        "settle_steps": settle_steps,
    }
