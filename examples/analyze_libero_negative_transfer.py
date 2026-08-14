#!/usr/bin/env python3
"""Replay LIBERO Spatial rollouts and save evidence for negative-transfer analysis.

This intentionally does not use LeRobot's ``eval.recording`` feature.  The
LIBERO observation feature names contain slashes, which are rejected by the
recording dataset schema in LeRobot 0.6.0.  Instead, this script mirrors the
evaluation step loop and writes failed videos plus compact JSONL transitions.

For an exact replay of tasks 3, 6 and 7, tasks 0 through 7 are executed in the
same order as the original evaluation.  This preserves the policy RNG stream;
only target-task failures are persisted.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import gymnasium as gym


DEFAULT_POLICY = Path(
    "/Users/takemo/PARC2026_outputs/20260814_121853/"
    "smolvla_libero_plus_spatial_lora_merged"
)
DEFAULT_OUTPUT = Path(
    "/Users/takemo/PARC2026_outputs/20260814_121853/"
    "failure_analysis/spatial_lora"
)
TARGET_TASK_IDS = (3, 6, 7)
SOURCE_OBJECT = "akita_black_bowl_1"
TARGET_OBJECT = "plate_1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if (
        value.__class__.__module__.startswith("torch")
        and hasattr(value, "detach")
        and hasattr(value, "cpu")
    ):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def _json_fallback(value: Any) -> Any:
    """Normalize scalar-like extension types missed by isinstance checks."""
    item = getattr(value, "item", None)
    if callable(item):
        normalized = item()
        if normalized is not value:
            return _jsonable(normalized)
    return repr(value)


def classify_failure(
    transitions: list[dict[str, Any]],
    *,
    reach_threshold_m: float = 0.10,
    lift_threshold_m: float = 0.03,
) -> dict[str, Any]:
    """Classify a failed pick-and-place episode using simulator state.

    Priority is sequential: the end effector must first reach the source,
    then establish a grasp or lift it, and finally place it.  ``grasped`` is
    read from MuJoCo contacts when available; a source-height increase is a
    fallback because contact inspection can vary across robosuite releases.
    """
    if not transitions:
        return {"category": "到達失敗", "reason": "状態遷移が空です"}

    distances = [
        item["state_before"].get("eef_to_source_distance_m")
        for item in transitions
    ]
    finite_distances = [
        float(value)
        for value in distances
        if value is not None and math.isfinite(float(value))
    ]
    reached = bool(finite_distances) and min(finite_distances) <= reach_threshold_m

    source_heights = [
        item["state_before"].get("source_height_m")
        for item in transitions
    ]
    finite_heights = [
        float(value)
        for value in source_heights
        if value is not None and math.isfinite(float(value))
    ]
    lift_delta = (
        max(finite_heights) - finite_heights[0]
        if finite_heights
        else 0.0
    )
    contact_grasp = any(
        bool(item["state_before"].get("grasp_contact"))
        for item in transitions
    )
    grasped = contact_grasp or lift_delta >= lift_threshold_m
    success = any(bool(item.get("is_success")) for item in transitions)

    if success:
        category = "成功"
        reason = "LIBEROの成功述語を満たしました"
    elif not reached:
        category = "到達失敗"
        reason = "手先が把持対象の10cm圏内へ到達していません"
    elif not grasped:
        category = "把持失敗"
        reason = "対象へ到達しましたが、接触把持も3cm以上の持ち上げもありません"
    else:
        category = "配置失敗"
        reason = "対象を把持または持ち上げましたが、plate上の成功述語を満たしていません"

    return {
        "category": category,
        "reason": reason,
        "reached_source": reached,
        "grasped_or_lifted": grasped,
        "contact_grasp_detected": contact_grasp,
        "min_eef_to_source_distance_m": min(finite_distances) if finite_distances else None,
        "max_source_lift_m": lift_delta,
        "thresholds": {
            "reach_m": reach_threshold_m,
            "lift_m": lift_threshold_m,
        },
    }


def _vector_success(info: dict[str, Any]) -> bool:
    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        value = final_info.get("is_success", False)
    elif final_info is not None and len(final_info) and isinstance(final_info[0], dict):
        value = final_info[0].get("is_success", False)
    else:
        value = info.get("is_success", False)
    array = np.asarray(value).reshape(-1)
    return bool(array[0]) if array.size else False


def _internal_state(vec_env: gym.vector.SyncVectorEnv) -> dict[str, Any]:
    wrapped = vec_env.envs[0]
    offscreen = getattr(wrapped, "_env", None)
    sim_env = getattr(offscreen, "env", None)
    if sim_env is None:
        return {}

    raw = sim_env._get_observations()
    eef_pos = np.asarray(raw.get("robot0_eef_pos"), dtype=float)
    source_pos = np.asarray(raw.get(f"{SOURCE_OBJECT}_pos"), dtype=float)
    target_pos = np.asarray(raw.get(f"{TARGET_OBJECT}_pos"), dtype=float)

    grasp_contact: bool | None = None
    source_model = getattr(sim_env, "objects_dict", {}).get(SOURCE_OBJECT)
    try:
        grasp_contact = bool(
            sim_env._check_grasp(
                gripper=sim_env.robots[0].gripper,
                object_geoms=source_model.contact_geoms,
            )
        )
    except (AttributeError, TypeError, ValueError):
        # Height-based detection remains available in classify_failure().
        pass

    def valid_xyz(value: np.ndarray) -> list[float] | None:
        return value.tolist() if value.shape == (3,) else None

    state = {
        "eef_pos": valid_xyz(eef_pos),
        "gripper_qpos": _jsonable(raw.get("robot0_gripper_qpos")),
        "gripper_qvel": _jsonable(raw.get("robot0_gripper_qvel")),
        "joint_pos": _jsonable(raw.get("robot0_joint_pos")),
        "joint_vel": _jsonable(raw.get("robot0_joint_vel")),
        "source_object": SOURCE_OBJECT,
        "source_pos": valid_xyz(source_pos),
        "source_quat": _jsonable(raw.get(f"{SOURCE_OBJECT}_quat")),
        "target_object": TARGET_OBJECT,
        "target_pos": valid_xyz(target_pos),
        "target_quat": _jsonable(raw.get(f"{TARGET_OBJECT}_quat")),
        "grasp_contact": grasp_contact,
    }
    if eef_pos.shape == source_pos.shape == (3,):
        state["eef_to_source_distance_m"] = float(np.linalg.norm(eef_pos - source_pos))
    else:
        state["eef_to_source_distance_m"] = None
    if source_pos.shape == (3,):
        state["source_height_m"] = float(source_pos[2])
    else:
        state["source_height_m"] = None
    if source_pos.shape == target_pos.shape == (3,):
        state["source_to_target_distance_m"] = float(np.linalg.norm(source_pos - target_pos))
    else:
        state["source_to_target_distance_m"] = None
    return state


def _write_jsonl(path: Path, transitions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for transition in transitions:
            handle.write(
                json.dumps(
                    _jsonable(transition),
                    ensure_ascii=False,
                    default=_json_fallback,
                )
                + "\n"
            )


def _mark_time_limit(transitions: list[dict[str, Any]]) -> None:
    """Mark the final transition when the local episode horizon is exhausted."""
    if not transitions:
        return
    transitions[-1]["done"] = True
    transitions[-1]["truncated"] = True
    transitions[-1]["termination_reason"] = "max_episode_steps"


def _run_episode(
    *,
    vec_env: gym.vector.SyncVectorEnv,
    policy: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    seed: int,
    capture: bool,
) -> tuple[bool, list[np.ndarray], list[dict[str, Any]]]:
    import torch

    from lerobot.envs import preprocess_observation
    from lerobot.utils.constants import ACTION

    policy.reset()
    observation, _ = vec_env.reset(seed=[seed])
    frames = [vec_env.envs[0].render()] if capture else []
    transitions: list[dict[str, Any]] = []
    max_steps = int(vec_env.call("_max_episode_steps")[0])

    for step in range(max_steps):
        state_before = _internal_state(vec_env) if capture else {}
        policy_observation = preprocess_observation(deepcopy(observation))
        try:
            policy_observation["task"] = list(vec_env.call("task_description"))
        except (AttributeError, NotImplementedError):
            policy_observation["task"] = list(vec_env.call("task"))
        policy_observation = env_preprocessor(policy_observation)
        policy_observation = preprocessor(policy_observation)
        with torch.inference_mode():
            action = policy.select_action(policy_observation)
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        action_numpy = action.detach().cpu().numpy()

        observation, reward, terminated, truncated, info = vec_env.step(action_numpy)
        is_success = _vector_success(info)
        is_done = bool(terminated[0] or truncated[0])
        if capture:
            state_after = _internal_state(vec_env)
            frames.append(vec_env.envs[0].render())
            transitions.append(
                {
                    "step": step,
                    "seed": seed,
                    "state_before": state_before,
                    "state_after": state_after,
                    "action": action_numpy[0],
                    "reward": float(reward[0]),
                    "done": is_done,
                    "terminated": bool(terminated[0]),
                    "truncated": bool(truncated[0]),
                    "is_success": is_success,
                    "info": info,
                }
            )
        if is_done:
            return is_success, frames, transitions

    if capture:
        _mark_time_limit(transitions)
    return False, frames, transitions


def run(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs import make_env, make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.io_utils import write_video
    from lerobot.utils.random_utils import set_seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_task_ids = list(range(max(args.target_task_ids) + 1))
    camera_mapping = {
        "agentview_image": "front",
        "robot0_eye_in_hand_image": "wrist",
    }
    env_cfg = LiberoEnv(
        task="libero_spatial",
        task_ids=replay_task_ids,
        observation_height=256,
        observation_width=256,
        control_mode="relative",
        camera_name_mapping=camera_mapping,
        is_libero_plus=True,
    )

    set_seed(args.seed)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)

    episode_summaries: list[dict[str, Any]] = []
    try:
        for task_id in replay_task_ids:
            vec_env = envs["libero_spatial"][task_id]
            task_name = str(vec_env.call("task")[0])
            for episode_index in range(args.episodes):
                seed = args.seed + episode_index
                capture = task_id in args.target_task_ids
                print(
                    f"task={task_id} episode={episode_index} seed={seed} capture={capture}",
                    flush=True,
                )
                success, frames, transitions = _run_episode(
                    vec_env=vec_env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    seed=seed,
                    capture=capture,
                )
                summary = {
                    "task_id": task_id,
                    "task_name": task_name,
                    "episode_index": episode_index,
                    "seed": seed,
                    "success": success,
                    "steps": len(transitions) if capture else None,
                }
                if capture and not success:
                    stem = f"task_{task_id}_episode_{episode_index}_seed_{seed}"
                    video_path = args.output_dir / f"{stem}.mp4"
                    transitions_path = args.output_dir / f"{stem}.jsonl"
                    write_video(str(video_path), np.asarray(frames), args.video_fps)
                    _write_jsonl(transitions_path, transitions)
                    summary.update(
                        {
                            "video_path": str(video_path),
                            "transitions_path": str(transitions_path),
                            "classification": classify_failure(transitions),
                        }
                    )
                episode_summaries.append(summary)
    finally:
        for task_envs in envs.values():
            for vec_env in task_envs.values():
                vec_env.close()

    result = {
        "policy_path": str(args.policy_path),
        "seed": args.seed,
        "episodes_per_task": args.episodes,
        "replay_task_ids": replay_task_ids,
        "target_task_ids": args.target_task_ids,
        "note": "tasks 0..max(target) are replayed to preserve the original policy RNG stream",
        "episodes": episode_summaries,
    }
    summary_path = args.output_dir / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(
            _jsonable(result),
            ensure_ascii=False,
            indent=2,
            default=_json_fallback,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved: {summary_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-task-ids", type=int, nargs="+", default=list(TARGET_TASK_IDS))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--video-fps", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
