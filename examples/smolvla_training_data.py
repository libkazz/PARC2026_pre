"""SmolVLA の追加学習に使うエピソードを選別する。"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeTrajectory:
    """1エピソード分の行動・状態系列。"""

    episode_index: int
    actions: np.ndarray
    states: np.ndarray


@dataclass(frozen=True)
class GripperQuality:
    """グリッパー操作の品質判定結果。"""

    passed: bool
    run_values: tuple[int, ...]
    run_lengths: tuple[int, ...]
    reason: str | None = None


@dataclass(frozen=True)
class EpisodeRejection:
    """品質判定で除外したエピソード。"""

    episode_index: int
    reason: str


@dataclass(frozen=True)
class EpisodeSelection:
    """行動軌跡単位のエピソード選択結果。"""

    selected_episode_indices: tuple[int, ...]
    selected_trajectory_fingerprints: tuple[str, ...]
    candidate_count: int
    quality_candidate_count: int
    unique_trajectory_count: int
    quality_rejections: tuple[EpisodeRejection, ...]


def build_episode_trajectories(
    frame_episode_indices: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray,
) -> dict[int, EpisodeTrajectory]:
    """フレーム列をエピソードごとの連続した系列に分割する。"""

    episode_indices = np.asarray(frame_episode_indices)
    action_values = np.asarray(actions)
    state_values = np.asarray(states)

    if episode_indices.ndim != 1:
        raise ValueError("frame_episode_indices は1次元である必要があります。")
    if action_values.ndim != 2 or state_values.ndim != 2:
        raise ValueError("actions と states は2次元である必要があります。")
    if not (
        len(episode_indices)
        == len(action_values)
        == len(state_values)
    ):
        raise ValueError("フレーム数が一致していません。")
    if len(episode_indices) == 0:
        return {}

    boundaries = np.flatnonzero(
        episode_indices[1:] != episode_indices[:-1]
    ) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_indices)]))
    trajectories: dict[int, EpisodeTrajectory] = {}

    for start, end in zip(starts, ends, strict=True):
        episode_index = int(episode_indices[start])
        if episode_index in trajectories:
            raise ValueError(
                f"episode {episode_index} のフレームが連続していません。"
            )
        trajectories[episode_index] = EpisodeTrajectory(
            episode_index=episode_index,
            actions=action_values[start:end],
            states=state_values[start:end],
        )

    return trajectories


def trajectory_fingerprint(trajectory: EpisodeTrajectory) -> str:
    """action と state の完全一致を判定できるハッシュを返す。"""

    actions = np.asarray(trajectory.actions, dtype="<f4", order="C")
    states = np.asarray(trajectory.states, dtype="<f4", order="C")

    if actions.ndim != 2 or states.ndim != 2:
        raise ValueError("actions と states は2次元である必要があります。")
    if len(actions) == 0 or len(actions) != len(states):
        raise ValueError("action と state の系列長が不正です。")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("action または state に NaN/Inf が含まれています。")

    digest = hashlib.sha256()
    digest.update(np.asarray(actions.shape, dtype="<i8").tobytes())
    digest.update(actions.tobytes())
    digest.update(np.asarray(states.shape, dtype="<i8").tobytes())
    digest.update(states.tobytes())
    return digest.hexdigest()


def evaluate_gripper_quality(
    actions: np.ndarray,
    *,
    min_run_frames: int = 3,
) -> GripperQuality:
    """開く→閉じる→開くの3区間になっているか判定する。"""

    action_values = np.asarray(actions)
    if action_values.ndim != 2 or action_values.shape[1] < 1:
        raise ValueError("actions はグリッパーを含む2次元配列が必要です。")
    if min_run_frames < 1:
        raise ValueError("min_run_frames は1以上である必要があります。")

    gripper = action_values[:, -1]
    if len(gripper) == 0 or not np.isfinite(gripper).all():
        raise ValueError("グリッパーactionが空、またはNaN/Infを含みます。")
    if np.any(gripper == 0):
        return GripperQuality(
            passed=False,
            run_values=(),
            run_lengths=(),
            reason="グリッパーactionに開閉を判定できない0が含まれます。",
        )

    signs = np.where(gripper > 0, 1, -1)
    boundaries = np.flatnonzero(signs[1:] != signs[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(signs)]))
    run_values = tuple(int(value) for value in signs[starts])
    run_lengths = tuple(
        int(end - start)
        for start, end in zip(starts, ends, strict=True)
    )

    expected = (-1, 1, -1)
    if run_values != expected:
        if run_values == (-1, 1):
            reason = "把持後の解放操作がありません。"
        elif len(run_values) > len(expected):
            reason = "余分なグリッパー反転があります。"
        else:
            reason = "グリッパー操作が開く→閉じる→開くの順ではありません。"
        return GripperQuality(
            passed=False,
            run_values=run_values,
            run_lengths=run_lengths,
            reason=reason,
        )

    if min(run_lengths) < min_run_frames:
        return GripperQuality(
            passed=False,
            run_values=run_values,
            run_lengths=run_lengths,
            reason=(
                f"{min_run_frames}フレーム未満の短い"
                "グリッパー操作区間があります。"
            ),
        )

    return GripperQuality(
        passed=True,
        run_values=run_values,
        run_lengths=run_lengths,
    )


def _evenly_spaced_positions(length: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("count は1以上である必要があります。")
    if length < count:
        raise ValueError(
            f"{count}件の選択に対して候補が{length}件しかありません。"
        )
    if count == 1:
        return [length // 2]

    return [
        round(index * (length - 1) / (count - 1))
        for index in range(count)
    ]


def select_episodes_by_trajectory(
    trajectories: Sequence[EpisodeTrajectory],
    count: int,
    *,
    require_gripper_quality: bool = False,
    min_gripper_run_frames: int = 3,
) -> EpisodeSelection:
    """異なる行動軌跡から代表エピソードを決定的に選ぶ。"""

    accepted: list[EpisodeTrajectory] = []
    rejections: list[EpisodeRejection] = []

    for trajectory in trajectories:
        if require_gripper_quality:
            quality = evaluate_gripper_quality(
                trajectory.actions,
                min_run_frames=min_gripper_run_frames,
            )
            if not quality.passed:
                rejections.append(
                    EpisodeRejection(
                        episode_index=trajectory.episode_index,
                        reason=quality.reason or "グリッパー品質判定に失敗しました。",
                    )
                )
                continue
        accepted.append(trajectory)

    grouped: dict[str, list[EpisodeTrajectory]] = defaultdict(list)
    for trajectory in accepted:
        grouped[trajectory_fingerprint(trajectory)].append(trajectory)

    fingerprints = list(grouped)
    selected_group_positions = _evenly_spaced_positions(
        len(fingerprints),
        count,
    )
    selected_fingerprints = [
        fingerprints[position]
        for position in selected_group_positions
    ]

    selected_episodes: list[int] = []
    for rank, fingerprint in enumerate(selected_fingerprints):
        variants = grouped[fingerprint]
        if count == 1:
            variant_position = len(variants) // 2
        else:
            variant_position = round(
                rank * (len(variants) - 1) / (count - 1)
            )
        selected_episodes.append(
            variants[variant_position].episode_index
        )

    return EpisodeSelection(
        selected_episode_indices=tuple(selected_episodes),
        selected_trajectory_fingerprints=tuple(selected_fingerprints),
        candidate_count=len(trajectories),
        quality_candidate_count=len(accepted),
        unique_trajectory_count=len(grouped),
        quality_rejections=tuple(rejections),
    )
