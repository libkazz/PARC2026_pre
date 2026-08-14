import numpy as np
import pytest

from examples.smolvla_training_data import (
    EpisodeTrajectory,
    build_episode_trajectories,
    evaluate_gripper_quality,
    select_episodes_by_trajectory,
    trajectory_fingerprint,
)


def actions_with_gripper(*runs):
    values = []
    for value, length in runs:
        for _ in range(length):
            values.append([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, value])
    return np.asarray(values, dtype=np.float32)


def trajectory(episode_index, value, *, gripper_runs=None):
    if gripper_runs is None:
        actions = np.full((4, 7), value, dtype=np.float32)
        actions[:, -1] = -1
    else:
        actions = actions_with_gripper(*gripper_runs)
    states = np.full((len(actions), 8), value, dtype=np.float32)
    return EpisodeTrajectory(episode_index, actions, states)


def test_builds_episode_trajectories_from_contiguous_frames():
    result = build_episode_trajectories(
        np.asarray([4, 4, 9]),
        np.arange(21, dtype=np.float32).reshape(3, 7),
        np.arange(24, dtype=np.float32).reshape(3, 8),
    )

    assert list(result) == [4, 9]
    assert result[4].actions.shape == (2, 7)
    assert result[9].states.shape == (1, 8)


def test_rejects_non_contiguous_episode_frames():
    with pytest.raises(ValueError, match="連続していません"):
        build_episode_trajectories(
            np.asarray([4, 9, 4]),
            np.zeros((3, 7), dtype=np.float32),
            np.zeros((3, 8), dtype=np.float32),
        )


def test_fingerprint_uses_action_and_state_sequences():
    original = trajectory(1, 0.1)
    same_sequence = trajectory(2, 0.1)
    different_state = EpisodeTrajectory(
        3,
        original.actions.copy(),
        original.states + 0.1,
    )

    assert trajectory_fingerprint(original) == trajectory_fingerprint(
        same_sequence
    )
    assert trajectory_fingerprint(original) != trajectory_fingerprint(
        different_state
    )


def test_accepts_open_close_release_gripper_sequence():
    result = evaluate_gripper_quality(
        actions_with_gripper((-1, 4), (1, 5), (-1, 3))
    )

    assert result.passed
    assert result.run_values == (-1, 1, -1)
    assert result.run_lengths == (4, 5, 3)


@pytest.mark.parametrize(
    ("runs", "reason"),
    [
        (((-1, 4), (1, 5)), "解放操作がありません"),
        (
            ((-1, 4), (1, 5), (-1, 1), (1, 1), (-1, 3)),
            "余分なグリッパー反転があります",
        ),
        (((-1, 2), (1, 5), (-1, 3)), "3フレーム未満"),
    ],
)
def test_rejects_low_quality_gripper_sequences(runs, reason):
    result = evaluate_gripper_quality(actions_with_gripper(*runs))

    assert not result.passed
    assert reason in result.reason


def test_selects_distinct_trajectories_and_spreads_visual_variants():
    candidates = []
    for behavior in range(5):
        for variant in range(3):
            candidates.append(
                trajectory(behavior * 10 + variant, behavior + 0.1)
            )

    result = select_episodes_by_trajectory(candidates, 3)

    assert result.selected_episode_indices == (0, 21, 42)
    assert result.unique_trajectory_count == 5
    assert len(set(result.selected_trajectory_fingerprints)) == 3


def test_applies_gripper_quality_only_when_requested():
    bad = trajectory(
        1,
        0.1,
        gripper_runs=((-1, 4), (1, 5)),
    )
    good = [
        trajectory(
            index,
            float(index),
            gripper_runs=((-1, 4), (1, 5), (-1, 3)),
        )
        for index in range(2, 5)
    ]

    unfiltered = select_episodes_by_trajectory([bad, *good], 1)
    filtered = select_episodes_by_trajectory(
        [bad, *good],
        1,
        require_gripper_quality=True,
    )

    assert unfiltered.quality_rejections == ()
    assert filtered.quality_rejections[0].episode_index == 1
    assert filtered.quality_candidate_count == 3
