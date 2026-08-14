import json

from examples.analyze_libero_negative_transfer import (
    _json_fallback,
    _mark_time_limit,
    classify_failure,
)


def transition(distance, height, *, grasp=False, success=False):
    return {
        "state_before": {
            "eef_to_source_distance_m": distance,
            "source_height_m": height,
            "grasp_contact": grasp,
        },
        "is_success": success,
    }


def test_classifies_reach_failure():
    result = classify_failure([transition(0.30, 0.10), transition(0.15, 0.10)])
    assert result["category"] == "到達失敗"


def test_classifies_grasp_failure():
    result = classify_failure([transition(0.08, 0.10), transition(0.05, 0.11)])
    assert result["category"] == "把持失敗"


def test_classifies_place_failure_from_lift():
    result = classify_failure([transition(0.08, 0.10), transition(0.05, 0.14)])
    assert result["category"] == "配置失敗"


def test_classifies_place_failure_from_contact():
    result = classify_failure([transition(0.08, 0.10, grasp=True)])
    assert result["category"] == "配置失敗"


def test_success_takes_precedence():
    result = classify_failure([transition(0.30, 0.10, success=True)])
    assert result["category"] == "成功"


def test_json_fallback_normalizes_extension_scalar():
    class ExtensionScalar:
        def item(self):
            return True

    assert json.dumps(ExtensionScalar(), default=_json_fallback) == "true"


def test_marks_final_transition_as_time_limit():
    transitions = [{"done": False, "truncated": False}]
    _mark_time_limit(transitions)
    assert transitions[-1] == {
        "done": True,
        "truncated": True,
        "termination_reason": "max_episode_steps",
    }
