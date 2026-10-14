import json

from slbjnv.dataset import OrderedInstanceDataset
from slbjnv.schema import FEATURE_DIMENSION, FEATURE_SLICES


def test_feature_schema_is_1000() -> None:
    assert FEATURE_DIMENSION == 1000
    assert sum(FEATURE_SLICES.values()) == 1000


def test_dataset_preserves_instance_time_order(tmp_path) -> None:
    path = tmp_path / "samples.txt"
    base = {
        "node_id": 1,
        "parent_node_id": 0,
        "branching_rule": 1,
        "branching_variable": {"item": 0},
        "feature_vector": [0.0] * FEATURE_DIMENSION,
        "target_score": 1.0,
        "key_node_sequence": [0, 1],
        "key_value_sequence": [0.0, 0.5, 0.75, 1.0],
        "root_objective": 10.0,
    }
    rows = [dict(base, instance_id="b", decision_index=2),
            dict(base, instance_id="a", decision_index=0),
            dict(base, instance_id="b", decision_index=1)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    groups = OrderedInstanceDataset(path)
    assert [sample.instance_id for sample in groups[0]] == ["a"]
    assert [sample.decision_index for sample in groups[1]] == [1, 2]
