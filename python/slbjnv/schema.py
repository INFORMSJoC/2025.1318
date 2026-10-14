"""Shared fixed-width input schema from Supplementary Materials Table 1."""

from collections import OrderedDict

FEATURE_SLICES = OrderedDict(
    item_equip=520,
    model_batch=5,
    added_branch_constraints=100,
    node_variable_pair=100,
    dual_values=100,
    pricing_reduced_costs=30,
    distance_from_root=1,
    leaf_depths=100,
    current_depth=1,
    integer_variable_ratio=1,
    bounds=2,
    objective_sequence=20,
    lower_bound_sequence=20,
)
FEATURE_DIMENSION = sum(FEATURE_SLICES.values())
assert FEATURE_DIMENSION == 1000
