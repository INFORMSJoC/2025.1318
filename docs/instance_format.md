# GBD instance text format

Each file is UTF-8 text and contains all data required by the Section 3 GBD
master and pricing models. IDs are zero-based and contiguous.

```text
SLBJNV_GBD_INSTANCE_V1
instance_id "gbd_2_6_example"
num_items 6
num_machines 2
items
0 10 5
...
machines
0 80
...
compatibility
110000
...
end
```

An item row is `item_id weight profit`. A machine row is
`machine_id capacity`. Compatibility row `k` encodes the master compatibility
set `I_k`: character `i` is one exactly when item `i` may be batched with
master item `k`. The diagonal must therefore be one.

The reader rejects malformed dimensions, non-binary compatibility data,
non-contiguous IDs, non-positive weights or capacities, negative profits,
items that fit no machine, and trailing content.

Generate the configured default set with:

```text
generate_instances --config configs/instance_generator.yaml
```

Use `--seed 1234` for a reproducible run or `--output DIRECTORY` to override
the configured destination. Validate files independently with:

```text
python scripts/validate_instances.py --config configs/instance_generator.yaml data/instances
```
