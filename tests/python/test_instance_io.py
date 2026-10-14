from pathlib import Path

import pytest

from slbjnv.instance_io import read_instance


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "gbd_2_6_debug.txt"


def test_read_and_validate_debug_instance() -> None:
    instance = read_instance(FIXTURE)
    assert instance.instance_id == "gbd_2_6_debug"
    assert len(instance.items) == 6
    assert len(instance.machines) == 2
    assert instance.feasible_master_item_pairs() == 6


def test_rejects_nonbinary_compatibility(tmp_path: Path) -> None:
    damaged = tmp_path / "damaged.txt"
    damaged.write_text(FIXTURE.read_text(encoding="utf-8").replace("110000", "1x0000", 1),
                       encoding="utf-8")
    with pytest.raises(ValueError, match="compatibility"):
        read_instance(damaged)
