import pytest

from concurrent_smoke import require_generation_checks


CHECKS = (
    "serial_repeat_matches",
    "outputs_match",
    "static_pp2_concurrent_matches",
    "static_pp3_concurrent_matches",
)


def test_all_static_and_migrating_lanes_are_required():
    require_generation_checks({name: [True, True] for name in CHECKS})


@pytest.mark.parametrize("name", CHECKS)
@pytest.mark.parametrize("values", [None, [], [True], [True, False]])
def test_missing_empty_or_failed_control_cannot_qualify(name, values):
    result = {key: [True, True] for key in CHECKS}
    result[name] = values
    with pytest.raises(AssertionError, match=name):
        require_generation_checks(result)
