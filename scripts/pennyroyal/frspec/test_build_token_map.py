import pytest
from build_token_map import select_ids


def test_deterministic_exact_size_and_specials():
    counts = {8: 10, 7: 10, 4: 2, 1: 99, 99: 10000}
    assert select_ids(counts, range(12), {11}, 6, 2) == [0, 1, 4, 7, 8, 11]
    assert select_ids(
        dict(reversed(list(counts.items()))), reversed(range(12)), {11}, 6, 2
    ) == [0, 1, 4, 7, 8, 11]


def test_fill_from_valid_tokenizer_ids_only():
    assert select_ids({}, [0, 2, 4, 6, 9], {9}, 4, 1) == [0, 2, 4, 9]


@pytest.mark.parametrize(
    "special,size,base", [({99}, 4, 2), ({4}, 2, 3), ({4}, 6, 1), ({3, 4}, 2, 2)]
)
def test_invalid_maps_fail(special, size, base):
    with pytest.raises(ValueError):
        select_ids({}, range(5), special, size, base)
