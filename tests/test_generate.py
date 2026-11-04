import pytest
import torch
from pixelart.generate import resolve_seed, seamless


def test_resolve_seed_int():
    assert resolve_seed(42.0) == 42 and isinstance(resolve_seed(42.0), int)


def test_resolve_seed_random():
    for s in (None, -1):
        assert 0 <= resolve_seed(s) < 2**31


def test_seamless_sets_and_restores():
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1), torch.nn.Linear(2, 2))
    with seamless(m):
        assert m[0].padding_mode == "circular"
    assert m[0].padding_mode == "zeros"


def test_seamless_restores_on_error():
    m = torch.nn.Conv2d(3, 3, 3, padding=1)
    with pytest.raises(RuntimeError):
        with seamless(m):
            raise RuntimeError
    assert m.padding_mode == "zeros"
