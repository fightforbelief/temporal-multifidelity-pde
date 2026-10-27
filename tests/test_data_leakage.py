import numpy as np

from temporal_mf.data.normalization import fit_standardizer
from temporal_mf.data.splits import assign_splits, ic_hash, near_duplicate_violations
from temporal_mf.data.windows import assert_no_future_in_features, one_step_windows


def test_splits_are_disjoint_and_seeded():
    ids = [f"t{i}" for i in range(20)]
    a = assign_splits(ids, seed=20261023)
    b = assign_splits(ids, seed=20261023)
    assert a == b
    groups = {"train": set(), "val": set(), "test": set()}
    for tid, split in a.items():
        groups[split].add(tid)
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert set().union(*groups.values()) == set(ids)


def test_near_duplicate_across_splits_is_detected():
    ids = ["a", "b"]
    labels = {"a": "train", "b": "test"}
    ics = {"a": "abc", "b": "abc"}
    assert near_duplicate_violations(labels, ics)


def test_same_ic_hash_for_same_field():
    u = np.linspace(0, 1, 16)
    assert ic_hash(u) == ic_hash(u.copy())


def test_windows_are_causal():
    full = np.arange(12, dtype=float).reshape(4, 3)
    inputs, targets = one_step_windows(full)
    assert_no_future_in_features(inputs, targets, full)


def test_scaler_ignores_later_perturbation():
    train = np.zeros((4, 8))
    scaler = fit_standardizer(train)
    _ = np.ones((4, 8)) * 100
    again = fit_standardizer(train)
    assert scaler.mean == again.mean
    assert scaler.std == again.std
