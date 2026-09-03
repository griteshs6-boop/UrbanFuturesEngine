"""PYTHONHASHSEED regression tests (spec Section 15.2 and Section 23 item 4).

Section 15.2 requires that the engine "sort before iterating over sets"; Section 23 item 4
requires byte-identical output from the same inputs, seed and parameter version.

Iterating a Python ``set``/``frozenset`` of strings visits its members in an order derived
from the strings' hashes, and string hashing is randomised per interpreter start unless
``PYTHONHASHSEED`` is fixed. Any code that *accumulates a float* while iterating such a set
is therefore not reproducible: floating-point addition is not associative, so the last bits
of the sum move from one run to the next.

That is exactly the bug these tests pin. ``ufe.layers.l6_price.decompose`` used to hand its
``run`` callable a ``frozenset`` of active factor names. A ``run`` that adds one
contribution per active factor — which is what ``ufe.sim.factors.decompose_run`` and the
Section 13.4 acceptance fixture both do — summed in hash order, so the FULL run and the
matching leave-one-out run disagreed in the last bits and
``test_acc_loo_reproduces_full_without_factor`` failed roughly one interpreter start in
three. ``decompose`` now passes a **sorted tuple**.

The tests spawn subprocesses because ``PYTHONHASHSEED`` is read once, at interpreter
startup: it cannot be changed from inside a running test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

#: Several distinct fixed hash seeds. With the frozenset bug in place these produce
#: different member orders and the identity below breaks; with the fix they must all agree.
HASH_SEEDS: tuple[str, ...] = ("0", "1", "12345", "98765")

#: The test that used to be flaky, as a pytest node id.
FLAKY_NODE = (
    "tests/unit/test_l6_price.py::test_acc_loo_reproduces_full_without_factor"
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Recomputes the Section 13.4 decomposition in a fresh interpreter and prints a digest of
#: the exact float bytes of every reported quantity. Byte-identical output means
#: byte-identical digests, whatever the hash seed.
DIGEST_SCRIPT = """
import hashlib
import sys

import numpy as np
import pandas as pd

from ufe.layers import l6_price as L6
from ufe.params import load_params

FACTORS = ("metro", "airport", "data_centres")
N = 40

index = pd.RangeIndex(N)
rng = np.random.default_rng(4242)
base = pd.Series(rng.normal(size=N), index=index)
effects = {
    name: pd.Series(rng.uniform(0.01, 0.2, size=N), index=index) for name in FACTORS
}


def run(active):
    total = base.copy()
    for name in active:
        total = total + effects[name]
    if {"metro", "airport"} <= set(active):
        total = total + effects["metro"] * effects["airport"]
    return total


params = load_params("vizag")
result = L6.decompose(run, FACTORS, params)

# The purity identity the acceptance test asserts, checked here too so a mismatch fails
# loudly in the child rather than showing up only as a differing digest.
for name in FACTORS:
    reduced = L6.decompose(run, [f for f in FACTORS if f != name], params)
    np.testing.assert_array_equal(
        reduced.ln_p_full.to_numpy(), result.loo[name].to_numpy()
    )

digest = hashlib.sha256()
for frame in (result.raw, result.normalised):
    for column in sorted(frame.columns):
        digest.update(column.encode("utf-8"))
        digest.update(frame[column].to_numpy(dtype=float).tobytes())
for series in (result.total, result.interaction, result.ln_p_base, result.ln_p_full):
    digest.update(series.to_numpy(dtype=float).tobytes())
sys.stdout.write(digest.hexdigest())
"""


def _child_env(hash_seed: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    return env


@pytest.mark.parametrize("hash_seed", HASH_SEEDS)
def test_the_previously_flaky_loo_test_passes_under_every_hash_seed(hash_seed):
    """The Section 13.4 LOO acceptance test, re-run in a child with a fixed hash seed."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", FLAKY_NODE],
        cwd=REPO_ROOT,
        env=_child_env(hash_seed),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"{FLAKY_NODE} failed under PYTHONHASHSEED={hash_seed}; set iteration order is "
        f"leaking into the result (spec Section 15.2).\n{completed.stdout}\n"
        f"{completed.stderr}"
    )


def test_decomposition_is_byte_identical_across_hash_seeds():
    """Section 23 item 4: same inputs, same params version -> byte-identical output."""
    digests: dict[str, str] = {}
    for hash_seed in HASH_SEEDS:
        completed = subprocess.run(
            [sys.executable, "-c", DIGEST_SCRIPT],
            cwd=REPO_ROOT,
            env=_child_env(hash_seed),
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            f"the decomposition child failed under PYTHONHASHSEED={hash_seed}:\n"
            f"{completed.stderr}"
        )
        digests[hash_seed] = completed.stdout.strip()

    assert len(set(digests.values())) == 1, (
        "the Section 13.4 decomposition is not byte-identical across hash seeds: "
        f"{digests}"
    )
