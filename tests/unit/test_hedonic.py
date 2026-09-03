"""Tests for the spatial hedonic (spec Section 13.0).

Section 13.0 is a headline change in spec v2: the hedonic **must** be a spatial model, not
OLS, and the specification must be chosen from the Lagrange-multiplier diagnostics of an
OLS residual.  These tests are therefore not smoke tests: each one builds a lattice with a
*known* spatial data-generating process and asserts that the decision procedure selects the
right specification.

Coverage map
------------
``test_select_specification_*``      the four branches of the Section 13.0 decision table
``test_dgp_iid_selects_ols``         no spatial dependence -> OLS, Moran's I recorded
``test_dgp_lag_selects_spatial_lag`` y = rho W y + Xb + e  -> ``spreg.GM_Lag``
``test_dgp_error_selects_spatial_error``  u = lam W u + e  -> ``spreg.GM_Error_Het``
``test_lag_total_effect_*``          gamma reported as the TOTAL effect, both reported
``test_report_records_*``            mandatory provenance: spec, LM diagnostics, Moran's I
``test_determinism_*``               byte-identical refits (CONTRACT.md rule 5)
``test_missing_parameter_paths``     the two paths Section 13.0 needs and price.yaml lacks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufe.errors import MissingParameter
from ufe.layers import hedonic as H
from ufe.params import load_params

from tests.fixtures.synthetic import (  # noqa: F401  (registers the `synthetic_city` fixture)
    synthetic_city,
)

CITY = "vizag"

# Estimation controls that Section 13.0 names but `config/params/price.yaml` does not
# define (see the build report).  Tests pass them explicitly; production callers must add
# them to the YAML.
K_NEIGHBOURS = 8  # Section 13.0: `libpysal.weights.KNN.from_dataframe(cells, k=8)`
ALPHA = 0.05

# Lattice geometry for the synthetic DGPs.  Roughly Visakhapatnam, so EPSG:32644 is valid.
LAT0, LON0 = 17.70, 83.30
STEP_DEG = 0.01
SIDE = 22  # 484 cells — big enough for the LM tests to have power


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


# --------------------------------------------------------------------------------------
# synthetic lattices with a known spatial DGP
# --------------------------------------------------------------------------------------


def _lattice(side: int = SIDE) -> pd.DataFrame:
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    lat = LAT0 + ii.ravel() * STEP_DEG
    lon = LON0 + jj.ravel() * STEP_DEG
    return pd.DataFrame(
        {
            "h3": [f"cell_{n:05d}" for n in range(side * side)],
            "lat": lat,
            "lon": lon,
        }
    )


def _design(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lnA": rng.normal(size=n),
            "builtup_frac": rng.uniform(size=n),
        }
    )


TRUE_BETA = np.array([1.0, -0.5])
NOISE_SD = 0.35


def _dgp(kind: str, *, seed: int, params, side: int = SIDE, coef: float = 0.6):
    """Build a lattice whose `price_res_inr_sqft` follows a known spatial process."""
    cells = _lattice(side)
    n = len(cells)
    rng = np.random.default_rng(seed)
    X = _design(rng, n)
    w = H.spatial_weights(cells, params, k=K_NEIGHBOURS)
    W = w.full()[0]
    signal = X.to_numpy() @ TRUE_BETA
    noise = rng.normal(scale=NOISE_SD, size=n)

    if kind == "iid":
        ln_y = signal + noise
    elif kind == "lag":
        ln_y = np.linalg.solve(np.eye(n) - coef * W, signal + noise)
    elif kind == "error":
        u = np.linalg.solve(np.eye(n) - coef * W, noise)
        ln_y = signal + u
    else:  # pragma: no cover - programmer error
        raise ValueError(kind)

    out = pd.concat([cells, X], axis=1)
    # ln P is what the hedonic models; the column carries the level, INR/sqft.
    out["price_res_inr_sqft"] = np.exp(ln_y)
    return out, w


X_COLS = ("lnA", "builtup_frac")


# --------------------------------------------------------------------------------------
# 13.0 — the decision table, tested branch by branch
# --------------------------------------------------------------------------------------


def test_select_specification_lag_only():
    spec, reason = H.select_specification(
        lm_lag_p=0.001, lm_error_p=0.4, rlm_lag_stat=1.0, rlm_error_stat=9.0, alpha=ALPHA
    )
    assert spec == H.SPEC_LAG
    assert "LM-lag significant" in reason


def test_select_specification_error_only():
    spec, reason = H.select_specification(
        lm_lag_p=0.9, lm_error_p=0.001, rlm_lag_stat=9.0, rlm_error_stat=1.0, alpha=ALPHA
    )
    assert spec == H.SPEC_ERROR
    assert "LM-error significant" in reason


def test_select_specification_both_prefers_larger_robust_lm():
    spec, reason = H.select_specification(
        lm_lag_p=0.001, lm_error_p=0.001, rlm_lag_stat=50.0, rlm_error_stat=3.0, alpha=ALPHA
    )
    assert spec == H.SPEC_LAG
    assert "robust" in reason

    spec, reason = H.select_specification(
        lm_lag_p=0.001, lm_error_p=0.001, rlm_lag_stat=3.0, rlm_error_stat=50.0, alpha=ALPHA
    )
    assert spec == H.SPEC_ERROR
    assert "robust" in reason


def test_select_specification_neither_falls_back_to_ols():
    spec, reason = H.select_specification(
        lm_lag_p=0.3, lm_error_p=0.4, rlm_lag_stat=1.0, rlm_error_stat=1.0, alpha=ALPHA
    )
    assert spec == H.SPEC_OLS
    assert "Moran" in reason


# --------------------------------------------------------------------------------------
# 13.0 — end to end against a known DGP
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_dgp_iid_selects_ols(params):
    cells, w = _dgp("iid", seed=101, params=params)
    fit = H.fit_hedonic(cells, params, x_cols=X_COLS, w=w, alpha=ALPHA, k=K_NEIGHBOURS)
    assert fit.specification == H.SPEC_OLS
    # "neither -> OLS is fine, record Moran's I and move on"
    assert fit.diagnostics.moran_ols_i is not None
    assert fit.diagnostics.moran_residual_i is not None
    assert not fit.diagnostics.misspecified
    # coefficients recovered
    assert fit.direct["lnA"] == pytest.approx(TRUE_BETA[0], abs=0.1)
    assert fit.direct["builtup_frac"] == pytest.approx(TRUE_BETA[1], abs=0.3)
    # OLS has no spillover: total == direct
    assert fit.total == fit.direct
    assert fit.spatial_multiplier == pytest.approx(1.0)


@pytest.mark.acceptance
def test_dgp_lag_selects_spatial_lag(params):
    cells, w = _dgp("lag", seed=202, params=params)
    fit = H.fit_hedonic(cells, params, x_cols=X_COLS, w=w, alpha=ALPHA, k=K_NEIGHBOURS)
    assert fit.specification == H.SPEC_LAG
    assert fit.rho == pytest.approx(0.6, abs=0.15)
    assert fit.lam is None
    assert fit.diagnostics.lm_lag_p < ALPHA
    # Positive spatial dependence must beat the iid Moran's I of the OLS residual.
    assert fit.diagnostics.moran_ols_i > 0


@pytest.mark.acceptance
def test_dgp_error_selects_spatial_error(params):
    cells, w = _dgp("error", seed=303, params=params, coef=0.7)
    fit = H.fit_hedonic(cells, params, x_cols=X_COLS, w=w, alpha=ALPHA, k=K_NEIGHBOURS)
    assert fit.specification == H.SPEC_ERROR
    assert fit.lam == pytest.approx(0.7, abs=0.2)
    assert fit.rho is None
    # Spatial error: "Coefficients are interpretable as usual once corrected."
    assert fit.direct["lnA"] == pytest.approx(TRUE_BETA[0], abs=0.15)
    assert fit.total == fit.direct
    # The correctly specified error model must NOT be flagged misspecified: its *filtered*
    # residual is white even though its prediction residual is autocorrelated by design.
    assert fit.diagnostics.misspecified is False
    assert abs(fit.diagnostics.moran_residual_i) < abs(fit.diagnostics.moran_ols_i)


def test_dgp_error_beats_ols_on_bias(params):
    """The whole point of 13.0: OLS on a lag DGP is biased, the spatial model is not."""
    cells, w = _dgp("lag", seed=404, params=params)
    fit = H.fit_hedonic(cells, params, x_cols=X_COLS, w=w, alpha=ALPHA, k=K_NEIGHBOURS)
    ols = H.fit_hedonic(
        cells, params, x_cols=X_COLS, w=w, alpha=ALPHA, k=K_NEIGHBOURS, force=H.SPEC_OLS
    )
    assert fit.specification == H.SPEC_LAG
    assert ols.specification == H.SPEC_OLS
    assert abs(fit.direct["lnA"] - TRUE_BETA[0]) < abs(ols.direct["lnA"] - TRUE_BETA[0])


# --------------------------------------------------------------------------------------
# 13.0 — direct vs total effects
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_lag_total_effect_is_larger_and_reported_alongside_direct(params):
    """"`gamma` must be reported as the **total** effect, not the direct one. Report both.\""""
    cells, w = _dgp("lag", seed=505, params=params)
    fit = H.fit_hedonic(cells, params, x_cols=X_COLS, w=w, alpha=ALPHA, k=K_NEIGHBOURS)
    assert fit.specification == H.SPEC_LAG

    multiplier = 1.0 / (1.0 - fit.rho)
    assert fit.spatial_multiplier == pytest.approx(multiplier, rel=1e-9)
    for name in X_COLS:
        assert fit.total[name] == pytest.approx(fit.direct[name] * multiplier, rel=1e-9)
    assert abs(fit.total["lnA"]) > abs(fit.direct["lnA"])

    assert fit.gamma("lnA") == fit.total["lnA"]
    assert fit.gamma("lnA", effect="direct") == fit.direct["lnA"]
    with pytest.raises(ValueError):
        fit.gamma("lnA", effect="indirect_only")


def test_gamma_unknown_variable_raises(params):
    cells, w = _dgp("iid", seed=606, params=params)
    fit = H.fit_hedonic(cells, params, x_cols=X_COLS, w=w, alpha=ALPHA, k=K_NEIGHBOURS)
    with pytest.raises(KeyError):
        fit.gamma("no_such_variable")


# --------------------------------------------------------------------------------------
# 13.0 — mandatory reporting
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
@pytest.mark.parametrize("kind,coef", [("iid", 0.6), ("lag", 0.6), ("error", 0.7)])
def test_report_records_specification_and_diagnostics(params, kind, coef):
    """"every fitted hedonic writes Moran's I on the residuals, the chosen specification,
    and the LM diagnostics into the fit report.\""""
    cells, w = _dgp(kind, seed=707, params=params, coef=coef)
    fit = H.fit_hedonic(cells, params, x_cols=X_COLS, w=w, alpha=ALPHA, k=K_NEIGHBOURS)
    report = fit.report()

    assert report["specification"] == fit.specification
    assert report["selection_rule"]
    diagnostics = report["diagnostics"]
    for key in (
        "lm_lag_stat",
        "lm_lag_p",
        "lm_error_stat",
        "lm_error_p",
        "rlm_lag_stat",
        "rlm_lag_p",
        "rlm_error_stat",
        "rlm_error_p",
        "moran_ols_i",
        "moran_ols_p",
        "moran_residual_i",
        "moran_residual_p",
        "misspecified",
        "alpha",
        "k_neighbours",
        "n_obs",
    ):
        assert key in diagnostics, key
        assert diagnostics[key] is not None, key
    assert report["direct"].keys() == report["total"].keys()
    assert report["n_obs"] == len(cells)


def test_misspecified_flag_set_when_residual_autocorrelation_survives(params):
    """"A model shipped with significant residual spatial autocorrelation is
    misspecified, and the manifest must say so.\""""
    cells, w = _dgp("lag", seed=808, params=params)
    forced = H.fit_hedonic(
        cells, params, x_cols=X_COLS, w=w, alpha=ALPHA, k=K_NEIGHBOURS, force=H.SPEC_OLS
    )
    assert forced.diagnostics.misspecified is True
    assert forced.report()["diagnostics"]["misspecified"] is True


# --------------------------------------------------------------------------------------
# determinism (CONTRACT.md rule 5, brief requirement 8)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind,coef", [("iid", 0.6), ("lag", 0.6), ("error", 0.7)])
def test_determinism_identical_refit(params, kind, coef):
    cells, w = _dgp(kind, seed=909, params=params, coef=coef)
    a = H.fit_hedonic(cells, params, x_cols=X_COLS, w=w, alpha=ALPHA, k=K_NEIGHBOURS)
    b = H.fit_hedonic(cells, params, x_cols=X_COLS, w=w, alpha=ALPHA, k=K_NEIGHBOURS)
    assert a.specification == b.specification
    assert a.direct == b.direct
    assert a.total == b.total
    assert a.diagnostics.as_dict() == b.diagnostics.as_dict()
    np.testing.assert_array_equal(a.residuals.to_numpy(), b.residuals.to_numpy())


def test_weights_are_built_in_a_metric_crs_not_degrees(params):
    """CONTRACT.md rule 7 / Section 0.3: neighbours are nearest in metres, not degrees."""
    cells = _lattice()
    w = H.spatial_weights(cells, params, k=K_NEIGHBOURS)
    assert w.n == len(cells)
    assert w.transform == "R"
    # A degree lattice with equal lat/lon steps is *not* square in metres: a degree of
    # longitude at 17.7N is ~0.95 of a degree of latitude, so the metric nearest
    # neighbours of an interior cell are its east/west pair before its north/south pair.
    interior = (SIDE // 2) * SIDE + (SIDE // 2)
    neighbours = w.neighbors[interior]
    assert len(neighbours) == K_NEIGHBOURS
    assert interior + SIDE in neighbours and interior - SIDE in neighbours


# --------------------------------------------------------------------------------------
# missing parameter paths — reported precisely, never hardcoded (brief requirement 7)
# --------------------------------------------------------------------------------------


def test_missing_parameter_paths(params):
    cells = _lattice(side=10)
    rng = np.random.default_rng(1)
    cells = pd.concat([cells, _design(rng, len(cells))], axis=1)
    cells["price_res_inr_sqft"] = np.exp(rng.normal(size=len(cells)))

    with pytest.raises(MissingParameter) as excinfo:
        H.fit_hedonic(cells, params, x_cols=X_COLS)
    message = str(excinfo.value)
    assert H.P_KNN_K in message
    assert H.P_ALPHA in message


def test_null_rows_are_dropped_and_index_preserved(params, synthetic_city):  # noqa: F811
    cells = synthetic_city.cells
    n_valid = int(cells["price_res_inr_sqft"].notna().sum())
    assert n_valid < len(cells)  # the fixture really does have missing prices
    fit = H.fit_hedonic(
        cells, params, x_cols=("lnA", "builtup_frac"), alpha=ALPHA, k=K_NEIGHBOURS
    )
    assert fit.diagnostics.n_obs == n_valid
    assert len(fit.residuals) == n_valid
    assert fit.residuals.index.isin(cells.index).all()


def test_too_few_observations_raises(params):
    cells = _lattice(side=2)
    rng = np.random.default_rng(2)
    cells = pd.concat([cells, _design(rng, len(cells))], axis=1)
    cells["price_res_inr_sqft"] = np.exp(rng.normal(size=len(cells)))
    with pytest.raises(ValueError):
        H.fit_hedonic(cells, params, x_cols=X_COLS, alpha=ALPHA, k=K_NEIGHBOURS)
