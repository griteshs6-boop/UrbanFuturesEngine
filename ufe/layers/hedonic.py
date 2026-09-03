"""The spatial hedonic (spec Section 13.0).

Section 13.0 corrects a defect in spec v1 and is emphatic about it:

    "Property prices are strongly spatially autocorrelated: neighbouring cells are not
    independent observations. Plain OLS on a cross-section of cells gives understated
    standard errors ... Where the autocorrelation is substantive rather than nuisance
    ... OLS coefficients are biased as well."

So this module never ships a bare OLS hedonic by accident. It always fits OLS *first*,
purely to obtain the Lagrange-multiplier diagnostics, and then picks the specification
from those diagnostics using the decision table the spec gives verbatim:

    - LM-lag significant, LM-error not   -> spatial lag  (``spreg.GM_Lag``)
    - LM-error significant, LM-lag not   -> spatial error (``spreg.GM_Error_Het``)
    - both significant                   -> compare robust LM, prefer the larger
    - neither                            -> OLS is fine, record Moran's I and move on

:func:`select_specification` is that table and nothing else, so it can be tested branch by
branch; :func:`fit_hedonic` wires it to the data.

Mandatory reporting
-------------------
    "every fitted hedonic writes Moran's I on the residuals, the chosen specification, and
    the LM diagnostics into the fit report. A model shipped with significant residual
    spatial autocorrelation is misspecified, and the manifest must say so."

:meth:`HedonicFit.report` is that fit report: a plain JSON-able dict carrying the chosen
specification, the sentence explaining *why* it was chosen, every LM statistic and p-value,
Moran's I on both the OLS residual and the fitted model's residual, and a ``misspecified``
boolean. It is intended to be embedded in the run manifest as-is.

Direct vs total effects
-----------------------
    "Spatial lag means a cell's price depends on its neighbours' prices — a genuine
    spillover. Its coefficients carry direct and indirect (feedback) effects, and `gamma`
    must be reported as the **total** effect, not the direct one. Report both."

Under a row-standardised weights matrix the average total effect of a unit change in
covariate *k* is ``beta_k / (1 - rho)``. :attr:`HedonicFit.direct` and
:attr:`HedonicFit.total` both exist for every specification (they are equal for OLS and for
spatial error, where there is no feedback), and :meth:`HedonicFit.gamma` returns the
**total** effect unless asked otherwise.

CRS discipline
--------------
The spec's snippet is ``libpysal.weights.KNN.from_dataframe(cells, k=8)``, which on a frame
of EPSG:4326 geometry would compute nearest neighbours *in degrees* — Section 21's
"Degrees used as metres" failure mode. :func:`spatial_weights` therefore projects cell
centroids into the city's ``crs_metric`` via :mod:`ufe.geo` before building the KNN graph.

Determinism
-----------
Nothing here draws a random number. ``spreg.OLS``, ``spreg.GM_Lag`` and
``spreg.GM_Error_Het`` are closed-form/GMM estimators with no stochastic component, and the
KD-tree behind ``KNN`` is deterministic given identical input coordinates. There is a test
that a refit is identical.

Numeric policy (CONTRACT.md rule 1)
-----------------------------------
The two estimation controls Section 13.0 names — the neighbour count ``k`` and the
significance level used to read the LM p-values — are parameters. They are read from
:data:`P_KNN_K` and :data:`P_ALPHA`. **Neither path exists in ``config/params/price.yaml``
at the time of writing**; :func:`fit_hedonic` raises :class:`ufe.errors.MissingParameter`
naming both, and callers who cannot yet edit the YAML must pass ``k=`` / ``alpha=``
explicitly. No default is invented here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from ufe import geo
from ufe.errors import MissingParameter

logger = logging.getLogger(__name__)

__all__ = [
    "SPEC_OLS",
    "SPEC_LAG",
    "SPEC_ERROR",
    "SPECIFICATIONS",
    "P_KNN_K",
    "P_ALPHA",
    "HedonicDiagnostics",
    "HedonicFit",
    "spatial_weights",
    "select_specification",
    "morans_i",
    "fit_hedonic",
]

# --------------------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------------------

#: "neither -> OLS is fine, record Moran's I and move on"
SPEC_OLS = "ols"
#: "LM-lag significant, LM-error not -> spatial lag (spreg.GM_Lag)"
SPEC_LAG = "spatial_lag"
#: "LM-error significant, LM-lag not -> spatial error (spreg.GM_Error_Het)"
SPEC_ERROR = "spatial_error"

SPECIFICATIONS: tuple[str, ...] = (SPEC_OLS, SPEC_LAG, SPEC_ERROR)

EFFECT_TOTAL = "total"
EFFECT_DIRECT = "direct"

# --------------------------------------------------------------------------------------
# parameter paths (Section 13.0). NEITHER EXISTS IN price.yaml — see the module docstring.
# --------------------------------------------------------------------------------------

#: Neighbour count for the KNN spatial weights. Section 13.0 shows ``k=8``.
P_KNN_K = "price.hedonic.spatial_weights_k"
#: Significance level at which an LM p-value counts as "significant" in the decision table.
P_ALPHA = "price.hedonic.lm_significance_alpha"

COL_LAT = "lat"
COL_LON = "lon"
COL_PRICE_RES = "price_res_inr_sqft"
CONSTANT_NAME = "CONSTANT"

#: A regression needs strictly more observations than parameters; the LM diagnostics need
#: rather more than that to have any power. Expressed structurally (no literal): at least
#: as many observations as the weights graph has neighbours per cell, times the number of
#: estimated parameters.
_MIN_OBS_FACTOR = "k * (n_regressors + 1)"


# --------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------


def _metric_coordinates(cells: pd.DataFrame, crs_metric: str) -> np.ndarray:
    """Cell centroids projected into `crs_metric`, as an ``(n, 2)`` array of metres."""
    for column in (COL_LON, COL_LAT):
        if column not in cells.columns:
            raise KeyError(f"spatial weights need a {column!r} column on `cells`")
    points = gpd.GeoSeries(
        gpd.points_from_xy(
            np.asarray(cells[COL_LON], dtype=float), np.asarray(cells[COL_LAT], dtype=float)
        ),
        crs=geo.GEOGRAPHIC_CRS,
    )
    projected = geo.to_metric(points, crs_metric)
    return np.column_stack(
        [projected.x.to_numpy(dtype=float), projected.y.to_numpy(dtype=float)]
    )


def spatial_weights(
    cells: pd.DataFrame,
    params: Any = None,
    *,
    k: int | None = None,
    crs_metric: str | None = None,
):
    """Row-standardised KNN spatial weights over cell centroids (Section 13.0).

    Section 13.0's snippet is ``KNN.from_dataframe(cells, k=8)``. Applied to EPSG:4326
    geometry that computes neighbours in degrees, which Section 21 lists as a named failure
    mode; centroids are therefore projected into the city's ``crs_metric`` first.

    Parameters
    ----------
    cells:
        Frame carrying ``lat`` / ``lon`` in EPSG:4326.
    params:
        Used only to resolve the city's ``crs_metric`` and, when `k` is omitted,
        :data:`P_KNN_K`.
    k:
        Neighbour count. When ``None`` it is read from :data:`P_KNN_K`, which raises
        :class:`ufe.errors.MissingParameter` while that path is absent from the YAML.
    crs_metric:
        Overrides the city's metric CRS (tests and multi-city tooling).
    """
    import libpysal  # imported lazily: heavy, and only estimation needs it

    if k is None:
        k = int(_require(params, P_KNN_K, "KNN neighbour count (Section 13.0 shows k=8)"))
    k = int(k)
    if k < 1:
        raise ValueError(f"{P_KNN_K} must be >= 1, got {k}")
    if len(cells) <= k:
        raise ValueError(
            f"cannot build {k}-nearest-neighbour weights over {len(cells)} cells"
        )

    if crs_metric is None:
        crs_metric = geo.city_metric_crs(params)
    coords = _metric_coordinates(cells, crs_metric)
    w = libpysal.weights.KNN.from_array(coords, k=k)
    w.transform = "r"
    return w


def morans_i(values: np.ndarray, w: Any) -> float:
    """Moran's I of `values` under weights `w`.

    ``I = (n / S0) * (u' W u) / (u' u)``, the textbook definition. Computed here rather
    than imported because ``esda`` is not a declared dependency; ``libpysal.weights.
    lag_spatial`` supplies ``W u`` without ever densifying the matrix.
    """
    from libpysal.weights import lag_spatial

    u = np.asarray(values, dtype=float).ravel()
    u = u - u.mean()
    denominator = float(u @ u)
    if denominator == 0:
        return 0.0
    numerator = float(u @ np.asarray(lag_spatial(w, u), dtype=float).ravel())
    return (len(u) / float(w.s0)) * (numerator / denominator)


# --------------------------------------------------------------------------------------
# 13.0 — the decision table
# --------------------------------------------------------------------------------------


def select_specification(
    *,
    lm_lag_p: float,
    lm_error_p: float,
    rlm_lag_stat: float,
    rlm_error_stat: float,
    alpha: float,
) -> tuple[str, str]:
    """The Section 13.0 decision procedure. Returns ``(specification, reason)``.

    The `reason` is a sentence written into the fit report so a reader can see not just
    which model was chosen but why — Section 13.0's "mandatory reporting".

    Ambiguity (reported in the build summary): the spec says "both significant -> compare
    robust LM, prefer the larger" without saying whether the *robust* statistics must
    themselves be significant, and without saying what to do on an exact tie. This
    implementation compares the raw robust statistics as written, and breaks an exact tie
    toward the spatial *error* model, on the grounds that an error specification makes the
    weaker substantive claim (nuisance autocorrelation rather than a genuine spillover).
    """
    lag_significant = float(lm_lag_p) < float(alpha)
    error_significant = float(lm_error_p) < float(alpha)

    if lag_significant and not error_significant:
        return SPEC_LAG, (
            f"LM-lag significant (p={lm_lag_p:.4g} < {alpha:g}) and LM-error not "
            f"(p={lm_error_p:.4g}); Section 13.0 selects the spatial lag model."
        )
    if error_significant and not lag_significant:
        return SPEC_ERROR, (
            f"LM-error significant (p={lm_error_p:.4g} < {alpha:g}) and LM-lag not "
            f"(p={lm_lag_p:.4g}); Section 13.0 selects the spatial error model."
        )
    if lag_significant and error_significant:
        if float(rlm_lag_stat) > float(rlm_error_stat):
            return SPEC_LAG, (
                f"both LM tests significant at {alpha:g}; the larger robust LM is the lag "
                f"test ({rlm_lag_stat:.4g} > {rlm_error_stat:.4g}), so Section 13.0 "
                "prefers the spatial lag model."
            )
        return SPEC_ERROR, (
            f"both LM tests significant at {alpha:g}; the larger robust LM is the error "
            f"test ({rlm_error_stat:.4g} >= {rlm_lag_stat:.4g}), so Section 13.0 prefers "
            "the spatial error model."
        )
    return SPEC_OLS, (
        f"neither LM test significant at {alpha:g} (LM-lag p={lm_lag_p:.4g}, LM-error "
        f"p={lm_error_p:.4g}); Section 13.0: OLS is fine, record Moran's I and move on."
    )


# --------------------------------------------------------------------------------------
# fit objects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HedonicDiagnostics:
    """Everything Section 13.0 makes mandatory in the fit report."""

    n_obs: int
    k_neighbours: int
    alpha: float
    specification: str
    selection_rule: str

    lm_lag_stat: float
    lm_lag_p: float
    lm_error_stat: float
    lm_error_p: float
    rlm_lag_stat: float
    rlm_lag_p: float
    rlm_error_stat: float
    rlm_error_p: float
    lm_sarma_stat: float
    lm_sarma_p: float

    #: Moran's I of the *OLS* residual, with its z-score and p-value (``spreg`` supplies
    #: all three).
    moran_ols_i: float
    moran_ols_z: float
    moran_ols_p: float

    #: Moran's I of the residual of the model actually shipped, and the p-value of the
    #: same test recomputed on that residual (see :func:`_residual_moran`).
    moran_residual_i: float
    moran_residual_p: float

    #: "A model shipped with significant residual spatial autocorrelation is
    #: misspecified, and the manifest must say so."
    misspecified: bool

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True)
class HedonicFit:
    """A fitted hedonic plus its Section 13.0 provenance."""

    specification: str
    variables: tuple[str, ...]
    #: Estimated coefficients, including ``CONSTANT``.
    direct: dict[str, float]
    #: Total (direct + indirect/feedback) effects. Equal to `direct` unless the chosen
    #: specification is the spatial lag.
    total: dict[str, float]
    #: Spatial autoregressive coefficient on ``W y`` (spatial lag only).
    rho: float | None
    #: Spatial autoregressive coefficient on the error (spatial error only).
    lam: float | None
    #: ``1 / (1 - rho)`` for the spatial lag, ``1.0`` otherwise.
    spatial_multiplier: float
    fitted: pd.Series
    residuals: pd.Series
    diagnostics: HedonicDiagnostics

    def gamma(self, variable: str, *, effect: str = EFFECT_TOTAL) -> float:
        """The elasticity on `variable`.

        Section 13.0: "`gamma` must be reported as the **total** effect, not the direct
        one", so `effect` defaults to ``"total"``.
        """
        if effect == EFFECT_TOTAL:
            table = self.total
        elif effect == EFFECT_DIRECT:
            table = self.direct
        else:
            raise ValueError(
                f"effect must be {EFFECT_TOTAL!r} or {EFFECT_DIRECT!r}, got {effect!r}"
            )
        if variable not in table:
            raise KeyError(
                f"{variable!r} is not a hedonic regressor; have {sorted(table)}"
            )
        return table[variable]

    def report(self) -> dict[str, Any]:
        """The JSON-able fit report embedded in the run manifest (Section 13.0)."""
        return {
            "specification": self.specification,
            "selection_rule": self.diagnostics.selection_rule,
            "n_obs": self.diagnostics.n_obs,
            "variables": list(self.variables),
            "direct": dict(self.direct),
            "total": dict(self.total),
            "rho": self.rho,
            "lambda": self.lam,
            "spatial_multiplier": self.spatial_multiplier,
            "diagnostics": self.diagnostics.as_dict(),
        }


# --------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------


def _require(params: Any, path: str, what: str) -> float:
    if params is None:
        raise MissingParameter(
            f"no Params supplied, so {path!r} ({what}) cannot be resolved; pass the value "
            "explicitly or load the city's parameters"
        )
    try:
        return params.value(path)
    except MissingParameter as exc:
        raise MissingParameter(
            f"{path!r} is not defined in config/params/price.yaml but Section 13.0 needs "
            f"it: {what}. Add it to the YAML (CONTRACT.md rule 1 forbids a default here) "
            f"or pass it explicitly to fit_hedonic()."
        ) from exc


def _resolve_estimation_controls(
    params: Any, k: int | None, alpha: float | None
) -> tuple[int, float]:
    """Resolve `k` and `alpha`, reporting *both* missing paths in one exception."""
    missing: list[str] = []
    resolved_k: int | None = None
    resolved_alpha: float | None = None

    if k is None:
        try:
            resolved_k = int(
                _require(params, P_KNN_K, "KNN neighbour count (Section 13.0 shows k=8)")
            )
        except MissingParameter:
            missing.append(f"{P_KNN_K} (KNN neighbour count; Section 13.0 shows k=8)")
    else:
        resolved_k = int(k)

    if alpha is None:
        try:
            resolved_alpha = float(
                _require(
                    params,
                    P_ALPHA,
                    "significance level for the LM decision table (Section 13.0)",
                )
            )
        except MissingParameter:
            missing.append(
                f"{P_ALPHA} (significance level for the Section 13.0 LM decision table)"
            )
    else:
        resolved_alpha = float(alpha)

    if missing:
        raise MissingParameter(
            "the spatial hedonic (spec Section 13.0) needs parameters that "
            "config/params/price.yaml does not define:\n  "
            + "\n  ".join(missing)
            + "\nCONTRACT.md rule 1 forbids defaulting them in Python. Add them to the "
            "YAML, or pass k= / alpha= explicitly."
        )
    assert resolved_k is not None and resolved_alpha is not None
    return resolved_k, resolved_alpha


def _pair(value: Any) -> tuple[float, float]:
    """``spreg`` returns LM diagnostics as ``(statistic, p-value)`` tuples."""
    stat, p = value
    return float(stat), float(p)


def _residual_moran(
    residuals: np.ndarray, design: np.ndarray, w: Any
) -> tuple[float, float]:
    """Moran's I on a fitted model's residual, with a p-value.

    ``spreg`` reports ``moran_res`` only for :class:`spreg.OLS`. For the GM estimators the
    statistic is computed directly by :func:`morans_i`; its p-value is obtained by running
    the same Moran test through an auxiliary ``spreg.OLS`` of the residual on the original
    design matrix. For a well-specified model the auxiliary regression explains nothing, so
    its residual is the input residual and the test is the test we want.
    """
    import spreg

    statistic = morans_i(residuals, w)
    # The auxiliary regression is deliberately degenerate (the residual is close to
    # orthogonal to the design), which makes some of spreg's *other* spatial diagnostics
    # divide by zero. Only `moran_res` is read, so those warnings are suppressed here
    # rather than allowed to reach the caller.
    with np.errstate(divide="ignore", invalid="ignore"):
        auxiliary = spreg.OLS(
            residuals.reshape(-1, 1), design, w=w, spat_diag=True, moran=True
        )
    _, _, p_value = (float(x) for x in auxiliary.moran_res)
    return statistic, p_value


def fit_hedonic(
    cells: pd.DataFrame,
    params: Any = None,
    *,
    x_cols: Sequence[str],
    y_col: str = COL_PRICE_RES,
    log_y: bool = True,
    k: int | None = None,
    alpha: float | None = None,
    w: Any = None,
    force: str | None = None,
) -> HedonicFit:
    """Fit the Section 13.0 spatial hedonic and return it with its diagnostics.

    The procedure, in the order the spec gives it:

    1. build row-standardised KNN weights (in metres, see :func:`spatial_weights`);
    2. fit ``spreg.OLS(y, X, w=w, spat_diag=True, moran=True)`` — **only** to obtain the
       Lagrange-multiplier diagnostics and Moran's I;
    3. run :func:`select_specification` on those diagnostics;
    4. refit with ``spreg.GM_Lag`` / ``spreg.GM_Error_Het``, or keep the OLS fit;
    5. record Moran's I on the *chosen* model's residual and set ``misspecified`` when
       significant spatial autocorrelation survives.

    Parameters
    ----------
    cells:
        Cross-section carrying `y_col`, every column in `x_cols`, and ``lat`` / ``lon``.
        Rows with a null in any of those are dropped; the returned residual/fitted series
        keep the surviving rows' index.
    log_y:
        Section 0.3 uses natural logs throughout, and the hedonic is specified in
        ``ln P``. Set ``False`` only if `y_col` is already logged.
    force:
        Skip step 3 and fit the named specification anyway. Used to demonstrate the OLS
        bias Section 13.0 warns about; never use it in production.
    """
    import spreg

    x_cols = tuple(x_cols)
    if not x_cols:
        raise ValueError("the hedonic needs at least one regressor")
    k, alpha = _resolve_estimation_controls(params, k, alpha)
    if force is not None and force not in SPECIFICATIONS:
        raise ValueError(f"force must be one of {SPECIFICATIONS}, got {force!r}")

    needed = [y_col, *x_cols, COL_LAT, COL_LON]
    missing_columns = [c for c in needed if c not in cells.columns]
    if missing_columns:
        raise KeyError(f"`cells` is missing hedonic columns {missing_columns}")

    usable = cells.loc[cells[needed].notna().all(axis=1)]
    n_obs = len(usable)
    n_params = len(x_cols) + 1  # + the constant
    if n_obs <= k * n_params:
        raise ValueError(
            f"the spatial hedonic needs more than {_MIN_OBS_FACTOR} = {k * n_params} "
            f"usable observations for the LM diagnostics to have any power; got {n_obs}"
        )

    y_raw = np.asarray(usable[y_col], dtype=float)
    if log_y:
        if not (y_raw > 0).all():
            raise ValueError(
                f"{y_col!r} has non-positive values; ln P is undefined (Section 0.3 uses "
                "natural log throughout)"
            )
        y_raw = np.log(y_raw)
    y = y_raw.reshape(-1, 1)
    X = np.column_stack([np.asarray(usable[c], dtype=float) for c in x_cols])

    if w is None:
        w = spatial_weights(usable, params, k=k)
    elif w.n != n_obs:
        raise ValueError(
            f"the supplied weights cover {w.n} units but {n_obs} observations are usable; "
            "rebuild the weights on the same rows"
        )

    # ---- step 2: OLS, for the diagnostics only -----------------------------------
    ols = spreg.OLS(y, X, w=w, spat_diag=True, moran=True, name_x=list(x_cols))
    lm_lag_stat, lm_lag_p = _pair(ols.lm_lag)
    lm_error_stat, lm_error_p = _pair(ols.lm_error)
    rlm_lag_stat, rlm_lag_p = _pair(ols.rlm_lag)
    rlm_error_stat, rlm_error_p = _pair(ols.rlm_error)
    lm_sarma_stat, lm_sarma_p = _pair(ols.lm_sarma)
    moran_ols_i, moran_ols_z, moran_ols_p = (float(x) for x in ols.moran_res)

    # ---- step 3: choose ----------------------------------------------------------
    specification, selection_rule = select_specification(
        lm_lag_p=lm_lag_p,
        lm_error_p=lm_error_p,
        rlm_lag_stat=rlm_lag_stat,
        rlm_error_stat=rlm_error_stat,
        alpha=alpha,
    )
    if force is not None:
        selection_rule = (
            f"specification forced to {force!r} by the caller; the Section 13.0 decision "
            f"table would have chosen {specification!r} because {selection_rule}"
        )
        specification = force

    # ---- step 4: refit -----------------------------------------------------------
    rho: float | None = None
    lam: float | None = None
    multiplier = 1.0
    if specification == SPEC_OLS:
        model = ols
        names = list(ols.name_x)
        betas = np.asarray(ols.betas, dtype=float).ravel()
    elif specification == SPEC_LAG:
        model = spreg.GM_Lag(
            y, X, w=w, spat_diag=True, spat_impacts="simple", name_x=list(x_cols)
        )
        betas_all = np.asarray(model.betas, dtype=float).ravel()
        rho = float(betas_all[-1])
        betas = betas_all[:-1]
        names = list(model.name_z)[: len(betas)]
        if rho >= 1.0:
            raise ValueError(
                f"estimated spatial autoregressive coefficient rho={rho} >= 1: the spatial "
                "multiplier 1/(1-rho) does not exist and prices would be explosive "
                "(Section 21, 'Agglomeration divergence')"
            )
        multiplier = 1.0 / (1.0 - rho)
    else:
        model = spreg.GM_Error_Het(y, X, w=w, name_x=list(x_cols))
        betas_all = np.asarray(model.betas, dtype=float).ravel()
        lam = float(betas_all[-1])
        betas = betas_all[:-1]
        names = list(model.name_x)[: len(betas)]

    direct = {str(name): float(value) for name, value in zip(names, betas)}
    total = {name: value * multiplier for name, value in direct.items()}

    residual_values = np.asarray(model.u, dtype=float).ravel()
    predicted_values = np.asarray(model.predy, dtype=float).ravel()

    # ---- step 5: mandatory Moran's I on the shipped model's residual --------------
    # For the spatial *error* model the prediction residual `u = y - Xb` is autocorrelated
    # by construction — that is what the model asserts. The residual that must be white is
    # the spatially filtered one, `e = u - lambda * W u`, which spreg exposes as
    # `e_filtered`. Testing `u` instead would flag every correctly specified error model as
    # misspecified.
    diagnostic_residual = residual_values
    if specification == SPEC_ERROR:
        filtered = getattr(model, "e_filtered", None)
        if filtered is not None:
            diagnostic_residual = np.asarray(filtered, dtype=float).ravel()
    moran_residual_i, moran_residual_p = _residual_moran(diagnostic_residual, X, w)
    misspecified = bool(moran_residual_p < alpha)
    if misspecified:
        logger.warning(
            "hedonic specification %r ships with significant residual spatial "
            "autocorrelation (Moran's I=%.4g, p=%.4g < alpha=%g): the model is "
            "misspecified and the manifest says so (spec Section 13.0)",
            specification,
            moran_residual_i,
            moran_residual_p,
            alpha,
        )

    diagnostics = HedonicDiagnostics(
        n_obs=int(n_obs),
        k_neighbours=int(k),
        alpha=float(alpha),
        specification=specification,
        selection_rule=selection_rule,
        lm_lag_stat=lm_lag_stat,
        lm_lag_p=lm_lag_p,
        lm_error_stat=lm_error_stat,
        lm_error_p=lm_error_p,
        rlm_lag_stat=rlm_lag_stat,
        rlm_lag_p=rlm_lag_p,
        rlm_error_stat=rlm_error_stat,
        rlm_error_p=rlm_error_p,
        lm_sarma_stat=lm_sarma_stat,
        lm_sarma_p=lm_sarma_p,
        moran_ols_i=moran_ols_i,
        moran_ols_z=moran_ols_z,
        moran_ols_p=moran_ols_p,
        moran_residual_i=moran_residual_i,
        moran_residual_p=moran_residual_p,
        misspecified=misspecified,
    )

    return HedonicFit(
        specification=specification,
        variables=x_cols,
        direct=direct,
        total=total,
        rho=rho,
        lam=lam,
        spatial_multiplier=float(multiplier),
        fitted=pd.Series(predicted_values, index=usable.index, name="ln_price_fitted"),
        residuals=pd.Series(residual_values, index=usable.index, name="ln_price_residual"),
        diagnostics=diagnostics,
    )
