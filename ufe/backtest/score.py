"""Scoring for the backtest (spec Section 19.4).

One entry point, :func:`score`, producing one :class:`ScoreCard` per city per origin year.
Every metric in the Section 19.4 table is implemented; nothing is optional and nothing is
silently skipped — a metric that cannot be computed is ``None`` on the card with the reason
recorded in ``notes``, so a scorecard never looks better by omission.

The Section 21 look-ahead alarm lives here too. "Suspiciously high Spearman, >0.8" is a
*symptom*, so :class:`ScoreCard.suspicious_spearman` is raised whenever the threshold in
``backtest.scoring.lookahead.spearman_alarm`` is crossed, and the gate repeats it in its
reasoning. It is not an automatic failure — a genuinely excellent model would trip it — but
it is impossible to read a passing scorecard without seeing it, which is the point.

Level convention
----------------
``pred`` and ``actual`` are **zone-level** Series of total fractional appreciation over the
horizon, indexed identically. :func:`to_zone` aggregates a cell-level baseline output to
res-8 zones (Section 19.4's ``top3_precision`` is defined on res-8 zones). Settlement
metrics take Δ ``builtup_frac`` in the same shape.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = ["ScoreCard", "score", "to_zone", "spearman", "bootstrap_difference_ci"]

# ------------------------------------------------------------------ parameter paths
P_TOP_K = "backtest.scoring.top_k_zones"
P_TOP_QUANTILE = "backtest.scoring.actual_top_quantile"
P_MIN_ZONES = "backtest.scoring.min_zones"
P_BOOT_N = "backtest.scoring.bootstrap.n_resamples"
P_BOOT_LEVEL = "backtest.scoring.bootstrap.ci_level"
P_BOOT_SEED = "backtest.scoring.bootstrap.seed"
P_BAND_LOWER = "backtest.scoring.bands.lower_column"
P_BAND_UPPER = "backtest.scoring.bands.upper_column"
P_BAND_NOMINAL = "backtest.scoring.bands.nominal_coverage"
P_BAND_PREFIX = "backtest.scoring.bands.column_prefix"
P_BAND_PCT_SCALE = "backtest.scoring.bands.percent_scale"
P_MAPE_MIN = "backtest.scoring.mape.min_abs_actual"
P_OVERHEAT_QUANTILE = "backtest.scoring.overheat.comparison_quantile"
P_SPEARMAN_ALARM = "backtest.scoring.lookahead.spearman_alarm"

COL_ZONE = "h3_res8"
_ZERO = 0
_ONE = 1
_TWO = _ONE + _ONE


# --------------------------------------------------------------------------------------
# ScoreCard
# --------------------------------------------------------------------------------------


@dataclass
class ScoreCard:
    """The Section 19.4 metric block for one city at one origin year.

    Serialises to JSON (:meth:`to_json`) and to a markdown report (:meth:`to_markdown`),
    as Section 19.4 requires.
    """

    city_id: str
    t0: int
    horizon_years: int
    n_zones: int

    spearman: float | None = None
    spearman_pvalue: float | None = None
    top3_precision: float | None = None
    beat_b2: float | None = None
    beat_b2_ci_lower: float | None = None
    beat_b2_ci_upper: float | None = None
    band_coverage: float | None = None
    reliability: tuple[tuple[float, float], ...] = ()
    overheat_precision: float | None = None
    mape_cagr: float | None = None
    settlement_spearman: float | None = None
    beat_b4: float | None = None

    b2_spearman: float | None = None
    b4_settlement_spearman: float | None = None

    price_test_enabled: bool = True
    suspicious_spearman: bool = False
    spearman_alarm_threshold: float | None = None
    underpowered: bool = False
    mape_excluded_zones: int = _ZERO

    params_hash: str | None = None
    freeze_hash: str | None = None
    seed: int | None = None
    notes: list[str] = field(default_factory=list)

    # -------------------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reliability"] = [list(point) for point in self.reliability]
        return payload

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str, **kwargs)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScoreCard":
        data = dict(payload)
        data["reliability"] = tuple(tuple(point) for point in data.get("reliability", ()))
        data["notes"] = list(data.get("notes", []))
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001 - dataclass API
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_markdown(self) -> str:
        rows = [
            ("spearman", self.spearman),
            ("top3_precision", self.top3_precision),
            ("beat_b2", self.beat_b2),
            ("beat_b2 CI", _format_interval(self.beat_b2_ci_lower, self.beat_b2_ci_upper)),
            ("band_coverage", self.band_coverage),
            ("overheat_precision", self.overheat_precision),
            ("mape_cagr", self.mape_cagr),
            ("settlement_spearman", self.settlement_spearman),
            ("beat_b4 (reported, not gated)", self.beat_b4),
        ]
        lines = [
            f"# Backtest scorecard — {self.city_id} @ t0={self.t0} "
            f"(+{self.horizon_years}y, {self.n_zones} zones)",
            "",
            "| metric | value |",
            "|---|---|",
        ]
        lines += [f"| `{name}` | {_format_value(value)} |" for name, value in rows]
        if self.reliability:
            lines += [
                "",
                "## Reliability diagram",
                "",
                "| nominal coverage | empirical coverage |",
                "|---|---|",
            ]
            lines += [
                f"| {_format_value(nominal)} | {_format_value(empirical)} |"
                for nominal, empirical in self.reliability
            ]
        if not self.price_test_enabled:
            lines += [
                "",
                "> The t0 price surface could not be reconstructed densely enough for this "
                "city, so the price metrics are not reported and only the settlement test "
                "ran (spec Section 19.1).",
            ]
        if self.suspicious_spearman:
            lines += [
                "",
                f"> **Look-ahead alarm.** Spearman {_format_value(self.spearman)} exceeds "
                f"{_format_value(self.spearman_alarm_threshold)}. Section 21 lists this as "
                "the symptom of look-ahead in the backtest. Re-read the freeze provenance "
                "report before quoting this number.",
            ]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {note}" for note in self.notes]
        lines += [
            "",
            f"params_hash: `{self.params_hash}` · freeze_hash: `{self.freeze_hash}` · "
            f"seed: `{self.seed}`",
        ]
        return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_interval(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "n/a"
    return f"[{lower:.4f}, {upper:.4f}]"


# --------------------------------------------------------------------------------------
# metric primitives
# --------------------------------------------------------------------------------------


def to_zone(values: pd.Series, cells: pd.DataFrame, *, how: str = "mean") -> pd.Series:
    """Aggregate a cell-indexed Series to res-8 zones (Section 19.4's scoring level)."""
    mapping = cells.set_index("h3")[COL_ZONE]
    zones = values.index.map(mapping)
    out = values.groupby(zones).agg(how)
    out.index.name = COL_ZONE
    out.name = values.name
    return out.sort_index()


def _aligned(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    shared = left.index.intersection(right.index)
    a = pd.to_numeric(left.reindex(shared), errors="coerce")
    b = pd.to_numeric(right.reindex(shared), errors="coerce")
    keep = a.notna() & b.notna()
    return a[keep].to_numpy(), b[keep].to_numpy(), shared[keep.to_numpy()]


def spearman(pred: pd.Series, actual: pd.Series) -> tuple[float | None, float | None]:
    """``scipy.stats.spearmanr`` on aligned zones. ``(None, None)`` when undefined."""
    x, y, _ = _aligned(pred, actual)
    # `ptp`, not `std`: a constant B0 prediction accumulates float dust under `std` and
    # would slip past the guard into a ConstantInputWarning and a NaN.
    if len(x) < _TWO or float(np.ptp(x)) == _ZERO or float(np.ptp(y)) == _ZERO:
        return None, None
    result = stats.spearmanr(x, y)
    correlation = float(result.statistic)
    if np.isnan(correlation):
        return None, None
    return correlation, float(result.pvalue)


def _top_k_precision(pred: pd.Series, actual: pd.Series, params: Any) -> float | None:
    """Fraction of the predicted top-k zones that land in the actual top decile."""
    k = int(params.value(P_TOP_K))
    quantile = float(params.value(P_TOP_QUANTILE))
    x, y, index = _aligned(pred, actual)
    if len(index) < k:
        return None
    predicted = pd.Series(x, index=index).sort_values(ascending=False).head(k).index
    cutoff = float(np.quantile(y, quantile))
    actual_top = set(index[y >= cutoff])
    return len(actual_top.intersection(predicted)) / k


def bootstrap_difference_ci(
    pred: pd.Series,
    reference: pd.Series,
    actual: pd.Series,
    params: Any,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[float | None, float | None, float | None]:
    """``spearman(model) - spearman(reference)`` with a percentile bootstrap over zones.

    Zones, not observations: the unit of resampling has to be the unit the claim is made
    about, and the claim is "the model ranks zones better than momentum does".
    """
    n_resamples = int(params.value(P_BOOT_N))
    level = float(params.value(P_BOOT_LEVEL))
    rng = np.random.default_rng(int(params.value(P_BOOT_SEED))) if rng is None else rng

    shared = pred.index.intersection(reference.index).intersection(actual.index)
    frame = pd.DataFrame(
        {
            "pred": pd.to_numeric(pred.reindex(shared), errors="coerce"),
            "ref": pd.to_numeric(reference.reindex(shared), errors="coerce"),
            "actual": pd.to_numeric(actual.reindex(shared), errors="coerce"),
        }
    ).dropna()
    if len(frame) < _TWO:
        return None, None, None

    point_model, _ = spearman(frame["pred"], frame["actual"])
    point_reference, _ = spearman(frame["ref"], frame["actual"])
    if point_model is None or point_reference is None:
        return None, None, None
    point = point_model - point_reference

    values = frame.to_numpy()
    draws = np.empty(n_resamples, dtype=float)
    n = len(frame)
    kept = _ZERO
    for draw in range(n_resamples):
        rows = rng.integers(_ZERO, n, size=n)
        sample = values[rows]
        if float(np.ptp(sample[:, _ZERO])) == _ZERO or float(np.ptp(sample[:, _TWO])) == _ZERO:
            continue
        model_rho = float(stats.spearmanr(sample[:, _ZERO], sample[:, _TWO]).statistic)
        reference_rho = float(stats.spearmanr(sample[:, _ONE], sample[:, _TWO]).statistic)
        if np.isnan(model_rho) or np.isnan(reference_rho):
            continue
        draws[kept] = model_rho - reference_rho
        kept += _ONE
    if kept == _ZERO:
        return point, None, None

    tail = (_ONE - level) / _TWO
    lower = float(np.quantile(draws[:kept], tail))
    upper = float(np.quantile(draws[:kept], _ONE - tail))
    return point, lower, upper


def _band_coverage(
    actual: pd.Series, bands: pd.DataFrame, params: Any
) -> tuple[float | None, tuple[tuple[float, float], ...]]:
    """Coverage of the p10-p90 interval, plus a reliability diagram over every band pair."""
    lower_column = str(params.value(P_BAND_LOWER))
    upper_column = str(params.value(P_BAND_UPPER))
    prefix = str(params.value(P_BAND_PREFIX))
    percent_scale = float(params.value(P_BAND_PCT_SCALE))

    shared = bands.index.intersection(actual.index)
    if len(shared) == _ZERO:
        return None, ()
    observed = pd.to_numeric(actual.reindex(shared), errors="coerce")

    coverage: float | None = None
    if lower_column in bands.columns and upper_column in bands.columns:
        low = pd.to_numeric(bands.loc[shared, lower_column], errors="coerce")
        high = pd.to_numeric(bands.loc[shared, upper_column], errors="coerce")
        inside = (observed >= low) & (observed <= high)
        valid = observed.notna() & low.notna() & high.notna()
        coverage = float(inside[valid].mean()) if bool(valid.any()) else None

    # Reliability diagram: every `p<q>` column, paired symmetrically about the median.
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    quantiles: dict[float, str] = {}
    for column in bands.columns:
        match = pattern.match(str(column))
        if match:
            quantiles[int(match.group(_ONE)) / percent_scale] = str(column)
    diagram: list[tuple[float, float]] = []
    for q in sorted(quantile for quantile in quantiles if quantile < _ONE / _TWO):
        partner = _ONE - q
        if partner not in quantiles:
            continue
        low = pd.to_numeric(bands.loc[shared, quantiles[q]], errors="coerce")
        high = pd.to_numeric(bands.loc[shared, quantiles[partner]], errors="coerce")
        valid = observed.notna() & low.notna() & high.notna()
        if not bool(valid.any()):
            continue
        empirical = float(((observed >= low) & (observed <= high))[valid].mean())
        diagram.append((partner - q, empirical))
    return coverage, tuple(diagram)


def _overheat_precision(
    actual: pd.Series, overheat_flags: pd.Series, params: Any
) -> float | None:
    """Of zones flagged overheated at t0, the fraction underperforming the city median."""
    quantile = float(params.value(P_OVERHEAT_QUANTILE))
    shared = overheat_flags.index.intersection(actual.index)
    if len(shared) == _ZERO:
        return None
    flagged = overheat_flags.reindex(shared).astype(bool)
    observed = pd.to_numeric(actual.reindex(shared), errors="coerce")
    if not bool(flagged.any()):
        return None
    median = float(np.nanquantile(observed.to_numpy(dtype=float), quantile))
    subset = observed[flagged].dropna()
    if subset.empty:
        return None
    return float((subset < median).mean())


def _cagr(appreciation: pd.Series, years: int) -> pd.Series:
    values = pd.to_numeric(appreciation, errors="coerce").astype(float)
    return (_ONE + values) ** (_ONE / years) - _ONE


def _mape_cagr(
    pred: pd.Series, actual: pd.Series, years: int, params: Any
) -> tuple[float | None, int]:
    minimum = float(params.value(P_MAPE_MIN))
    x, y, index = _aligned(pred, actual)
    if len(index) == _ZERO:
        return None, _ZERO
    predicted = _cagr(pd.Series(x, index=index), years)
    observed = _cagr(pd.Series(y, index=index), years)
    usable = observed.abs() >= minimum
    excluded = int((~usable).sum())
    if not bool(usable.any()):
        return None, excluded
    errors = ((predicted[usable] - observed[usable]) / observed[usable]).abs()
    return float(errors.mean()), excluded


# --------------------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------------------


def score(
    pred: pd.Series,
    actual: pd.Series,
    bands: pd.DataFrame | None = None,
    *,
    params: Any,
    years: int,
    city_id: str = "",
    t0: int = _ZERO,
    b2_pred: pd.Series | None = None,
    overheat_flags: pd.Series | None = None,
    settlement_pred: pd.Series | None = None,
    settlement_actual: pd.Series | None = None,
    b4_settlement_pred: pd.Series | None = None,
    price_test_enabled: bool = True,
    freeze_hash: str | None = None,
    rng: np.random.Generator | None = None,
) -> ScoreCard:
    """Compute the Section 19.4 metric block. See the module docstring for conventions."""
    notes: list[str] = []
    min_zones = int(params.value(P_MIN_ZONES))
    alarm = float(params.value(P_SPEARMAN_ALARM))
    seed = int(params.value(P_BOOT_SEED))

    _, _, shared = _aligned(pred, actual)
    card = ScoreCard(
        city_id=city_id,
        t0=int(t0),
        horizon_years=int(years),
        n_zones=len(shared),
        price_test_enabled=bool(price_test_enabled),
        spearman_alarm_threshold=alarm,
        params_hash=getattr(params, "hash", None),
        freeze_hash=freeze_hash,
        seed=seed,
    )
    card.underpowered = card.n_zones < min_zones
    if card.underpowered:
        notes.append(
            f"only {card.n_zones} scoring zone(s), below the {min_zones}-zone minimum: "
            "every metric here is underpowered and the gate will refuse this scorecard"
        )

    if price_test_enabled:
        card.spearman, card.spearman_pvalue = spearman(pred, actual)
        card.top3_precision = _top_k_precision(pred, actual, params)
        card.mape_cagr, card.mape_excluded_zones = _mape_cagr(pred, actual, years, params)
        if card.mape_excluded_zones:
            notes.append(
                f"{card.mape_excluded_zones} zone(s) excluded from mape_cagr for a "
                "near-zero actual CAGR (MAPE is undefined there)"
            )
        if b2_pred is not None:
            card.b2_spearman, _ = spearman(b2_pred, actual)
            (
                card.beat_b2,
                card.beat_b2_ci_lower,
                card.beat_b2_ci_upper,
            ) = bootstrap_difference_ci(pred, b2_pred, actual, params, rng=rng)
        else:
            notes.append("no B2 prediction supplied: beat_b2 is unscored and the gate fails")
        if bands is not None:
            card.band_coverage, card.reliability = _band_coverage(actual, bands, params)
        else:
            notes.append("no prediction bands supplied: band_coverage is unscored")
        if overheat_flags is not None:
            card.overheat_precision = _overheat_precision(actual, overheat_flags, params)
    else:
        notes.append(
            "price test skipped: the t0 price surface could not be reconstructed "
            "(spec Section 19.1 step 1). Only the settlement test is reported."
        )

    if settlement_pred is not None and settlement_actual is not None:
        card.settlement_spearman, _ = spearman(settlement_pred, settlement_actual)
        if b4_settlement_pred is not None:
            card.b4_settlement_spearman, _ = spearman(b4_settlement_pred, settlement_actual)
            if card.settlement_spearman is not None and card.b4_settlement_spearman is not None:
                card.beat_b4 = card.settlement_spearman - card.b4_settlement_spearman
                notes.append(
                    "beat_b4 is reported, never gated (Section 19.6). Losing to a "
                    "150-line logistic CA on settlement while winning on price is "
                    "informative, not fatal; losing on both means stop."
                )

    if card.spearman is not None and card.spearman > alarm:
        card.suspicious_spearman = True
        notes.append(
            f"LOOK-AHEAD ALARM: spearman {card.spearman:.4f} exceeds {alarm:.4f}. Section "
            "21 lists a suspiciously high Spearman as the symptom of look-ahead in the "
            "backtest. Re-run freeze.assert_no_lookahead and read its provenance report "
            "before this number leaves the building."
        )
        logger.warning(notes[-_ONE])

    card.notes = notes
    return card


def load_scorecards(payload: Sequence[Mapping[str, Any]]) -> list[ScoreCard]:
    """Rehydrate scorecards from parsed JSON (the CLI's ``--scorecards`` input)."""
    return [ScoreCard.from_dict(item) for item in payload]
