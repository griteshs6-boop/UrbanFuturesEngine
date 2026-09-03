"""The ship gate (spec Section 19.6).

Section 20's build sequence calls step 14 "a real stop": if the model does not beat momentum
out of sample, nothing else gets built. :func:`ship_gate` is what makes that judgement, so
it is written to be boring, total and auditable — every criterion is evaluated even after
one has already failed, so the printed reasoning lists *all* the problems rather than the
first one, and every threshold comes out of ``config/params/backtest.yaml``.

Section 19.6, transcribed::

    def ship_gate(scorecards: list[ScoreCard]) -> bool:
        return (
            median(s.spearman for s in holdout) > 0.55
            and median(s.beat_b2 for s in holdout) > 0
            and bootstrap_ci(beat_b2).lower > 0
            and 0.70 <= median(s.band_coverage) <= 0.90
        )

plus Section 23 item 3 — "``ufe backtest gate`` prints PASS on at least three hold-out
cities" — which is enforced here as a criterion in its own right rather than left to whoever
assembles the scorecard list. ``beat_b4`` on ``settlement_spearman`` is **reported and never
gated** (Section 19.6), unless ``backtest.gate.gate_on_beat_b4`` is deliberately flipped.

A missing metric fails its criterion. It never passes it, and it is never skipped: a
scorecard with ``beat_b2 = None`` is a scorecard that did not measure whether the model beat
momentum, and the whole point of the gate is that this is not good enough.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ufe.backtest.score import ScoreCard

logger = logging.getLogger(__name__)

__all__ = ["Criterion", "GateResult", "ship_gate", "bootstrap_ci"]

P_MIN_SPEARMAN = "backtest.gate.min_median_spearman"
P_MIN_BEAT_B2 = "backtest.gate.min_median_beat_b2"
P_MIN_CI_LOWER = "backtest.gate.min_beat_b2_ci_lower"
P_MIN_COVERAGE = "backtest.gate.min_median_band_coverage"
P_MAX_COVERAGE = "backtest.gate.max_median_band_coverage"
P_MIN_CITIES = "backtest.gate.min_holdout_cities"
P_GATE_ON_B4 = "backtest.gate.gate_on_beat_b4"
P_MIN_ZONES = "backtest.scoring.min_zones"
P_BOOT_N = "backtest.scoring.bootstrap.n_resamples"
P_BOOT_LEVEL = "backtest.scoring.bootstrap.ci_level"
P_BOOT_SEED = "backtest.scoring.bootstrap.seed"

PASS = "PASS"
FAIL = "FAIL"

_ZERO = 0
_ONE = 1
_TWO = _ONE + _ONE


@dataclass(frozen=True)
class Criterion:
    """One gate condition, its observed value and whether it held."""

    name: str
    requirement: str
    observed: float | int | None
    passed: bool
    detail: str = ""

    def line(self) -> str:
        mark = PASS if self.passed else FAIL
        observed = "not measured" if self.observed is None else f"{self.observed:.4f}"
        suffix = f" — {self.detail}" if self.detail else ""
        return f"[{mark}] {self.name}: {observed} (requires {self.requirement}){suffix}"


@dataclass(frozen=True)
class GateResult:
    """The gate's verdict, with the reasoning it is required to print."""

    passed: bool
    criteria: tuple[Criterion, ...]
    n_cities: int
    warnings: tuple[str, ...] = ()
    reported: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return PASS if self.passed else FAIL

    @property
    def failing(self) -> tuple[Criterion, ...]:
        return tuple(c for c in self.criteria if not c.passed)

    def reasoning(self) -> str:
        lines = [f"{self.verdict}: backtest ship gate over {self.n_cities} hold-out city/ies", ""]
        lines += [criterion.line() for criterion in self.criteria]
        if self.reported:
            lines += ["", "Reported, not gated (Section 19.6):"]
            lines += [
                f"  {name}: {'n/a' if value is None else f'{value:.4f}'}"
                for name, value in sorted(self.reported.items())
            ]
        if self.warnings:
            lines += [""] + [f"WARNING: {warning}" for warning in self.warnings]
        if not self.passed:
            lines += [
                "",
                "Nothing ships on FAIL (spec Section 19.6). The failing criteria above are "
                "the whole message: do not proceed to the next build step, and do not "
                "re-run with a different origin year until the reason is understood.",
            ]
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return self.passed


# --------------------------------------------------------------------------------------


def _median(values: Iterable[float | None]) -> float | None:
    usable = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not usable:
        return None
    return float(np.median(usable))


def bootstrap_ci(
    values: Sequence[float | None], params: Any, *, rng: np.random.Generator | None = None
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI of the median of ``values`` across hold-out cities.

    Section 19.6's ``bootstrap_ci(beat_b2).lower > 0`` is evaluated at the *gate* level,
    over cities. The per-city zone-level CI already lives on each scorecard
    (``beat_b2_ci_lower``); this one asks the different and harder question of whether the
    advantage survives resampling which cities we happened to hold out. With a single city
    that question has no answer, and the gate says so rather than inventing one.
    """
    usable = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if len(usable) < _TWO:
        return None, None
    n_resamples = int(params.value(P_BOOT_N))
    level = float(params.value(P_BOOT_LEVEL))
    rng = np.random.default_rng(int(params.value(P_BOOT_SEED))) if rng is None else rng

    sample = np.asarray(usable, dtype=float)
    draws = np.median(
        sample[rng.integers(_ZERO, len(sample), size=(n_resamples, len(sample)))], axis=_ONE
    )
    tail = (_ONE - level) / _TWO
    return float(np.quantile(draws, tail)), float(np.quantile(draws, _ONE - tail))


def ship_gate(
    scorecards: Sequence[ScoreCard],
    params: Any,
    *,
    rng: np.random.Generator | None = None,
) -> GateResult:
    """Evaluate Section 19.6 over the hold-out scorecards. Returns the full reasoning."""
    cards = list(scorecards)
    cities = sorted({card.city_id for card in cards})
    min_cities = int(params.value(P_MIN_CITIES))
    min_zones = int(params.value(P_MIN_ZONES))

    warnings: list[str] = []
    criteria: list[Criterion] = []

    criteria.append(
        Criterion(
            name="holdout_cities",
            requirement=f">= {min_cities}",
            observed=len(cities),
            passed=len(cities) >= min_cities,
            detail="Section 23 item 3: PASS on at least three hold-out cities"
            + (f" (have {', '.join(cities)})" if cities else " (have none)"),
        )
    )

    price_cards = [card for card in cards if card.price_test_enabled]
    if len(price_cards) < len(cards):
        warnings.append(
            f"{len(cards) - len(price_cards)} scorecard(s) ran the settlement test only "
            "because the t0 price surface could not be reconstructed (Section 19.1). They "
            "contribute nothing to the price criteria below."
        )

    underpowered = [card for card in cards if card.underpowered or card.n_zones < min_zones]
    criteria.append(
        Criterion(
            name="scorecards_powered",
            requirement=f"every scorecard has >= {min_zones} zones",
            observed=len(cards) - len(underpowered),
            passed=not underpowered,
            detail=(
                "underpowered: "
                + ", ".join(f"{c.city_id}@{c.t0} ({c.n_zones} zones)" for c in underpowered)
            )
            if underpowered
            else "",
        )
    )

    median_spearman = _median(card.spearman for card in price_cards)
    minimum = float(params.value(P_MIN_SPEARMAN))
    criteria.append(
        Criterion(
            name="median_spearman",
            requirement=f"> {minimum}",
            observed=median_spearman,
            passed=median_spearman is not None and median_spearman > minimum,
        )
    )

    beat_values = [card.beat_b2 for card in price_cards]
    median_beat = _median(beat_values)
    minimum_beat = float(params.value(P_MIN_BEAT_B2))
    criteria.append(
        Criterion(
            name="median_beat_b2",
            requirement=f"> {minimum_beat}",
            observed=median_beat,
            passed=median_beat is not None and median_beat > minimum_beat,
            detail="the model must out-rank momentum out of sample; this is the real stop",
        )
    )

    ci_lower, ci_upper = bootstrap_ci(beat_values, params, rng=rng)
    minimum_ci = float(params.value(P_MIN_CI_LOWER))
    per_card_lowers = [card.beat_b2_ci_lower for card in price_cards]
    criteria.append(
        Criterion(
            name="beat_b2_ci_lower",
            requirement=f"> {minimum_ci}",
            observed=ci_lower,
            passed=ci_lower is not None and ci_lower > minimum_ci,
            detail=(
                f"across-city bootstrap of the median (upper {ci_upper:.4f}); "
                "per-city zone-level lower bounds "
                + ", ".join("n/a" if v is None else f"{v:.4f}" for v in per_card_lowers)
            )
            if ci_lower is not None
            else "fewer than two hold-out cities with a measured beat_b2: the "
            "across-city CI is undefined, which is a FAIL, not a pass",
        )
    )

    median_coverage = _median(card.band_coverage for card in price_cards)
    low = float(params.value(P_MIN_COVERAGE))
    high = float(params.value(P_MAX_COVERAGE))
    criteria.append(
        Criterion(
            name="median_band_coverage",
            requirement=f"{low} <= x <= {high}",
            observed=median_coverage,
            passed=median_coverage is not None and low <= median_coverage <= high,
            detail="too low means the bands lie; too high means they say nothing",
        )
    )

    median_beat_b4 = _median(card.beat_b4 for card in cards)
    reported = {
        "median_beat_b4_settlement": median_beat_b4,
        "median_settlement_spearman": _median(card.settlement_spearman for card in cards),
        "median_top3_precision": _median(card.top3_precision for card in price_cards),
        "median_mape_cagr": _median(card.mape_cagr for card in price_cards),
        "median_overheat_precision": _median(card.overheat_precision for card in price_cards),
    }
    if bool(params.value(P_GATE_ON_B4)):
        criteria.append(
            Criterion(
                name="median_beat_b4",
                requirement=f"> {minimum_beat}",
                observed=median_beat_b4,
                passed=median_beat_b4 is not None and median_beat_b4 > minimum_beat,
                detail="gated because backtest.gate.gate_on_beat_b4 is true; Section 19.6 "
                "reports this by default",
            )
        )
    elif median_beat_b4 is not None and median_beat_b4 <= _ZERO:
        warnings.append(
            "the engine loses to the in-house logistic-CA baseline (B4) on settlement "
            "prediction. Section 19.6: informative rather than fatal if it still wins on "
            "price — the value is then in the credibility and factor machinery, which "
            "changes what you sell. If it loses on both, stop."
        )

    alarmed = [card for card in cards if card.suspicious_spearman]
    if alarmed:
        warnings.append(
            "LOOK-AHEAD ALARM on "
            + ", ".join(f"{c.city_id}@{c.t0} (spearman {c.spearman:.4f})" for c in alarmed)
            + ". Section 21 lists a suspiciously high Spearman as the symptom of "
            "look-ahead. A PASS carrying this warning is not a PASS until the freeze "
            "provenance report has been re-read."
        )

    result = GateResult(
        passed=all(criterion.passed for criterion in criteria),
        criteria=tuple(criteria),
        n_cities=len(cities),
        warnings=tuple(warnings),
        reported=reported,
    )
    logger.info("ship gate verdict: %s over %d city/ies", result.verdict, result.n_cities)
    return result
