"""Tests for Module 6 — Layer 3, credibility (spec Section 10).

Written before the implementation, per CONTRACT.md rule 6.  Every item in the Section 10
ACCEPTANCE block is marked ``@pytest.mark.acceptance``.

The hand-computed expectations below are derived from ``config/params/credibility.yaml`` as
it stands on disk; each one shows its arithmetic so a parameter change makes the test fail
loudly rather than silently drifting.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from ufe.errors import MissingParameter
from ufe.layers import l3_credibility as cred
from ufe.params import load_params
from tests.fixtures.synthetic import (  # noqa: F401  (registers the session fixture)
    synthetic_announcers,
    synthetic_city,
    synthetic_projects,
)

CITY = "vizag"

# The synthetic fixture's archetypes are not all present in credibility.delay_lognormal
# (reported as a gap); tests that need a delay family for every synthetic project pass this.
SYNTHETIC_FAMILY_MAP = {
    "metro_phase1": "metro_phase1",
    "highway": "highway",
    "data_centre": "data_centre",
    "airport_greenfield": "airport_greenfield",
    "trunk_utilities": "trunk_utilities",
    "industrial_park": "private_industrial",
    "port_expansion": "govt_mega",
    "it_campus": "private_industrial",
    "township": "private_industrial",
    "university": "govt_mega",
}


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


# --------------------------------------------------------------------------------------
# frame builders — everything is derived from tests/fixtures/synthetic.py
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def project_template() -> pd.Series:
    """A single synthetic project row, used as the schema-complete template."""
    return synthetic_projects().iloc[0].copy()


def make_projects(template: pd.Series, specs: list[dict]) -> pd.DataFrame:
    """Build a projects frame with `len(specs)` rows from the synthetic template."""
    rows = []
    for i, spec in enumerate(specs):
        row = template.copy()
        row["project_id"] = spec.pop("project_id", f"t-{i:03d}")
        row["name"] = f"Test project {i}"
        row["modifiers"] = []
        for key, value in spec.items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def make_announcers(template_frame: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    row = template_frame.iloc[0]
    rows = []
    for spec in specs:
        new = row.copy()
        for key, value in spec.items():
            new[key] = value
        rows.append(new)
    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.fixture(scope="module")
def announcer_template() -> pd.DataFrame:
    return synthetic_announcers()


# --------------------------------------------------------------------------------------
# 10.1 / 10.2 — hand-computed completion probabilities
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acceptance_summit_mou_weak_announcer_below_ten_percent(
    params, project_template, announcer_template
):
    """Section 10 ACCEPTANCE 1.

    "A summit-MoU project from an announcer with 0.2 delivery ratio and 5x capacity ratio
    yields `p < 0.10`."

    Hand computation against credibility.yaml as it stands:
        delivery_score      delivery_ratio 0.20 -> band {min: 0.00}      = 0.25
        lag_score           1 - 30/60                                    = 0.50
        capacity_score      5.0 -> band {max: 6.0}                       = 0.30
        hardness_component  commitment_hardness.summit_mou               = 0.15
        cycle_score         no sector_capacity_util -> 1 - centre        = 0.50
        ACS_raw = .30*.25 + .15*.50 + .20*.30 + .20*.15 + .15*.50        = 0.315
        ACS     = 0.55 + 0.315*(1.35-0.55)                               = 0.802
        p_base  = 0.15 * 0.802                                           = 0.1203
        p       = 0.1203 * modifiers.political_event_announce (0.80)     = 0.09624

    NOTE (reported): with NO modifier the same project scores 0.1203, i.e. the bare
    Section 10.1/10.2 formula on the on-disk parameters cannot reach p < 0.10 for a
    summit MoU — the floor is acs_bounds.min * commitment_hardness.summit_mou = 0.0825
    and the stated inputs land above 0.10.  The acceptance therefore only holds once a
    plausible summit-MoU modifier is attached.  Both values are asserted below.
    """
    announcers = make_announcers(
        announcer_template,
        [
            {
                "announcer_id": "weak",
                "delivery_ratio": 0.20,
                "median_slip_months": 30.0,
                "mean_annual_capex_3y_inr_cr": 100.0,
            }
        ],
    )
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "summit-with-modifier",
                "is_public": False,
                "announcer_id": "weak",
                "commitment_form": "summit_mou",
                "capex_inr_cr": 500.0,
                "modifiers": ["political_event_announce"],
                "physical_state": None,
                "physical_asof": pd.NaT,
            },
            {
                "project_id": "summit-bare",
                "is_public": False,
                "announcer_id": "weak",
                "commitment_form": "summit_mou",
                "capex_inr_cr": 500.0,
                "modifiers": [],
                "physical_state": None,
                "physical_asof": pd.NaT,
            },
        ],
    )

    out = cred.completion_probability(projects, announcers, params)
    row = out.set_index("project_id")

    assert row.loc["summit-with-modifier", "delivery_score"] == pytest.approx(0.25)
    assert row.loc["summit-with-modifier", "lag_score"] == pytest.approx(0.50)
    assert row.loc["summit-with-modifier", "capacity_score"] == pytest.approx(0.30)
    assert row.loc["summit-with-modifier", "hardness_component"] == pytest.approx(0.15)
    assert row.loc["summit-with-modifier", "cycle_score"] == pytest.approx(0.50)
    assert row.loc["summit-with-modifier", "acs_raw"] == pytest.approx(0.315)
    assert row.loc["summit-with-modifier", "acs"] == pytest.approx(0.802)
    assert row.loc["summit-with-modifier", "p_base"] == pytest.approx(0.1203)

    assert row.loc["summit-with-modifier", "p_completion"] == pytest.approx(0.09624)
    assert row.loc["summit-with-modifier", "p_completion"] < 0.10

    # The documented shortfall of the bare formula.
    assert row.loc["summit-bare", "p_completion"] == pytest.approx(0.1203)


@pytest.mark.acceptance
def test_acceptance_public_construction_land_secured_above_ninety_percent(
    params, project_template, announcer_template
):
    """Section 10 ACCEPTANCE 2: construction-stage public project with land secured -> p > 0.90.

    p_base = stage_probability.construction = 0.92
    p      = min(0.92 * modifiers.land_over_70pct (1.20), p_cap 0.97) = 0.97
    """
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "public-construction",
                "is_public": True,
                "announcer_id": None,
                "commitment_form": None,
                "stage": "construction",
                "land_possession_pct": 0.85,
                "modifiers": ["land_over_70pct"],
                "physical_state": "structure",
                "physical_asof": pd.NaT,
            }
        ],
    )
    out = cred.completion_probability(projects, announcer_template, params)
    assert out.loc[0, "p_base"] == pytest.approx(0.92)
    assert out.loc[0, "p_completion"] == pytest.approx(0.97)
    assert out.loc[0, "p_completion"] > 0.90


@pytest.mark.acceptance
def test_acceptance_p_never_exceeds_p_cap(params, synthetic_city):
    """Section 10 ACCEPTANCE 3: `p` never exceeds `p_cap`."""
    p_cap = params.value("credibility.p_cap")
    out = cred.completion_probability(
        synthetic_city.projects,
        synthetic_city.announcers,
        params,
        unknown_modifiers="ignore",
    )
    assert (out["p_completion"] <= p_cap).all()
    assert (out["p_completion"] >= 0).all()


@pytest.mark.acceptance
def test_acceptance_physical_divergence_penalty_and_staleness(
    params, project_template, announcer_template
):
    """Section 10 ACCEPTANCE 4.

    "Physical divergence halves `p` for a project claiming construction with no satellite
    activity, and does nothing when `physical_asof` is older than `stale_after_days`."
    """
    penalty = params.value("credibility.physical_divergence.penalty_mult")
    stale_after = params.value("credibility.physical_divergence.stale_after_days")
    as_of = datetime(2025, 1, 1)

    base = {
        "is_public": True,
        "announcer_id": None,
        "commitment_form": None,
        "stage": "construction",
        "modifiers": [],
    }
    projects = make_projects(
        project_template,
        [
            {
                **base,
                "project_id": "fresh-divergent",
                "physical_state": "none",
                "physical_asof": as_of - timedelta(days=int(stale_after) // 2),
            },
            {
                **base,
                "project_id": "stale-divergent",
                "physical_state": "none",
                "physical_asof": as_of - timedelta(days=int(stale_after) * 2),
            },
            {
                **base,
                "project_id": "no-divergence",
                "physical_state": "structure",
                "physical_asof": as_of - timedelta(days=int(stale_after) // 2),
            },
        ],
    )
    out = cred.completion_probability(
        projects, announcer_template, params, as_of=as_of
    ).set_index("project_id")

    undiverged = out.loc["no-divergence", "p_completion"]
    assert out.loc["fresh-divergent", "p_completion"] == pytest.approx(
        undiverged * penalty
    )
    assert "physical_divergence" in out.loc["fresh-divergent", "credibility_flags"]

    # Stale observation: the penalty does nothing.
    assert out.loc["stale-divergent", "p_completion"] == pytest.approx(undiverged)
    assert "physical_divergence" not in out.loc["stale-divergent", "credibility_flags"]


def test_physical_divergence_also_fires_on_commitment_form(
    params, project_template, announcer_template
):
    """`claimed_construction_stages` mixes the stage and commitment_form vocabularies.

    `epc_appointed` / `equipment_ordered` are commitment forms, not stages, so membership is
    tested against both columns.
    """
    as_of = datetime(2025, 1, 1)
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "epc-no-activity",
                "is_public": False,
                "announcer_id": None,
                "commitment_form": "epc_appointed",
                "stage": "funded",
                "physical_state": "cleared",
                "physical_asof": as_of - timedelta(days=1),
                "modifiers": [],
            }
        ],
    )
    out = cred.completion_probability(projects, announcer_template, params, as_of=as_of)
    assert "physical_divergence" in out.loc[0, "credibility_flags"]


# --------------------------------------------------------------------------------------
# monotonicity
# --------------------------------------------------------------------------------------


def test_completion_probability_monotone_in_stage(
    params, project_template, announcer_template
):
    """A later stage must not lower completion probability."""
    stages = list(params.get("credibility.stage_probability"))
    projects = make_projects(
        project_template,
        [
            {
                "project_id": f"stage-{s}",
                "is_public": True,
                "announcer_id": None,
                "commitment_form": None,
                "stage": s,
                "modifiers": [],
                "physical_state": None,
                "physical_asof": pd.NaT,
            }
            for s in stages
        ],
    )
    out = cred.completion_probability(projects, announcer_template, params)
    p = out["p_completion"].to_numpy()
    assert np.all(np.diff(p) >= 0)


def test_completion_probability_monotone_in_delivery_record(
    params, project_template, announcer_template
):
    """A better delivery record must not lower completion probability."""
    ratios = [0.05, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]
    announcers = make_announcers(
        announcer_template,
        [
            {
                "announcer_id": f"a{i}",
                "delivery_ratio": r,
                "median_slip_months": 24.0,
                "mean_annual_capex_3y_inr_cr": 1000.0,
            }
            for i, r in enumerate(ratios)
        ],
    )
    projects = make_projects(
        project_template,
        [
            {
                "project_id": f"p{i}",
                "is_public": False,
                "announcer_id": f"a{i}",
                "commitment_form": "board_approved",
                "capex_inr_cr": 900.0,
                "modifiers": [],
                "physical_state": None,
                "physical_asof": pd.NaT,
            }
            for i in range(len(ratios))
        ],
    )
    out = cred.completion_probability(projects, announcers, params)
    p = out["p_completion"].to_numpy()
    assert np.all(np.diff(p) >= 0)
    assert p[-1] > p[0]


def test_completion_probability_monotone_in_commitment_hardness(
    params, project_template, announcer_template
):
    forms = list(params.get("credibility.commitment_hardness"))
    announcers = make_announcers(
        announcer_template,
        [
            {
                "announcer_id": "a",
                "delivery_ratio": 0.6,
                "median_slip_months": 12.0,
                "mean_annual_capex_3y_inr_cr": 1000.0,
            }
        ],
    )
    projects = make_projects(
        project_template,
        [
            {
                "project_id": f"f-{f}",
                "is_public": False,
                "announcer_id": "a",
                "commitment_form": f,
                "capex_inr_cr": 500.0,
                "modifiers": [],
                "physical_state": None,
                "physical_asof": pd.NaT,
            }
            for f in forms
        ],
    )
    out = cred.completion_probability(projects, announcers, params)
    assert np.all(np.diff(out["p_completion"].to_numpy()) >= 0)


def test_acs_is_bounded_by_construction(params, synthetic_city):
    lo = params.value("credibility.acs_bounds.min")
    hi = params.value("credibility.acs_bounds.max")
    out = cred.completion_probability(
        synthetic_city.projects,
        synthetic_city.announcers,
        params,
        unknown_modifiers="ignore",
    )
    acs = out["acs"].dropna()
    assert len(acs) > 0
    assert acs.between(lo, hi).all()
    assert out["acs_raw"].dropna().between(0, 1).all()


def test_missing_announcer_uses_unknown_announcer_score(
    params, project_template, announcer_template
):
    unknown = params.value("credibility.delivery_ratio_score.unknown_announcer")
    missing = params.value("credibility.capacity_score.missing_value")
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "orphan",
                "is_public": False,
                "announcer_id": None,
                "commitment_form": "verbal",
                "capex_inr_cr": 100.0,
                "modifiers": [],
                "physical_state": None,
                "physical_asof": pd.NaT,
            }
        ],
    )
    out = cred.completion_probability(projects, announcer_template, params)
    assert out.loc[0, "delivery_score"] == pytest.approx(unknown)
    assert out.loc[0, "capacity_score"] == pytest.approx(missing)
    assert bool(out.loc[0, "capacity_data_missing"])
    assert "capacity_unknown" in out.loc[0, "credibility_flags"]


def test_unknown_modifier_raises_missing_parameter(
    params, project_template, announcer_template
):
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "bad-modifier",
                "is_public": True,
                "announcer_id": None,
                "commitment_form": None,
                "stage": "funded",
                "modifiers": ["not_a_real_modifier"],
            }
        ],
    )
    with pytest.raises(MissingParameter, match="credibility.modifiers.not_a_real_modifier"):
        cred.completion_probability(projects, announcer_template, params)


def test_sector_capacity_util_requires_missing_params(
    params, project_template, announcer_template
):
    """Section 10.2's cycle score needs a mean and sd that credibility.yaml does not carry."""
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "cyclical",
                "is_public": False,
                "announcer_id": None,
                "commitment_form": "board_approved",
                "modifiers": [],
            }
        ],
    )
    projects["sector_capacity_util"] = [0.9]
    with pytest.raises(MissingParameter, match="sector_capacity_util_mean"):
        cred.completion_probability(projects, announcer_template, params)


def test_purity_input_frame_untouched(params, synthetic_city):
    before = synthetic_city.projects.copy(deep=True)
    out = cred.completion_probability(
        synthetic_city.projects,
        synthetic_city.announcers,
        params,
        unknown_modifiers="ignore",
    )
    pd.testing.assert_frame_equal(synthetic_city.projects, before)
    assert out is not synthetic_city.projects
    assert len(out) == len(before)
    assert list(out.index) == list(before.index)


# --------------------------------------------------------------------------------------
# 10.3 delay distribution
# --------------------------------------------------------------------------------------


def test_slip_cdf_unimodal_matches_lognormal(params):
    median = params.value("credibility.delay_lognormal.highway.median")
    sigma = params.value("credibility.delay_lognormal.highway.sigma")
    x = np.linspace(0.001, 20.0, 200)
    got = cred.slip_cdf(params, "highway", x)
    want = stats.lognorm.cdf(x, s=sigma, scale=median)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_slip_distribution_integrates_to_one(params):
    """The density integrates to 1 (unimodal) — i.e. the CDF runs 0 -> 1 monotonically."""
    for family in ("highway", "metro_phase1", "data_centre", "trunk_utilities"):
        median = params.value(f"credibility.delay_lognormal.{family}.median")
        sigma = params.value(f"credibility.delay_lognormal.{family}.sigma")
        pdf = lambda t: stats.lognorm.pdf(t, s=sigma, scale=median)  # noqa: E731
        mass, _ = integrate.quad(pdf, 0.0, np.inf, limit=400)
        assert mass == pytest.approx(1.0, abs=1e-6)

        assert cred.slip_cdf(params, family, np.array([0.0]))[0] == pytest.approx(0.0)
        assert cred.slip_cdf(params, family, np.array([1e6]))[0] == pytest.approx(1.0)
        grid = np.linspace(0.0, 50.0, 500)
        assert np.all(np.diff(cred.slip_cdf(params, family, grid)) >= 0)
        # Median of a lognormal is exp(mu) = median parameter.
        assert cred.slip_median(params, family) == pytest.approx(median)


def test_govt_mega_is_bimodal_and_integrates_to_one(params):
    """`govt_mega` is a point mass at fast_slip mixed with a lognormal slow branch."""
    fast_p = params.value("credibility.delay_lognormal.govt_mega.fast_p")
    fast_slip = params.value("credibility.delay_lognormal.govt_mega.fast_slip")
    slow_slip = params.value("credibility.delay_lognormal.govt_mega.slow_slip")
    sigma = params.value("credibility.delay_lognormal.govt_mega.sigma")

    eps = 1e-9
    jump = (
        cred.slip_cdf(params, "govt_mega", np.array([fast_slip]))[0]
        - cred.slip_cdf(params, "govt_mega", np.array([fast_slip - eps]))[0]
    )
    assert jump == pytest.approx(fast_p, abs=1e-6)

    # Total mass: point mass + continuous branch = 1.
    pdf = lambda t: (1 - fast_p) * stats.lognorm.pdf(t, s=sigma, scale=slow_slip)  # noqa: E731
    continuous, _ = integrate.quad(pdf, 0.0, np.inf, limit=400)
    assert fast_p + continuous == pytest.approx(1.0, abs=1e-6)
    assert cred.slip_cdf(params, "govt_mega", np.array([1e9]))[0] == pytest.approx(1.0)

    # Mixture median solves 0.35 + 0.65*Phi((ln x - ln 2.0)/0.8) = 0.5.
    q = (0.5 - fast_p) / (1 - fast_p)
    want = slow_slip * math.exp(sigma * stats.norm.ppf(q))
    assert cred.slip_median(params, "govt_mega") == pytest.approx(want)
    assert cred.slip_cdf(params, "govt_mega", np.array([want]))[0] == pytest.approx(0.5)


def test_monte_carlo_samples_match_the_cdf(params, project_template, announcer_template):
    projects = make_projects(
        project_template,
        [
            {
                "project_id": f"mc-{i}",
                "archetype": "highway",
                "announced_date": datetime(2020, 1, 1),
                "stated_completion": datetime(2026, 1, 1),
                "modifiers": [],
            }
            for i in range(5000)
        ],
    )
    rng = np.random.default_rng(7)
    out = cred.delay_distribution(projects, params, monte_carlo=True, rng=rng)
    slip = out["slip"].to_numpy()
    result = stats.kstest(
        slip,
        lambda x: cred.slip_cdf(params, "highway", np.asarray(x, dtype=float)),
    )
    assert result.pvalue > 0.01
    assert np.median(slip) == pytest.approx(cred.slip_median(params, "highway"), rel=0.05)


def test_delay_is_deterministic_for_a_given_seed(
    params, project_template, announcer_template
):
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "d1",
                "archetype": "govt_mega",
                "announced_date": datetime(2020, 1, 1),
                "stated_completion": datetime(2025, 1, 1),
                "modifiers": [],
            }
        ]
        * 50,
    )
    projects["project_id"] = [f"d{i}" for i in range(len(projects))]
    a = cred.delay_distribution(
        projects, params, monte_carlo=True, rng=np.random.default_rng(11)
    )
    b = cred.delay_distribution(
        projects, params, monte_carlo=True, rng=np.random.default_rng(11)
    )
    pd.testing.assert_frame_equal(a, b)
    c = cred.delay_distribution(
        projects, params, monte_carlo=True, rng=np.random.default_rng(12)
    )
    assert not np.allclose(a["slip"].to_numpy(), c["slip"].to_numpy())


def test_deterministic_open_year_hand_computed(params, project_template):
    """announced_duration 6.0 yr, highway median slip 0.35, no modifiers."""
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "hwy",
                "archetype": "highway",
                "announced_date": datetime(2020, 1, 1),
                "stated_completion": datetime(2026, 1, 1),
                "modifiers": [],
            }
        ],
    )
    out = cred.delay_distribution(projects, params, days_per_year=365.25)
    duration = (datetime(2026, 1, 1) - datetime(2020, 1, 1)).days / 365.25
    assert out.loc[0, "announced_duration_yr"] == pytest.approx(duration)
    assert out.loc[0, "slip"] == pytest.approx(0.35)
    assert out.loc[0, "actual_duration_yr"] == pytest.approx(duration * 1.35)
    assert out.loc[0, "open_year"] == pytest.approx(2020 + duration * 1.35)


def test_delay_modifiers_multiply_the_slip(params, project_template):
    mult = params.value("credibility.modifiers.state_budget_only.delay_mult")
    specs = []
    for pid, mods in (("plain", []), ("slow", ["state_budget_only"])):
        specs.append(
            {
                "project_id": pid,
                "archetype": "highway",
                "announced_date": datetime(2020, 1, 1),
                "stated_completion": datetime(2026, 1, 1),
                "modifiers": mods,
            }
        )
    out = cred.delay_distribution(
        make_projects(project_template, specs), params
    ).set_index("project_id")
    assert out.loc["slow", "slip"] == pytest.approx(out.loc["plain", "slip"] * mult)


def test_unmapped_archetype_reports_the_missing_delay_path(params, project_template):
    projects = make_projects(
        project_template,
        [{"project_id": "x", "archetype": "township", "modifiers": []}],
    )
    with pytest.raises(MissingParameter, match=r"credibility\.delay_lognormal\.township"):
        cred.delay_distribution(projects, params)


# --------------------------------------------------------------------------------------
# 10.4 activation weight
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def opening_project(params, project_template, announcer_template):
    """One public project that opens, with p_completion and open_year attached."""
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "opener",
                "archetype": "metro_rail",
                "is_public": True,
                "announcer_id": None,
                "commitment_form": None,
                "stage": "funded",
                "announced_date": datetime(2020, 1, 1),
                "stated_completion": datetime(2026, 1, 1),
                "modifiers": [],
                "physical_state": None,
                "physical_asof": pd.NaT,
            }
        ],
    )
    frame = cred.completion_probability(projects, announcer_template, params)
    frame = cred.delay_distribution(
        frame, params, family_map={"metro_rail": "metro_phase1"}, days_per_year=365.25
    )
    return frame


@pytest.mark.acceptance
def test_acceptance_activation_weight_is_monotone_in_t(params, opening_project):
    """Section 10 ACCEPTANCE 5: w(t) is monotonically non-decreasing in t."""
    years = list(range(2015, 2061))
    out = cred.activation_weight(opening_project, params, years)
    w = out.sort_values("year")["activation_weight"].to_numpy()
    assert np.all(np.diff(w) >= -1e-12)


@pytest.mark.acceptance
def test_acceptance_phase_curve_sums_to_one_gives_w_equal_p(params, opening_project):
    """Section 10 ACCEPTANCE 6: sum(phase_curve) == 1 implies w(t -> inf) == p exactly."""
    curve = params.get("archetypes.metro_rail.phase_curve")
    total = sum(leaf["value"] for leaf in curve.values())
    assert total == pytest.approx(1.0)

    p = float(opening_project.loc[0, "p_completion"])
    far = cred.activation_weight(opening_project, params, 2200)
    assert far.loc[0, "phase_weight"] == pytest.approx(1.0)
    assert far.loc[0, "discount"] == pytest.approx(1.0)
    assert far.loc[0, "activation_weight"] == p


def test_activation_weight_is_zero_before_announcement(params, opening_project):
    announced_year = int(opening_project.loc[0, "announced_date"].year)
    for year in (announced_year - 5, announced_year - 1):
        out = cred.activation_weight(opening_project, params, year)
        assert out.loc[0, "phase_weight"] == 0
        assert out.loc[0, "activation_weight"] == 0
    assert cred.activation_weight(opening_project, params, announced_year).loc[
        0, "activation_weight"
    ] > 0


def test_activation_weight_phases_hand_computed(params, opening_project):
    """Announcement / construction / operational plateaus and the discount factor."""
    curve = params.get("archetypes.metro_rail.phase_curve")
    f_ann = curve["announcement"]["value"]
    f_cs = curve["construction_start"]["value"]
    f_op = curve["operational"]["value"]
    ramp_years = params.value("archetypes.metro_rail.operational_ramp_years")
    r = params.value("credibility.discount_rate")

    p = float(opening_project.loc[0, "p_completion"])
    open_year = float(opening_project.loc[0, "open_year"])
    construction_years = params.value("archetypes._defaults.construction_years")
    construction_start = open_year - construction_years

    # Just after announcement, before construction start.
    t = 2020
    assert t < construction_start
    out = cred.activation_weight(opening_project, params, t)
    assert out.loc[0, "phase_weight"] == pytest.approx(f_ann)
    assert out.loc[0, "discount"] == pytest.approx((1 + r) ** (t - open_year))
    assert out.loc[0, "activation_weight"] == pytest.approx(
        p * f_ann * (1 + r) ** (t - open_year)
    )

    # Construction under way but not open.
    t = int(math.ceil(construction_start))
    out = cred.activation_weight(opening_project, params, t)
    assert out.loc[0, "phase_weight"] == pytest.approx(f_ann + f_cs)

    # Part-way up the operational ramp.
    t = int(math.floor(open_year)) + 1
    out = cred.activation_weight(opening_project, params, t)
    ramp = min(max((t - open_year) / ramp_years, 0.0), 1.0)
    assert out.loc[0, "phase_weight"] == pytest.approx(f_ann + f_cs + f_op * ramp)
    assert out.loc[0, "discount"] == pytest.approx(1.0)


def test_activation_weight_uses_the_yaml_default_phase_curve(
    params, project_template, announcer_template
):
    """An archetype with no phase_curve falls back to archetypes._defaults (Section 10.4)."""
    default = params.get("archetypes._defaults.phase_curve")
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "nocurve",
                "archetype": "township",
                "is_public": True,
                "announcer_id": None,
                "commitment_form": None,
                "stage": "funded",
                "announced_date": datetime(2020, 1, 1),
                "stated_completion": datetime(2026, 1, 1),
                "modifiers": [],
                "physical_state": None,
                "physical_asof": pd.NaT,
            }
        ],
    )
    frame = cred.completion_probability(projects, announcer_template, params)
    frame = cred.delay_distribution(
        frame, params, family_map={"township": "private_industrial"}
    )
    out = cred.activation_weight(frame, params, 2021)
    assert out.loc[0, "phase_weight"] == pytest.approx(default["announcement"]["value"])


def test_activation_weight_requires_its_inputs(params, synthetic_city):
    with pytest.raises(ValueError, match="p_completion"):
        cred.activation_weight(synthetic_city.projects, params, 2025)


# --------------------------------------------------------------------------------------
# 10.5 counterfactual mode
# --------------------------------------------------------------------------------------


@pytest.mark.acceptance
def test_acceptance_force_fails_produces_zero_effect_in_every_year(
    params, project_template, announcer_template
):
    """Section 10 ACCEPTANCE 7: force_project_state='fails' -> zero effect in every year."""
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "doomed",
                "archetype": "metro_rail",
                "is_public": True,
                "announcer_id": None,
                "commitment_form": None,
                "stage": "half_complete",
                "announced_date": datetime(2020, 1, 1),
                "stated_completion": datetime(2026, 1, 1),
                "modifiers": [],
                "physical_state": None,
                "physical_asof": pd.NaT,
            }
        ],
    )
    forced = {"doomed": "fails"}
    frame = cred.completion_probability(
        projects, announcer_template, params, force_project_state=forced
    )
    assert frame.loc[0, "p_completion"] == 0
    frame = cred.delay_distribution(
        frame,
        params,
        family_map={"metro_rail": "metro_phase1"},
        force_project_state=forced,
    )
    out = cred.activation_weight(
        frame, params, list(range(2015, 2081)), force_project_state=forced
    )
    assert (out["activation_weight"] == 0).all()


def test_force_happens_uses_stated_completion_and_p_one(
    params, project_template, announcer_template
):
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "blessed",
                "archetype": "metro_rail",
                "is_public": False,
                "announcer_id": None,
                "commitment_form": "verbal",
                "announced_date": datetime(2020, 1, 1),
                "stated_completion": datetime(2026, 1, 1),
                "modifiers": [],
                "physical_state": None,
                "physical_asof": pd.NaT,
            }
        ],
    )
    forced = {"blessed": "happens"}
    frame = cred.completion_probability(
        projects, announcer_template, params, force_project_state=forced
    )
    assert frame.loc[0, "p_completion"] == 1
    frame = cred.delay_distribution(
        frame,
        params,
        family_map={"metro_rail": "metro_phase1"},
        force_project_state=forced,
        days_per_year=365.25,
    )
    assert frame.loc[0, "slip"] == 0
    duration = (datetime(2026, 1, 1) - datetime(2020, 1, 1)).days / 365.25
    assert frame.loc[0, "open_year"] == pytest.approx(2020 + duration)
    out = cred.activation_weight(frame, params, 2200)
    assert out.loc[0, "activation_weight"] == pytest.approx(1.0)


def test_counterfactual_is_an_argument_not_a_global(params, project_template, announcer_template):
    """Scenario A and Scenario B must be obtainable from the same inputs in one process."""
    projects = make_projects(
        project_template,
        [
            {
                "project_id": "swing",
                "is_public": True,
                "announcer_id": None,
                "commitment_form": None,
                "stage": "funded",
                "modifiers": [],
                "physical_state": None,
                "physical_asof": pd.NaT,
            }
        ],
    )
    a = cred.completion_probability(
        projects, announcer_template, params, force_project_state={"swing": "happens"}
    )
    b = cred.completion_probability(
        projects, announcer_template, params, force_project_state={"swing": "fails"}
    )
    c = cred.completion_probability(projects, announcer_template, params)
    assert a.loc[0, "p_completion"] == 1
    assert b.loc[0, "p_completion"] == 0
    assert 0 < c.loc[0, "p_completion"] < 1


def test_force_project_state_rejects_unknown_state(
    params, project_template, announcer_template
):
    projects = make_projects(
        project_template,
        [{"project_id": "x", "is_public": True, "announcer_id": None,
          "commitment_form": None, "stage": "funded", "modifiers": []}],
    )
    with pytest.raises(ValueError, match="maybe"):
        cred.completion_probability(
            projects, announcer_template, params, force_project_state={"x": "maybe"}
        )


# --------------------------------------------------------------------------------------
# Section 21 failure mode — survivorship in the frozen pipeline
# --------------------------------------------------------------------------------------


def _pipeline_with_outcomes(projects: pd.DataFrame) -> pd.DataFrame:
    """Attach a Section 17 `outcome` column; every third project is abandoned."""
    frame = projects.copy()
    outcomes = []
    for i in range(len(frame)):
        outcomes.append("abandoned" if i % 3 == 0 else "under_construction")
    frame["outcome"] = outcomes
    return frame


def test_abandoned_projects_get_zero_probability_and_zero_weight(
    params, synthetic_city
):
    projects = _pipeline_with_outcomes(synthetic_city.projects)
    out = cred.completion_probability(
        projects, synthetic_city.announcers, params, unknown_modifiers="ignore"
    )
    dead = out["outcome"] == "abandoned"
    assert dead.any()
    assert (out.loc[dead, "p_completion"] == 0).all()
    assert (out.loc[~dead, "p_completion"] > 0).all()
    assert out.loc[dead, "credibility_flags"].map(lambda f: "dead_project" in f).all()

    frame = cred.delay_distribution(
        out,
        params,
        family_map=SYNTHETIC_FAMILY_MAP,
        monte_carlo=False,
        unknown_modifiers="ignore",
    )
    weights = cred.activation_weight(frame, params, list(range(2015, 2051)))
    dead_ids = set(out.loc[dead, "project_id"])
    assert (
        weights.loc[weights["project_id"].isin(dead_ids), "activation_weight"] == 0
    ).all()


def test_survivorship_a_pipeline_with_dead_projects_differs_materially(
    params, synthetic_city
):
    """Section 21: dropping dead projects must not leave the credibility layer's output alone.

    The frozen-pipeline guard lives in the backtest module; this is the layer-side proof
    that a survivor-only pipeline scores systematically differently, so a contaminated
    freeze would make the credibility layer look like it adds nothing.
    """
    full = _pipeline_with_outcomes(synthetic_city.projects)
    survivors = full.loc[full["outcome"] != "abandoned"].reset_index(drop=True)
    assert len(survivors) < len(full)

    kw = dict(unknown_modifiers="ignore")
    p_full = cred.completion_probability(full, synthetic_city.announcers, params, **kw)
    p_surv = cred.completion_probability(
        survivors, synthetic_city.announcers, params, **kw
    )

    # 1. The pipeline-level expected completion rate is materially higher when the dead
    #    projects are silently dropped.
    mean_full = p_full["p_completion"].mean()
    mean_surv = p_surv["p_completion"].mean()
    assert mean_surv > mean_full
    assert (mean_surv - mean_full) / mean_surv > 0.10

    # 2. The credibility discount — total activation weight against the naive p=1 pipeline
    #    of baseline B3 — is materially deeper on the honest pipeline.
    def discount(frame: pd.DataFrame) -> float:
        frame = cred.delay_distribution(
            frame,
            params,
            family_map=SYNTHETIC_FAMILY_MAP,
            monte_carlo=False,
            unknown_modifiers="ignore",
        )
        w = cred.activation_weight(frame, params, list(range(2015, 2051)))
        naive = frame.assign(p_completion=1.0)
        w_naive = cred.activation_weight(naive, params, list(range(2015, 2051)))
        return w["activation_weight"].sum() / w_naive["activation_weight"].sum()

    d_full = discount(p_full)
    d_surv = discount(p_surv)
    assert d_full < d_surv
    assert (d_surv - d_full) / d_surv > 0.10


def test_assert_pipeline_contains_dead_projects(synthetic_city):
    clean = _pipeline_with_outcomes(synthetic_city.projects)
    cred.assert_pipeline_contains_dead_projects(clean)  # does not raise

    contaminated = clean.loc[clean["outcome"] != "abandoned"]
    with pytest.raises(Exception, match="contaminated"):
        cred.assert_pipeline_contains_dead_projects(contaminated)


# --------------------------------------------------------------------------------------
# housekeeping
# --------------------------------------------------------------------------------------


def test_no_unseeded_randomness_in_monte_carlo_mode(params, project_template):
    projects = make_projects(
        project_template,
        [{"project_id": "x", "archetype": "highway", "modifiers": []}],
    )
    with pytest.raises(ValueError, match="rng"):
        cred.delay_distribution(projects, params, monte_carlo=True)


def test_module_does_not_import_ufe_ai():
    import ufe.layers.l3_credibility as module

    source = module.__file__
    with open(source, "r", encoding="utf-8") as fh:
        text = fh.read()
    assert "ufe.ai" not in text
