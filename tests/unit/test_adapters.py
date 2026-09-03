"""The state adapter pattern (spec Section 6.0) and the Section 23 item 11 guarantee.

The headline test here is
:func:`test_a_second_state_needs_no_code_outside_the_adapters_package`. Section 23 item 11
requires that onboarding a second city needs "no code changes outside
``ufe/ingest/adapters/``", so the test *literally* writes a fictional state adapter as a new
module inside that package at run time — touching nothing else — and then drives the whole
state-tier pipeline through it. If any dispatch table, import list or ``if state == ...``
existed anywhere outside the package, that test would fail.

The module is written to the real package directory (and removed in teardown) rather than to
a temporary package, because a temporary package would not demonstrate the claim: the point
is that dropping a file into ``ufe/ingest/adapters/`` is sufficient.
"""

from __future__ import annotations

import importlib
import textwrap
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from ufe.errors import DataRightsViolation
from ufe.ingest import cadastral, prices, rera, runner
from ufe.ingest.adapters import ap, base
from ufe.ingest.core import CityConfig, MissingSource
from ufe.params import load_params
from tests.fixtures import raster_fixtures as rf

#: The fictional state the second-city test onboards. Kerala/Telangana/Karnataka are named
#: in Section 6.0 as real future adapters, so a clearly fake code is used instead.
FICTIONAL_STATE_CODE = "ZZ"
_ADAPTERS_DIR = Path(base.__file__).parent
_FICTIONAL_MODULE = _ADAPTERS_DIR / "zz_fictional_state.py"
#: ``discover`` skips ``_``-prefixed modules (they are private helpers), so the fictional
#: adapter must be a normal module name to prove the claim.

#: The whole of a new state's onboarding: one file, inside the adapters package.
_FICTIONAL_SOURCE = textwrap.dedent(
    '''
    """A fictional state adapter, written by tests/unit/test_adapters.py.

    Exists to prove the Section 23 item 11 claim: a new state is one module in this
    package and nothing else.
    """

    from __future__ import annotations

    import geopandas as gpd
    import pandas as pd

    from ufe.ingest.adapters.base import (
        CAP_GUIDANCE_VALUES,
        CAP_RERA_PROJECTS,
        AccessTerms,
        StateAdapterBase,
        register,
    )


    @register
    class FictionalStateAdapter(StateAdapterBase):
        state_code = "ZZ"
        state_name = "Zzland"
        # Deliberately narrower than AP: no cadastral parcels, no industrial allotments.
        provides = frozenset({CAP_GUIDANCE_VALUES, CAP_RERA_PROJECTS})

        def __init__(self, reader=None):
            self.reader = reader

        def guidance_values(self, city):
            frame = self.reader.vector("zz/guidance_values")
            out = gpd.GeoDataFrame(frame.copy(), geometry=frame.geometry.name, crs=frame.crs)
            if "guidance_inr_sqft" not in out.columns:
                # Zzland publishes per square foot already: no conversion needed.
                out["guidance_inr_sqft"] = out["guidance_inr_sqyd"].astype(float)
            return out

        def rera_projects(self, city):
            return self.reader.table("zz/rera_projects")

        def access_terms(self) -> dict:
            return AccessTerms(
                state_code=self.state_code,
                portals={"registration": "https://registration.zz.example/"},
                bulk_access_allowed=True,
                licence="Per-Portal-Terms",
                notes="Fictional state used by the test suite.",
            ).as_dict()
    '''
).lstrip()


@pytest.fixture
def fictional_state():
    """Write the fictional adapter into the package, discover it, then remove it."""
    _FICTIONAL_MODULE.write_text(_FICTIONAL_SOURCE)
    try:
        base.discover(force=True)
        yield FICTIONAL_STATE_CODE
    finally:
        _FICTIONAL_MODULE.unlink(missing_ok=True)
        for name in list(base._REGISTRY):
            if name == FICTIONAL_STATE_CODE:
                base._REGISTRY.pop(name)
        module = f"{base.__package__}.zz_fictional_state"
        if module in importlib.sys.modules:
            del importlib.sys.modules[module]
        for cached in _ADAPTERS_DIR.glob("__pycache__/zz_fictional_state*"):
            cached.unlink(missing_ok=True)
        base.discover(force=True)


@pytest.fixture(scope="module")
def params():
    return load_params("vizag")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return rf.build_reader(tmp_path_factory.mktemp("adapter_fixtures"))


@pytest.fixture(scope="module")
def reader(bundle):
    return bundle[0]


@pytest.fixture(scope="module")
def cells(bundle):
    return bundle[1]


@pytest.fixture(scope="module")
def city(params):
    return CityConfig.from_params(params)


# --------------------------------------------------------------------------------------
# The protocol itself
# --------------------------------------------------------------------------------------


def test_protocol_declares_every_section_6_0_method():
    """The seven members of Section 6.0's ``StateAdapter``, no more and no less."""
    expected = {
        "guidance_values",
        "registration_transactions",
        "rera_projects",
        "cadastral_parcels",
        "industrial_allotments",
        "capabilities",
        "access_terms",
    }
    assert set(base.ADAPTER_METHODS) == expected


def test_capability_vocabulary_matches_the_data_returning_methods():
    assert set(base.CAPABILITIES) == set(base.ADAPTER_METHODS) - {
        "capabilities",
        "access_terms",
    }


def test_ap_adapter_satisfies_the_protocol(reader):
    adapter = ap.AndhraPradeshAdapter(reader=reader)
    assert base.satisfies_protocol(adapter)
    assert isinstance(adapter, base.StateAdapter)
    base.assert_adapter(adapter)  # raises on any deviation


def test_a_non_conforming_adapter_is_rejected():
    class Broken:
        state_code = "XX"

        def capabilities(self):
            return set()

    with pytest.raises(base.AdapterError):
        base.assert_adapter(Broken())


def test_an_adapter_claiming_an_unknown_capability_is_rejected():
    with pytest.raises(base.AdapterError):

        class Bogus(base.StateAdapterBase):
            state_code = "XY"
            provides = frozenset({"teleportation"})


def test_unknown_state_code_raises(reader):
    with pytest.raises(base.UnknownStateAdapter) as excinfo:
        base.get_adapter("QQ", reader=reader)
    assert "ufe/ingest/adapters" in str(excinfo.value)


def test_adapter_is_selected_by_the_city_config(params, reader):
    """Section 6.0: "State data is reached through an adapter, selected by city.state_code"."""
    city = CityConfig.from_params(params)
    assert city.state_code == "AP"
    adapter = base.get_adapter(city.state_code, reader=reader)
    assert adapter.state_code == city.state_code


# --------------------------------------------------------------------------------------
# Honesty: None, capabilities() and data_conf
# --------------------------------------------------------------------------------------


def test_ap_returns_none_for_a_dataset_the_state_does_not_publish(reader, city):
    """Section 6.0: "Any method may return None where the state does not publish it"."""
    adapter = ap.AndhraPradeshAdapter(reader=reader)
    assert adapter.registration_transactions(city) is None
    assert base.CAP_REGISTRATION_TRANSACTIONS not in adapter.capabilities()


def test_capabilities_and_provides_cannot_disagree(reader):
    adapter = ap.AndhraPradeshAdapter(reader=reader)
    assert adapter.capabilities() == set(ap.AndhraPradeshAdapter.provides)


def test_missing_capability_lowers_data_conf_and_never_imputes(reader, cells):
    """Section 6.0: "A missing capability lowers data_conf; it never silently imputes"."""
    from ufe.ingest import core

    adapter = ap.AndhraPradeshAdapter(reader=reader)
    absent = base.missing_capabilities(adapter)
    assert absent == {base.CAP_REGISTRATION_TRANSACTIONS}
    empty = pd.DataFrame(columns=list(core.CELL_IMPUTATION_COLUMNS))
    baseline = core.data_conf(empty, cells)
    penalised = core.data_conf(empty, cells, missing_capabilities=absent)
    penalty = float(core.cfg("data_conf.missing_capability_penalty"))
    assert (baseline - penalised).round(9).eq(round(penalty, 9)).all()


def test_capabilities_go_into_the_manifest_shape(reader):
    """"``capabilities()`` is written into the manifest so a report can state honestly
    which layers were unavailable"."""
    adapter = ap.AndhraPradeshAdapter(reader=reader)
    manifest = {
        "state_code": adapter.state_code,
        "capabilities": sorted(adapter.capabilities()),
        "unavailable": sorted(base.missing_capabilities(adapter)),
        "access_terms": adapter.access_terms(),
    }
    assert manifest["unavailable"] == [base.CAP_REGISTRATION_TRANSACTIONS]
    assert manifest["access_terms"]["licence"]


# --------------------------------------------------------------------------------------
# access_terms() is not decorative (Sections 6.0, 22.2)
# --------------------------------------------------------------------------------------


def test_access_terms_declares_portals_terms_and_a_rate_limit(reader):
    terms = ap.AndhraPradeshAdapter(reader=reader).access_terms()
    assert terms["portals"], "an adapter must name the portals it reads"
    assert terms["licence"] == "Per-Portal-Terms"
    assert terms["rate_limit_s"] > 0
    assert terms["bulk_access_allowed"] is False


def test_default_rate_limit_comes_from_yaml():
    from ufe.ingest.core import cfg

    terms = base.AccessTerms(state_code="AP")
    assert terms.rate_limit_s() == float(
        cfg("reader.default_min_seconds_between_requests")
    )


def test_bulk_access_is_refused_when_the_terms_forbid_it(reader):
    terms = base.AccessTerms(**{"state_code": "AP", "bulk_access_allowed": False})
    with pytest.raises(DataRightsViolation, match="bulk"):
        terms.assert_bulk_access("the registration portal")


def test_bulk_access_is_allowed_when_the_terms_permit_it():
    base.AccessTerms(state_code="ZZ", bulk_access_allowed=True).assert_bulk_access("a portal")


def test_ap_licence_string_is_in_the_licence_registry(reader):
    """Section 22.2: every source's licence must resolve in the registry."""
    import yaml

    registry = yaml.safe_load(
        (Path(base.__file__).parents[3] / "config" / "data_sources_licences.yaml").read_text()
    )
    known = {str(spec["licence"]) for spec in registry["sources"].values()}
    assert ap.AndhraPradeshAdapter(reader=reader).access_terms()["licence"] in known


# --------------------------------------------------------------------------------------
# AP specifics
# --------------------------------------------------------------------------------------


def test_ap_normalises_guidance_values_to_inr_per_sqft(reader, city):
    """A state's publishing convention must not leak into a national-tier ingester."""
    from ufe.ingest.core import cfg

    guidance = ap.AndhraPradeshAdapter(reader=reader).guidance_values(city)
    per_sqyd = [float(v) for v in rf.FIXTURE_CONFIG["guidance"]["inr_per_sqyd"]]
    factor = float(cfg("prices.sqft_per_sqyd"))
    assert sorted(guidance["guidance_inr_sqft"]) == pytest.approx(
        sorted(v / factor for v in per_sqyd)
    )
    for column in ap.GUIDANCE_COLUMNS:
        assert column in guidance.columns


def test_ap_rera_extract_is_normalised(reader, city):
    frame = ap.AndhraPradeshAdapter(reader=reader).rera_projects(city)
    for column in ap.RERA_COLUMNS:
        assert column in frame.columns


def test_ap_without_a_reader_raises_rather_than_returning_empty(city):
    with pytest.raises(MissingSource):
        ap.AndhraPradeshAdapter().guidance_values(city)


def test_ap_optional_layers_return_none_when_absent(reader, city, tmp_path):
    stripped, _, _ = rf.build_reader(tmp_path)
    stripped.drop(ap.KEY_PARCELS, ap.KEY_ALLOTMENTS)
    adapter = ap.AndhraPradeshAdapter(reader=stripped)
    assert adapter.cadastral_parcels(city) is None
    assert adapter.industrial_allotments(city) is None


# --------------------------------------------------------------------------------------
# Section 23 item 11 — the whole point of the pattern
# --------------------------------------------------------------------------------------


def test_discovery_finds_every_module_in_the_package():
    table = base.registry()
    assert "AP" in table
    assert table["AP"] is ap.AndhraPradeshAdapter


def test_a_second_state_needs_no_code_outside_the_adapters_package(
    fictional_state, reader, cells, params, tmp_path
):
    """Section 23 item 11, tested literally.

    A brand-new state is onboarded by writing ONE file into ``ufe/ingest/adapters/``. This
    test then runs the entire state tier — RERA and cadastral — plus the price ingester's
    guidance-value leg through it, without importing it, naming it in a registry or
    touching any other file.
    """
    assert fictional_state in base.available_state_codes()

    # The new state's extracts are handed to the same injectable reader interface.
    reader.add_vector("zz/guidance_values", rf.synthetic_guidance_values(cells))
    reader.add_table("zz/rera_projects", rf.synthetic_rera_projects(cells))

    zz_city = CityConfig(
        city_id="zzville",
        state_code=fictional_state,
        crs_metric=params.city_config["crs_metric"],
        base_year=int(params.city_config["base_year"]),
        coastal=False,
        cbd_lat=float(params.city_config["cbd_point"]["lat"]),
        cbd_lon=float(params.city_config["cbd_point"]["lon"]),
    )
    adapter = base.get_adapter(zz_city.state_code, reader=reader)
    assert adapter.state_code == fictional_state

    # 1. The state tier runs unchanged through the new adapter.
    ingesters = runner.ingesters_for_tier(
        "state", reader=reader, city=zz_city, params=params, adapter=adapter
    )
    result = runner.run_ingesters(
        ingesters, cells=cells, city=zz_city, params=params, adapter=adapter
    )
    assert not result.failures, result.failures
    assert {"parcel_count", "mean_parcel_sqm"} <= result.columns

    # 2. Zzland publishes no parcels, so the cadastral ingester falls back and flags it —
    #    the honest degradation Section 6.9 asks for, with no code change.
    cad = [i for i in ingesters if isinstance(i, cadastral.CadastralIngester)][0]
    assert cad.used_fallback

    # 3. The price ingester's registration leg works through the new adapter too.
    price_ingester = prices.PricesIngester(
        reader, adapter=adapter, city=zz_city, params=params
    )
    priced = price_ingester.to_cells(
        price_ingester.parse(price_ingester.fetch(zz_city)), cells
    )
    assert priced["price_res_inr_sqft"].notna().any()

    # 4. And its narrower capability set flows straight into data_conf and the manifest.
    assert base.missing_capabilities(adapter) == {
        base.CAP_REGISTRATION_TRANSACTIONS,
        base.CAP_CADASTRAL_PARCELS,
        base.CAP_INDUSTRIAL_ALLOTMENTS,
    }


def test_no_ingester_branches_on_a_state_code():
    """The negative form of the same claim: no state code appears outside the package."""
    ingest_dir = Path(prices.__file__).parent
    offenders: list[str] = []
    for path in ingest_dir.rglob("*.py"):
        if "adapters" in path.parts:
            continue
        text = path.read_text()
        for token in ('"AP"', "'AP'", '"KL"', '"TG"', '"KA"', FICTIONAL_STATE_CODE):
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, offenders


def test_state_tier_ingesters_only_use_protocol_methods():
    """A state-tier ingester may call only the seven Section 6.0 methods on its adapter."""
    import re

    allowed = set(base.ADAPTER_METHODS)
    for module in (prices, rera, cadastral):
        calls = set(re.findall(r"self\.adapter\.(\w+)", Path(module.__file__).read_text()))
        assert calls <= allowed, f"{module.__name__} calls {calls - allowed}"


def test_registering_without_a_state_code_is_an_error():
    class Nameless:
        pass

    with pytest.raises(base.AdapterError):
        base.register(Nameless)


def test_fictional_adapter_is_gone_after_teardown():
    """The fixture must leave the package exactly as it found it."""
    assert not _FICTIONAL_MODULE.exists()
    assert FICTIONAL_STATE_CODE not in base.available_state_codes()
