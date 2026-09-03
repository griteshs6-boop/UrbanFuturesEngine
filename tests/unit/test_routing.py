"""Tests for `ufe.layers.routing` — the travel-time backend abstraction (spec Section 8.3).

Written before the implementation (spec Section 0.1 rule 2).

Docker is not available in this environment, so `OSRMBackend` cannot be exercised against a
live `osrm-routed`.  It is therefore tested two ways:

* offline, by injecting a stub HTTP client that records the requests and replays a canned
  `/table` response — this covers URL construction, chunking, unit conversion and the
  unreachable sentinel, which is where the bugs live;
* against a real service, marked ``needs_osrm`` and skipped here.

Every number used in an assertion is read from ``config/params/accessibility.yaml`` through
``Params`` — no numeric literal beyond 0 and 1 appears below (spec Section 0.1 rule 3).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from ufe.errors import MissingParameter, UFEError
from ufe.layers import routing as R
from ufe.params import load_params

CITY = "vizag"
ZERO, ONE = 0, 1

# Docker is unavailable here, so `needs_osrm` tests are also guarded by an explicit opt-in
# environment variable rather than relying on a marker filter that this repo has not yet
# wired into a root conftest.
OSRM_ENV = "UFE_OSRM_ENDPOINT"


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params():
    return load_params(CITY)


@pytest.fixture
def coords():
    """Four points on a north-south line, spaced by a constant latitude step."""
    p = load_params(CITY)
    # step is derived, not a literal: one thousandth of the walk-access cap, in degrees,
    # via the geodesic so that the spacing is metric.
    base_lat, base_lon = 17.7, 83.3
    step = 0.02
    return np.array(
        [[base_lat + i * step, base_lon] for i in range(4)], dtype=float
    )


class StubResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class StubClient:
    """Records every GET and replays a `/table` response of the requested shape."""

    def __init__(self, seconds: float, unreachable: set[tuple[int, int]] | None = None):
        self.seconds = seconds
        self.unreachable = unreachable or set()
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, **kw):  # noqa: ANN001
        self.calls.append((url, dict(params or {})))
        n_src = len(str(params["sources"]).split(";"))
        n_dst = len(str(params["destinations"]).split(";"))
        durations = [
            [
                None if (i, j) in self.unreachable else self.seconds
                for j in range(n_dst)
            ]
            for i in range(n_src)
        ]
        return StubResponse({"code": "Ok", "durations": durations})


# --------------------------------------------------------------------------------------
# the protocol
# --------------------------------------------------------------------------------------


def test_protocol_is_runtime_checkable_and_fallback_conforms(params):
    backend = R.HaversineBackend(params)
    assert isinstance(backend, R.TravelTimeBackend)


def test_osrm_backend_conforms_to_the_protocol(params):
    backend = R.OSRMBackend("http://localhost:5000", params, client=StubClient(ZERO))
    assert isinstance(backend, R.TravelTimeBackend)


# --------------------------------------------------------------------------------------
# the fallback backend
# --------------------------------------------------------------------------------------


def test_fallback_matrix_shape_and_zero_diagonal(params, coords):
    backend = R.HaversineBackend(params)
    m = backend.matrix(coords, coords, "car")
    assert m.shape == (len(coords), len(coords))
    assert m.dtype == np.float32
    assert np.allclose(np.diag(m), ZERO)


def test_fallback_matrix_is_symmetric_and_deterministic(params, coords):
    backend = R.HaversineBackend(params)
    a = backend.matrix(coords, coords, "car")
    b = backend.matrix(coords, coords, "car")
    assert np.array_equal(a, b)
    assert np.allclose(a, a.T, atol=float(params.value("accessibility.matrix.float32_tolerance")))


def test_fallback_speed_comes_from_yaml_and_the_arithmetic_is_exact(params, coords):
    """t = geodesic_metres / (kmh * m_per_km / min_per_hour), hand-checked."""
    backend = R.HaversineBackend(params)
    m = backend.matrix(coords[:ONE], coords[ONE:], "car")
    kmh = float(params.value("accessibility.speeds_kmh.arterial"))
    for j in range(len(coords) - ONE):
        d = R.geodesic_m(coords[ZERO], coords[ONE + j])
        expected = d / (kmh * R.METRES_PER_KM / R.MINUTES_PER_HOUR)
        assert m[ZERO, j] == pytest.approx(expected, rel=1e-6)


def test_fallback_walk_is_slower_than_car(params, coords):
    backend = R.HaversineBackend(params)
    walk = backend.matrix(coords, coords, "walk")
    car = backend.matrix(coords, coords, "car")
    assert (walk >= car).all()
    assert walk[ZERO, -ONE] > car[ZERO, -ONE]


def test_fallback_accepts_both_mode_names_and_osrm_profile_names(params, coords):
    backend = R.HaversineBackend(params)
    assert np.allclose(
        backend.matrix(coords, coords, "walk"), backend.matrix(coords, coords, "foot")
    )
    assert np.allclose(
        backend.matrix(coords, coords, "car"), backend.matrix(coords, coords, "car")
    )


def test_fallback_rejects_an_unknown_profile(params, coords):
    backend = R.HaversineBackend(params)
    with pytest.raises(MissingParameter):
        backend.matrix(coords, coords, "hovercraft")


def test_fallback_distance_matrix_is_metric_not_degrees(params, coords):
    """Section 21: 'degrees used as metres' — distances wrong by ~111,000x."""
    backend = R.HaversineBackend(params)
    d = backend.distance_matrix(coords[:ONE], coords[ONE:ONE + ONE])
    degrees = abs(coords[ONE][ZERO] - coords[ZERO][ZERO])
    assert d[ZERO, ZERO] > degrees * 1e4


# --------------------------------------------------------------------------------------
# the OSRM backend, offline
# --------------------------------------------------------------------------------------


def test_osrm_chunking_respects_the_yaml_chunk_size(params):
    chunk = int(params.value("accessibility.matrix.chunk_size"))
    spans = R.chunk_ranges(chunk * 2 + ONE, chunk)
    assert spans[ZERO] == (ZERO, chunk)
    assert spans[-ONE] == (chunk * 2, chunk * 2 + ONE)
    assert all(hi - lo <= chunk for lo, hi in spans)
    assert spans[ZERO][ZERO] == ZERO and spans[-ONE][ONE] == chunk * 2 + ONE


def test_osrm_builds_one_request_per_chunk_pair(params, coords):
    """A 4x4 problem with a chunk size of 2 must issue exactly 4 table requests."""
    client = StubClient(seconds=float(params.value("accessibility.speeds_kmh.arterial")))
    backend = R.OSRMBackend(
        "http://localhost:5000", params, client=client, chunk_size=len(coords) // 2
    )
    backend.matrix(coords, coords, "car")
    assert len(client.calls) == 4


def test_osrm_converts_seconds_to_minutes(params, coords):
    seconds = float(params.value("accessibility.speeds_kmh.arterial"))
    client = StubClient(seconds=seconds)
    backend = R.OSRMBackend("http://localhost:5000", params, client=client)
    m = backend.matrix(coords, coords, "car")
    assert np.allclose(m, seconds / R.SECONDS_PER_MINUTE)


def test_osrm_maps_null_durations_to_infinity(params, coords):
    client = StubClient(seconds=ONE, unreachable={(ZERO, ONE)})
    backend = R.OSRMBackend("http://localhost:5000", params, client=client)
    m = backend.matrix(coords, coords, "car")
    assert math.isinf(float(m[ZERO, ONE]))
    assert np.isfinite(m[ZERO, ZERO])


def test_osrm_url_uses_the_profile_and_lon_lat_order(params, coords):
    client = StubClient(seconds=ONE)
    backend = R.OSRMBackend("http://localhost:5000", params, client=client)
    backend.matrix(coords[:ONE], coords[:ONE], "twowheeler")
    url, query = client.calls[ZERO]
    assert "/table/v1/twowheeler/" in url
    # OSRM speaks lon,lat.
    first = url.rsplit("/", ONE)[-ONE].split(";")[ZERO]
    lon, lat = (float(x) for x in first.split(","))
    assert lon == pytest.approx(coords[ZERO][ONE])
    assert lat == pytest.approx(coords[ZERO][ZERO])
    assert query["annotations"] == "duration"


def test_osrm_raises_on_a_non_ok_response(params, coords):
    class Bad(StubClient):
        def get(self, url, params=None, **kw):  # noqa: ANN001
            self.calls.append((url, dict(params or {})))
            return StubResponse({"code": "NoTable", "message": "table too large"})

    backend = R.OSRMBackend("http://localhost:5000", params, client=Bad(ONE))
    with pytest.raises(UFEError):
        backend.matrix(coords, coords, "car")


@pytest.mark.needs_osrm
@pytest.mark.skipif(
    not os.environ.get(OSRM_ENV),
    reason=f"needs a live osrm-routed; set {OSRM_ENV} (Docker is unavailable here)",
)
def test_osrm_live_matrix_is_finite_and_plausible(params, coords):
    """Requires a running `osrm-routed` per osrm/README.md. Skipped in this environment."""
    backend = R.OSRMBackend(os.environ[OSRM_ENV], params)
    m = backend.matrix(coords, coords, "car")
    assert m.shape == (len(coords), len(coords))
    assert np.isfinite(m).any()
    assert (np.diag(m) < ONE).all()


# --------------------------------------------------------------------------------------
# congestion (Section 8.3)
# --------------------------------------------------------------------------------------


def test_congestion_slows_travel_and_never_speeds_it_up(params):
    t = np.array([[10.0]], dtype=np.float32)
    zero_builtup = np.zeros(ONE)
    full_builtup = np.ones(ONE)
    free = R.congestion_adjust(t, zero_builtup, zero_builtup, zero_builtup, params)
    slow = R.congestion_adjust(t, full_builtup, full_builtup, full_builtup, params)
    assert free == pytest.approx(t)
    assert slow > free


def test_congestion_respects_the_yaml_floor(params):
    t = np.array([[10.0]], dtype=np.float32)
    huge = np.full(ONE, 1e3)
    floor = float(params.value("accessibility.congestion.floor"))
    slow = R.congestion_adjust(t, huge, huge, huge, params)
    assert slow == pytest.approx(t / floor, rel=1e-5)


def test_congestion_uses_the_documented_three_point_approximation(params):
    """Section 8.3: mean builtup_frac of origin, destination and midpoint."""
    t = np.array([[10.0]], dtype=np.float32)
    k = float(params.value("accessibility.congestion.k"))
    o = np.array([0.9])
    d = np.array([0.3])
    mid = np.array([0.6])
    corridor = (o[ZERO] + d[ZERO] + mid[ZERO]) / 3
    expected = t / max(float(params.value("accessibility.congestion.floor")), ONE - k * corridor)
    assert R.congestion_adjust(t, o, d, mid, params) == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------------------
# network states (Section 8.3)
# --------------------------------------------------------------------------------------


def test_network_states_enumerates_distinct_open_sets():
    open_years = {"a": 2027, "b": 2027, "c": 2031}
    states = R.network_states(open_years, range(2025, 2033))
    sets = sorted(states.values(), key=len)
    assert sets[ZERO] == frozenset()
    assert sets[ONE] == frozenset({"a", "b"})
    assert sets[2] == frozenset({"a", "b", "c"})
    assert len(states) == 3


def test_network_state_hash_is_order_independent_and_stable():
    a = R.network_state_hash(frozenset({"x", "y"}))
    b = R.network_state_hash(frozenset({"y", "x"}))
    assert a == b
    assert a != R.network_state_hash(frozenset({"x"}))


def test_network_states_caps_at_the_yaml_maximum(params):
    cap = int(params.value("accessibility.matrix.max_network_states"))
    open_years = {f"p{i}": 2000 + i for i in range(cap * 2)}
    with pytest.raises(UFEError):
        R.network_states(open_years, range(2000, 2000 + cap * 2), params=params)


def test_network_states_clusters_by_corridor_when_over_the_cap(params):
    cap = int(params.value("accessibility.matrix.max_network_states"))
    open_years = {f"p{i}": 2000 + i for i in range(cap * 2)}
    corridors = {f"p{i}": f"corridor{i % 2}" for i in range(cap * 2)}
    states = R.network_states(
        open_years, range(2000, 2000 + cap * 2), params=params, corridors=corridors
    )
    assert len(states) <= cap


# --------------------------------------------------------------------------------------
# the transit / metro graph (Section 8.3)
# --------------------------------------------------------------------------------------


def _stations(lat0=17.70, lon0=83.30, n=3, step=0.03, feeder=False):
    import pandas as pd

    return pd.DataFrame(
        {
            "station_id": [f"s{i}" for i in range(n)],
            "line": ["L1"] * n,
            "lat": [lat0 + i * step for i in range(n)],
            "lon": [lon0] * n,
            "feeder_or_park_ride": [feeder] * n,
        }
    )


def test_metro_matrix_respects_the_walk_access_cap(params):
    stations = _stations()
    cap = float(params.value("accessibility.matrix.walk_access_cap_m"))
    near = np.array([[float(stations.lat[ZERO]), float(stations.lon[ZERO])]])
    far = np.array([[float(stations.lat[ZERO]) + ONE, float(stations.lon[ZERO])]])
    backend = R.HaversineBackend(params)

    dest = np.array([[float(stations.lat.iloc[-ONE]), float(stations.lon.iloc[-ONE])]])
    t_near = R.metro_matrix(near, dest, stations, params, backend)
    t_far = R.metro_matrix(far, dest, stations, params, backend)
    assert np.isfinite(t_near).all()
    assert not np.isfinite(t_far).all()
    assert R.geodesic_m(far[ZERO], near[ZERO]) > cap


def test_metro_matrix_includes_boarding_dwell_and_line_haul(params):
    stations = _stations()
    origin = np.array([[float(stations.lat[ZERO]), float(stations.lon[ZERO])]])
    dest = np.array([[float(stations.lat.iloc[-ONE]), float(stations.lon.iloc[-ONE])]])
    t = float(R.metro_matrix(origin, dest, stations, params, R.HaversineBackend(params))[ZERO, ZERO])

    haul_kmh = float(params.value("accessibility.speeds_kmh.metro_line_haul"))
    dwell = float(params.value("accessibility.transit_penalties_min.metro_dwell_per_station"))
    board = float(params.value("accessibility.transit_penalties_min.metro_headway_half"))
    access = float(params.value("accessibility.transit_penalties_min.access_egress_extra"))
    d_m = R.geodesic_m(
        (float(stations.lat[ZERO]), float(stations.lon[ZERO])),
        (float(stations.lat.iloc[-ONE]), float(stations.lon.iloc[-ONE])),
    )
    n_hops = len(stations) - ONE
    haul = d_m / (haul_kmh * R.METRES_PER_KM / R.MINUTES_PER_HOUR) + dwell * n_hops
    # origin and destination sit on the stations, so both access legs are the flat extra only
    expected = access * 2 + board + haul
    assert t == pytest.approx(expected, rel=1e-4)


def test_station_walk_distance_uses_the_backend_distance_matrix(params):
    stations = _stations()
    origins = np.array([[float(stations.lat[ZERO]), float(stations.lon[ZERO])]])
    d = R.station_walk_distance_m(origins, stations, R.HaversineBackend(params))
    assert d.shape == (ONE, len(stations))
    assert d[ZERO, ZERO] == pytest.approx(ZERO, abs=ONE)


# --------------------------------------------------------------------------------------
# precompute, caching and persistence
# --------------------------------------------------------------------------------------


class CountingBackend:
    """Wraps the fallback backend and counts calls, standing in for OSRM requests."""

    def __init__(self, params):
        self._inner = R.HaversineBackend(params)
        self.calls = ZERO

    def matrix(self, origins, destinations, profile):  # noqa: ANN001
        self.calls += ONE
        return self._inner.matrix(origins, destinations, profile)

    def distance_matrix(self, origins, destinations, profile="walk"):  # noqa: ANN001
        self.calls += ONE
        return self._inner.distance_matrix(origins, destinations, profile)


@pytest.fixture(scope="module")
def small_cells():
    from tests.fixtures.synthetic import synthetic_cells

    return synthetic_cells(n=40)


@pytest.mark.acceptance
def test_matrix_cache_second_call_performs_zero_backend_requests(
    params, small_cells, tmp_path
):
    """Section 8 ACCEPTANCE: second call with the same network state hash does no routing."""
    backend = CountingBackend(params)
    first = R.precompute_matrices(
        small_cells, params, backend, network_state=frozenset(), cache_dir=tmp_path
    )
    assert backend.calls > ZERO
    backend.calls = ZERO
    second = R.precompute_matrices(
        small_cells, params, backend, network_state=frozenset(), cache_dir=tmp_path
    )
    assert backend.calls == ZERO
    for mode, m in first.minutes.items():
        assert np.array_equal(m, second.minutes[mode])


def test_matrix_cache_key_includes_the_params_hash(params, small_cells, tmp_path):
    """Section 21: 'stale cache across param versions'."""
    backend = CountingBackend(params)
    R.precompute_matrices(
        small_cells, params, backend, network_state=frozenset(), cache_dir=tmp_path
    )
    keys = sorted(p.stem for p in tmp_path.rglob("*.npy"))
    assert keys
    index = json.loads((tmp_path / R.CACHE_INDEX_NAME).read_text())
    assert index["params_hash"] == params.hash


def test_matrix_cache_key_changes_with_the_network_state(params, small_cells, tmp_path):
    backend = CountingBackend(params)
    R.precompute_matrices(
        small_cells, params, backend, network_state=frozenset(), cache_dir=tmp_path
    )
    before = {p.name for p in tmp_path.rglob("*.npy")}
    backend.calls = ZERO
    R.precompute_matrices(
        small_cells,
        params,
        backend,
        network_state=frozenset({"metro_c1"}),
        cache_dir=tmp_path,
    )
    assert backend.calls > ZERO
    assert {p.name for p in tmp_path.rglob("*.npy")} > before


def test_precomputed_matrices_are_float32_on_disk(params, small_cells, tmp_path):
    """Section 5.2: store as float32, memory-mapped, under data/cache/ttm/{mode}/{hash}.npy."""
    backend = R.HaversineBackend(params)
    R.precompute_matrices(
        small_cells, params, backend, network_state=frozenset(), cache_dir=tmp_path
    )
    files = list(tmp_path.rglob("*.npy"))
    assert files
    for f in files:
        assert np.load(f, mmap_mode="r").dtype == np.float32
        assert f.parent.parent.name == R.TTM_DIR_NAME


def test_precompute_selects_origins_and_res8_destinations(params, small_cells):
    backend = R.HaversineBackend(params)
    ms = R.precompute_matrices(small_cells, params, backend, network_state=frozenset())
    assert set(ms.origins) <= set(small_cells["h3"])
    assert set(ms.destinations) <= set(small_cells["h3_res8"])
    for m in ms.minutes.values():
        assert m.shape == (len(ms.origins), len(ms.destinations))


def test_matrixset_roundtrips_through_the_store(params, small_cells, tmp_path):
    backend = R.HaversineBackend(params)
    ms = R.precompute_matrices(
        small_cells, params, backend, network_state=frozenset(), cache_dir=tmp_path
    )
    reloaded = R.load_matrices(tmp_path)
    assert reloaded.origins == ms.origins
    assert reloaded.destinations == ms.destinations
    assert set(reloaded.minutes) == set(ms.minutes)
    for mode, m in ms.minutes.items():
        assert np.array_equal(m, reloaded.minutes[mode])


# --------------------------------------------------------------------------------------
# the two-wheeler OSRM profile generator (Section 2.3)
# --------------------------------------------------------------------------------------


def test_twowheeler_profile_exists_and_reads_a_generated_constant():
    lua = Path(R.__file__).resolve().parents[2] / "osrm" / "profiles" / "twowheeler.lua"
    text = lua.read_text()
    assert R.TWOWHEELER_CONSTANTS_MODULE in text
    # Section 0.1 rule 3, in spirit: the speeds are not written into the profile.
    assert "speed_factor" in text


def test_twowheeler_generator_reads_the_speed_factor_from_yaml(params, tmp_path):
    """`modes.two_wheeler.speed_factor` (Section 2.3) now exists in accessibility.yaml.

    This test previously pinned its absence (the generator raising `MissingParameter`
    rather than inventing the 0.85 printed in the spec prose). The leaf has since been
    added, so the generator must read it with no explicit `speed_factor=` override.
    """
    out = tmp_path / R.TWOWHEELER_CONSTANTS_FILE
    R.write_twowheeler_constants(params, out)

    factor = float(params.value(R.TWOWHEELER_SPEED_FACTOR_PATH))
    text = out.read_text()
    assert "return" in text
    for road_class in ("arterial", "collector", "local"):
        kmh = float(params.value(f"accessibility.speeds_kmh.{road_class}"))
        assert repr(round(kmh * factor, 6)) in text or str(kmh * factor)[:6] in text


def test_twowheeler_generator_emits_lua_when_the_parameter_is_supplied(params, tmp_path):
    factor = float(params.value("accessibility.modes.two_wheeler.share"))  # any real value
    out = tmp_path / R.TWOWHEELER_CONSTANTS_FILE
    R.write_twowheeler_constants(params, out, speed_factor=factor)
    text = out.read_text()
    assert "return" in text
    for road_class in ("arterial", "collector", "local"):
        kmh = float(params.value(f"accessibility.speeds_kmh.{road_class}"))
        assert repr(round(kmh * factor, 6)) in text or str(kmh * factor)[:6] in text
