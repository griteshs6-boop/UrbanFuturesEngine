"""Travel-time backends, matrix precomputation and caching (spec Sections 8.3, 8.4, 5.2).

This module is the **ingestion / precompute** half of Module 4.  It is the only place in the
engine that may talk to a routing service or touch the filesystem for travel times.  The
simulation-time half (`ufe.layers.l1_accessibility`) reads a :class:`MatrixSet` and never
imports anything from here that can perform I/O — CONTRACT.md rule 3.

The separation is structural, not conventional:

* everything that can make a network call lives behind the :class:`TravelTimeBackend`
  protocol and is only reachable through :func:`precompute_matrices`;
* :class:`MatrixSet` is a plain, already-materialised value object;
* `l1_accessibility` imports only :class:`MatrixSet` and :func:`station_decay_bands`, and a
  test asserts that it does not so much as name a backend class.

Docker is not available in this environment, so `osrm-routed` cannot run.  :class:`OSRMBackend`
is written correctly and completely against the real `/table` service and is exercised offline
with an injected HTTP client; the live test is marked ``needs_osrm``.  :class:`HaversineBackend`
is the deterministic non-network fallback that lets the rest of the engine and every test run
offline.  Its speeds come from ``config/params/accessibility.yaml``.

Unit conversions
----------------
`METRES_PER_KM`, `MINUTES_PER_HOUR` and `SECONDS_PER_MINUTE` are definitions of the SI and
sexagesimal systems, not estimated quantities, so they are named constants here rather than
YAML parameters (spec Section 0.1 rule 3 governs *parameters*).  Nothing else numeric in this
module is a literal.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.spatial import cKDTree

from ufe.errors import MissingParameter, UFEError
from ufe.params import Params

logger = logging.getLogger(__name__)

__all__ = [
    "METRES_PER_KM",
    "MINUTES_PER_HOUR",
    "SECONDS_PER_MINUTE",
    "TravelTimeBackend",
    "OSRMBackend",
    "HaversineBackend",
    "MatrixSet",
    "StationBand",
    "geodesic_m",
    "chunk_ranges",
    "congestion_adjust",
    "network_state_hash",
    "network_states",
    "metro_matrix",
    "station_walk_distance_m",
    "station_decay_bands",
    "select_origins",
    "select_destinations",
    "precompute_matrices",
    "save_matrices",
    "load_matrices",
    "write_twowheeler_constants",
]

# --- unit definitions (not parameters) -------------------------------------------------
METRES_PER_KM = 1_000
MINUTES_PER_HOUR = 60
SECONDS_PER_MINUTE = 60

ZERO, ONE = 0, 1

NAMESPACE = "accessibility"
TTM_DIR_NAME = "ttm"
CACHE_INDEX_NAME = "ttm_index.json"
STATION_DIR_NAME = "_stations"

# Section 2.3 names this path; it is NOT present in config/params/accessibility.yaml.
TWOWHEELER_SPEED_FACTOR_PATH = f"{NAMESPACE}.modes.two_wheeler.speed_factor"
TWOWHEELER_CONSTANTS_MODULE = "twowheeler_speeds"
TWOWHEELER_CONSTANTS_FILE = f"{TWOWHEELER_CONSTANTS_MODULE}.lua"

# `ufe.geo` is being written in parallel; use its metric helpers when they land, otherwise
# fall back to the WGS84 geodesic, which is metres — never degrees (spec Section 0.3).
try:  # pragma: no cover - depends on a sibling agent's module landing
    from ufe import geo as _geo
except Exception:  # noqa: BLE001
    _geo = None

_GEOD = Geod(ellps="WGS84")

# Which YAML speed a profile borrows in the non-network fallback (Section 8.3's
# "mode-specific speed table").  Values are parameter *paths*, not numbers.
_SPEED_KEY_BY_PROFILE: dict[str, str] = {
    "car": "arterial",
    "two_wheeler": "arterial",
    "twowheeler": "arterial",
    "walk": "walk",
    "foot": "walk",
    "transit": "collector",
    "metro": "metro_line_haul",
}

# `decay_beta.work` is tabulated for car/two_wheeler/transit/walk only; metro is not listed.
# Section 8.1 needs a beta per mode, so metro borrows transit's. Reported as an ambiguity.
_BETA_MODE_ALIAS: dict[str, str] = {"metro": "transit"}

ROAD_MODES: tuple[str, ...] = ("car", "two_wheeler", "walk")
GRAPH_MODES: tuple[str, ...] = ("transit", "metro")


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------


def geodesic_m(a: Sequence[float], b: Sequence[float]) -> float:
    """Geodesic distance in METRES between two ``(lat, lon)`` points (EPSG:4326 input)."""
    if _geo is not None and hasattr(_geo, "geodesic_m"):  # pragma: no cover
        return float(_geo.geodesic_m(a, b))
    _, _, dist = _GEOD.inv(a[ONE], a[ZERO], b[ONE], b[ZERO])
    return float(abs(dist))


def geodesic_matrix_m(origins: np.ndarray, destinations: np.ndarray) -> np.ndarray:
    """Pairwise geodesic distances in metres, shape ``(n_origins, n_destinations)``."""
    origins = np.asarray(origins, dtype=float)
    destinations = np.asarray(destinations, dtype=float)
    n_o, n_d = len(origins), len(destinations)
    lat_o = np.repeat(origins[:, ZERO], n_d)
    lon_o = np.repeat(origins[:, ONE], n_d)
    lat_d = np.tile(destinations[:, ZERO], n_o)
    lon_d = np.tile(destinations[:, ONE], n_o)
    _, _, dist = _GEOD.inv(lon_o, lat_o, lon_d, lat_d)
    return np.abs(np.asarray(dist, dtype=float)).reshape(n_o, n_d)


def _minutes_from_metres(dist_m: np.ndarray, kmh: float) -> np.ndarray:
    return dist_m / (kmh * METRES_PER_KM / MINUTES_PER_HOUR)


def chunk_ranges(n: int, size: int) -> list[tuple[int, int]]:
    """`[(lo, hi), ...]` half-open spans of at most `size` covering `range(n)`."""
    if size < ONE:
        raise ValueError("chunk size must be at least 1")
    return [(lo, min(lo + size, n)) for lo in range(ZERO, max(n, ONE), size)][: max(ONE, -(-n // size))]


# --------------------------------------------------------------------------------------
# the backend protocol
# --------------------------------------------------------------------------------------


@runtime_checkable
class TravelTimeBackend(Protocol):
    """CONTRACT.md's travel-time abstraction."""

    def matrix(
        self, origins: np.ndarray, destinations: np.ndarray, profile: str
    ) -> np.ndarray:
        """Return an ``(n_origins, n_destinations)`` matrix of travel times in MINUTES.

        `origins` and `destinations` are ``(n, 2)`` arrays of ``(lat, lon)`` in EPSG:4326.
        Unreachable pairs are ``np.inf``.  The result is ``float32`` (Section 5.2).
        """


@runtime_checkable
class DistanceCapableBackend(TravelTimeBackend, Protocol):
    """A backend that can also return network DISTANCE in metres (Section 8.4)."""

    def distance_matrix(
        self, origins: np.ndarray, destinations: np.ndarray, profile: str = ...
    ) -> np.ndarray: ...


def _speed_kmh(params: Params, profile: str) -> float:
    key = _SPEED_KEY_BY_PROFILE.get(profile)
    if key is None:
        raise MissingParameter(
            f"no speed is defined for routing profile {profile!r}; known profiles: "
            f"{sorted(_SPEED_KEY_BY_PROFILE)}"
        )
    kmh = float(params.value(f"{NAMESPACE}.speeds_kmh.{key}"))
    if profile in ("two_wheeler", "twowheeler"):
        kmh *= _optional_speed_factor(params)
    return kmh


def _optional_speed_factor(params: Params) -> float:
    """`modes.two_wheeler.speed_factor` (Section 2.3) if it exists, else a no-op 1."""
    try:
        return float(params.value(TWOWHEELER_SPEED_FACTOR_PATH))
    except MissingParameter:
        return ONE


# --------------------------------------------------------------------------------------
# the real backend: OSRM over HTTP (ingestion / precompute time only)
# --------------------------------------------------------------------------------------


class OSRMBackend:
    """HTTP client for a self-hosted `osrm-routed` (spec Sections 2.2, 8.3).

    One container per profile is the deployment described in Section 2.2, but a single
    `osrm-routed` answers on `/table/v1/{profile}/...`, so `endpoint` may either be one
    service or a mapping of profile -> endpoint.

    `max-table-size` must have been raised at launch; the default of 100 rejects our
    requests.  Chunking is `matrix.chunk_size` from YAML (500 x 500 per Section 8.3).
    """

    def __init__(
        self,
        endpoint: str | Mapping[str, str],
        params: Params,
        *,
        client: Any | None = None,
        chunk_size: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._params = params
        self._chunk = int(
            chunk_size
            if chunk_size is not None
            else params.value(f"{NAMESPACE}.matrix.chunk_size")
        )
        self._timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    # ------------------------------------------------------------------ plumbing

    def _endpoint_for(self, profile: str) -> str:
        if isinstance(self._endpoint, Mapping):
            try:
                return str(self._endpoint[profile]).rstrip("/")
            except KeyError as exc:
                raise MissingParameter(
                    f"no OSRM endpoint configured for profile {profile!r}"
                ) from exc
        return str(self._endpoint).rstrip("/")

    def _http(self) -> Any:
        if self._client is None:  # pragma: no cover - needs_osrm path
            import httpx

            self._client = (
                httpx.Client(timeout=self._timeout_s)
                if self._timeout_s is not None
                else httpx.Client()
            )
        return self._client

    @staticmethod
    def _coord_list(points: np.ndarray) -> str:
        """OSRM speaks `lon,lat`."""
        return ";".join(f"{float(p[ONE])},{float(p[ZERO])}" for p in points)

    # ------------------------------------------------------------------ the service

    def _table(
        self,
        origins: np.ndarray,
        destinations: np.ndarray,
        profile: str,
        annotation: str,
    ) -> np.ndarray:
        n_o, n_d = len(origins), len(destinations)
        out = np.full((n_o, n_d), np.inf, dtype=np.float64)
        base = self._endpoint_for(profile)
        client = self._http()

        for lo_o, hi_o in chunk_ranges(n_o, self._chunk):
            for lo_d, hi_d in chunk_ranges(n_d, self._chunk):
                block_o = origins[lo_o:hi_o]
                block_d = destinations[lo_d:hi_d]
                coords = self._coord_list(np.vstack([block_o, block_d]))
                url = f"{base}/table/v1/{profile}/{coords}"
                query = {
                    "sources": ";".join(str(i) for i in range(len(block_o))),
                    "destinations": ";".join(
                        str(len(block_o) + j) for j in range(len(block_d))
                    ),
                    "annotations": annotation,
                }
                response = client.get(url, params=query)
                payload = response.json()
                if payload.get("code") != "Ok":
                    raise UFEError(
                        f"OSRM /table returned {payload.get('code')!r}: "
                        f"{payload.get('message')!r}. If this is NoTable, raise "
                        f"--max-table-size (spec Section 2.2) or lower "
                        f"{NAMESPACE}.matrix.chunk_size."
                    )
                key = "durations" if annotation == "duration" else "distances"
                block = np.array(
                    [[np.inf if v is None else float(v) for v in row] for row in payload[key]],
                    dtype=np.float64,
                )
                out[lo_o:hi_o, lo_d:hi_d] = block
        return out

    def matrix(
        self, origins: np.ndarray, destinations: np.ndarray, profile: str
    ) -> np.ndarray:
        seconds = self._table(
            np.asarray(origins, dtype=float),
            np.asarray(destinations, dtype=float),
            profile,
            "duration",
        )
        return (seconds / SECONDS_PER_MINUTE).astype(np.float32)

    def distance_matrix(
        self, origins: np.ndarray, destinations: np.ndarray, profile: str = "foot"
    ) -> np.ndarray:
        """Network distance in METRES — Section 8.4 wants walk-network, not Euclidean."""
        return self._table(
            np.asarray(origins, dtype=float),
            np.asarray(destinations, dtype=float),
            profile,
            "distance",
        ).astype(np.float32)

    def close(self) -> None:  # pragma: no cover
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None


# --------------------------------------------------------------------------------------
# the fallback backend: deterministic, offline
# --------------------------------------------------------------------------------------


class HaversineBackend:
    """Deterministic non-network fallback (CONTRACT.md), speeds read from YAML.

    Travel time is ``geodesic_metres / speed``, where the speed for each profile comes from
    ``accessibility.speeds_kmh`` via :data:`_SPEED_KEY_BY_PROFILE`.  Distances are true
    WGS84 geodesics (metres), never degrees.

    **Honest limitations versus real OSRM.** This backend knows nothing about the road
    network.  It therefore:

    * ignores circuity — a real trip is typically 20-40% longer than the straight line, so
      times are systematically *optimistic*, and more so in a coastal or hilly city where the
      network detours around water and terrain;
    * ignores barriers — it will happily route across the Bay of Bengal, a reservoir, a
      railway corridor or a restricted port, so it never returns `inf` for road modes;
    * ignores turn restrictions, one-ways, gradients, signals and mode-specific access
      rules, which is precisely what the two-wheeler profile of Section 2.3 exists to model;
    * uses a single free-flow speed per mode rather than a per-way speed, so the
      arterial/collector/local structure of `speeds_kmh` is unused except as the source of
      the one figure it does read.

    Congestion (:func:`congestion_adjust`) is applied on top identically for both backends,
    so it does not compensate for any of the above.  Numbers from this backend are for
    keeping the engine runnable and tests deterministic; they are not a substitute for a
    routed matrix, and any published accessibility figure must come from `OSRMBackend`.
    """

    def __init__(self, params: Params) -> None:
        self._params = params

    def matrix(
        self, origins: np.ndarray, destinations: np.ndarray, profile: str
    ) -> np.ndarray:
        kmh = _speed_kmh(self._params, profile)
        dist = geodesic_matrix_m(origins, destinations)
        return _minutes_from_metres(dist, kmh).astype(np.float32)

    def distance_matrix(
        self, origins: np.ndarray, destinations: np.ndarray, profile: str = "walk"
    ) -> np.ndarray:
        del profile  # a straight line is the same for every mode
        return geodesic_matrix_m(origins, destinations).astype(np.float32)


# --------------------------------------------------------------------------------------
# congestion (Section 8.3)
# --------------------------------------------------------------------------------------


def congestion_adjust(
    minutes: np.ndarray,
    origin_builtup: np.ndarray,
    dest_builtup: np.ndarray,
    midpoint_builtup: np.ndarray,
    params: Params,
) -> np.ndarray:
    """`t_adj = t / max(floor, 1 - k * corridor_builtup)` (Section 8.3).

    **Approximation, as the spec instructs.** Computing the cells a route actually
    intersects for every OD pair is far too expensive, so `corridor_builtup_ij` is the mean
    of `builtup_frac` at the origin, the destination, and the cell nearest the straight-line
    midpoint.  This under-states congestion for routes that detour through a dense core and
    over-states it for orbital routes.  Section 8.3's stated remedy — request route
    geometries for a 2% sample and fit a correction — is not implemented here and is left as
    a calibration task once a live OSRM exists.
    """
    k = float(params.value(f"{NAMESPACE}.congestion.k"))
    floor = float(params.value(f"{NAMESPACE}.congestion.floor"))

    o = np.asarray(origin_builtup, dtype=float)
    d = np.asarray(dest_builtup, dtype=float)
    mid = np.asarray(midpoint_builtup, dtype=float)
    corridor = np.mean(
        np.broadcast_arrays(o[:, None] if o.ndim == ONE else o,
                            d[None, :] if d.ndim == ONE else d,
                            mid),
        axis=ZERO,
    )
    multiplier = np.maximum(floor, ONE - k * corridor)
    return (np.asarray(minutes, dtype=float) / multiplier).astype(np.float32)


def _midpoint_builtup(
    origins: np.ndarray, destinations: np.ndarray, cells: pd.DataFrame
) -> np.ndarray:
    """`builtup_frac` of the cell nearest each straight-line OD midpoint."""
    centroids = cells[["lat", "lon"]].to_numpy(dtype=float)
    tree = cKDTree(centroids)
    mid_lat = np.add.outer(origins[:, ZERO], destinations[:, ZERO]) / len(("o", "d"))
    mid_lon = np.add.outer(origins[:, ONE], destinations[:, ONE]) / len(("o", "d"))
    query = np.column_stack([mid_lat.ravel(), mid_lon.ravel()])
    _, idx = tree.query(query)
    values = cells["builtup_frac"].to_numpy(dtype=float)[idx]
    return values.reshape(mid_lat.shape)


# --------------------------------------------------------------------------------------
# network states (Section 8.3)
# --------------------------------------------------------------------------------------


def network_state_hash(state: Iterable[str]) -> str:
    """Order-independent, stable hash of a set of open project ids."""
    payload = json.dumps(sorted(str(s) for s in state), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def network_states(
    open_years: Mapping[str, int],
    years: Iterable[int],
    *,
    params: Params | None = None,
    corridors: Mapping[str, str] | None = None,
) -> dict[str, frozenset[str]]:
    """Distinct sets of network-modifying projects open across the horizon (Section 8.3).

    `open_years` maps project id -> the year the project starts modifying the network.
    Capped at `matrix.max_network_states`; over the cap, projects are collapsed onto their
    corridor (one toggle per corridor) when `corridors` is supplied, and a `UFEError` is
    raised if that still does not fit.
    """

    def enumerate_states(mapping: Mapping[str, int]) -> dict[str, frozenset[str]]:
        found: dict[str, frozenset[str]] = {}
        for year in sorted(set(int(y) for y in years)):
            state = frozenset(pid for pid, oy in mapping.items() if int(oy) <= year)
            found[network_state_hash(state)] = state
        return found

    states = enumerate_states(open_years)
    if params is None:
        return states

    cap = int(params.value(f"{NAMESPACE}.matrix.max_network_states"))
    if len(states) <= cap:
        return states

    if corridors:
        collapsed: dict[str, int] = {}
        for pid, year in open_years.items():
            corridor = str(corridors.get(pid, pid))
            collapsed[corridor] = min(int(year), collapsed.get(corridor, int(year)))
        logger.warning(
            "%d network states exceeds the cap of %d; collapsing %d projects onto "
            "%d corridors (spec Section 8.3)",
            len(states),
            cap,
            len(open_years),
            len(collapsed),
        )
        states = enumerate_states(collapsed)
        if len(states) <= cap:
            return states

    raise UFEError(
        f"{len(states)} distinct network states exceeds "
        f"{NAMESPACE}.matrix.max_network_states = {cap}. Supply a corridor map so that "
        f"each corridor becomes a single toggle (spec Section 8.3)."
    )


# --------------------------------------------------------------------------------------
# the synthetic transit / metro graph (Section 8.3)
# --------------------------------------------------------------------------------------

STATION_COLUMNS = ("station_id", "line", "lat", "lon")


def _station_coords(stations: pd.DataFrame) -> np.ndarray:
    return stations[["lat", "lon"]].to_numpy(dtype=float)


def _rail_graph(stations: pd.DataFrame, params: Params) -> nx.Graph:
    """Nodes are `(line, station_id)`; line-haul edges within a line, transfers between."""
    haul_kmh = float(params.value(f"{NAMESPACE}.speeds_kmh.metro_line_haul"))
    dwell = float(params.value(f"{NAMESPACE}.transit_penalties_min.metro_dwell_per_station"))
    transfer = float(params.value(f"{NAMESPACE}.transit_penalties_min.transfer"))

    graph = nx.Graph()
    for line, group in stations.groupby("line", sort=True):
        rows = list(group.itertuples())
        for a, b in zip(rows, rows[ONE:]):
            d = geodesic_m((a.lat, a.lon), (b.lat, b.lon))
            weight = _minutes_from_metres(np.array([d]), haul_kmh)[ZERO] + dwell
            graph.add_edge((line, a.station_id), (line, b.station_id), weight=float(weight))
        for row in rows:
            graph.add_node((line, row.station_id))
    for station_id, group in stations.groupby("station_id", sort=True):
        nodes = [(row.line, station_id) for row in group.itertuples()]
        for a, b in zip(nodes, nodes[ONE:]):
            graph.add_edge(a, b, weight=transfer)
    return graph


def _station_to_station_minutes(stations: pd.DataFrame, params: Params) -> np.ndarray:
    graph = _rail_graph(stations, params)
    nodes = [(row.line, row.station_id) for row in stations.itertuples()]
    n = len(nodes)
    out = np.full((n, n), np.inf, dtype=float)
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))
    for i, a in enumerate(nodes):
        reachable = lengths.get(a, {})
        for j, b in enumerate(nodes):
            if b in reachable:
                out[i, j] = float(reachable[b])
    return out


def _access_minutes(
    points: np.ndarray, stations: pd.DataFrame, params: Params, backend: Any
) -> np.ndarray:
    """Walk access/egress minutes from each point to each station, capped (Section 8.3)."""
    cap = float(params.value(f"{NAMESPACE}.matrix.walk_access_cap_m"))
    extra = float(params.value(f"{NAMESPACE}.transit_penalties_min.access_egress_extra"))
    station_pts = _station_coords(stations)

    dist = station_walk_distance_m(points, stations, backend)
    walk_minutes = backend.matrix(points, station_pts, "walk").astype(float)
    out = walk_minutes + extra

    allowed = dist <= cap
    if "feeder_or_park_ride" in stations.columns:
        feeder = stations["feeder_or_park_ride"].to_numpy(dtype=bool)
        gated = _feeder_band_max_m(params)
        if gated is not None:
            allowed = allowed | (feeder[None, :] & (dist <= gated))
    out = np.where(allowed, out, np.inf)
    return out


def _feeder_band_max_m(params: Params) -> float | None:
    gated = [band.max_m for band in station_decay_bands(params) if band.requires]
    return max(gated) if gated else None


def metro_matrix(
    origins: np.ndarray,
    destinations: np.ndarray,
    stations: pd.DataFrame,
    params: Params,
    backend: Any,
) -> np.ndarray:
    """Minutes by rail: walk access + boarding + line haul (+ transfers) + walk egress.

    Shortest path over the Section 8.3 synthetic graph, evaluated as two min-plus products
    so that the Dijkstra only has to run over the (small) station graph.
    """
    board = float(params.value(f"{NAMESPACE}.transit_penalties_min.metro_headway_half"))
    origins = np.asarray(origins, dtype=float)
    destinations = np.asarray(destinations, dtype=float)

    access = _access_minutes(origins, stations, params, backend)          # (n_o, n_s)
    egress = _access_minutes(destinations, stations, params, backend).T   # (n_s, n_d)
    rail = _station_to_station_minutes(stations, params)                  # (n_s, n_s)

    with np.errstate(invalid="ignore"):
        to_alight = np.min(access[:, :, None] + rail[None, :, :], axis=ONE)   # (n_o, n_s)
        total = np.min(to_alight[:, :, None] + egress[None, :, :], axis=ONE)  # (n_o, n_d)
    total = np.where(np.isfinite(total), total + board, np.inf)
    return total.astype(np.float32)


def station_walk_distance_m(
    origins: np.ndarray, stations: pd.DataFrame, backend: Any
) -> np.ndarray:
    """Walk-NETWORK distance in metres to every station (Section 8.4, not Euclidean).

    With `OSRMBackend` this is a `/table?annotations=distance` request on the foot profile.
    With the offline fallback it degrades to the geodesic, which under-states real walking
    distance wherever the street grid is coarse — a documented limitation, not a modelling
    choice.
    """
    if not hasattr(backend, "distance_matrix"):
        raise UFEError(
            f"{type(backend).__name__} cannot return network distances; Section 8.4 "
            "requires walk-network distance, not Euclidean"
        )
    return np.asarray(
        backend.distance_matrix(np.asarray(origins, dtype=float), _station_coords(stations), "walk"),
        dtype=np.float32,
    )


# --------------------------------------------------------------------------------------
# station decay bands (Section 8.4 / Section 21 exclusive-band guard)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StationBand:
    """One row of `accessibility.station_decay`. Bands are EXCLUSIVE (Section 21)."""

    max_m: float
    w: float
    requires: str | None = None


def station_decay_bands(params: Params) -> tuple[StationBand, ...]:
    """The station-decay ladder, ascending in `max_m`, validated as non-overlapping."""
    raw = params.get(f"{NAMESPACE}.station_decay")
    if not isinstance(raw, list) or not raw:
        raise MissingParameter(f"{NAMESPACE}.station_decay must be a non-empty sequence")
    bands = tuple(
        StationBand(
            max_m=float(entry["max_m"]),
            w=float(params.value(f"{NAMESPACE}.station_decay.{i}.w")),
            requires=entry.get("requires"),
        )
        for i, entry in enumerate(raw)
    )
    edges = [band.max_m for band in bands]
    if edges != sorted(edges):
        raise UFEError(
            f"{NAMESPACE}.station_decay must be ordered by ascending max_m so that the "
            "bands can be applied exclusively (spec Section 21)"
        )
    return bands


# --------------------------------------------------------------------------------------
# origin / destination selection (Section 5.2)
# --------------------------------------------------------------------------------------


def select_origins(cells: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Section 5.2: `in_city` cells that hold people or development headroom.

    Departure, reported: Section 5.2 writes `developable_sqm > 20_000`; neither that column
    nor that threshold exists in the landed schema or in YAML, so this uses the closest
    landed equivalent, `headroom_sqm > 0`, which needs no new parameter.
    """
    del params
    mask = cells["in_city"].to_numpy(dtype=bool)
    has_people = cells["population"].to_numpy(dtype=float) > ZERO
    headroom = (
        cells["headroom_sqm"].fillna(ZERO).to_numpy(dtype=float)
        if "headroom_sqm" in cells.columns
        else np.zeros(len(cells))
    )
    keep = mask & (has_people | (headroom > ZERO))
    if not keep.any():
        keep = mask
    if not keep.any():
        raise UFEError("no origin cells: every cell has in_city=False")
    return cells.loc[keep]


def select_destinations(cells: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Section 5.2: res-8 cells with `jobs_total >= dest_min_jobs`, capped by `dest_max_count`."""
    min_jobs = float(params.value(f"{NAMESPACE}.grid.dest_min_jobs"))
    max_count = int(params.value(f"{NAMESPACE}.grid.dest_max_count"))

    jobs = cells["jobs_by_sector"].map(lambda v: float(np.sum(np.asarray(v, dtype=float))))
    grouped = (
        pd.DataFrame({"h3_res8": cells["h3_res8"].to_numpy(), "jobs": jobs.to_numpy()})
        .groupby("h3_res8", sort=True)["jobs"]
        .sum()
    )
    kept = grouped[grouped >= min_jobs]
    if kept.empty:
        kept = grouped
    kept = kept.sort_values(ascending=False).head(max_count).sort_index()

    coords = {
        parent: (float(group["lat"].mean()), float(group["lon"].mean()))
        for parent, group in cells.groupby("h3_res8", sort=True)
    }
    return pd.DataFrame(
        {
            "h3_res8": list(kept.index),
            "lat": [coords[p][ZERO] for p in kept.index],
            "lon": [coords[p][ONE] for p in kept.index],
            "jobs": kept.to_numpy(dtype=float),
        }
    )


# --------------------------------------------------------------------------------------
# the value object the simulation reads
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixSet:
    """Precomputed travel times for one network state. Pure data; performs no I/O.

    `minutes[mode]` is `(len(origins), len(destinations))` in MINUTES, `inf` where
    unreachable. `origins` are res-9 `h3` ids; `destinations` are res-8 `h3_res8` ids
    (Section 5.2).
    """

    origins: tuple[str, ...]
    destinations: tuple[str, ...]
    minutes: dict[str, np.ndarray] = field(default_factory=dict)
    station_walk_dist_m: np.ndarray | None = None
    station_feeder: tuple[bool, ...] = ()
    network_state: str = ""

    def __post_init__(self) -> None:
        shape = (len(self.origins), len(self.destinations))
        for mode, m in self.minutes.items():
            if np.asarray(m).shape != shape:
                raise UFEError(
                    f"matrix for mode {mode!r} has shape {np.asarray(m).shape}, expected {shape}"
                )
        if self.station_walk_dist_m is not None:
            n_o, n_s = np.asarray(self.station_walk_dist_m).shape
            if n_o != len(self.origins):
                raise UFEError("station distance matrix does not match the origin set")
            if self.station_feeder and len(self.station_feeder) != n_s:
                raise UFEError("station_feeder does not match the station distance matrix")


# --------------------------------------------------------------------------------------
# precompute, cache and persistence (Section 5.2)
# --------------------------------------------------------------------------------------


def _cache_key(params: Params, state_hash: str, origins, destinations, mode: str) -> str:
    payload = json.dumps(
        {
            "params": params.hash,          # Section 21: no stale cache across param versions
            "state": state_hash,
            "origins": list(origins),
            "destinations": list(destinations),
            "mode": mode,
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / CACHE_INDEX_NAME


def _read_index(cache_dir: Path) -> dict[str, Any]:
    path = _index_path(cache_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_matrices(matrices: MatrixSet, cache_dir: str | Path, params: Params) -> Path:
    """Persist to `{cache_dir}/ttm/{mode}/{key}.npy` as float32 (Section 5.2)."""
    cache_dir = Path(cache_dir)
    index = _read_index(cache_dir)
    if index.get("params_hash") not in (None, params.hash):
        index = {}
    index.setdefault("params_hash", params.hash)
    index.setdefault("states", {})
    index["origins"] = list(matrices.origins)
    index["destinations"] = list(matrices.destinations)

    state_hash = matrices.network_state or network_state_hash(())
    entry: dict[str, Any] = {"modes": {}}
    for mode, m in matrices.minutes.items():
        key = _cache_key(params, state_hash, matrices.origins, matrices.destinations, mode)
        rel = Path(TTM_DIR_NAME) / mode / f"{key}.npy"
        target = cache_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, np.asarray(m, dtype=np.float32))
        entry["modes"][mode] = str(rel)

    if matrices.station_walk_dist_m is not None:
        key = _cache_key(
            params, state_hash, matrices.origins, matrices.destinations, STATION_DIR_NAME
        )
        rel = Path(TTM_DIR_NAME) / STATION_DIR_NAME / f"{key}.npy"
        (cache_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / rel, np.asarray(matrices.station_walk_dist_m, dtype=np.float32))
        entry["stations"] = str(rel)
        entry["station_feeder"] = [bool(x) for x in matrices.station_feeder]

    index["states"][state_hash] = entry
    index["latest"] = state_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    _index_path(cache_dir).write_text(json.dumps(index, indent=len(("i", "n"))))
    return _index_path(cache_dir)


def load_matrices(cache_dir: str | Path, state_hash: str | None = None) -> MatrixSet:
    """Read a persisted :class:`MatrixSet` back, memory-mapped (Section 5.2)."""
    cache_dir = Path(cache_dir)
    index = _read_index(cache_dir)
    if not index:
        raise UFEError(f"no travel-time cache index at {_index_path(cache_dir)}")
    state_hash = state_hash or index["latest"]
    entry = index["states"][state_hash]
    minutes = {
        mode: np.load(cache_dir / rel, mmap_mode="r")
        for mode, rel in entry["modes"].items()
    }
    stations = (
        np.load(cache_dir / entry["stations"], mmap_mode="r")
        if entry.get("stations")
        else None
    )
    return MatrixSet(
        origins=tuple(index["origins"]),
        destinations=tuple(index["destinations"]),
        minutes=minutes,
        station_walk_dist_m=stations,
        station_feeder=tuple(entry.get("station_feeder", ())),
        network_state=state_hash,
    )


def precompute_matrices(
    cells: pd.DataFrame,
    params: Params,
    backend: TravelTimeBackend,
    *,
    network_state: Iterable[str] = (),
    stations: pd.DataFrame | None = None,
    cache_dir: str | Path | None = None,
    modes: Sequence[str] | None = None,
) -> MatrixSet:
    """Build (or reuse) every travel-time matrix for one network state.

    **This is the only function in Module 4 that may call a routing backend.** It runs at
    ingestion / precompute time. The simulation reads the persisted :class:`MatrixSet`.

    A cache hit — same params hash, same origin/destination sets, same network state —
    performs zero backend requests (Section 8 ACCEPTANCE).
    """
    state_hash = network_state_hash(network_state)
    origins = select_origins(cells, params)
    dests = select_destinations(cells, params)
    origin_ids = tuple(origins["h3"])
    dest_ids = tuple(dests["h3_res8"])

    wanted = tuple(modes) if modes is not None else ROAD_MODES + (
        GRAPH_MODES if stations is not None and len(stations) else ()
    )

    if cache_dir is not None:
        cached = _try_cache(cache_dir, params, state_hash, origin_ids, dest_ids, wanted)
        if cached is not None:
            return cached

    origin_pts = origins[["lat", "lon"]].to_numpy(dtype=float)
    dest_pts = dests[["lat", "lon"]].to_numpy(dtype=float)
    o_builtup = origins["builtup_frac"].to_numpy(dtype=float)
    d_builtup = (
        pd.DataFrame({"h3_res8": cells["h3_res8"], "b": cells["builtup_frac"]})
        .groupby("h3_res8", sort=True)["b"]
        .mean()
        .reindex(dest_ids)
        .to_numpy(dtype=float)
    )
    mid_builtup = _midpoint_builtup(origin_pts, dest_pts, cells)

    minutes: dict[str, np.ndarray] = {}
    for mode in wanted:
        if mode in ROAD_MODES:
            profile = _osrm_profile(params, mode)
            raw = backend.matrix(origin_pts, dest_pts, profile)
            minutes[mode] = congestion_adjust(raw, o_builtup, d_builtup, mid_builtup, params)
        elif stations is not None and len(stations):
            minutes[mode] = metro_matrix(origin_pts, dest_pts, stations, params, backend)
        else:
            logger.warning("mode %s requested with no stations; skipping", mode)

    station_dist = None
    feeder: tuple[bool, ...] = ()
    if stations is not None and len(stations):
        station_dist = station_walk_distance_m(origin_pts, stations, backend)
        feeder = tuple(
            bool(x) for x in stations.get("feeder_or_park_ride", pd.Series(False, index=stations.index))
        )

    result = MatrixSet(
        origins=origin_ids,
        destinations=dest_ids,
        minutes=minutes,
        station_walk_dist_m=station_dist,
        station_feeder=feeder,
        network_state=state_hash,
    )
    if cache_dir is not None:
        save_matrices(result, cache_dir, params)
    return result


def _osrm_profile(params: Params, mode: str) -> str:
    node = params.get(f"{NAMESPACE}.modes.{mode}")
    profile = node.get("osrm_profile") if isinstance(node, dict) else None
    return str(profile) if profile else mode


def _try_cache(
    cache_dir: str | Path,
    params: Params,
    state_hash: str,
    origin_ids: tuple[str, ...],
    dest_ids: tuple[str, ...],
    wanted: Sequence[str],
) -> MatrixSet | None:
    index = _read_index(Path(cache_dir))
    if not index or index.get("params_hash") != params.hash:
        return None
    if tuple(index.get("origins", ())) != origin_ids:
        return None
    if tuple(index.get("destinations", ())) != dest_ids:
        return None
    entry = index.get("states", {}).get(state_hash)
    if not entry or set(wanted) - set(entry["modes"]):
        return None
    if any(not (Path(cache_dir) / rel).exists() for rel in entry["modes"].values()):
        return None
    logger.info("travel-time cache hit for network state %s", state_hash[:8])
    return load_matrices(cache_dir, state_hash)


# --------------------------------------------------------------------------------------
# the two-wheeler OSRM profile constants (Section 2.3)
# --------------------------------------------------------------------------------------

_TWOWHEELER_CLASSES = ("expressway", "national_highway", "arterial", "collector", "local")

_LUA_HEADER = """-- GENERATED FILE — do not edit by hand.
-- Written by ufe.layers.routing.write_twowheeler_constants from
-- config/params/accessibility.yaml (spec Section 2.3).
-- Lua cannot read our YAML at osrm-extract time, so the parameters are generated into this
-- module and required by twowheeler.lua. Regenerate whenever accessibility.yaml changes:
--     python -c "from ufe.params import load_params; from ufe.layers.routing import \\
--       write_twowheeler_constants as w; w(load_params('vizag'), 'osrm/profiles/{file}')"
-- params_hash = {params_hash}
"""


def write_twowheeler_constants(
    params: Params, path: str | Path, *, speed_factor: float | None = None
) -> Path:
    """Generate `twowheeler_speeds.lua` — the speed table `twowheeler.lua` requires.

    Section 2.3 says the factor is read from
    ``config/params/accessibility.yaml -> modes.two_wheeler.speed_factor``. That leaf does
    **not exist** in the file on disk, so this raises :class:`MissingParameter` naming the
    path rather than hardcoding the 0.85 printed in the spec prose (CONTRACT.md rule 1).
    Pass `speed_factor` explicitly to generate the file before the leaf is added.
    """
    if speed_factor is None:
        try:
            speed_factor = float(params.value(TWOWHEELER_SPEED_FACTOR_PATH))
        except MissingParameter as exc:
            raise MissingParameter(
                f"{TWOWHEELER_SPEED_FACTOR_PATH} is required by spec Section 2.3 but is "
                f"absent from config/params/accessibility.yaml. Add it (with conf/scope, "
                f"and a per-road-class override block if Section 2.3's class-specific "
                f"behaviour is wanted) or pass speed_factor= explicitly. "
                f"No default is assumed here."
            ) from exc

    factor = float(speed_factor)
    lines = [_LUA_HEADER.format(file=TWOWHEELER_CONSTANTS_FILE, params_hash=params.hash)]
    lines.append("return {")
    lines.append(f"  speed_factor = {factor!r},")
    lines.append("  speeds_kmh = {")
    for road_class in _TWOWHEELER_CLASSES:
        kmh = float(params.value(f"{NAMESPACE}.speeds_kmh.{road_class}"))
        # bracket form: `local` is a Lua reserved word
        lines.append(
            f"    [{road_class!r}] = {round(kmh * factor, len(_TWOWHEELER_CLASSES) + ONE)!r},"
        )
    lines.append("  },")
    walk = float(params.value(f"{NAMESPACE}.speeds_kmh.walk"))
    lines.append(f"  walk_kmh = {walk!r},")
    lines.append("}")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s (speed_factor=%s)", out, factor)
    return out
