"""6.7 Prices — three sub-ingesters, blended.

    (a) listing portals   -> asking prices and rents, by locality polygon or by point
                             with a 500 m Gaussian smear
    (b) registration      -> guidance values by SRO and village/ward, reached through the
                             state adapter (Section 6.0), joined by administrative polygon
    (c) broker panel      -> the calibration anchor, a manual CSV upload

    ask_adj_i   = ask_i * (1 - haircut)          # price.price_data.ask_haircut_*
    reg_adj_i   = reg_i * regional_uplift_i      # fitted, not assumed
    price_res_i = w * ask_adj_i + (1 - w) * reg_adj_i    # price.price_data.blend_weight_ask

``regional_uplift`` is fitted exactly as Section 6.7 sets out: ``uplift_hat = median(broker
/ reg)`` on cells with broker observations, regressed on ``(dist_cbd, zone_class)`` and
predicted elsewhere; with fewer than 30 broker observations it collapses to the single
median and ``data_conf`` for the price column is pinned at 0.4. Both figures come from
``config/ingest.yaml``, and the haircut and blend weight from ``config/params/price.yaml``.

**The legal constraint, which is load-bearing (Sections 6.7, 22.2, 22.3).** Listing portal
data is proprietary and their terms of service prohibit automated collection; Section 22.3
ranks this as the single largest legal exposure in the system, above every software licence.
:func:`assert_listing_source_permitted` implements the spec's rule literally: an ingester
run in ``production`` mode against a source marked ``tos_restricted`` **refuses to run** and
raises :class:`ufe.errors.DataRightsViolation`. ``provenance()['licence']`` records which of
"licensed feed / licensed aggregator / internal calibration only" applies.

Rents (Section 6.7, "Rents") are always from listing portals and are required for the yield
check. A missing rent is imputed from the res-8 parent median; where that is missing too,
the cell is marked ``overheat_unavailable`` rather than imputed further — hence the
``rent_res_inr_sqft_mo__impute_method`` value ``overheat_detector_unavailable``, which
Layer 6 must honour.

Genuinely complete: the blend, the uplift fit, the Gaussian smear and the rent cascade are
all implemented and tested, including a hand-computed blend fixture (Section 6 ACCEPTANCE).
The portal *acquisition* is deliberately not implemented — it is a licensing decision, not a
coding one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from ufe import geo
from ufe.errors import DataRightsViolation
from ufe.ingest.core import CityConfig, Ingester, cells_gdf, cfg, mark_imputed

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_LISTINGS",
    "KEY_LISTING_POINTS",
    "KEY_BROKER_PANEL",
    "PRICE_RES_COLUMN",
    "PRICE_LAND_COLUMN",
    "RENT_COLUMN",
    "BROKER_PANEL_SCHEMA",
    "MARKET_STATE_HAIRCUT_PATHS",
    "PricesIngester",
    "assert_listing_source_permitted",
    "gaussian_smear",
    "listings_to_cells",
    "guidance_to_cells",
    "broker_panel_to_cells",
    "fit_regional_uplift",
    "blend_prices",
    "impute_rents",
]

KEY_LISTINGS = "listings/localities"
KEY_LISTING_POINTS = "listings/points"
KEY_BROKER_PANEL = "broker_panel"

PRICE_RES_COLUMN = "price_res_inr_sqft"
PRICE_LAND_COLUMN = "price_land_inr_sqft"
RENT_COLUMN = "rent_res_inr_sqft_mo"

#: Section 6.7c, verbatim: the manual broker CSV schema.
BROKER_PANEL_SCHEMA: tuple[str, ...] = (
    "date",
    "lat",
    "lon",
    "area_sqft",
    "total_price_inr",
    "property_type",
    "transaction_type",
)

#: Market state -> the haircut parameter path in ``config/params/price.yaml``.
MARKET_STATE_HAIRCUT_PATHS: Mapping[str, str] = {
    "stable": "price.price_data.ask_haircut_stable",
    "soft": "price.price_data.ask_haircut_soft",
}
BLEND_WEIGHT_PATH = "price.price_data.blend_weight_ask"

_ASK, _REG, _RENT, _BROKER = "_ask", "_reg", "_rent", "_broker"
_TOS_RESTRICTED = "tos_restricted"

#: The 2 in the Gaussian kernel ``exp(-d^2 / (2 sigma^2))`` and the squaring of the
#: coordinate differences: the definition of a Gaussian, not a model parameter.
_GAUSSIAN_VARIANCE_FACTOR = 2
_SQUARE = 2


# --------------------------------------------------------------------------------------
# The legal gate
# --------------------------------------------------------------------------------------


def assert_listing_source_permitted(mode: str, licence_status: str, *, config: Any = None) -> None:
    """Section 6.7: refuse to run in ``production`` against a ``tos_restricted`` source.

    ``licence_status`` is what the operator asserts about the feed —
    ``licensed_feed`` / ``licensed_aggregator`` / ``internal_calibration_only`` /
    ``tos_restricted``. Only the last is blocked, and only in a restricted mode, because
    scraped data confined to internal calibration is the spec's third permitted option.
    """
    restricted_modes = {str(m) for m in cfg("prices.restricted_modes", config)}
    if str(licence_status) == _TOS_RESTRICTED and str(mode) in restricted_modes:
        raise DataRightsViolation(
            "refusing to ingest listing-portal data marked tos_restricted in "
            f"{mode!r} mode. Licence the feed, use a licensed aggregator, or restrict the "
            "data to internal calibration that is never redistributed and never surfaced "
            "in a paid product (spec Sections 6.7, 22.2, 22.3)."
        )


# --------------------------------------------------------------------------------------
# (a) listing portals
# --------------------------------------------------------------------------------------


def gaussian_smear(
    cells: pd.DataFrame,
    points: gpd.GeoDataFrame,
    value_column: str,
    *,
    crs_metric: str,
    config: Any = None,
) -> pd.Series:
    """Section 6.7a: a point observation smeared over nearby cells with a Gaussian kernel.

    ``weight = exp(-d^2 / (2 sigma^2))`` with ``sigma = prices.point_smear_sigma_m``,
    truncated below ``prices.smear_weight_floor``. Distances are metric. The result is the
    weighted mean of every contributing observation, which is order-independent and hence
    deterministic.
    """
    sigma = float(cfg("prices.point_smear_sigma_m", config))
    floor = float(cfg("prices.smear_weight_floor", config))
    index = pd.Index(cells["h3"].astype(str), name="h3")
    if points is None or not len(points):
        return pd.Series(np.nan, index=index)

    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    centroids = np.column_stack(
        [hexes.geometry.centroid.x.to_numpy(), hexes.geometry.centroid.y.to_numpy()]
    )
    src = points if points.crs is not None else points.set_crs(geo.GEOGRAPHIC_CRS)
    src_m = geo.to_metric(src[[value_column, src.geometry.name]], crs_metric)
    coords = np.column_stack(
        [src_m.geometry.x.to_numpy(dtype=float), src_m.geometry.y.to_numpy(dtype=float)]
    )
    values = np.asarray(src_m[value_column], dtype=float)

    dx = centroids[:, 0][:, None] - coords[:, 0][None, :]
    dy = centroids[:, 1][:, None] - coords[:, 1][None, :]
    weights = np.exp(
        -(dx**_SQUARE + dy**_SQUARE) / (_GAUSSIAN_VARIANCE_FACTOR * sigma**_SQUARE)
    )
    weights[weights < floor] = 0.0
    totals = weights.sum(axis=1)
    smeared = np.divide(
        weights @ values, totals, out=np.full(len(centroids), np.nan), where=totals > 0
    )
    return pd.Series(smeared, index=index)


def listings_to_cells(
    cells: pd.DataFrame,
    *,
    crs_metric: str,
    localities: gpd.GeoDataFrame | None = None,
    points: gpd.GeoDataFrame | None = None,
    ask_column: str = "ask_inr_sqft",
    rent_column: str = "rent_inr_sqft_mo",
    config: Any = None,
) -> pd.DataFrame:
    """Asking price and rent per cell (Section 6.7a).

    Locality polygons win where they exist — "assign to cells within the locality polygon"
    — and the Gaussian smear fills the rest from point observations.
    """
    index = pd.Index(cells["h3"].astype(str), name="h3")
    ask = pd.Series(np.nan, index=index)
    rent = pd.Series(np.nan, index=index)

    if localities is not None and len(localities):
        polygons = (
            localities if localities.crs is not None else localities.set_crs(geo.GEOGRAPHIC_CRS)
        )
        keep = [c for c in (ask_column, rent_column) if c in polygons.columns]
        polygons = geo.to_metric(polygons[keep + [polygons.geometry.name]], crs_metric)
        hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
        centroids = gpd.GeoDataFrame(
            {"h3": hexes["h3"].to_numpy()}, geometry=hexes.geometry.centroid, crs=hexes.crs
        )
        joined = gpd.sjoin(centroids, polygons, how="left", predicate="within")
        joined = joined.drop_duplicates("h3").set_index("h3")
        if ask_column in keep:
            ask = pd.to_numeric(joined[ask_column], errors="coerce").reindex(index)
        if rent_column in keep:
            rent = pd.to_numeric(joined[rent_column], errors="coerce").reindex(index)

    if points is not None and len(points):
        if ask_column in points.columns:
            smeared = gaussian_smear(
                cells, points, ask_column, crs_metric=crs_metric, config=config
            )
            ask = ask.where(ask.notna(), smeared)
        if rent_column in points.columns:
            smeared_rent = gaussian_smear(
                cells, points, rent_column, crs_metric=crs_metric, config=config
            )
            rent = rent.where(rent.notna(), smeared_rent)

    return pd.DataFrame({"h3": index.to_numpy(), _ASK: ask.to_numpy(), _RENT: rent.to_numpy()})


# --------------------------------------------------------------------------------------
# (b) registration / guidance values
# --------------------------------------------------------------------------------------


def guidance_to_cells(
    guidance: gpd.GeoDataFrame,
    cells: pd.DataFrame,
    *,
    crs_metric: str,
    value_column: str = "guidance_inr_sqft",
) -> pd.DataFrame:
    """Join guidance values to cells by administrative polygon (Section 6.7b).

    The frame comes from the state adapter, already normalised to INR/sqft; this function
    knows nothing about which state published it.
    """
    index = pd.Index(cells["h3"].astype(str), name="h3")
    if guidance is None or not len(guidance):
        return pd.DataFrame({"h3": index.to_numpy(), _REG: np.nan})
    polygons = guidance if guidance.crs is not None else guidance.set_crs(geo.GEOGRAPHIC_CRS)
    polygons = geo.to_metric(polygons[[value_column, polygons.geometry.name]], crs_metric)
    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    centroids = gpd.GeoDataFrame(
        {"h3": hexes["h3"].to_numpy()}, geometry=hexes.geometry.centroid, crs=hexes.crs
    )
    joined = (
        gpd.sjoin(centroids, polygons, how="left", predicate="within")
        .drop_duplicates("h3")
        .set_index("h3")
    )
    values = pd.to_numeric(joined[value_column], errors="coerce").reindex(index)
    return pd.DataFrame({"h3": index.to_numpy(), _REG: values.to_numpy()})


# --------------------------------------------------------------------------------------
# (c) broker panel — the calibration anchor
# --------------------------------------------------------------------------------------


def broker_panel_to_cells(
    panel: pd.DataFrame, cells: pd.DataFrame, *, crs_metric: str
) -> pd.DataFrame:
    """Median observed INR/sqft per cell from the broker CSV (Section 6.7c).

    Validates the CSV against :data:`BROKER_PANEL_SCHEMA` and raises on a missing column —
    this is the calibration anchor, so a malformed upload must fail loudly.
    """
    index = pd.Index(cells["h3"].astype(str), name="h3")
    if panel is None or not len(panel):
        return pd.DataFrame({"h3": index.to_numpy(), _BROKER: np.nan})
    missing = [c for c in BROKER_PANEL_SCHEMA if c not in panel.columns]
    if missing:
        raise ValueError(
            f"broker panel is missing required column(s) {missing}; the Section 6.7c "
            f"schema is {list(BROKER_PANEL_SCHEMA)}"
        )
    observations = gpd.GeoDataFrame(
        panel.copy(),
        geometry=gpd.points_from_xy(
            panel["lon"].astype(float), panel["lat"].astype(float)
        ),
        crs=geo.GEOGRAPHIC_CRS,
    )
    observations["_inr_sqft"] = observations["total_price_inr"].astype(float) / observations[
        "area_sqft"
    ].astype(float)
    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    observations = geo.to_metric(observations[["_inr_sqft", "geometry"]], crs_metric)
    joined = gpd.sjoin(observations, hexes, how="inner", predicate="within")
    medians = joined.groupby("h3")["_inr_sqft"].median().reindex(index)
    return pd.DataFrame({"h3": index.to_numpy(), _BROKER: medians.to_numpy()})


# --------------------------------------------------------------------------------------
# regional_uplift — fitted, not assumed
# --------------------------------------------------------------------------------------


def fit_regional_uplift(
    broker: pd.Series | np.ndarray,
    registration: pd.Series | np.ndarray,
    cells: pd.DataFrame,
    *,
    config: Any = None,
) -> tuple[np.ndarray, bool, int]:
    """Section 6.7's uplift procedure, verbatim.

    ``uplift_hat = median(broker_i / reg_i)`` over cells with a broker observation; fit
    ``uplift ~ dist_cbd + zone_class`` by ordinary least squares and predict for cells with
    no observation. Returns ``(uplift_per_cell, low_confidence, n_observations)``;
    ``low_confidence`` is True when there are fewer than
    ``prices.min_broker_observations`` observations, in which case the uplift is the single
    median everywhere and the caller pins ``data_conf`` for the price column.
    """
    min_obs = int(cfg("prices.min_broker_observations", config))
    min_per_zone = int(cfg("prices.min_observations_per_zone", config))

    broker = np.asarray(broker, dtype=float)
    reg = np.asarray(registration, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((reg > 0) & np.isfinite(broker), broker / reg, np.nan)
    observed = np.isfinite(ratio)
    n_obs = int(observed.sum())
    if n_obs == 0:
        return np.ones(len(cells), dtype=float), True, 0

    single = float(np.nanmedian(ratio[observed]))
    if n_obs < min_obs:
        logger.warning(
            "only %d broker observation(s) (< %d): regional_uplift collapses to the single "
            "median %.3f and data_conf for the price column is pinned (Section 6.7)",
            n_obs,
            min_obs,
            single,
        )
        return np.full(len(cells), single, dtype=float), True, n_obs

    dist = np.asarray(cells["dist_cbd_m"], dtype=float)
    zones = cells["zone_class"].astype(str).to_numpy()
    counts = pd.Series(zones[observed]).value_counts()
    zone_terms = [z for z in sorted(counts.index) if counts[z] >= min_per_zone]
    columns = [np.ones(len(cells)), dist] + [
        (zones == z).astype(float) for z in zone_terms[1:]
    ]
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design[observed], ratio[observed], rcond=None)
    predicted = design @ coefficients
    uplift = np.where(observed, ratio, predicted)
    # An uplift must be positive; a regression can extrapolate below zero at the fringe.
    uplift = np.where(np.isfinite(uplift) & (uplift > 0), uplift, single)
    return uplift, False, n_obs


# --------------------------------------------------------------------------------------
# The blend
# --------------------------------------------------------------------------------------


def blend_prices(
    ask: pd.Series | np.ndarray,
    registration: pd.Series | np.ndarray,
    uplift: np.ndarray,
    *,
    params: Any,
    market_state: str = "stable",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``price_res = w * ask * (1 - haircut) + (1 - w) * reg * uplift`` (Section 6.7).

    Returns ``(price_res, ask_adj, reg_adj)``. Where one leg is missing the other carries
    the whole weight — a null ask must not zero the price of a cell with a good guidance
    value — and the caller flags that cell as partly imputed.
    """
    if market_state not in MARKET_STATE_HAIRCUT_PATHS:
        raise ValueError(
            f"unknown market_state {market_state!r}; known: "
            f"{sorted(MARKET_STATE_HAIRCUT_PATHS)}"
        )
    haircut = float(params.value(MARKET_STATE_HAIRCUT_PATHS[market_state]))
    weight = float(params.value(BLEND_WEIGHT_PATH))

    ask_adj = np.asarray(ask, dtype=float) * (1 - haircut)
    reg_adj = np.asarray(registration, dtype=float) * np.asarray(uplift, dtype=float)
    have_ask, have_reg = np.isfinite(ask_adj), np.isfinite(reg_adj)

    blended = np.full(len(ask_adj), np.nan, dtype=float)
    both = have_ask & have_reg
    blended[both] = weight * ask_adj[both] + (1 - weight) * reg_adj[both]
    blended[have_ask & ~have_reg] = ask_adj[have_ask & ~have_reg]
    blended[~have_ask & have_reg] = reg_adj[~have_ask & have_reg]
    return blended, ask_adj, reg_adj


def impute_rents(
    rent: pd.Series | np.ndarray, cells: pd.DataFrame, *, config: Any = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Section 6.7 "Rents": res-8 parent median, then mark the detector unavailable.

    Returns ``(rent, imputed_from_parent_mask, overheat_unavailable_mask)``. Nothing is
    imputed beyond the parent median: a cell with no rent anywhere in its res-8 parent
    keeps a null rent and is marked so Layer 6 can disable the overheating detector for it
    rather than scoring a fabricated yield.
    """
    min_obs = int(cfg("prices.rent_parent_min_observations", config))
    values = pd.Series(np.asarray(rent, dtype=float), index=cells.index)
    parents = cells["h3_res8"].astype(str)
    parent_median = values.groupby(parents).transform(
        lambda s: s.median() if s.notna().sum() >= min_obs else np.nan
    )
    from_parent = values.isna() & parent_median.notna()
    filled = values.where(~from_parent, parent_median)
    unavailable = filled.isna()
    return filled.to_numpy(), from_parent.to_numpy(), unavailable.to_numpy()


# --------------------------------------------------------------------------------------
# The ingester
# --------------------------------------------------------------------------------------


class PricesIngester(Ingester):
    """Listings + guidance values + broker panel -> ``price_res``, ``price_land``, ``rent``.

    The registration leg is reached through the state adapter, so a new state needs no
    change here (Section 23 item 11).
    """

    source_id = "listing_portals"
    tier = "city"
    fills = (PRICE_RES_COLUMN, PRICE_LAND_COLUMN, RENT_COLUMN)
    spatial_res = "locality polygon, or point with a 500 m Gaussian smear"
    temporal_res = "as published (asking prices are a snapshot)"
    notes = (
        "Largest legal exposure in the system (Section 22.3). Blend of ask (haircut from "
        "price.yaml), guidance values via the state adapter (uplift fitted against the "
        "broker panel) and the broker panel itself as the anchor."
    )

    def __init__(
        self,
        reader: Any,
        *,
        adapter: Any = None,
        city: CityConfig | None = None,
        params: Any = None,
        config: Mapping[str, Any] | None = None,
        licence_status: str = "internal_calibration_only",
        market_state: str = "stable",
    ) -> None:
        super().__init__(reader, city=city, config=config)
        self.adapter = adapter
        self.params = params
        self.licence_status = licence_status
        self.market_state = market_state
        #: Set by :meth:`to_cells`: True when the uplift fell back to the single median.
        self.low_confidence = False
        self.n_broker_observations = 0

    def keys(self, city: CityConfig) -> tuple[str, ...]:
        return (KEY_LISTINGS, KEY_LISTING_POINTS, KEY_BROKER_PANEL)

    def fetch(self, city: CityConfig, force: bool = False) -> Path:
        """Refuses outright in a restricted mode against a ToS-restricted source."""
        assert_listing_source_permitted(city.mode, self.licence_status, config=self.config)
        for key in self.keys(city):
            if self.reader.exists(key):
                if not force and key in self._fetched:
                    return self._fetched[key]
                path = self.reader.path(key, force=force)
                self._fetched[key] = path
                return path
        from ufe.ingest.core import MissingSource

        raise MissingSource(f"no price source available for {city.city_id}")

    def parse(self, raw: Path) -> pd.DataFrame:
        """The broker panel, validated against the Section 6.7c schema."""
        if not self.reader.exists(KEY_BROKER_PANEL):
            return pd.DataFrame(columns=list(BROKER_PANEL_SCHEMA))
        panel = self.reader.table(KEY_BROKER_PANEL)
        missing = [c for c in BROKER_PANEL_SCHEMA if c not in panel.columns]
        if missing:
            raise ValueError(
                f"broker panel is missing required column(s) {missing}; the Section 6.7c "
                f"schema is {list(BROKER_PANEL_SCHEMA)}"
            )
        return panel

    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.city is None or self.params is None:
            raise ValueError("PricesIngester needs both a CityConfig and Params")
        crs = self.city.crs_metric
        listings = listings_to_cells(
            cells,
            crs_metric=crs,
            localities=self.reader.vector(KEY_LISTINGS)
            if self.reader.exists(KEY_LISTINGS)
            else None,
            points=self.reader.vector(KEY_LISTING_POINTS)
            if self.reader.exists(KEY_LISTING_POINTS)
            else None,
            config=self.config,
        )
        guidance = (
            self.adapter.guidance_values(self.city) if self.adapter is not None else None
        )
        registration = guidance_to_cells(guidance, cells, crs_metric=crs)
        broker = broker_panel_to_cells(df, cells, crs_metric=crs)

        uplift, low_conf, n_obs = fit_regional_uplift(
            broker[_BROKER], registration[_REG], cells, config=self.config
        )
        self.low_confidence, self.n_broker_observations = low_conf, n_obs

        price, ask_adj, reg_adj = blend_prices(
            listings[_ASK],
            registration[_REG],
            uplift,
            params=self.params,
            market_state=self.market_state,
        )
        rent, rent_from_parent, overheat_unavailable = impute_rents(
            listings[_RENT], cells, config=self.config
        )

        out = pd.DataFrame(
            {
                "h3": cells["h3"].astype(str).to_numpy(),
                PRICE_RES_COLUMN: price,
                # The guidance value *is* a land value; it is the only land observation
                # available, so price_land is reg_adj and is flagged as derived from it.
                PRICE_LAND_COLUMN: reg_adj,
                RENT_COLUMN: rent,
            }
        )
        no_ask = ~np.isfinite(np.asarray(ask_adj, dtype=float))
        out = mark_imputed(out, PRICE_RES_COLUMN, no_ask, "no_listing_observation")
        out = mark_imputed(
            out,
            PRICE_RES_COLUMN,
            np.full(len(out), bool(low_conf)),
            "uplift_single_median_low_confidence",
        )
        out = mark_imputed(
            out, PRICE_LAND_COLUMN, np.ones(len(out), dtype=bool), "guidance_value_as_land_price"
        )
        out = mark_imputed(out, RENT_COLUMN, rent_from_parent, "res8_parent_median")
        out = mark_imputed(
            out, RENT_COLUMN, overheat_unavailable, "overheat_detector_unavailable"
        )
        return out

    def price_column_data_conf(self) -> float | None:
        """Section 6.7: ``data_conf`` for the price column when the uplift is low-confidence."""
        if not self.low_confidence:
            return None
        return float(cfg("prices.low_confidence_data_conf", self.config))
