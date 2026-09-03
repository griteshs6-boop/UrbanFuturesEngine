"""6.8 RERA — the state RERA portal's registered-project listings.

Reached through the state adapter (:meth:`StateAdapter.rera_projects`), so onboarding a new
state needs no change in this module (Section 23 item 11).

Section 6.8 asks for three outputs from one source:

``supply_pipeline``      forward residential supply by cell and year -> feeds Layer 4
``absorption_observed``  booked units per quarter per project -> calibrates
                         ``supply.absorption.base_growth``
delivery record          per private developer -> feeds ``announcers`` for the township
                         archetypes

Per-project fields extracted: name, promoter, location, total units, unit mix, declared
start and completion, quarterly progress percentage and, where published, units booked.

Structural stub versus complete: the three transforms below are complete and tested against
a synthetic portal extract. What is *not* implemented is portal acquisition — AP RERA has no
bulk endpoint and its terms do not permit automated collection, so
:meth:`AccessTerms.assert_bulk_access` blocks it and the extract must be obtained by hand or
by formal request (Section 22.2). No cell column comes from this ingester; its outputs are
tables of their own, which is why ``fills`` is empty.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from ufe import geo
from ufe.ingest.core import CityConfig, Ingester, cells_gdf, cfg

logger = logging.getLogger(__name__)

__all__ = [
    "SUPPLY_PIPELINE_COLUMNS",
    "ABSORPTION_COLUMNS",
    "DELIVERY_RECORD_COLUMNS",
    "ReraIngester",
    "assign_projects_to_cells",
    "supply_pipeline",
    "absorption_observed",
    "delivery_record",
]

SUPPLY_PIPELINE_COLUMNS: tuple[str, ...] = (
    "h3",
    "year",
    "rera_id",
    "units",
    "source",
)
ABSORPTION_COLUMNS: tuple[str, ...] = (
    "rera_id",
    "quarter",
    "units_booked",
    "progress_pct",
)
DELIVERY_RECORD_COLUMNS: tuple[str, ...] = (
    "promoter",
    "projects_registered",
    "units_declared",
    "units_delivered",
    "delivery_ratio",
    "median_slip_months",
)

def _quarter_end(quarter: pd.Series, *, config: Any = None) -> pd.Series:
    """``"2024-Q1"`` -> the last month of that quarter, as a Timestamp.

    Deterministic by construction: the slip is measured against the last quarter the
    portal actually reported, never against "now" (CONTRACT.md rule 5).
    """
    months_per_quarter = int(cfg("calendar.months_per_quarter", config))
    text = quarter.astype(str).str.upper().str.extract(r"(?P<year>\d{4}).*?Q(?P<q>[1-4])")
    year = pd.to_numeric(text["year"], errors="coerce")
    q = pd.to_numeric(text["q"], errors="coerce")
    month = q * months_per_quarter
    stamps = pd.to_datetime(
        pd.DataFrame({"year": year, "month": month, "day": 1}), errors="coerce"
    )
    return stamps


def assign_projects_to_cells(
    projects: pd.DataFrame, cells: pd.DataFrame, *, crs_metric: str
) -> pd.DataFrame:
    """Attach an ``h3`` id to every RERA project from its ``(lat, lon)``.

    Projects outside the cell set get a null ``h3`` and are dropped from the pipeline with
    a warning — a project sited outside the analysis halo has no cell to supply.
    """
    if not len(projects):
        return projects.assign(h3=pd.Series(dtype=str))
    points = gpd.GeoDataFrame(
        projects.copy(),
        geometry=gpd.points_from_xy(
            pd.to_numeric(projects["lon"], errors="coerce"),
            pd.to_numeric(projects["lat"], errors="coerce"),
        ),
        crs=geo.GEOGRAPHIC_CRS,
    )
    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    points_m = geo.to_metric(points, crs_metric)
    joined = gpd.sjoin(points_m, hexes, how="left", predicate="within")
    joined = joined.drop(columns=[c for c in ("index_right", "geometry") if c in joined.columns])
    outside = joined["h3"].isna().sum()
    if outside:
        logger.warning("%d RERA project(s) fall outside the cell set and are dropped", outside)
    return pd.DataFrame(joined)


def supply_pipeline(
    projects: pd.DataFrame, cells: pd.DataFrame, *, crs_metric: str
) -> pd.DataFrame:
    """Forward residential supply by cell and completion year (Section 6.8).

    Units are attributed to the declared completion year. No phasing curve is applied here:
    the archetype phase curves live in ``archetypes.yaml`` and are Layer 4's business.
    """
    sited = assign_projects_to_cells(projects, cells, crs_metric=crs_metric)
    sited = sited[sited["h3"].notna()]
    if not len(sited):
        return pd.DataFrame(columns=list(SUPPLY_PIPELINE_COLUMNS))
    completion = pd.to_datetime(sited["declared_completion"], errors="coerce")
    out = pd.DataFrame(
        {
            "h3": sited["h3"].astype(str).to_numpy(),
            "year": completion.dt.year.to_numpy(),
            "rera_id": sited["rera_id"].astype(str).to_numpy(),
            "units": pd.to_numeric(sited["total_units"], errors="coerce").to_numpy(),
            "source": "rera",
        }
    )
    return out.dropna(subset=["year", "units"]).reset_index(drop=True)


def absorption_observed(projects: pd.DataFrame) -> pd.DataFrame:
    """Booked units per quarter per project — the ``absorption.base_growth`` calibrator."""
    if not len(projects):
        return pd.DataFrame(columns=list(ABSORPTION_COLUMNS))
    out = pd.DataFrame(
        {
            "rera_id": projects["rera_id"].astype(str).to_numpy(),
            "quarter": projects["quarter"].astype(str).to_numpy(),
            "units_booked": pd.to_numeric(
                projects.get("units_booked"), errors="coerce"
            ).to_numpy(),
            "progress_pct": pd.to_numeric(
                projects.get("progress_pct"), errors="coerce"
            ).to_numpy(),
        }
    )
    return out[out["quarter"].notna() & (out["quarter"] != "nan")].reset_index(drop=True)


def delivery_record(projects: pd.DataFrame, *, config: Any = None) -> pd.DataFrame:
    """Per-promoter delivery record, for the ``announcers`` table (Sections 6.8, 3.4).

    ``units_delivered`` is ``total_units * progress_pct`` summed over the promoter's
    projects; ``median_slip_months`` is the median of
    ``(last reported quarter) - (declared completion)`` in months, positive meaning late,
    counted only for projects reporting less than full progress. The slip is measured
    against the last quarter the portal actually reported, never against "now", so the
    record is deterministic (CONTRACT.md rule 5).
    """
    if not len(projects):
        return pd.DataFrame(columns=list(DELIVERY_RECORD_COLUMNS))
    months_per_year = int(cfg("calendar.months_per_year", config))
    frame = projects.copy()
    frame["units"] = pd.to_numeric(frame["total_units"], errors="coerce")
    frame["progress"] = pd.to_numeric(frame["progress_pct"], errors="coerce")
    frame["delivered"] = frame["units"] * frame["progress"]
    completion = pd.to_datetime(frame["declared_completion"], errors="coerce")
    reported = _quarter_end(frame["quarter"], config=config)
    elapsed = (reported.dt.year - completion.dt.year) * months_per_year + (
        reported.dt.month - completion.dt.month
    )
    frame["slip_months"] = pd.Series(
        np.where(frame["progress"] < 1, elapsed.to_numpy(dtype="float64"), 0.0),
        index=frame.index,
    ).clip(lower=0)

    grouped = frame.groupby("promoter", dropna=True)
    out = pd.DataFrame(
        {
            "projects_registered": grouped["rera_id"].nunique(),
            "units_declared": grouped["units"].sum(min_count=1),
            "units_delivered": grouped["delivered"].sum(min_count=1),
            "median_slip_months": grouped["slip_months"].median(),
        }
    ).reset_index()
    out["delivery_ratio"] = np.where(
        out["units_declared"] > 0, out["units_delivered"] / out["units_declared"], np.nan
    )
    return out[list(DELIVERY_RECORD_COLUMNS)]


class ReraIngester(Ingester):
    """State RERA -> ``supply_pipeline``, ``absorption_observed``, delivery record."""

    source_id = "registration_rera_ec_portals"
    tier = "state"
    fills = ()
    spatial_res = "project point location as registered"
    temporal_res = "quarterly progress reports"
    notes = (
        "Reached through the state adapter. AP RERA has no bulk endpoint and its terms do "
        "not permit automated collection, so the extract is obtained manually or by formal "
        "request and handed to the SourceReader (Section 22.2)."
    )

    def __init__(
        self,
        reader: Any,
        *,
        adapter: Any = None,
        city: CityConfig | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reader, city=city, config=config)
        self.adapter = adapter
        self.side_tables: dict[str, pd.DataFrame] = {}

    def keys(self, city: CityConfig) -> tuple[str, ...]:
        from ufe.ingest.adapters.ap import KEY_RERA

        return (KEY_RERA,)

    def parse(self, raw: Path) -> pd.DataFrame:
        """The adapter's normalised project listing (the adapter owns the portal's shape)."""
        if self.adapter is None or self.city is None:
            raise ValueError("ReraIngester needs a StateAdapter and a CityConfig")
        return self.adapter.rera_projects(self.city)

    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        """RERA fills no ``cells`` column; it produces three tables of its own."""
        if self.city is None:
            raise ValueError("ReraIngester needs a CityConfig")
        self.side_tables = {
            "supply_pipeline": supply_pipeline(df, cells, crs_metric=self.city.crs_metric),
            "absorption_observed": absorption_observed(df),
            "developer_delivery_record": delivery_record(df, config=self.config),
        }
        return pd.DataFrame({"h3": cells["h3"].astype(str).to_numpy()})
