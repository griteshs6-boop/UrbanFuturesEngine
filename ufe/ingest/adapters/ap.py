"""Andhra Pradesh state adapter (spec Section 6.0).

This module is the *entire* state-tier surface for every AP city. Adding Visakhapatnam,
Vijayawada or Tirupati needs no change outside ``ufe/ingest/adapters/`` — the ingesters see
only :class:`ufe.ingest.adapters.base.StateAdapter`.

What Andhra Pradesh actually publishes
--------------------------------------
=========================== ==================================================== ==========
Capability                  Portal                                               Published?
=========================== ==================================================== ==========
``guidance_values``         Registration & Stamps Dept, market-value assistance  yes
``rera_projects``           AP RERA registered-project search                     yes
``cadastral_parcels``       Bhu-Naksha AP (survey-number parcel maps)             yes
``industrial_allotments``   APIIC land-allotment disclosures                      yes
``registration_transactions`` — AP publishes no deed-level bulk extract          **no**
=========================== ==================================================== ==========

``registration_transactions()`` therefore returns ``None``. Per Section 6.0 that lowers
``data_conf`` and is named in the report as unavailable; it is never imputed.

Access terms (Section 22.2)
---------------------------
All four portals are interactive, session-based government search interfaces. None of them
publishes a bulk-download endpoint or an API, and none grants permission for bulk automated
collection, so :attr:`AccessTerms.bulk_access_allowed` is ``False``: every read here goes
through the injected :class:`~ufe.ingest.core.SourceReader`, which in production is fed by
a **manually obtained or formally requested** extract, not by a scraper. An ingester that
tries a bulk portal read gets :class:`ufe.errors.DataRightsViolation`. The declared licence
is ``Per-Portal-Terms``, matching ``registration_rera_ec_portals`` in
``config/data_sources_licences.yaml``.

Units
-----
AP registration guidance values are published per square yard. Section 0.3 fixes the
engine's price unit as INR per square foot, so the conversion happens *here* — a
state-specific publishing convention must never leak into a national-tier ingester. The
factor is read from ``config/ingest.yaml`` (``prices.sqft_per_sqyd``).
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import pandas as pd

from ufe.ingest.adapters.base import (
    CAP_CADASTRAL_PARCELS,
    CAP_GUIDANCE_VALUES,
    CAP_INDUSTRIAL_ALLOTMENTS,
    CAP_REGISTRATION_TRANSACTIONS,
    CAP_RERA_PROJECTS,
    AccessTerms,
    StateAdapterBase,
    register,
)
from ufe.ingest.core import MissingSource, SourceReader, cfg

logger = logging.getLogger(__name__)

__all__ = ["AndhraPradeshAdapter", "KEY_GUIDANCE", "KEY_RERA", "KEY_PARCELS", "KEY_ALLOTMENTS"]

#: Reader keys this adapter asks its :class:`SourceReader` for. Logical names, not URLs.
KEY_GUIDANCE = "ap/guidance_values"
KEY_RERA = "ap/rera_projects"
KEY_PARCELS = "ap/cadastral_parcels"
KEY_ALLOTMENTS = "ap/industrial_allotments"

_PORTALS = {
    "registration": "https://registration.ap.gov.in/",
    "rera": "https://rera.ap.gov.in/",
    "bhunaksha": "https://bhunaksha.ap.gov.in/",
    "apiic": "https://apiic.in/",
}

#: Columns the adapter guarantees to its callers, whatever the portal extract looks like.
GUIDANCE_COLUMNS = (
    "locality_id",
    "sro_code",
    "locality_name",
    "guidance_inr_sqft",
    "effective_year",
)
RERA_COLUMNS = (
    "rera_id",
    "project_name",
    "promoter",
    "lat",
    "lon",
    "total_units",
    "units_1bhk",
    "units_2bhk",
    "units_3bhk",
    "declared_start",
    "declared_completion",
    "progress_pct",
    "quarter",
    "units_booked",
)
ALLOTMENT_COLUMNS = ("allotment_id", "allottee", "park_name", "lat", "lon", "area_acres", "year")


@register
class AndhraPradeshAdapter(StateAdapterBase):
    """Andhra Pradesh. ``state_code = "AP"`` — selected by ``config/cities/vizag.yaml``."""

    state_code = "AP"
    state_name = "Andhra Pradesh"
    provides = frozenset(
        {
            CAP_GUIDANCE_VALUES,
            CAP_RERA_PROJECTS,
            CAP_CADASTRAL_PARCELS,
            CAP_INDUSTRIAL_ALLOTMENTS,
        }
    )

    def __init__(self, reader: SourceReader | None = None) -> None:
        self.reader = reader

    # -- helpers ---------------------------------------------------------------------

    def _require_reader(self, capability: str) -> SourceReader:
        if self.reader is None:
            raise MissingSource(
                f"{self.state_code} adapter has no SourceReader; cannot supply "
                f"{capability}. Inject one (spec Section 6: fetch is the injectable half)."
            )
        return self.reader

    def _optional(self, key: str, capability: str) -> Any:
        if self.reader is None or not self.reader.exists(key):
            return self._unavailable(capability)
        return self.reader

    # -- the protocol ----------------------------------------------------------------

    def guidance_values(self, city: Any) -> pd.DataFrame:
        """Guidance values by SRO and village/ward, as locality polygons (Section 6.7b).

        Returned as a GeoDataFrame (which is a DataFrame) so the price ingester can join it
        to cells by administrative polygon without knowing anything about AP. Values are
        converted from the published INR/sq-yd to the engine's INR/sqft here.
        """
        reader = self._require_reader(CAP_GUIDANCE_VALUES)
        frame = reader.vector(KEY_GUIDANCE)
        out = gpd.GeoDataFrame(frame.copy(), geometry=frame.geometry.name, crs=frame.crs)
        if "guidance_inr_sqft" not in out.columns:
            if "guidance_inr_sqyd" not in out.columns:
                raise MissingSource(
                    f"{KEY_GUIDANCE} must carry guidance_inr_sqyd or guidance_inr_sqft; "
                    f"got {sorted(out.columns)}"
                )
            out["guidance_inr_sqft"] = out["guidance_inr_sqyd"].astype(float) / float(
                cfg("prices.sqft_per_sqyd")
            )
        for column in GUIDANCE_COLUMNS:
            if column not in out.columns:
                out[column] = pd.NA
        return out

    def registration_transactions(self, city: Any) -> pd.DataFrame | None:
        """``None``: AP publishes no deed-level bulk extract (see the module docstring)."""
        return self._unavailable(CAP_REGISTRATION_TRANSACTIONS)

    def rera_projects(self, city: Any) -> pd.DataFrame:
        """AP RERA registered projects, normalised to :data:`RERA_COLUMNS` (Section 6.8)."""
        reader = self._require_reader(CAP_RERA_PROJECTS)
        frame = reader.table(KEY_RERA).copy()
        for column in RERA_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        return frame

    def cadastral_parcels(self, city: Any) -> gpd.GeoDataFrame | None:
        """Bhu-Naksha parcel polygons with survey-number attributes (Section 6.9)."""
        if self._optional(KEY_PARCELS, CAP_CADASTRAL_PARCELS) is None:
            return None
        return self._require_reader(CAP_CADASTRAL_PARCELS).vector(KEY_PARCELS)

    def industrial_allotments(self, city: Any) -> pd.DataFrame | None:
        """APIIC land allotments — feeds the industrial-park announcer record."""
        if self._optional(KEY_ALLOTMENTS, CAP_INDUSTRIAL_ALLOTMENTS) is None:
            return None
        frame = self._require_reader(CAP_INDUSTRIAL_ALLOTMENTS).table(KEY_ALLOTMENTS).copy()
        for column in ALLOTMENT_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        return frame

    def access_terms(self) -> dict:
        return AccessTerms(
            state_code=self.state_code,
            portals=_PORTALS,
            tos_urls={
                "registration": _PORTALS["registration"] + "Disclaimer",
                "rera": _PORTALS["rera"] + "disclaimer",
            },
            bulk_access_allowed=False,
            licence="Per-Portal-Terms",
            notes=(
                "All four AP portals are interactive government search interfaces with no "
                "bulk endpoint and no granted permission for automated collection. Extracts "
                "must be obtained manually or by formal request and handed to the "
                "SourceReader; bulk automated reads raise DataRightsViolation "
                "(Sections 6.0, 22.2)."
            ),
        ).as_dict()
