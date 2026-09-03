"""Module 2 — data ingestion (spec Section 6).

Every ingester implements the Section 6 protocol and is split into an injectable fetch/read
half (:class:`ufe.ingest.core.SourceReader`) and a pure transform half
(``parse`` / ``to_cells``). See :mod:`ufe.ingest.core` for the shared machinery, the
imputation-flagging convention and the ``ingest_runs`` ledger, and
:mod:`ufe.ingest.adapters.base` for the state adapter pattern.

Which ``cells`` columns come from where
--------------------------------------
============================================= ==========================
``elev_m``, ``slope_pct``                     :mod:`~ufe.ingest.terrain`
``landcover``, ``undevelopable_frac``         :mod:`~ufe.ingest.landcover`
``builtup_frac``, ``floorspace_res_sqm``,     :mod:`~ufe.ingest.buildings`
``floorspace_com_sqm``
``population``                                :mod:`~ufe.ingest.population`
``util_power``, ``dist_cbd_m``,               :mod:`~ufe.ingest.osm`
``dist_coast_m``, ``dist_arterial_m``,
``jobs_by_sector``, ``*_poi_count``
``nightlight``                                :mod:`~ufe.ingest.nightlights`
``price_res_inr_sqft``,                       :mod:`~ufe.ingest.prices`
``price_land_inr_sqft``,
``rent_res_inr_sqft_mo``
``parcel_count``, ``mean_parcel_sqm``         :mod:`~ufe.ingest.cadastral`
``zone_class``, ``permitted_far``,            :mod:`~ufe.ingest.zoning`
``crz_class``, ``util_water``, ``util_sewer``
``data_conf``                                 :func:`ufe.ingest.core.data_conf`
============================================= ==========================

``h3``, ``h3_res8``, ``in_city``, ``geometry``, ``lat``, ``lon`` and ``area_sqm`` come from
Module 1 (:mod:`ufe.grid.build`). ``households`` and ``hh_by_band`` cannot be derived until
``behaviour.persons_per_household_by_band`` is populated — see
:func:`ufe.ingest.population.households_from_population`.

Imports here are lazy by design: this package must be importable without pulling in
rasterio, tobler and exactextract.
"""

from __future__ import annotations

__all__ = [
    "core",
    "adapters",
    "terrain",
    "landcover",
    "buildings",
    "population",
    "osm",
    "nightlights",
    "prices",
    "rera",
    "cadastral",
    "zoning",
    "projects",
    "coverage",
]
