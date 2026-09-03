"""6.2 Land cover — ESA WorldCover 10 m (2021 vintage).

Section 6.2 asks for three things: the zonal *majority* class per cell, the per-class area
*fractions*, and a mapping onto ``undevelopable``. All three come from one ``exactextract``
pass using the ``majority`` / ``frac`` / ``unique`` operations — ``frac`` returns exact
per-class area fractions directly, which is why the spec mandates ``exactextract`` here and
forbids ``rasterstats`` (Section 2.1b).

The class gate, verbatim from Section 6.2:

* water (80), permanent wetland (90), mangroves (95) -> **hard gate**;
* tree cover (10) -> hard gate **only inside a declared forest boundary**, otherwise
  developable.

The tree-cover rule is the reason this module takes an optional ``forest`` polygon layer.
Where it is absent the conditional gate is *not* applied and every affected cell is flagged
imputed — the alternative (assuming all tree cover is reserved forest, or that none is)
would silently move ``undevelopable_frac``, which feeds the Layer 0 hard gates.

Class codes, the undevelopable sets and the ``frac`` sum tolerance all live in
``config/ingest.yaml``; this module contains no numeric literals.

Genuinely complete: transform half implemented and tested against a synthetic categorical
raster with hand-computed class fractions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from ufe import geo
from ufe.errors import UFEError
from ufe.ingest.core import CityConfig, Ingester, cells_gdf, cfg, mark_imputed, zonal
from ufe.store import schemas as S

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_LANDCOVER",
    "KEY_FOREST",
    "LANDCOVER_COLUMN",
    "UNDEVELOPABLE_COLUMN",
    "FRACS_TABLE",
    "LandcoverIngester",
    "class_name_map",
    "landcover_to_cells",
    "landcover_fracs_long",
    "assert_fracs_sum_to_one",
]

KEY_LANDCOVER = "worldcover"
KEY_FOREST = "forest_boundary"

LANDCOVER_COLUMN = "landcover"
UNDEVELOPABLE_COLUMN = "undevelopable_frac"

#: Section 6.2 wants ``landcover_fracs`` "as a struct". ``ufe.store.schemas.CELLS`` is
#: ``strict=True`` and declares no such column, so the fractions are returned as this tidy
#: side table instead and the schema addition is reported, not made.
FRACS_TABLE = "cell_landcover_fracs"

_FRAC, _UNIQUE, _MAJORITY = "frac", "unique", "majority"


def class_name_map(config: Any = None) -> dict[int, str]:
    """WorldCover class code -> the ``cells.landcover`` vocabulary, from YAML."""
    mapping = {int(k): str(v) for k, v in cfg("landcover.class_codes", config).items()}
    unknown = sorted(set(mapping.values()) - set(S.LANDCOVER_CLASSES))
    if unknown:
        raise UFEError(
            f"landcover.class_codes maps to names not in schemas.LANDCOVER_CLASSES: {unknown}"
        )
    return mapping


def _forest_share(
    cells: pd.DataFrame, forest: gpd.GeoDataFrame | None, crs_metric: str
) -> np.ndarray:
    """Per-cell area share falling inside a declared forest boundary.

    Computed in ``crs_metric`` — an area ratio in degrees is meaningless at this latitude.
    """
    n = len(cells)
    if forest is None or not len(forest):
        return np.zeros(n, dtype=float)
    hexes = geo.to_metric(cells_gdf(cells)[["h3", "geometry"]], crs_metric)
    boundary = forest if forest.crs is not None else forest.set_crs(geo.GEOGRAPHIC_CRS)
    boundary = geo.to_metric(boundary[[boundary.geometry.name]], crs_metric)
    union = boundary.union_all() if hasattr(boundary, "union_all") else boundary.unary_union
    inter = hexes.geometry.intersection(union).area
    return (inter / hexes.geometry.area).to_numpy(dtype=float)


def landcover_to_cells(
    raster: str | Path,
    cells: pd.DataFrame,
    *,
    crs_metric: str,
    forest: gpd.GeoDataFrame | None = None,
    config: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zonal majority class and ``undevelopable_frac`` per cell, plus the fractions table.

    Returns ``(cell_columns, fracs_long)`` where ``fracs_long`` is
    ``(h3, landcover, frac)``.
    """
    names = class_name_map(config)
    hard = [int(c) for c in cfg("landcover.undevelopable_classes", config)]
    conditional = [int(c) for c in cfg("landcover.conditional_undevelopable_classes", config)]

    stats = zonal(raster, cells, [_MAJORITY, _FRAC, _UNIQUE])
    stats = stats.set_index("h3").reindex(cells["h3"].astype(str))

    forest_share = _forest_share(cells, forest, crs_metric)
    rows: list[dict[str, Any]] = []
    frac_rows: list[dict[str, Any]] = []
    for position, (h3_id, record) in enumerate(stats.iterrows()):
        codes = np.asarray(record.get(_UNIQUE) if record.get(_UNIQUE) is not None else [])
        fracs = np.asarray(record.get(_FRAC) if record.get(_FRAC) is not None else [], dtype=float)
        total = float(fracs.sum()) if fracs.size else 0.0
        per_class: dict[str, float] = {}
        if total > 0:
            for code, frac in zip(codes.tolist(), (fracs / total).tolist()):
                name = names.get(int(code))
                if name is None:
                    logger.warning("unmapped WorldCover class %s in cell %s", code, h3_id)
                    continue
                per_class[name] = per_class.get(name, 0.0) + float(frac)
        hard_frac = sum(
            per_class.get(names[c], 0.0) for c in hard if c in names
        )
        cond_frac = sum(per_class.get(names[c], 0.0) for c in conditional if c in names)
        gated = hard_frac + cond_frac * float(forest_share[position])
        majority_code = record.get(_MAJORITY)
        majority = names.get(int(majority_code)) if pd.notna(majority_code) else None
        rows.append(
            {
                "h3": h3_id,
                LANDCOVER_COLUMN: majority,
                UNDEVELOPABLE_COLUMN: min(gated, 1),
                "_covered": total > 0,
                "_tree_frac": cond_frac,
            }
        )
        for name, frac in sorted(per_class.items()):
            frac_rows.append({"h3": h3_id, LANDCOVER_COLUMN: name, "frac": frac})

    out = pd.DataFrame(rows)
    uncovered = ~out["_covered"].to_numpy(dtype=bool)
    if uncovered.any():
        fallback = (
            out.loc[~uncovered, LANDCOVER_COLUMN].mode().iat[0]
            if (~uncovered).any()
            else S.LANDCOVER_CLASSES[0]
        )
        out.loc[uncovered, LANDCOVER_COLUMN] = fallback
        out.loc[uncovered, UNDEVELOPABLE_COLUMN] = 0.0
    out = mark_imputed(out, LANDCOVER_COLUMN, uncovered, "worldcover_gap_city_majority")
    out = mark_imputed(out, UNDEVELOPABLE_COLUMN, uncovered, "worldcover_gap_zero")

    # Tree cover with no declared forest boundary: the conditional gate cannot be resolved.
    if forest is None or not len(forest):
        unresolved = out["_tree_frac"].to_numpy(dtype=float) > 0
        out = mark_imputed(
            out, UNDEVELOPABLE_COLUMN, unresolved, "tree_cover_no_forest_boundary"
        )

    fracs = pd.DataFrame(frac_rows, columns=["h3", LANDCOVER_COLUMN, "frac"])
    return out.drop(columns=["_covered", "_tree_frac"]), fracs


def landcover_fracs_long(fracs: pd.DataFrame) -> pd.DataFrame:
    """The ``cell_landcover_fracs`` side table, one row per (cell, class)."""
    return fracs.reset_index(drop=True)


def assert_fracs_sum_to_one(fracs: pd.DataFrame, *, config: Any = None) -> None:
    """Section 6 ACCEPTANCE: "sum of ``landcover_fracs`` = 1 +- 1e-3"."""
    tolerance = float(cfg("landcover.frac_sum_tolerance", config))
    totals = fracs.groupby("h3")["frac"].sum()
    bad = totals[(totals - 1).abs() > tolerance]
    if len(bad):
        raise UFEError(
            f"{len(bad)} cell(s) have landcover fractions that do not sum to 1 within "
            f"{tolerance}: worst {bad.abs().max()}"
        )


class LandcoverIngester(Ingester):
    """ESA WorldCover 10 m -> ``landcover``, ``undevelopable_frac`` (+ fractions table)."""

    source_id = "esa_worldcover"
    tier = "national"
    fills = (LANDCOVER_COLUMN, UNDEVELOPABLE_COLUMN)
    spatial_res = "10 m"
    temporal_res = "annual (2021 vintage)"
    notes = (
        "Majority class and per-class area fractions from a single exactextract pass "
        "(majority/frac/unique). Tree cover is gated only inside a declared forest "
        "boundary; without one the conditional gate is flagged imputed (Section 6.2)."
    )

    def keys(self, city: CityConfig) -> tuple[str, ...]:
        return (KEY_LANDCOVER, KEY_FOREST)

    def parse(self, raw: Path) -> pd.DataFrame:
        """A one-row manifest naming the raster; the raster itself is read in ``to_cells``."""
        return pd.DataFrame([{"path": str(raw)}])

    def to_cells(self, df: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
        if self.city is None:
            raise ValueError("LandcoverIngester needs a CityConfig to know crs_metric")
        forest = (
            self.reader.vector(KEY_FOREST) if self.reader.exists(KEY_FOREST) else None
        )
        columns, fracs = landcover_to_cells(
            df["path"].iat[0],
            cells,
            crs_metric=self.city.crs_metric,
            forest=forest,
            config=self.config,
        )
        #: Side output, keyed for the caller to persist as ``cell_landcover_fracs``.
        self.side_tables: Mapping[str, pd.DataFrame] = {FRACS_TABLE: fracs}
        return columns
