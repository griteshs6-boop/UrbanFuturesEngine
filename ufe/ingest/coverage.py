"""Section 20.2 step 9 — the coverage report and the refusal gate.

    9. [check] Coverage report: fraction of cells with real vs imputed values, per column.
               Refuse to proceed if price coverage < 40% of populated cells.

Two functions, one gate:

:func:`coverage_report`   per column, over the *populated* cells: how many values are real,
                          how many imputed, how many missing entirely, and the fractions.
:func:`assert_coverage`   raises :class:`ufe.errors.CoverageError` when a column's real
                          fraction is below its threshold. The threshold is read from
                          ``config/ingest.yaml`` (``coverage.min_real_fraction``), never a
                          literal, so the spec's 40% is data and can be changed without
                          touching Python.

"Real vs imputed" is answered from the imputation ledger, not by counting nulls. Every
ingester marks what it filled but did not observe (see
:func:`ufe.ingest.core.mark_imputed`), those flags are melted into
``cell_imputation(h3, column, imputed, method)``, and this module reads that. A column can
therefore be fully populated and still show 0% real coverage — which is exactly the
situation the gate exists to catch, and which a null-count report would miss entirely.

"Populated cells" means in-city cells carrying at least
``coverage.populated_min_population`` people, which is the denominator Section 20.2 names
("40% of populated cells"). The report also carries the unrestricted counts so an operator
can see whether a low fraction is a data problem or a denominator problem.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ufe.errors import CoverageError
from ufe.ingest.core import cfg, imputation_long

logger = logging.getLogger(__name__)

__all__ = [
    "REPORT_COLUMNS",
    "COVERAGE_TABLE",
    "populated_mask",
    "coverage_report",
    "assert_coverage",
    "coverage_thresholds",
    "write_coverage_report",
    "format_report",
]

COVERAGE_TABLE = "coverage_report"
REPORT_COLUMNS: tuple[str, ...] = (
    "column",
    "cells_total",
    "cells_populated",
    "n_real",
    "n_imputed",
    "n_missing",
    "frac_real",
    "frac_imputed",
    "frac_missing",
    "threshold",
    "passes",
    "methods",
)


def coverage_thresholds(config: Any = None) -> dict[str, float]:
    """``{column: minimum real fraction}`` from ``config/ingest.yaml``."""
    return {
        str(k): float(v) for k, v in cfg("coverage.min_real_fraction", config).items()
    }


def populated_mask(cells: pd.DataFrame, *, config: Any = None) -> np.ndarray:
    """In-city cells with at least ``coverage.populated_min_population`` people."""
    minimum = float(cfg("coverage.populated_min_population", config))
    population = pd.to_numeric(cells.get("population"), errors="coerce").fillna(0.0)
    in_city = (
        cells["in_city"].astype(bool).to_numpy()
        if "in_city" in cells.columns
        else np.ones(len(cells), dtype=bool)
    )
    return in_city & (population.to_numpy(dtype=float) >= minimum)


def _imputation_index(imputation: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ledger to one row per (h3, column) with the union of the flags."""
    if imputation is None or not len(imputation):
        return pd.DataFrame(columns=["h3", "column", "imputed", "method"])
    frame = imputation.copy()
    frame["imputed"] = frame["imputed"].astype(bool)
    frame["method"] = frame.get("method", "").fillna("")
    aggregated = (
        frame.groupby(["h3", "column"], as_index=False)
        .agg(
            imputed=("imputed", "max"),
            method=("method", lambda s: ";".join(sorted({x for x in s if x}))),
        )
    )
    return aggregated


def coverage_report(
    cells: pd.DataFrame,
    imputation: pd.DataFrame | None = None,
    *,
    columns: Sequence[str] | None = None,
    config: Any = None,
) -> pd.DataFrame:
    """The Section 20.2 step 9 report: real vs imputed vs missing, per column.

    ``imputation`` is the tidy ``cell_imputation`` ledger. If a frame carrying
    ``__imputed`` companion columns is passed instead (i.e. raw ingester output), it is
    melted first, so a caller can report on one ingester's result without persisting
    anything.
    """
    thresholds = coverage_thresholds(config)
    always = [str(c) for c in cfg("coverage.always_report", config)]
    mask = populated_mask(cells, config=config)
    populated = cells.loc[mask]
    n_populated = int(mask.sum())

    ledger = imputation
    if ledger is not None and len(ledger) and "column" not in ledger.columns:
        ledger = imputation_long(ledger, source_id="")
    ledger = _imputation_index(ledger if ledger is not None else pd.DataFrame())

    wanted = list(columns) if columns is not None else list(always)
    populated_ids = set(populated["h3"].astype(str)) if "h3" in populated.columns else set()

    rows: list[dict[str, Any]] = []
    for column in wanted:
        threshold = thresholds.get(column)
        if column not in cells.columns:
            rows.append(
                {
                    "column": column,
                    "cells_total": len(cells),
                    "cells_populated": n_populated,
                    "n_real": 0,
                    "n_imputed": 0,
                    "n_missing": n_populated,
                    "frac_real": 0.0,
                    "frac_imputed": 0.0,
                    "frac_missing": 1.0 if n_populated else 0.0,
                    "threshold": threshold,
                    "passes": threshold is None,
                    "methods": "column_absent",
                }
            )
            continue

        present = populated[column].notna().to_numpy()
        flags = ledger[ledger["column"] == column]
        flagged = set(flags.loc[flags["imputed"], "h3"].astype(str)) & populated_ids
        is_flagged = populated["h3"].astype(str).isin(flagged).to_numpy()

        n_real = int((present & ~is_flagged).sum())
        n_imputed = int((present & is_flagged).sum())
        n_missing = int((~present).sum())
        denominator = n_populated if n_populated else 1
        methods = ";".join(sorted({m for m in flags["method"] if m}))
        frac_real = n_real / denominator
        rows.append(
            {
                "column": column,
                "cells_total": len(cells),
                "cells_populated": n_populated,
                "n_real": n_real,
                "n_imputed": n_imputed,
                "n_missing": n_missing,
                "frac_real": frac_real,
                "frac_imputed": n_imputed / denominator,
                "frac_missing": n_missing / denominator,
                "threshold": threshold,
                "passes": True if threshold is None else frac_real >= threshold,
                "methods": methods,
            }
        )
    return pd.DataFrame(rows, columns=list(REPORT_COLUMNS))


def assert_coverage(report: pd.DataFrame) -> None:
    """Section 20.2 step 9's refusal. Raises :class:`ufe.errors.CoverageError`.

    Only columns with a declared threshold can fail; everything else is informational. The
    message names the column, its real fraction and its threshold, because the operator's
    next action ("licence a price feed for these localities") depends on which one failed.
    """
    failures = report[(report["threshold"].notna()) & (~report["passes"].astype(bool))]
    if not len(failures):
        return
    detail = "; ".join(
        f"{row['column']}: {row['frac_real']:.1%} real of {int(row['cells_populated'])} "
        f"populated cells, below the required {float(row['threshold']):.0%}"
        for _, row in failures.iterrows()
    )
    raise CoverageError(
        "refusing to proceed past the Section 20.2 step 9 coverage gate — " + detail
    )


def format_report(report: pd.DataFrame) -> str:
    """The report as a table, for the CLI. Uses ``rich`` where available."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:  # pragma: no cover - rich is a declared dependency
        return report.to_string(index=False)
    table = Table(title="Coverage report (Section 20.2 step 9)")
    for column in ("column", "n_real", "n_imputed", "n_missing", "frac_real", "threshold", "passes"):
        table.add_column(column)
    for _, row in report.iterrows():
        table.add_row(
            str(row["column"]),
            str(int(row["n_real"])),
            str(int(row["n_imputed"])),
            str(int(row["n_missing"])),
            f"{float(row['frac_real']):.1%}",
            "-" if pd.isna(row["threshold"]) else f"{float(row['threshold']):.0%}",
            "yes" if bool(row["passes"]) else "NO",
        )
    console = Console(record=True)
    console.print(table)
    return console.export_text()


def write_coverage_report(con: Any, report: pd.DataFrame, *, city_id: str) -> pd.DataFrame:
    """Persist the report so a run's coverage is auditable after the fact."""
    payload = report.copy()
    payload.insert(0, "city_id", city_id)
    con.execute(
        f'CREATE TABLE IF NOT EXISTS "{COVERAGE_TABLE}" ('
        "city_id VARCHAR, \"column\" VARCHAR, cells_total BIGINT, cells_populated BIGINT, "
        "n_real BIGINT, n_imputed BIGINT, n_missing BIGINT, frac_real DOUBLE, "
        "frac_imputed DOUBLE, frac_missing DOUBLE, threshold DOUBLE, passes BOOLEAN, "
        "methods VARCHAR)"
    )
    con.register("_ufe_coverage", payload)
    quoted = ", ".join(f'"{c}"' for c in payload.columns)
    con.execute(f'INSERT INTO "{COVERAGE_TABLE}" ({quoted}) SELECT {quoted} FROM _ufe_coverage')
    con.unregister("_ufe_coverage")
    return payload
