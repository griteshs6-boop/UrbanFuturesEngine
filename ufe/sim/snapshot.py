"""Snapshot access and run provenance (spec Section 3.8, Section 15.1, Section 23 item 5).

Section 23 item 5: "Every number in the output traces to a snapshot hash, a params hash,
and a git commit." This module is where those three come from.

``ufe/store/db.py`` already owns the snapshot *format* — ``SnapshotRef``, ``write_snapshot``,
``read_snapshot_table``, ``read_manifest``, ``snapshot_hash``. Nothing here duplicates it.
What this module adds is the read side a simulation needs:

``open_snapshot``
    Turn a path (or an existing ``SnapshotRef``) into a verified ``SnapshotRef``, checking
    the recorded ``snapshot_hash`` against the bytes actually on disk.
``SnapshotData``
    The three frames a run consumes — ``cells``, ``projects``, ``announcers`` — loaded once.
``Provenance`` / ``resolve_provenance``
    snapshot hash + params hash + git commit, with the Section 23 refusal to run against an
    unknown or dirty code state unless the caller overrides it explicitly.

**No numeric literals.** The one policy knob (``require_clean_git``) and the short-hash
length come from ``config/params/simulation.yaml``.

**No I/O beyond the local store.** ``git rev-parse`` is a local subprocess against the
working copy, run once before the simulation starts, never inside the annual loop.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ufe.errors import UFEError
from ufe.store.db import (
    SnapshotRef,
    file_hash,
    read_manifest,
    read_snapshot_table,
    snapshot_hash,
)

__all__ = [
    "ProvenanceError",
    "Provenance",
    "SnapshotData",
    "SnapshotRef",
    "open_snapshot",
    "load_snapshot_data",
    "git_commit",
    "git_is_dirty",
    "resolve_provenance",
    "UNKNOWN_COMMIT",
    "REPO_ROOT",
]

logger = logging.getLogger(__name__)

#: Placeholder recorded in the manifest when the code version cannot be established.
UNKNOWN_COMMIT = "unknown"

#: Repository root, derived from this file's location (``ufe/sim/snapshot.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Tables a simulation reads out of a snapshot, in load order.
SNAPSHOT_TABLES: tuple[str, ...] = ("cells", "projects", "announcers")

#: Files inside a snapshot directory that are not part of the hashed payload.
MANIFEST_NAME = "MANIFEST.json"

P_REQUIRE_CLEAN_GIT = "simulation.provenance.require_clean_git"
P_SHORT_HASH_LENGTH = "simulation.provenance.short_hash_length"


class ProvenanceError(UFEError):
    """The run's provenance is incomplete or unreproducible (Section 23 item 5).

    Lives here rather than in ``ufe/errors.py`` only because that file belongs to another
    agent; it is reported for promotion in the build summary.
    """


# --------------------------------------------------------------------------------------
# snapshot access
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotData:
    """The frames one simulation reads, loaded once from an immutable snapshot."""

    ref: SnapshotRef
    cells: pd.DataFrame
    projects: pd.DataFrame
    announcers: pd.DataFrame
    manifest: Mapping[str, Any]

    @property
    def snapshot_hash(self) -> str:
        return self.ref.snapshot_hash


def _iter_payload_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)


def open_snapshot(
    snapshot: SnapshotRef | str | Path, *, verify: bool = True
) -> SnapshotRef:
    """Resolve `snapshot` to a :class:`SnapshotRef`, optionally re-hashing the bytes.

    `verify=True` recomputes ``snapshot_hash`` from the files on disk and raises
    :class:`ProvenanceError` if it disagrees with ``MANIFEST.json``. That is the check that
    makes "every number traces to a snapshot hash" mean something: a snapshot edited after
    the fact no longer opens.
    """
    if isinstance(snapshot, SnapshotRef):
        ref = snapshot
    else:
        root = Path(snapshot)
        if not root.is_dir():
            raise ProvenanceError(f"snapshot directory does not exist: {root}")
        manifest = read_manifest(root)
        ref = SnapshotRef(
            snapshot_id=str(manifest["snapshot_id"]),
            snapshot_hash=str(manifest["snapshot_hash"]),
            path=root,
            city_id=str(manifest["city_id"]),
            params_hash=str(manifest.get("params_hash") or ""),
        )
    if verify:
        recomputed = snapshot_hash(
            file_hash(p) for p in _iter_payload_files(Path(ref.path))
        )
        if recomputed != ref.snapshot_hash:
            raise ProvenanceError(
                f"snapshot {ref.snapshot_id} has been modified since it was written: "
                f"recomputed hash {recomputed} != recorded {ref.snapshot_hash} "
                "(spec Section 3.8: a snapshot is immutable)"
            )
    return ref


def load_snapshot_data(
    snapshot: SnapshotRef | str | Path, *, verify: bool = True
) -> SnapshotData:
    """Read `cells`, `projects` and `announcers` out of a snapshot. The only legal read."""
    ref = open_snapshot(snapshot, verify=verify)
    frames = {name: read_snapshot_table(ref, name) for name in SNAPSHOT_TABLES}
    return SnapshotData(
        ref=ref,
        cells=frames["cells"],
        projects=frames["projects"],
        announcers=frames["announcers"],
        manifest=read_manifest(ref),
    )


# --------------------------------------------------------------------------------------
# git provenance
# --------------------------------------------------------------------------------------


def _git(args: list[str], repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError) as exc:  # pragma: no cover - git absent from the image
        logger.warning("git %s failed: %s", " ".join(args), exc)
        return None
    if completed.returncode != 0:
        logger.info(
            "git %s exited %d: %s",
            " ".join(args),
            completed.returncode,
            completed.stderr.strip(),
        )
        return None
    return completed.stdout


def git_commit(repo_root: Path | str = REPO_ROOT) -> str:
    """``git rev-parse HEAD``, or :data:`UNKNOWN_COMMIT` when it cannot be determined.

    A repository with no commits yet, a tarball export, or an image without git all yield
    :data:`UNKNOWN_COMMIT` — which :func:`resolve_provenance` then refuses to run against
    unless the caller overrides.
    """
    out = _git(["rev-parse", "HEAD"], Path(repo_root))
    return out.strip() if out and out.strip() else UNKNOWN_COMMIT


def git_is_dirty(repo_root: Path | str = REPO_ROOT) -> bool:
    """True when the working tree has uncommitted or untracked changes.

    ``None`` is not an option: if git cannot answer, the state is *not known clean*, so this
    returns True and the caller must decide.
    """
    out = _git(["status", "--porcelain"], Path(repo_root))
    if out is None:
        return True
    return out.strip() != ""


# --------------------------------------------------------------------------------------
# the provenance triple
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Section 23 item 5's triple, plus the audit trail for how it was obtained."""

    snapshot_id: str
    snapshot_hash: str
    params_hash: str
    code_version: str
    code_dirty: bool
    city_id: str
    #: True when the caller explicitly accepted an unknown/dirty code state.
    dirty_override: bool = False

    @property
    def complete(self) -> bool:
        """Every number in the output can be traced: all three identifiers are known."""
        return bool(
            self.snapshot_hash
            and self.params_hash
            and self.code_version
            and self.code_version != UNKNOWN_COMMIT
        )

    def short(self, length: int) -> dict[str, str]:
        """Truncated identifiers, for report headers."""
        n = int(length)
        return {
            "snapshot_hash": self.snapshot_hash[:n],
            "params_hash": self.params_hash[:n],
            "code_version": self.code_version[:n],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            "params_hash": self.params_hash,
            "code_version": self.code_version,
            "code_dirty": bool(self.code_dirty),
            "city_id": self.city_id,
            "dirty_override": bool(self.dirty_override),
            "complete": bool(self.complete),
        }

    def to_json(self) -> str:
        """Stable, sorted JSON — part of the byte-identical run output."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def resolve_provenance(
    snapshot: SnapshotRef,
    params: Any,
    *,
    allow_dirty: bool = False,
    repo_root: Path | str = REPO_ROOT,
    code_version: str | None = None,
    code_dirty: bool | None = None,
) -> Provenance:
    """Assemble the provenance triple and enforce Section 23 item 5.

    Raises :class:`ProvenanceError` when ``simulation.provenance.require_clean_git`` is
    true and the code version is unknown or the working tree is dirty, unless
    `allow_dirty=True`. The override is recorded on the returned :class:`Provenance` so the
    manifest — and therefore any report built from it — says so out loud.

    `code_version` / `code_dirty` exist so a Monte Carlo worker process, or a test, can pass
    an already-resolved answer instead of re-shelling out to git for every draw.
    """
    commit = git_commit(repo_root) if code_version is None else str(code_version)
    dirty = git_is_dirty(repo_root) if code_dirty is None else bool(code_dirty)

    provenance = Provenance(
        snapshot_id=snapshot.snapshot_id,
        snapshot_hash=snapshot.snapshot_hash,
        params_hash=params.hash,
        code_version=commit,
        code_dirty=dirty,
        city_id=snapshot.city_id,
        dirty_override=bool(allow_dirty),
    )

    require_clean = bool(params.value(P_REQUIRE_CLEAN_GIT))
    if not require_clean:
        return provenance

    problems: list[str] = []
    if commit == UNKNOWN_COMMIT:
        problems.append("the git commit could not be determined")
    if dirty:
        problems.append("the working tree has uncommitted changes")
    if not provenance.params_hash:
        problems.append("the params hash is empty")
    if not provenance.snapshot_hash:
        problems.append("the snapshot hash is empty")

    if problems and not allow_dirty:
        raise ProvenanceError(
            "refusing to run: "
            + "; ".join(problems)
            + ". Spec Section 23 item 5 requires every number in the output to trace to a "
            "snapshot hash, a params hash and a git commit. Commit the working tree, or "
            f"pass allow_dirty=True (CLI: --allow-dirty) to record the gap in the manifest "
            f"instead. Set {P_REQUIRE_CLEAN_GIT} to false to disable this check entirely."
        )
    if problems:
        logger.warning(
            "provenance is incomplete but allow_dirty=True: %s", "; ".join(problems)
        )
    return provenance


def with_params_hash(snapshot: SnapshotRef, params: Any) -> SnapshotRef:
    """A ``SnapshotRef`` carrying `params`' hash, for snapshots written without one."""
    return replace(snapshot, params_hash=params.hash)
