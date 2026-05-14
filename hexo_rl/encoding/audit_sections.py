"""Encoding audit section emitters — §176 P60 extract.

Pure-move extraction from hexo_rl/encoding/audit.py (P59 left audit.py at
939 LOC; this split keeps audit.py focused on AuditReport dataclasses +
top-level orchestration).

Section emitters (canonical order):
    §1 _section_registered    — registered encodings
    §2 _section_checkpoints   — .pt files declared vs inferred
    §3 _section_corpora       — .npz files sidecar vs filename heuristic
    §4 _section_variants      — variant configs resolve under registry
    §6 _section_cross_table   — ckpts ↔ corpora via sha256 (INV-1..6)

§5 (_section_hardcode) lives in hexo_rl/encoding/_hardcode_scan.py (P59).

Output format is byte-identical to pre-P60 — tests strict-mode pin format.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Optional

from hexo_rl.encoding import (
    EncodingRegistryError,
    all_specs,
    resolve_from_config,
)
from hexo_rl.encoding.audit import (
    AuditReport,
    AuditSection,
    CheckpointEntry,
    CorpusEntry,
    Severity,
)
from hexo_rl.encoding.compat import infer_encoding_from_state_dict
from hexo_rl.encoding.registry import _load as _load_registry


# §173 T10 — deliberately-unstamped dead checkpoint directories.
# These prefixes are skipped in §2 checkpoint audit (info, not error).
_DEAD_CKPT_PREFIXES: tuple[str, ...] = (
    "checkpoints/broken/",
    "checkpoints/collapsed_",
    "checkpoints/olod_20k/",
    "checkpoints/chain_planes",
    "checkpoints/v9_s3/",
    "checkpoints/pretrain/",
    "checkpoints/w4c_smoke_v7_laptop_preflight/",
)


# ---------------------------------------------------------------------------
# Section 1 — Registered encodings
# ---------------------------------------------------------------------------


def _section_registered(report: AuditReport) -> None:
    sect = AuditSection(
        title="§1 Registered encodings",
        headers=(
            "name",
            "board_size",
            "n_planes",
            "policy_logits",
            "multi_window",
            "schema_v",
        ),
    )
    try:
        specs = list(all_specs())
    except EncodingRegistryError as e:
        report.add_finding("error", "§1", f"registry load failed: {e}")
        sect.notes.append(f"REGISTRY ERROR: {e}")
        report.sections["§1"] = sect
        return
    for s in specs:
        sect.rows.append(
            [
                s.name,
                str(s.board_size),
                str(s.n_planes),
                str(s.policy_logit_count),
                str(s.is_multi_window),
                str(s.schema_version),
            ]
        )
    report.add_finding("info", "§1", f"{len(specs)} encoding(s) registered")
    report.sections["§1"] = sect


# ---------------------------------------------------------------------------
# Section 2 — Checkpoints
# ---------------------------------------------------------------------------


def _extract_state_dict(obj: object) -> dict | None:
    if isinstance(obj, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # Fallback: maybe the dict itself is a state_dict (tensor values).
        try:
            import torch

            if all(hasattr(v, "shape") for v in obj.values()):
                return obj  # type: ignore[return-value]
        except Exception:
            pass
    return None


def _section_checkpoints(
    report: AuditReport,
    ckpt_dir: Path,
    out_entries: list["CheckpointEntry"],
) -> None:
    sect = AuditSection(
        title="§2 Checkpoints",
        headers=("path", "declared", "inferred", "status"),
    )
    if not ckpt_dir.is_dir():
        sect.notes.append(f"checkpoints dir {ckpt_dir} not found — skipped")
        report.add_finding("warn", "§2", f"checkpoints dir {ckpt_dir} missing")
        report.sections["§2"] = sect
        return

    pts = sorted(p for p in ckpt_dir.rglob("*.pt") if p.is_file())
    if not pts:
        sect.notes.append("no .pt files found")
        report.sections["§2"] = sect
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        sect.notes.append("torch not importable — checkpoint load skipped")
        report.add_finding("warn", "§2", "torch import failed; cannot load .pt files")
        report.sections["§2"] = sect
        return

    import torch

    def _is_dead_ckpt(rel_path: str) -> bool:
        return any(rel_path.startswith(prefix) for prefix in _DEAD_CKPT_PREFIXES)

    for p in pts:
        rel = str(p.relative_to(ckpt_dir.parent) if p.is_relative_to(ckpt_dir.parent) else p)

        if _is_dead_ckpt(rel):
            sect.rows.append([rel, "-", "-", "ALLOWLISTED"])
            report.add_finding("info", "§2", f"{rel}: skipped (allowlisted dead dir)")
            continue

        declared = "-"
        inferred = "-"
        status = "?"
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            sect.rows.append([rel, "-", "-", f"LOAD-ERR ({type(e).__name__})"])
            report.add_finding("error", "§2", f"failed to load {rel}: {e}")
            continue

        meta = obj.get("metadata") if isinstance(obj, dict) else None
        enc_name: Optional[str] = None
        corpus_sha: Optional[str] = None
        has_meta = isinstance(meta, dict)
        if has_meta:
            enc_name = meta.get("encoding_name") if isinstance(meta.get("encoding_name"), str) else None  # type: ignore[union-attr]
            corpus_sha = meta.get("corpus_sha256") if isinstance(meta.get("corpus_sha256"), str) else None  # type: ignore[union-attr]
            if enc_name:
                declared = enc_name

        out_entries.append(
            CheckpointEntry(
                path=p,
                encoding_name=enc_name,
                corpus_sha256=corpus_sha,
                has_metadata=has_meta,
            )
        )

        sd = _extract_state_dict(obj) or {}
        try:
            inferred = infer_encoding_from_state_dict(sd, str(p))
        except EncodingRegistryError:
            inferred = "?"

        if declared != "-" and inferred not in ("-", "?"):
            if declared == inferred:
                status = "OK"
                report.add_finding("info", "§2", f"{rel}: declared==inferred ({declared})")
            else:
                status = "MISMATCH"
                report.add_finding(
                    "error",
                    "§2",
                    f"{rel}: declared={declared} inferred={inferred}",
                )
        elif declared != "-":
            # Declared but inference unavailable — still acceptable.
            status = "OK (no-infer)"
            report.add_finding("info", "§2", f"{rel}: declared={declared} (no-infer)")
        elif inferred not in ("-", "?"):
            status = "LEGACY"
            report.add_finding(
                "warn",
                "§2",
                f"{rel}: no metadata, inferred={inferred} (stamp via §172 A5 migration)",
            )
        else:
            status = "UNKNOWN"
            report.add_finding(
                "error",
                "§2",
                f"{rel}: no metadata, no inference — encoding indeterminate",
            )

        sect.rows.append([rel, declared, inferred, status])

    report.sections["§2"] = sect


# ---------------------------------------------------------------------------
# Section 3 — Corpora
# ---------------------------------------------------------------------------


_CORPUS_FILENAME_HEURISTIC: tuple[tuple[str, str], ...] = (
    # Order matters — most-specific first.
    ("v8_canvas_realness", "v8_canvas_realness"),
    ("v6w25", "v6w25"),
    ("v7full", "v7full"),
    ("v7e30", "v7full"),  # historical alias — closest spec
    ("v8full", "v8"),
    ("_v8", "v8"),
    ("_v7", "v7full"),
    ("_v6", "v6"),
)


def _infer_corpus_from_filename(name: str) -> str:
    """Best-effort filename→encoding heuristic; default v6."""
    for needle, encoding in _CORPUS_FILENAME_HEURISTIC:
        if needle in name:
            return encoding
    return "v6"


def _sha256_of_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _section_corpora(
    report: AuditReport,
    corpora_dir: Path,
    out_entries: list["CorpusEntry"],
) -> None:
    sect = AuditSection(
        title="§3 Corpora",
        headers=("path", "declared", "inferred", "sha", "status"),
    )
    if not corpora_dir.is_dir():
        sect.notes.append(f"corpora dir {corpora_dir} not found — skipped")
        report.add_finding("warn", "§3", f"corpora dir {corpora_dir} missing")
        report.sections["§3"] = sect
        return

    npzs = sorted(p for p in corpora_dir.rglob("*.npz") if p.is_file())
    if not npzs:
        sect.notes.append("no .npz files found")
        report.sections["§3"] = sect
        return

    for p in npzs:
        rel = str(p.relative_to(corpora_dir.parent) if p.is_relative_to(corpora_dir.parent) else p)
        sidecar = p.with_suffix(p.suffix + ".metadata.json")
        declared = "-"
        sha_status = "no-sidecar"
        registered = sorted(_load_registry())
        actual_sha: Optional[str] = None
        sidecar_enc: Optional[str] = None
        has_sidecar = sidecar.is_file()
        if has_sidecar:
            try:
                meta = json.loads(sidecar.read_text())
                if not isinstance(meta, dict):
                    raise ValueError("metadata is not a JSON object")
                if "encoding_name" not in meta:
                    raise ValueError("missing encoding_name")
                declared = str(meta["encoding_name"])
                sidecar_enc = declared
                expected_sha = meta.get("sha256")
                if isinstance(expected_sha, str):
                    actual_sha = _sha256_of_file(p)
                    if actual_sha == expected_sha:
                        sha_status = "OK"
                    else:
                        sha_status = "MISMATCH"
                        report.add_finding(
                            "error",
                            "§3",
                            f"{rel}: sidecar sha256={expected_sha[:12]}… "
                            f"actual={actual_sha[:12]}…",
                        )
                else:
                    sha_status = "no-sha-in-sidecar"
                    # Still compute actual sha for cross-table joins.
                    actual_sha = _sha256_of_file(p)
                    report.add_finding(
                        "warn",
                        "§3",
                        f"{rel}: sidecar lacks sha256 field",
                    )
                if declared not in registered:
                    report.add_finding(
                        "error",
                        "§3",
                        f"{rel}: sidecar encoding_name={declared!r} not registered "
                        f"(registered={registered})",
                    )
            except Exception as e:
                report.add_finding(
                    "error",
                    "§3",
                    f"{rel}: sidecar parse failed: {e}",
                )
                declared = "PARSE-ERR"
                sha_status = "?"
        else:
            # No sidecar — still compute sha for cross-table join.
            actual_sha = _sha256_of_file(p)
            report.add_finding(
                "warn",
                "§3",
                f"{rel}: no sidecar (.metadata.json); stamp via §172 A5",
            )

        out_entries.append(
            CorpusEntry(
                path=p,
                encoding_name=sidecar_enc,
                sha256=actual_sha,
                has_sidecar=has_sidecar,
            )
        )

        inferred = _infer_corpus_from_filename(p.name)
        if declared not in ("-", "PARSE-ERR") and inferred != declared:
            report.add_finding(
                "warn",
                "§3",
                f"{rel}: filename heuristic ({inferred}) disagrees with sidecar "
                f"({declared}); operator should confirm.",
            )

        if sha_status == "OK" and declared not in ("-", "PARSE-ERR"):
            status = "OK"
        elif declared in ("-", "PARSE-ERR"):
            status = "LEGACY"
        else:
            status = "PARTIAL"

        sect.rows.append([rel, declared, inferred, sha_status, status])

    report.sections["§3"] = sect


# ---------------------------------------------------------------------------
# Section 4 — Variants
# ---------------------------------------------------------------------------


def _section_variants(report: AuditReport, variants_dir: Path) -> None:
    sect = AuditSection(
        title="§4 Variants",
        headers=("path", "resolved", "status"),
    )
    if not variants_dir.is_dir():
        sect.notes.append(f"variants dir {variants_dir} not found — skipped")
        report.add_finding("warn", "§4", f"variants dir {variants_dir} missing")
        report.sections["§4"] = sect
        return

    yamls = sorted(p for p in variants_dir.glob("*.yaml") if p.is_file())
    if not yamls:
        sect.notes.append("no variant yaml files found")
        report.sections["§4"] = sect
        return

    try:
        import yaml
    except ImportError:
        sect.notes.append("PyYAML not installed — variants skipped")
        report.add_finding("warn", "§4", "PyYAML import failed")
        report.sections["§4"] = sect
        return

    for p in yamls:
        rel = str(p.relative_to(variants_dir.parent) if p.is_relative_to(variants_dir.parent) else p)
        try:
            cfg = yaml.safe_load(p.read_text()) or {}
        except Exception as e:
            sect.rows.append([rel, "-", f"YAML-ERR ({type(e).__name__})"])
            report.add_finding("error", "§4", f"{rel}: yaml parse failed: {e}")
            continue

        if not isinstance(cfg, dict):
            sect.rows.append([rel, "-", "NOT-A-MAPPING"])
            report.add_finding("error", "§4", f"{rel}: top-level is not a mapping")
            continue

        resolved = "-"
        status = "?"
        with warnings.catch_warnings(record=True) as wlog:
            warnings.simplefilter("always")
            try:
                spec = resolve_from_config(cfg)
                resolved = spec.name
                deprecation = [
                    str(w.message)
                    for w in wlog
                    if issubclass(w.category, DeprecationWarning)
                ]
                if deprecation:
                    status = "DEPRECATED"
                    for msg in deprecation:
                        report.add_finding(
                            "warn",
                            "§4",
                            f"{rel}: deprecation — {msg.splitlines()[0]}",
                        )
                else:
                    status = "OK"
                    report.add_finding(
                        "info",
                        "§4",
                        f"{rel}: resolved={resolved}",
                    )
            except EncodingRegistryError as e:
                status = "SCHEMA-ERR"
                report.add_finding("error", "§4", f"{rel}: {e}")
            except Exception as e:
                status = f"ERR ({type(e).__name__})"
                report.add_finding(
                    "error", "§4", f"{rel}: unexpected error: {e}"
                )

        sect.rows.append([rel, resolved, status])

    report.sections["§4"] = sect


# ---------------------------------------------------------------------------
# Section 6 — Cross-table consistency (ckpts ↔ corpora via sha256)
# ---------------------------------------------------------------------------


def _section_cross_table(
    report: AuditReport,
    ckpts: list[CheckpointEntry],
    corpora: list[CorpusEntry],
    corpora_dir: Path,
) -> None:
    """INV-1..6 per amended design §11.6.

    Joins ckpt.corpus_sha256 → corpus.sha256 (actual file hash).
    Emits per-row findings plus an INV-6 sweep for orphan corpora.
    """
    sect = AuditSection(
        title="§6 Cross-table consistency",
        headers=("checkpoint", "ckpt_enc", "corpus_sha", "corpus_enc", "status"),
    )

    # Skip rule: corpora_dir directory is empty (no entries at all) AND at
    # least one ckpt cites a corpus_sha256.  An empty dir means there is
    # genuinely nothing to join against; a dir that has files but no .npz
    # is a different situation (sha simply resolves to nothing — INV-2).
    _dir_empty = corpora_dir.is_dir() and not any(corpora_dir.iterdir())
    if _dir_empty and any(ck.corpus_sha256 for ck in ckpts):
        report.add_finding(
            "warn",
            "§6",
            "§6 skipped — no corpora under --corpora-dir to join against",
        )
        report.sections["§6"] = sect
        return

    by_sha: dict[str, CorpusEntry] = {
        c.sha256: c for c in corpora if c.sha256 is not None
    }

    referenced_shas: set[str] = set()

    for ck in ckpts:
        # Use path stem as display name (keep short).
        rel = ck.path.name

        if not ck.has_metadata:  # INV-4
            sect.rows.append([rel, "-", "-", "-", "NO-META"])
            report.add_finding(
                "warn",
                "§6",
                f"{rel}: no metadata — cannot cross-reference corpus",
            )
            continue

        if ck.corpus_sha256 is None:  # INV-3
            sect.rows.append(
                [rel, ck.encoding_name or "-", "-", "-", "NO-CORPUS-SHA"]
            )
            report.add_finding(
                "warn",
                "§6",
                f"{rel}: metadata lacks corpus_sha256",
            )
            continue

        match = by_sha.get(ck.corpus_sha256)
        if match is None:  # INV-2
            sha_short = ck.corpus_sha256[:12] + "…"
            sect.rows.append(
                [rel, ck.encoding_name or "-", sha_short, "?", "ORPHAN-SHA"]
            )
            report.add_finding(
                "error",
                "§6",
                f"{rel}: corpus_sha256={sha_short} matches no corpus",
            )
            continue

        referenced_shas.add(match.sha256)  # type: ignore[arg-type]
        sha_short = match.sha256[:12] + "…"  # type: ignore[index]

        if match.encoding_name != ck.encoding_name:  # INV-1
            sect.rows.append(
                [
                    rel,
                    ck.encoding_name or "-",
                    sha_short,
                    match.encoding_name or "-",
                    "ENC-MISMATCH",
                ]
            )
            report.add_finding(
                "error",
                "§6",
                f"{rel}: ckpt encoding={ck.encoding_name!r} but corpus encoding={match.encoding_name!r}",
            )
        else:  # INV-5
            sect.rows.append(
                [
                    rel,
                    ck.encoding_name or "-",
                    sha_short,
                    match.encoding_name or "-",
                    "OK",
                ]
            )
            report.add_finding(
                "info",
                "§6",
                f"{rel} ↔ {match.path.name}: {ck.encoding_name}",
            )

    # INV-6 — orphan corpora (sha unreferenced by any stamped ckpt).
    for co in corpora:
        if co.sha256 is not None and co.sha256 not in referenced_shas:
            sev: Severity = "warn" if report.strict else "info"
            report.add_finding(
                sev,
                "§6",
                f"{co.path.name}: orphan corpus — unused by any stamped checkpoint",
            )

    report.sections["§6"] = sect
