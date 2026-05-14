"""Encoding audit CLI — §172 A5 subagent 3.

Single-shot operator tool surfacing project-wide encoding posture.
Pre-flight gate before sustained runs; post-A8 wired into CI.

Design: docs/designs/encoding_registry_design.md §11.

Five report sections:
    1. Registered encodings — dump of every spec from registry.
    2. Checkpoints       — `.pt` files, declared (metadata) vs inferred shape.
    3. Corpora           — `.npz` files, sidecar declared vs filename inferred,
                           sha256 reconciliation.
    4. Variants          — variant config resolves under registry.
    5. Hardcoded literals — grep for `19`, `25`, `361`, `5`, `8` outside
                           an allowlist.

Exit codes (per §11):
    0 — every section clean (only `info` findings).
    1 — at least one `warn`, no `error`.
    2 — at least one `error`.

Invocation:
    python -m hexo_rl.encoding audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

from hexo_rl.encoding import (
    EncodingRegistryError,
    all_specs,
    resolve_from_config,
)
from hexo_rl.encoding.compat import infer_encoding_from_state_dict
from hexo_rl.encoding.registry import _load as _load_registry


Severity = Literal["info", "warn", "error"]


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


_HARDCODE_HITS_DUMP = Path("/tmp/encoding_audit_hardcode_hits.txt")

# JSON output format (--format=json):
# {
#   "registry_specs": [{"name":..., "board_size":..., "n_planes":...,
#                        "policy_logits":..., "multi_window":..., "schema_v":...}, ...],
#   "checkpoints": [{"path":..., "declared":..., "inferred":..., "status":...}, ...],
#   "corpora": [{"path":..., "declared":..., "inferred":..., "sha":..., "status":...}, ...],
#   "variants": [{"path":..., "resolved":..., "status":...}, ...],
#   "hardcode_hits": [{"file":..., "line":..., "hits":[...]}, ...],
#   "cross_table": [{"checkpoint":..., "ckpt_enc":..., "corpus_sha":...,
#                    "corpus_enc":..., "status":...}, ...],
#   "findings": [{"severity":..., "section":..., "message":...}, ...],
#   "summary": {"info":..., "warn":..., "error":..., "strict":..., "exit_code":...}
# }
# hardcode_hits replaces the /tmp side-channel; each entry is a per-line record
# (not per-file-summary), allowing callers to process the raw hit list.
# Text-mode behaviour (print(report)) is unchanged.


# ---------------------------------------------------------------------------
# AuditReport / AuditFinding
# ---------------------------------------------------------------------------


@dataclass
class AuditFinding:
    severity: Severity
    section: str
    message: str


@dataclass
class AuditSection:
    title: str
    rows: list[list[str]] = field(default_factory=list)
    headers: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    sections: dict[str, AuditSection] = field(default_factory=dict)
    findings: list[AuditFinding] = field(default_factory=list)
    strict: bool = False
    # Raw hardcode hits collected by _section_hardcode; populated when
    # json_mode=True so the caller can embed them in JSON output instead
    # of writing to /tmp.  Format: list of {"file", "line", "hits"} dicts.
    _raw_hardcode_hits: list[dict] = field(default_factory=list)

    def add_finding(self, severity: Severity, section: str, message: str) -> None:
        self.findings.append(AuditFinding(severity, section, message))

    def has_error(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def has_warn(self) -> bool:
        return any(f.severity == "warn" for f in self.findings)

    def exit_code(self) -> int:
        if self.has_error():
            return 2
        if self.has_warn():
            return 1
        return 0

    def __str__(self) -> str:
        out: list[str] = []
        out.append("=" * 76)
        out.append("encoding audit — hexo_rl.encoding")
        out.append("=" * 76)
        for sect in self.sections.values():
            out.append("")
            out.append(f"-- {sect.title} " + "-" * max(0, 70 - len(sect.title)))
            if sect.headers:
                out.append(_render_table(sect.headers, sect.rows))
            elif sect.rows:
                for r in sect.rows:
                    out.append("  " + " | ".join(r))
            for note in sect.notes:
                out.append(f"  {note}")
        out.append("")
        out.append("-- summary " + "-" * 65)
        ctr = Counter(f.severity for f in self.findings)
        out.append(
            f"  info={ctr.get('info', 0)} warn={ctr.get('warn', 0)} "
            f"error={ctr.get('error', 0)}  strict={self.strict}"
        )
        for f in self.findings:
            if f.severity == "info":
                continue
            out.append(f"  [{f.severity.upper()}] {f.section}: {f.message}")
        out.append(f"  exit_code = {self.exit_code()}")
        return "\n".join(out)

    def to_json_dict(self) -> dict:
        """Serialise audit results to a JSON-compatible dict.

        Shape mirrors the text sections; see module-level docstring for
        full field list.  hardcode_hits carries the raw per-line records
        (replaces the /tmp side-channel when --format=json is active).
        """
        # §1 — registry specs
        registry_rows = self.sections.get("§1")
        registry_specs: list[dict] = []
        if registry_rows and registry_rows.rows:
            keys = ("name", "board_size", "n_planes", "policy_logits",
                    "multi_window", "schema_v")
            for row in registry_rows.rows:
                registry_specs.append(dict(zip(keys, row)))

        # §2 — checkpoints
        ckpt_section = self.sections.get("§2")
        checkpoints: list[dict] = []
        if ckpt_section and ckpt_section.rows:
            keys2 = ("path", "declared", "inferred", "status")
            for row in ckpt_section.rows:
                checkpoints.append(dict(zip(keys2, row)))

        # §3 — corpora
        corpus_section = self.sections.get("§3")
        corpora: list[dict] = []
        if corpus_section and corpus_section.rows:
            keys3 = ("path", "declared", "inferred", "sha", "status")
            for row in corpus_section.rows:
                corpora.append(dict(zip(keys3, row)))

        # §4 — variants
        var_section = self.sections.get("§4")
        variants: list[dict] = []
        if var_section and var_section.rows:
            keys4 = ("path", "resolved", "status")
            for row in var_section.rows:
                variants.append(dict(zip(keys4, row)))

        # §5 — hardcode hits: raw per-line records (replaces /tmp dump)
        hardcode_hits: list[dict] = list(self._raw_hardcode_hits)

        # §6 — cross-table
        xt_section = self.sections.get("§6")
        cross_table: list[dict] = []
        if xt_section and xt_section.rows:
            keys6 = ("checkpoint", "ckpt_enc", "corpus_sha", "corpus_enc", "status")
            for row in xt_section.rows:
                cross_table.append(dict(zip(keys6, row)))

        # findings + summary
        ctr = Counter(f.severity for f in self.findings)
        return {
            "registry_specs": registry_specs,
            "checkpoints": checkpoints,
            "corpora": corpora,
            "variants": variants,
            "hardcode_hits": hardcode_hits,
            "cross_table": cross_table,
            "findings": [
                {"severity": f.severity, "section": f.section, "message": f.message}
                for f in self.findings
            ],
            "summary": {
                "info": ctr.get("info", 0),
                "warn": ctr.get("warn", 0),
                "error": ctr.get("error", 0),
                "strict": self.strict,
                "exit_code": self.exit_code(),
            },
        }


@dataclass(frozen=True)
class CheckpointEntry:
    path: Path
    encoding_name: Optional[str]   # from metadata block
    corpus_sha256: Optional[str]   # from metadata block
    has_metadata: bool


@dataclass(frozen=True)
class CorpusEntry:
    path: Path
    encoding_name: Optional[str]   # from sidecar
    sha256: Optional[str]          # actual file sha (computed)
    has_sidecar: bool


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "  (empty)"
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    lines: list[str] = []
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    lines.append(fmt.format(*headers))
    lines.append("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        padded = list(r) + [""] * (len(widths) - len(r))
        lines.append(fmt.format(*padded))
    return "\n".join(lines)


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
# Section 5 — Hardcoded literals
# ---------------------------------------------------------------------------


# Word-boundary number matcher; integer-only. Captures `19`, `25`, `361`, `5`, `8`.
_NUM_PATTERN = re.compile(r"\b(19|25|361|5|8)\b")
_HARDCODE_TARGETS: tuple[str, ...] = ("19", "25", "361", "5", "8")
_SKIP_DIRS: frozenset[str] = frozenset(
    {"__pycache__", "target", ".venv", "node_modules", "data", "checkpoints", ".git", "vendor", "build"}
)
_ALLOW_TOKENS: tuple[str, ...] = (
    "spec.",
    "EncodingSpec",
    "encoding.lookup",
    "registry::lookup",
    "enc-literal-ok",
    "audit: legacy-v6-fallback",   # §173 A7 — doc-commented legacy PyO3 fallback paths
    "#[pyo3(signature",            # §173 A7 — PyO3 signature defaults are not geometry violations
    "GroupNorm",                   # §173 A7 — GroupNorm group count is NN arch, not encoding geometry
    "GN_GROUPS",                   # §173 A7 — same
    "gn_group",                    # §173 A7 — same (fn param)
)
_TEST_FILE_HINTS: tuple[str, ...] = ("test", "fixtures", "fixture")
# Patterns that look like version tokens or in-string literals; allow.
_VERSION_RE = re.compile(r"v\d+\b")

# ---------------------------------------------------------------------------
# New allowlist constants (rules 1–10)
# ---------------------------------------------------------------------------

# Rule 5 — tunable hyperparameter tokens: skip any line containing these names.
_TUNABLE_TOKENS: frozenset[str] = frozenset({
    "c_puct", "fpu_reduction", "dirichlet_alpha", "dirichlet_epsilon",
    "temp_min", "eta_min", "timeout", "interval", "poll_interval",
    "figsize", "linewidth", "weight_for",
    # §173 A7 additions — tunable MCTS / training knobs + display/logging params
    "leaf_batch_size",     # MCTS batch size (tuned per host via sweep)
    "zoi_margin",          # zone-of-interest search margin (tunable)
    "max_train_burst",     # training throughput cap (tunable)
    "hard_abort_grad_norm_steps",  # gradient-norm abort window (tunable)
    "backup_count",        # log-rotation file count (infra, not geometry)
    "batch_size",          # generic inference batch (tunable unless guarded)
    "max_frames",          # animation/display frame count (not geometry)
    "fail_gb",             # disk-guard threshold (not geometry)
    "gn_groups",           # GroupNorm groups (NN arch knob, not encoding)
    "gn_group",                    # §173 A7 — same (fn param)
    "epochs",              # training epoch count (tunable CLI arg)
    "n_workers",           # worker pool count (tunable, used in CPU budget)
    "budget",              # CPU thread budget expression
    "divisor",             # CPU budget divisor expression
    "skipped_nonfinite",   # error counter (training loop)
    "len(batch)",          # batch accumulator size check
    "len(wdr)",            # display worker count for terminal UI
    "board.ply",           # game-ply threshold (not encoding geometry)
    "human_seeding_max_move",  # corpus opening-move seeding param (not encoding)
    "max_move",            # opening game length limit (not encoding geometry)
    "has_player_long_run", # game-rule threat probe (run-length ≠ encoding geometry)
    "hex_distance",        # game-coordinate distance function (not encoding geometry)
    "max_pages",           # scraper page limit (infra, not geometry)
    "jitter",              # MCTS jitter radii (tunable)
    "stride5",             # stride-5 detector step (game-rule constant)
    "pages",               # CLI page argument (infra)
})
# Guard: if a line also contains these tokens, do NOT suppress even if _TUNABLE_TOKENS hit.
# These are high-risk encoding constants that must still flag.
_TUNABLE_SKIP_GUARD_TOKENS: frozenset[str] = frozenset({
    "feature_len", "policy_len",
})

# Rule 2 — float tolerances like 1e-5, 1e-8, 2.5e-4.
_FLOAT_TOL_RE = re.compile(r"\d+(?:\.\d+)?[eE]-\d+")

# Rule 2b — decimal fraction literals like 0.5, 1.5, 0.25, 0.8 (§173 A7).
# These are probability/float scalars; `\b5\b` inside `0.5` is a false positive.
# Pattern: digit(s) DOT digit(s) — strips the whole token so `0.5` doesn't leave
# a bare `5` for the word-boundary scanner.
_DECIMAL_FRAC_RE = re.compile(r"\d+\.\d+")

# Rule 10 — display / infra context patterns (§173 A7): strip before scanning.
#   10a: Python sequence-slice notation `[:N]` and `[N:]` — display limits, not geometry.
#   10b: Rust Vec::with_capacity(N) — pre-allocation hint, not geometry.
#   10c: matplotlib / display kwargs like fontsize=N, dpi=N, alpha=N, linewidth=N.
#   10d: round(expr, N) precision argument.
#   10e: most_common(N), top-N display slices.
_SLICE_RE     = re.compile(r"\[:\s*\d+\s*\]")                    # 10a
_WITH_CAP_RE  = re.compile(r"\bwith_capacity\s*\(\s*\d+\s*\)")   # 10b
_DISPLAY_KW_RE = re.compile(                                       # 10c
    r"\b(?:fontsize|dpi|alpha|linewidth|markersize|rotation|zorder"
    r"|s=|edgelinewidth)\s*=\s*\d+"
)
_ROUND_PREC_RE = re.compile(r"\bround\s*\([^,]+,\s*\d+\s*\)")    # 10d
_TOP_N_RE      = re.compile(r"\bmost_common\s*\(\s*\d+\s*\)")     # 10e
# 10f — Rust byte-buffer declarations like `[0u8; 8]`, `[0i32; 5]`.
_BYTE_BUF_RE  = re.compile(r"\[\s*0[ui]\d+\s*;\s*\d+\s*\]")      # 10f
# 10g — Rust multi-line string continuation lines: these start with whitespace
#   and the content is text (no `=`, `let`, `fn`, etc.).  Heuristic: lines that
#   start with ≥12 spaces and contain only non-code prose patterns after the
#   spaces.  Use a simpler signal: lines ending with `\` (Rust string continue).
_RUST_STR_CONT_RE = re.compile(r"\\\s*$")                          # 10g
# 10h — Python sequence constructor `[N, N, N]` list literals that are clearly
#   not geometry (mixed values, or values outside the target set).
#   Strip repetition multiplier like `"-" * 25`.
_STR_REPEAT_RE = re.compile(r"\"\s*-*\s*\"\s*\*\s*\d+")           # 10h
# 10i — Python indexing `variable[N]` where context suggests tuple/list field
#   access, not an encoding plane index.  Conservative: only match `row[N]`,
#   `p[N]` in for-comprehension contexts, not `tensor[k, N]` (geometry-likely).
_ROW_IDX_RE = re.compile(r"\b(?:row|p)\[(\d+)\]")                 # 10i
# 10j — Rust match arms `N => {` — the integer is a channel/enum discriminant,
#   not an encoding geometry value being passed to a constructor.
_RUST_MATCH_ARM_RE = re.compile(r"^\s*\d+\s*=>\s*\{")             # 10j
# 10k — §N section references in report paths and inline text (`§5`, `§122`).
#   These survive multi-line-string continuation after the `\` is stripped.
_SECTION_REF_RE = re.compile(r"§\d+")                              # 10k
# 10l — byte-size arithmetic like `4 + 8 + 2` (u32 + u64 + u16 byte widths).
#   Pattern: `N + M` where N and M are single-digit; only in `.rs` files.
#   (Already covered by actual hits in persist.rs L199 only; keep conservative.)
_ENTRY_BYTES_RE = re.compile(r"\bpolicy_bytes\s*\+")               # 10l
# 10m — mixed-value collection literals with ≥ 4 elements containing a value
#   outside the encoding target set (i.e. a number ≥ 10 that is not 19, 25, or
#   361).  These are ply-count / step-number / milestone sequences, not
#   geometry constructors.  Guard: ONLY strip when element count ≥ 4 (safe
#   threshold that avoids `(8, 19, 19)` shape tuples with 3 elements).
# Pattern:  [\(\[] number (, number){3,} [\)\]]  — 4+ element sequence.
_MIXED_COLLECTION_RE = re.compile(
    r"[\(\[]\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+(?:\s*,\s*\d+)*\s*[\)\]]"
)

# Rule 4 — range bounds like `0..5`, `0..=8`, `0..19`.
_RANGE_BOUND_RE = re.compile(r"\b\d+\s*\.\.\s*=?\s*\d+\b")

# Rule 9 — whole-file allowlist (these ARE the sources of truth).
# §173 A7: added encoding/mod.rs (canonical TOML parser; EncodingSpec V6/V6W25
# const structs are the source of truth), and dataset_v{6w25,8}.py (corpus
# dataset files whose encoding parameters are intentional dataset attributes).
_FULL_FILE_ALLOWLIST: frozenset[str] = frozenset({
    "hexo_rl/utils/constants.py",
    "engine/src/encoding/mod.rs",          # §173 A7 — canonical TOML parser / EncodingSpec defs
    "hexo_rl/bootstrap/dataset_v6w25.py",  # §173 A7 — intentional v6w25 dataset attributes
    "hexo_rl/bootstrap/dataset_v8.py",     # §173 A7 — intentional v8 dataset attributes
    "hexo_rl/model/tf32.py",               # §173 A7 — CUDA compute-capability detection; no encoding geometry
    "hexo_rl/utils/encoding.py",           # §174 — canonical v6/v6w25/v8 EncodingSpec factory
    "engine/src/replay_buffer/sym_tables.rs",  # §174 — canonical static sym-table data
})

# Rule 3 — coordinate-call patterns; strip the match before scanning.
_COORD_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\bapply_move\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)"),
    re.compile(r"\bcells\.insert\s*\(\s*\("),
    re.compile(r"\bset_(?:row|col|diag)_\w+\s*\("),
    re.compile(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\.into\(\)"),
)

# Rule 8 — canonical-define lines are the source of truth; skip them.
# Matches both `BOARD_SIZE: usize = 19` (no prefix) and `pub const BOARD_SIZE: usize = 19`
# and `static SIZE19_8: Lazy<SymTables> = ...` (Rust static lazy).
_CANONICAL_DEFINE_RE = re.compile(
    r"^\s*(?:pub\s+(?:const\s+)?|const\s+|static\s+)?"
    r"(BOARD_SIZE|NUM_CELLS|BUFFER_CHANNELS|N_ACTIONS|MARGIN_M|HISTORY_LEN"
    r"|N_PLANES|N_CHAIN_PLANES|BOARD_H|BOARD_W|TOTAL_CELLS"
    # §173 A7 additions — v6w25 canonical constants + derived helper bodies
    r"|BOARD_H_V6W25|BOARD_W_V6W25|N_CELLS_V6W25|N_PLANES_V6W25"
    r"|N_ACTIONS_V6W25|STATE_STRIDE_V6W25|CHAIN_STRIDE_V6W25"
    r"|POLICY_STRIDE_V6W25|AUX_STRIDE_V6W25"
    r"|CLUSTER_THRESHOLD_V6W25|LEGAL_MOVE_RADIUS_V6W25"
    r"|KEPT_PLANE_INDICES"
    r"|DEFAULT_LEGAL_MOVE_RADIUS|DEFAULT_CLUSTER_THRESHOLD"
    # §174 additions — additional canonical constants used in production / Rust
    r"|WIRE_CHANNELS|_V8_OFF_WINDOW_PLANE_DEFAULT|_STRIDE5_STEP"
    r"|JITTER_RADII|MAX_PAGES"
    r"|OPP_STONE_PLANE|MOVES_REMAINING_PLANE|PLY_PARITY_PLANE"
    # Pure infrastructure constants unrelated to encoding geometry
    r"|MAX_RETRIES"
    r")\s*[:=\[]"
)

# Rule 7 — trailing comment patterns.
_RUST_LINE_COMMENT_RE = re.compile(r"//.*$")
_PYTHON_LINE_COMMENT_RE = re.compile(r"#.*$")


def _is_test_path(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()
    if any(h in name for h in _TEST_FILE_HINTS):
        return True
    return any(h in parts for h in _TEST_FILE_HINTS)


def _looks_like_comment(line: str, suffix: str) -> bool:
    stripped = line.lstrip()
    if suffix == ".py" and stripped.startswith("#"):
        return True
    if suffix == ".rs" and stripped.startswith("//"):
        return True
    return False


def _line_is_allowlisted(line: str, suffix: str) -> bool:
    if any(tok in line for tok in _ALLOW_TOKENS):
        return True
    if _looks_like_comment(line, suffix):
        return True
    return False


def _hits_outside_strings(line: str) -> list[str]:
    """Find target literals in `line` that are NOT inside string-quoted spans
    and not preceded by `v` (version tokens like v6, v8 → skip even if not
    matched by `\\b`).  Conservative: misses some edge cases but doesn't fire
    on `"v6"`, `"size_19"`, etc."""
    # Strip quoted spans first.
    cleaned = re.sub(r"\"[^\"]*\"|'[^']*'", "", line)
    # Drop version tokens.
    cleaned = _VERSION_RE.sub("", cleaned)
    return _NUM_PATTERN.findall(cleaned)


# ---------------------------------------------------------------------------
# Rule 1 — test-range helpers
# ---------------------------------------------------------------------------


def _test_ranges_rust(lines: list[str]) -> frozenset[int]:
    """Return 1-based line numbers inside #[cfg(test)] mod tests { ... } blocks."""
    in_test = False
    brace_depth = 0
    skip: set[int] = set()
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not in_test:
            if stripped in ("#[cfg(test)]", "#[cfg(test)]") or "mod tests" in stripped:
                in_test = True
                brace_depth = stripped.count("{") - stripped.count("}")
                skip.add(i)
                continue
        if in_test:
            skip.add(i)
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth <= 0:
                in_test = False
                brace_depth = 0
    return frozenset(skip)


def _test_and_docstring_ranges_python(lines: list[str]) -> frozenset[int]:
    """Return 1-based line numbers inside class Test*/def test_* bodies and docstrings."""
    skip: set[int] = set()
    in_docstring = False
    in_test_block = False
    test_indent: Optional[int] = None

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # --- docstring tracking (rule 6) ---
        if not in_docstring:
            if '"""' in stripped:
                count = stripped.count('"""')
                if count >= 2:
                    # Opened and closed on same line — skip just this line.
                    skip.add(i)
                else:
                    in_docstring = True
                    skip.add(i)
            elif "'''" in stripped:
                count = stripped.count("'''")
                if count >= 2:
                    skip.add(i)
                else:
                    in_docstring = True
                    skip.add(i)
        else:
            skip.add(i)
            if '"""' in stripped or "'''" in stripped:
                in_docstring = False
            continue  # inside docstring — no further analysis needed

        # --- test-block tracking (rule 1) ---
        if stripped.startswith("class Test") or stripped.startswith("def test_"):
            # Measure indent of this header line.
            leading = len(line) - len(line.lstrip())
            in_test_block = True
            test_indent = leading
            skip.add(i)
        elif in_test_block:
            current_indent = len(line) - len(line.lstrip()) if stripped else test_indent + 1  # type: ignore[operator]
            if stripped == "" or current_indent > test_indent:  # type: ignore[operator]
                skip.add(i)
            else:
                in_test_block = False
                test_indent = None

    return frozenset(skip)


# ---------------------------------------------------------------------------
# Line-level transform helpers (rules 2, 3, 4, 7)
# ---------------------------------------------------------------------------


def _strip_trailing_comment_rust(line: str) -> str:
    """Remove `// ...` comment from a Rust line (not inside strings)."""
    # Find `//` that is NOT preceded by a string-open. Simple heuristic:
    # walk left-to-right tracking string state.
    in_str = False
    str_char = ""
    for idx in range(len(line) - 1):
        ch = line[idx]
        if not in_str:
            if ch in ('"', "'"):
                in_str = True
                str_char = ch
            elif ch == "/" and line[idx + 1] == "/":
                return line[:idx]
        else:
            if ch == str_char and (idx == 0 or line[idx - 1] != "\\"):
                in_str = False
    return line


def _strip_trailing_comment_python(line: str) -> str:
    """Remove `# ...` comment from a Python line (not inside strings)."""
    in_str = False
    str_char = ""
    for idx, ch in enumerate(line):
        if not in_str:
            if ch in ('"', "'"):
                in_str = True
                str_char = ch
            elif ch == "#":
                return line[:idx]
        else:
            if ch == str_char and (idx == 0 or line[idx - 1] != "\\"):
                in_str = False
    return line


def _line_has_coord_pattern(line: str) -> bool:
    """Return True if line contains any coordinate-call pattern (rule 3)."""
    return any(pat.search(line) for pat in _COORD_PATTERNS)


def _apply_line_transforms(line: str, suffix: str) -> str:
    """Apply strip transforms to reduce false positives before scanning."""
    # Rule 7: strip trailing comments.
    if suffix == ".rs":
        line = _strip_trailing_comment_rust(line)
    elif suffix == ".py":
        line = _strip_trailing_comment_python(line)
    # Rule 2: strip float tolerance patterns.
    line = _FLOAT_TOL_RE.sub("", line)
    # Rule 2b: strip decimal fraction literals (0.5, 1.5, 0.25, 0.8, …).
    line = _DECIMAL_FRAC_RE.sub("", line)
    # Rule 4: strip range bounds.
    line = _RANGE_BOUND_RE.sub("", line)
    # Rule 10: strip display / infra context patterns.
    line = _SLICE_RE.sub("[]", line)       # 10a — sequence slices
    line = _WITH_CAP_RE.sub("", line)      # 10b — Rust pre-alloc hints
    line = _DISPLAY_KW_RE.sub("", line)    # 10c — matplotlib kwargs
    line = _ROUND_PREC_RE.sub("", line)    # 10d — round() precision
    line = _TOP_N_RE.sub("", line)         # 10e — most_common()
    line = _BYTE_BUF_RE.sub("", line)      # 10f — Rust byte-buffer declarations
    if _RUST_STR_CONT_RE.search(line):     # 10g — Rust string continuation lines
        return ""
    line = _STR_REPEAT_RE.sub("", line)    # 10h — string repeat `"-" * N`
    line = _ROW_IDX_RE.sub("", line)       # 10i — row/tuple field indexing
    if _RUST_MATCH_ARM_RE.match(line):     # 10j — Rust match arm `N => {`
        return ""
    line = _SECTION_REF_RE.sub("", line)  # 10k — §N section references
    if _ENTRY_BYTES_RE.search(line):       # 10l — buffer entry byte-layout arithmetic
        return ""
    if _MIXED_COLLECTION_RE.search(line):  # 10m — mixed-value collections (not pure geometry)
        return ""
    return line


def _scan_file(path: Path) -> list[tuple[int, str, list[str]]]:
    """Return [(lineno, line, hits)] for unjustified hits in `path`."""
    suffix = path.suffix
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    lines = text.splitlines()

    # Pre-compute skip ranges (rule 1).
    if suffix == ".rs":
        skip_linenos = _test_ranges_rust(lines)
    elif suffix == ".py":
        skip_linenos = _test_and_docstring_ranges_python(lines)
    else:
        skip_linenos = frozenset()

    out: list[tuple[int, str, list[str]]] = []
    for i, line in enumerate(lines, start=1):
        # Rule 1: skip test/docstring ranges.
        if i in skip_linenos:
            continue
        # Existing allowlist checks (spec., EncodingSpec, comment-only lines).
        if _line_is_allowlisted(line, suffix):
            continue
        # Rule 5: skip lines with tunable tokens (unless guard tokens also present).
        if (any(tok in line for tok in _TUNABLE_TOKENS)
                and not any(g in line for g in _TUNABLE_SKIP_GUARD_TOKENS)):
            continue
        # Rule 8: skip canonical-define lines in both Rust and Python.
        # Python canonical source files (constants.py, encoding.py) are handled
        # by rule 9 whole-file allowlist; this catches stray module-level
        # canonical constants elsewhere.
        if suffix in (".py", ".rs") and _CANONICAL_DEFINE_RE.match(line):
            continue
        # Rule 3: skip lines containing coordinate-call patterns.
        if _line_has_coord_pattern(line):
            continue
        # Apply strip transforms (rules 2, 4, 7).
        transformed = _apply_line_transforms(line, suffix)
        hits = _hits_outside_strings(transformed)
        if hits:
            out.append((i, line.rstrip(), hits))
    return out


def _section_hardcode(report: AuditReport, repo_root: Path, *, collect_raw: bool = False) -> None:
    sect = AuditSection(
        title="§5 Hardcoded literals",
        headers=("file", "hits", "lines (top 3)"),
    )
    targets = [
        repo_root / "engine" / "src",
        repo_root / "hexo_rl",
    ]
    file_hits: dict[Path, list[tuple[int, str, list[str]]]] = {}
    total_hits = 0
    for root in targets:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix not in (".py", ".rs"):
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            if _is_test_path(p):
                continue
            # Rule 9: whole-file allowlist.
            rel_str = str(p.relative_to(repo_root) if p.is_relative_to(repo_root) else p)
            # Normalise to forward-slash for cross-platform matching.
            if rel_str.replace("\\", "/") in _FULL_FILE_ALLOWLIST:
                continue
            hits = _scan_file(p)
            if hits:
                file_hits[p] = hits
                total_hits += sum(len(h[2]) for h in hits)

    if not file_hits:
        sect.notes.append("no unjustified hardcode hits found")
        report.add_finding("info", "§5", "no unjustified hits")
        report.sections["§5"] = sect
        return

    severity: Severity = "error" if report.strict else "warn"

    # collect_raw=True: store per-line dicts on report (used by --format=json).
    # Also skips the /tmp side-channel write — JSON payload carries the data.
    if collect_raw:
        for p in sorted(file_hits):
            rel = str(p.relative_to(repo_root) if p.is_relative_to(repo_root) else p)
            for lineno, line, hits in file_hits[p]:
                report._raw_hardcode_hits.append(
                    {"file": rel, "line": lineno, "content": line, "hits": hits}
                )
    else:
        # Text mode: persist full dump to /tmp side-channel.
        try:
            with _HARDCODE_HITS_DUMP.open("w") as fh:
                for p in sorted(file_hits):
                    fh.write(f"# {p}\n")
                    for lineno, line, hits in file_hits[p]:
                        fh.write(f"  L{lineno}  hits={hits}  | {line}\n")
                    fh.write("\n")
        except OSError:
            pass

    # Render compact table (top-N per file) — same for both modes.
    for p in sorted(file_hits):
        hits = file_hits[p]
        rel = str(p.relative_to(repo_root) if p.is_relative_to(repo_root) else p)
        n = sum(len(h[2]) for h in hits)
        preview = "; ".join(f"L{ln}={hh}" for ln, _, hh in hits[:3])
        sect.rows.append([rel, str(n), preview])

    if collect_raw:
        report.add_finding(
            severity,
            "§5",
            f"{total_hits} unjustified literal hit(s) across {len(file_hits)} file(s); "
            f"full dump in JSON hardcode_hits",
        )
        sect.notes.append(f"strict={report.strict} → severity={severity}")
        sect.notes.append("full dump: included in JSON hardcode_hits field")
    else:
        report.add_finding(
            severity,
            "§5",
            f"{total_hits} unjustified literal hit(s) across {len(file_hits)} file(s); "
            f"full dump → {_HARDCODE_HITS_DUMP}",
        )
        sect.notes.append(f"strict={report.strict} → severity={severity}")
        sect.notes.append(f"full dump: {_HARDCODE_HITS_DUMP}")
    report.sections["§5"] = sect


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


# ---------------------------------------------------------------------------
# Top-level audit + CLI
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Walk up from this module to repo root (engine/ + hexo_rl/ siblings)."""
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "engine").is_dir() and (ancestor / "hexo_rl").is_dir():
            return ancestor
    return here.parent.parent.parent


def audit(
    checkpoints_dir: Path,
    corpora_dir: Path,
    variants_dir: Optional[Path] = None,
    *,
    strict: bool = False,
    repo_root: Path | None = None,
    collect_raw_hardcode: bool = False,
) -> AuditReport:
    """Run the full 6-section audit; return AuditReport.

    collect_raw_hardcode: when True, store per-line hardcode hit dicts on
    report._raw_hardcode_hits (used by --format=json) and skip /tmp write.
    """
    report = AuditReport(strict=strict)
    _section_registered(report)
    ckpt_entries: list[CheckpointEntry] = []
    corpus_entries: list[CorpusEntry] = []
    _section_checkpoints(report, checkpoints_dir, ckpt_entries)
    _section_corpora(report, corpora_dir, corpus_entries)
    if variants_dir is None:
        # Try repo-root-relative default; fall back gracefully to a non-existent path
        # (§4 skips cleanly when dir missing).
        _rroot = repo_root or _repo_root()
        _effective_variants = _rroot / "configs" / "variants"
    else:
        _effective_variants = variants_dir
    _section_variants(report, _effective_variants)
    _section_hardcode(report, repo_root or _repo_root(), collect_raw=collect_raw_hardcode)
    _section_cross_table(report, ckpt_entries, corpus_entries, corpora_dir)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hexo_rl.encoding",
        description="Project-wide encoding-state audit (§172 A5).",
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        default="audit",
        choices=("audit",),
        help="subcommand (default: audit)",
    )
    root = _repo_root()
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=root / "checkpoints",
        help="directory of .pt checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--corpora-dir",
        type=Path,
        default=root / "data",
        help="directory of .npz corpora (default: ./data)",
    )
    parser.add_argument(
        "--variants-dir",
        type=Path,
        default=root / "configs" / "variants",
        help="directory of variant yaml files (default: ./configs/variants)",
    )
    parser.add_argument(
        "--hardcodes-only",
        action="store_true",
        help="run only the hardcoded-literals scan (skips checkpoints, corpora, variants, cross-table)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="promote hardcode hits from warn → error",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="override hardcode-grep root (default: auto-detected repo root)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        dest="output_format",
        help=(
            "output format: 'text' (default, human-readable) or 'json' "
            "(structured JSON to stdout; hardcode hits in 'hardcode_hits' "
            "key instead of /tmp side-channel)"
        ),
    )

    ns = parser.parse_args(list(argv) if argv is not None else None)
    json_mode = ns.output_format == "json"

    if ns.hardcodes_only:
        report = AuditReport(strict=ns.strict)
        _section_hardcode(report, ns.repo_root or _repo_root(), collect_raw=json_mode)
        if json_mode:
            print(json.dumps(report.to_json_dict(), indent=2))
        else:
            print(report)
        return report.exit_code()
    report = audit(
        ns.checkpoints_dir,
        ns.corpora_dir,
        ns.variants_dir,
        strict=ns.strict,
        repo_root=ns.repo_root,
        collect_raw_hardcode=json_mode,
    )
    if json_mode:
        print(json.dumps(report.to_json_dict(), indent=2))
    else:
        print(report)
    return report.exit_code()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
