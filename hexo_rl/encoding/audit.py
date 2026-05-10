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


_HARDCODE_HITS_DUMP = Path("/tmp/encoding_audit_hardcode_hits.txt")


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

    for p in pts:
        rel = str(p.relative_to(ckpt_dir.parent) if p.is_relative_to(ckpt_dir.parent) else p)
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
)
_TEST_FILE_HINTS: tuple[str, ...] = ("test", "fixtures", "fixture")
# Patterns that look like version tokens or in-string literals; allow.
_VERSION_RE = re.compile(r"v\d+\b")


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


def _scan_file(path: Path) -> list[tuple[int, str, list[str]]]:
    """Return [(lineno, line, hits)] for unjustified hits in `path`."""
    suffix = path.suffix
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    out: list[tuple[int, str, list[str]]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if _line_is_allowlisted(line, suffix):
            continue
        hits = _hits_outside_strings(line)
        if hits:
            out.append((i, line.rstrip(), hits))
    return out


def _section_hardcode(report: AuditReport, repo_root: Path) -> None:
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
    # Persist full dump to /tmp.
    try:
        with _HARDCODE_HITS_DUMP.open("w") as fh:
            for p in sorted(file_hits):
                fh.write(f"# {p}\n")
                for lineno, line, hits in file_hits[p]:
                    fh.write(f"  L{lineno}  hits={hits}  | {line}\n")
                fh.write("\n")
    except OSError:
        pass

    # Render compact table (top-N per file).
    for p in sorted(file_hits):
        hits = file_hits[p]
        rel = str(p.relative_to(repo_root) if p.is_relative_to(repo_root) else p)
        n = sum(len(h[2]) for h in hits)
        preview = "; ".join(f"L{ln}={hh}" for ln, _, hh in hits[:3])
        sect.rows.append([rel, str(n), preview])

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
) -> AuditReport:
    """Run the full 6-section audit; return AuditReport."""
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
    _section_hardcode(report, repo_root or _repo_root())
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

    ns = parser.parse_args(list(argv) if argv is not None else None)
    report = audit(
        ns.checkpoints_dir,
        ns.corpora_dir,
        ns.variants_dir,
        strict=ns.strict,
        repo_root=ns.repo_root,
    )
    print(report)
    return report.exit_code()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
