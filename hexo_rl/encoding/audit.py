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
    6. Cross-table — ckpts ↔ corpora via sha256 (INV-1..6).

Exit codes (per §11):
    0 — every section clean (only `info` findings).
    1 — at least one `warn`, no `error`.
    2 — at least one `error`.

Invocation:
    python -m hexo_rl.encoding audit

Implementation split (§176 P59 + P60):
    §1-§4 + §6 emitters → hexo_rl/encoding/audit_sections.py
    §5 emitter         → hexo_rl/encoding/_hardcode_scan.py
This module keeps AuditReport dataclasses, _render_table, top-level audit()
orchestration, and the argparse main().
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence


Severity = Literal["info", "warn", "error"]


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
# Section emitters — extracted to sibling modules (§176 P59 + P60).
# Re-exported here so existing callers / tests keep working.
# ---------------------------------------------------------------------------
from hexo_rl.encoding._hardcode_scan import _section_hardcode  # noqa: E402,F401
from hexo_rl.encoding.audit_sections import (  # noqa: E402,F401
    _section_checkpoints,
    _section_corpora,
    _section_cross_table,
    _section_registered,
    _section_variants,
)


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
