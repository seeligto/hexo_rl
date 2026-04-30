import pathlib
import re

STALE_PATTERNS = [
    r'24\s*\*\s*19\s*\*\s*19',
    r'24,\s*19,\s*19',
    r'24-plane',
    r'planes 18\.\.23',
]

EXCLUDE = (
    'reports/',
    'archive/',
    'docs/07_PHASE4_SPRINT_LOG.md',
    'tests/test_no_stale_plane_refs.py',
    # historical release artifact — accurate record of v0.4.0 24-plane era
    'releases/',
    'docs/releases/',
    # backward-compat corpus loader (intentionally handles legacy 24-plane NPZ)
    'hexo_rl/training/batch_assembly.py',
    # persist.rs v3/v4 version history comment
    'engine/src/replay_buffer/persist.rs',
    # batcher regression test documents the old wrong default
    'engine/tests/batcher_default.rs',
    # preflight test validates 24-plane checkpoint rejection
    'tests/test_preflight.py',
    # Q25 resolution history in open questions doc
    'docs/06_OPEN_QUESTIONS.md',
    # phase 3.5 historical benchmark note in roadmap
    'docs/02_roadmap.md',
    # CLAUDE.md Q25 resolved-checklist entry (historical context)
    'CLAUDE.md',
    # §115 moved CLAUDE.md Q25 entry here (historical context preserved)
    'docs/rules/phase-4-architecture.md',
    # FUTURE_REFACTORS.md tracks done/pending items including resolved 24-plane tasks
    'FUTURE_REFACTORS.md',
    # torchinductor compiled cache — generated, not source
    '.torchinductor-cache/',
    # agent memory files — not source, not reviewed
    '.claude/',
    # historical fixture changelog — documents v1→v6 24-plane→18-plane migration
    'fixtures/threat_probe_baseline.CHANGELOG.md',
    # P3 migration handoff — references A-003 (24-plane regression test) as historical context
    'docs/notes/p3_model_migration_handoff.md',
)


def test_no_stale_plane_refs() -> None:
    root = pathlib.Path(__file__).parent.parent
    bad = []
    for ext in ('.py', '.rs', '.md', '.yaml', '.yml'):
        for p in root.rglob(f'*{ext}'):
            rel = str(p.relative_to(root))
            if any(rel.startswith(e) or rel == e for e in EXCLUDE):
                continue
            text = p.read_text(errors='ignore')
            for pat in STALE_PATTERNS:
                for m in re.finditer(pat, text):
                    bad.append(f'{rel}: {m.group(0)!r}')
    assert not bad, 'Stale 24-plane references found:\n' + '\n'.join(bad)
