"""D-F HEADSWAP — WP2 regen tests (RECIPE §"WP2 board recovery").

Non-GPU only. Run:
  .venv/bin/python -m pytest scripts/headswap/test_wp2_regen.py -q

Covers:
  1. recover() zobrist matcher on the 41 WP1 positives using the EXISTING retro_slope
     248k games — the KNOWN-GOOD oracle (RECIPE: WP1 boards reconstruct 41/41 exact).
     If recover_one() matches 41/41 here, the matcher is sound for the WP2 regen path.
  2. book-loading (the 5 WP2 books load + carry the fields run_arm needs).
  3. schema of an emitted negative row == negatives_v1.jsonl keys (harvest_neg output).
  4. generate() is importable/callable + dry-runnable (prints planned run_arm, no exec).

Does NOT run generate()/harvest_neg GPU work — those need the box.
"""
import json
from pathlib import Path

import pytest

from scripts.headswap import wp2_regen
from scripts.headswap.wp2_regen import (
    recover_one, load_regen_games, _game_key, build_parser, cmd_generate,
)

REPO = Path(__file__).resolve().parent.parent.parent
ENC = "v6_live2_ls"
RETRO_GAMES = REPO / "reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl"
PROBE_SET = REPO / "reports/valprobe/probe_set_v1.jsonl"
WP2_BOOKS = REPO / "reports/valprobe/wp2"
NEG_V1 = REPO / "reports/valprobe/negatives_v1.jsonl"


# ── fixtures ──────────────────────────────────────────────────────────────────

def _load_probe(wp):
    rows = [json.loads(l) for l in open(PROBE_SET) if l.strip()]
    return [r for r in rows if r.get("wp") == wp]


def _retro_index():
    """Index the retro_slope games by (None, opening_idx, head_as_p1) — the oracle
    keys the WP1 positives match on (they have no per-batch book_id)."""
    games = [json.loads(l) for l in open(RETRO_GAMES) if l.strip()]
    index = {}
    for g in games:
        index[_game_key(None, g["opening_idx"], g["head_as_p1"])] = g
    return games, index


# ── 1. recover() oracle: 41/41 WP1 exact ─────────────────────────────────────

@pytest.mark.skipif(not RETRO_GAMES.exists() or not PROBE_SET.exists(),
                    reason="oracle data absent")
def test_recover_matches_41_wp1_oracle_exact():
    """recover_one must reconstruct all 41 WP1 boards at the EXACT ply t
    (retro_slope 248k games are the byte-identical source — no Gumbel drift)."""
    wp1 = _load_probe("WP1")
    assert len(wp1) == 41, f"expected 41 WP1 positives, got {len(wp1)}"
    _games, index = _retro_index()

    n_rec = 0
    n_exact = 0
    for target in wp1:
        rec = recover_one(target, index, ENC, match_book_id=False)
        assert rec["zobrist"] == str(target["zobrist"])
        if rec["recovered"]:
            n_rec += 1
            if rec["matched_ply"] == target["t"]:
                n_exact += 1
    assert n_rec == 41, f"recovered {n_rec}/41 WP1 oracle positions"
    # byte-identical source → every match is at the exact ply, not a neighbor
    assert n_exact == 41, f"only {n_exact}/41 matched at exact t"


@pytest.mark.skipif(not RETRO_GAMES.exists() or not PROBE_SET.exists(),
                    reason="oracle data absent")
def test_recover_reports_miss_on_wrong_zobrist():
    """A corrupted target zobrist must be reported recovered=False (matcher not a rubber-stamp)."""
    wp1 = _load_probe("WP1")
    _games, index = _retro_index()
    bad = dict(wp1[0])
    bad["zobrist"] = "123456789"   # not a real board hash
    rec = recover_one(bad, index, ENC, match_book_id=False)
    assert rec["recovered"] is False
    assert rec["matched_ply"] is None


@pytest.mark.skipif(not RETRO_GAMES.exists() or not PROBE_SET.exists(),
                    reason="oracle data absent")
def test_recover_neighbor_scan_finds_shifted_ply():
    """If the target t is off-by-2 but the zobrist is real, the ±neighbor scan recovers it
    (this is the Gumbel-non-determinism drift path the WP2 regen relies on)."""
    wp1 = _load_probe("WP1")
    _games, index = _retro_index()
    shifted = dict(wp1[0])
    shifted["t"] = int(wp1[0]["t"]) + 2   # zobrist still the real one at true t
    rec = recover_one(shifted, index, ENC, match_book_id=False, neighbor_scan=3)
    assert rec["recovered"] is True
    assert rec["matched_ply"] == int(wp1[0]["t"])   # found the true ply via the scan


@pytest.mark.skipif(not RETRO_GAMES.exists() or not PROBE_SET.exists(),
                    reason="oracle data absent")
def test_recover_no_matching_game():
    """No game for the (book_id, opening_idx, head_as_p1) key → recovered=False, reason set."""
    wp1 = _load_probe("WP1")
    _games, index = _retro_index()
    orphan = dict(wp1[0])
    orphan["opening_idx"] = 99999
    rec = recover_one(orphan, index, ENC, match_book_id=False)
    assert rec["recovered"] is False
    assert rec["reason"] == "no_matching_game"


# ── 2. book-loading ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not WP2_BOOKS.exists(), reason="WP2 books absent")
def test_wp2_books_load_and_carry_run_arm_fields():
    books = sorted(WP2_BOOKS.glob("evalfair_r5_wp2_b*.json"))
    assert len(books) == 5, f"expected 5 WP2 books, got {len(books)}"
    for bp in books:
        book = json.loads(bp.read_text())
        # run_arm consumes: book_id, seed, radius_stage, openings[*].moves
        assert book["book_id"] == bp.stem
        assert isinstance(book["seed"], int)
        assert book["radius_stage"] == 5
        assert len(book["openings"]) == 64
        o0 = book["openings"][0]
        assert "moves" in o0 and len(o0["moves"]) == 3


# ── 3. emitted negative-row schema == negatives_v1 keys ──────────────────────

@pytest.mark.skipif(not NEG_V1.exists(), reason="negatives_v1 absent")
def test_harvest_row_schema_matches_negatives_v1():
    """Build one harvest-shaped negative row from constants and assert its key-set +
    sealbot_verify key-set match negatives_v1.jsonl exactly (schema parity gate)."""
    ref = json.loads(next(l for l in open(NEG_V1) if l.strip()))
    ref_keys = set(ref.keys())
    ref_sv_keys = set(ref["sealbot_verify"].keys())

    # Reconstruct the exact row dict harvest_neg emits (mirror cmd_harvest_neg body).
    from scripts.valprobe.measure_recognition_lag import turn_of_ply
    from scripts.valprobe.run_valprobe_sealbot import WINDOW_HALF

    row = {
        "arm": "248k",
        "ckpt_step": wp2_regen.CKPT_STEP,
        "ckpt_sha": wp2_regen.EXPECTED_CKPT_SHA,
        "opening_idx": 0,
        "head_as_p1": True,
        "set": "safe",
        "t": 62,
        "turn": turn_of_ply(62),
        "side_to_move": "head",
        "moves_remaining": 2,
        "zobrist": "12345",
        "grid": "head_turn_start",
        "v_raw": 0.1,
        "ply_band": 6,
        "source": "wp2_regen_b0",
        "wp": "NEG",
        "sealbot_verify": {
            "safe": True,
            "head_score": 1.0,
            "last_score": 1.0,
            "side_to_move_is_head": True,
            "depth": 7,
            "window_half": WINDOW_HALF,
            "colony_skip": False,
            "rung": "sealbot_d7",
            "wall_s": 0.01,
        },
    }
    assert set(row.keys()) == ref_keys, (
        f"harvest row keys differ:\n  extra={set(row)-ref_keys}\n  missing={ref_keys-set(row)}"
    )
    assert set(row["sealbot_verify"].keys()) == ref_sv_keys, (
        f"sealbot_verify keys differ:\n  extra={set(row['sealbot_verify'])-ref_sv_keys}"
        f"\n  missing={ref_sv_keys-set(row['sealbot_verify'])}"
    )


# ── 4. generate() importable + dry-runnable (no GPU) ─────────────────────────

@pytest.mark.skipif(not WP2_BOOKS.exists(), reason="WP2 books absent")
def test_generate_dry_run_prints_planned_calls(capsys, tmp_path):
    """generate --dry-run must enumerate the 5 planned run_arm calls WITHOUT executing
    (no model load, no GPU). Confirms the entrypoint is callable on a non-GPU box."""
    ap = build_parser()
    args = ap.parse_args([
        "generate",
        "--books-dir", str(WP2_BOOKS),
        "--out", str(tmp_path / "regen"),
        "--dry-run",
    ])
    args.func(args)
    out = capsys.readouterr().out
    # one planned line per book, each naming run_arm + ArmSpec('simsdeploy') + sealbot_depth=5
    assert out.count("[GENERATE][dry-run]") == 5
    assert "ArmSpec('simsdeploy')" in out
    assert "sealbot_depth=5" in out
    assert "book_seed=" in out


def test_dry_run_makes_no_games_files(tmp_path):
    """A dry run must not create any games.jsonl (guards accidental live-exec in tests)."""
    ap = build_parser()
    args = ap.parse_args([
        "generate",
        "--books-dir", str(WP2_BOOKS),
        "--out", str(tmp_path / "regen"),
        "--dry-run",
    ])
    args.func(args)
    assert list((tmp_path / "regen").rglob("games.jsonl")) == []


# ── argparse wiring ──────────────────────────────────────────────────────────

def test_parser_has_three_subcommands():
    ap = build_parser()
    for cmd in ("generate", "recover", "harvest_neg"):
        args = ap.parse_args([cmd] + (["--dry-run"] if cmd == "generate" else []))
        assert hasattr(args, "func")
