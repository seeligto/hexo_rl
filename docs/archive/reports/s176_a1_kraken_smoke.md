# Wave A1 — KrakenBot smoke

Branch: `phase4.5/s176_phase_a_validation`. Host: laptop Ryzen 7 8845HS, .venv Python 3.14.2.

## (a) Submodule status

```
 d9c5bfb2282f43c933065c081cafac26f89d245b vendor/bots/krakenbot (heads/main)
 c94749c21c16c3b072fff6da49762dd5f92f3986 vendor/bots/sealbot (heads/master)
```

KrakenBot pinned at `d9c5bfb` ("better eval"); matches prompt-stated SHA. `main` branch.

## (b) Weights inventory

| Artifact | Path (relative to `vendor/bots/krakenbot/`) | Size (B) | Status | sha256 |
|---|---|---:|---|---|
| `pattern_values.json` | `data/pattern_values.json` | 12,688 | PRESENT — real JSON | `9fb8620c1b6a27b16…` |
| `resnet best.pt` | `training/resnet_results/best.pt` | — | **MISSING — never committed** (`.pt` in `.gitignore:8`) | n/a |
| `mcts best.pt` | `training/mcts_results/best.pt` | — | **MISSING — never committed** (`.gitignore:8`) | n/a |

No Git LFS in repo (`git lfs` not even installed on host). `.pt` files are in `vendor/bots/krakenbot/.gitignore:8`; upstream excluded them by design. `training/mcts_results/` and `training/resnet_results/` directories do not exist locally; `find … -name '*.pt'` returns empty. No HF / S3 / mirror URL referenced in `README`, `requirements.txt`, or `pyproject.toml`.

Bot source SHAs (for forensics):

```
2bedb9b701461d993877c46ad12684a5dcd813e961b5f64f74a7827e479355de  minimax_bot.py
20024938e65c93e99ba4c1b084130cdd2d17aee02ae3e991a1125a0b7bdc41a4  mcts_bot.py
b0331b9b551e73e5f7cbdc8c86f4160071049b5a26f89fdc6c2b27c125e6a455  bot.py
0ae6b685ee85ec0e0a93adc00169dcb75a06c8e06c21d547a1e7163c42b7d241  mcts/pattern_table.py
5333c222d1e76205e711ada82b7b457fb121741a0bdab70017d90aa43a3b1314  mcts/_puct_cy.pyx
3a4137a6571eefb93d3f4cd5bd0511a730733350d2884141482a47b97fa6bcfd  mcts/_mcts_cy.pyx
```

## (c) Build outcome

**No `pip install -e .` attempted; not needed for MinimaxBot or RandomBot.**

`setup.py:14-21` builds a pybind11 C++ extension named `ai_cpp` from `cpp/ai_cpp.cpp`. `grep ai_cpp` across `bot.py`, `minimax_bot.py`, `mcts_bot.py`, `mcts/tree.py`, `model/*.py` returns **zero hits** — extension is orphaned (not imported by any production bot path). `pyproject.toml` is bare (build-system only, no project metadata) — no `pip install -e .` package definition exposed.

`setup_puct.py:1-26` builds two Cython modules (`mcts._puct_cy`, `mcts._mcts_cy`). `_puct_cy` is imported only at `mcts/tree.py:379` inside `MCTSBot`'s search loop. **Not on MinimaxBot's path** — confirmed: MinimaxBot's only Cython-adjacent import is `from mcts.pattern_table import build_arrays` (`minimax_bot.py:153`), which is pure-Python `mcts/pattern_table.py:1-80`.

| Component | Status in `.venv` | Source |
|---|---|---|
| `torch 2.11.0+cu130` | installed | `requirements.txt:1` |
| `numpy 2.2.6` | installed | `requirements.txt:2` |
| `tqdm 4.67.3` | installed | `requirements.txt:4` |
| `pybind11 3.0.4` | installed | `requirements.txt:5` |
| `pandas` | NOT installed | `requirements.txt:3` — unused on bot runtime paths (only in `tools/`/training) |

**Net build status:** no build needed for MinimaxBot or RandomBot. MCTSBot would require Cython build (`python setup_puct.py build_ext --inplace`) + `best.pt` download; neither attempted (out of scope for INTEGRABLE verdict).

## (d) Per-bot init smoke (fresh subprocess each)

| Bot | Module | Import (ms) | Init (ms) | Status |
|---|---|---:|---:|---|
| `RandomBot` | `bot` (`bot.py:42`) | 0.5 | <0.01 | PASS |
| `MinimaxBot(time_limit=0.1)` | `minimax_bot` (`minimax_bot.py:195`) | 2.2 | 1.0 | PASS — `eval_length=6`, `pv_len=729` (`3^6`) |
| `MCTSBot(time_limit=1.0, n_sims=10, device="cpu")` | `mcts_bot` (`mcts_bot.py:24`) | 654 | — | **FAIL — `FileNotFoundError: …/training/resnet_results/best.pt`** at `mcts_bot.py:46` (`torch.load`) |

MCTSBot traceback (truncated):
```
File ".../mcts_bot.py", line 46, in __init__
    ckpt = torch.load(model_path, map_location=self.device,
                      weights_only=True)
…
FileNotFoundError: [Errno 2] No such file or directory:
'.../vendor/bots/krakenbot/training/resnet_results/best.pt'
```

`mcts_bot.py:19-21` hardcodes `_DEFAULT_MODEL_PATH = training/resnet_results/best.pt` — same path as the gitignored artifact.

## (e) Per-move latency table (4-stone position)

Position: A@(0,0); B@(2,0),(1,1); A@(0,2) — 4 stones, A to move next, single-move turn. Constructed via `vendor/bots/krakenbot/game.py:28 HexGame.make_move`. 5 samples each — first is warm-up, mean+stdev across samples 2–5.

### Cold bot per call (fresh `MinimaxBot()` per sample — realistic tourney harness)

| Bot dial | mean ms | stdev ms | warm ms (sample 1) | `last_depth` (per sample) | sample-2 move |
|---|---:|---:|---:|---|---|
| `MinimaxBot(time_limit=0.1)` | 221.8 | 3.0 | 204.1 | `[3, 4, 4, 4, 4]` | `[(1, 2), (-2, 2)]` |
| `MinimaxBot(time_limit=0.5)` | 224.3 | 2.5 | 228.7 | `[4, 4, 4, 4, 4]` | `[(1, 2), (-2, 2)]` |
| `MinimaxBot(time_limit=1.0)` | 226.6 | 2.7 | 230.0 | `[4, 4, 4, 4, 4]` | `[(1, 2), (-2, 2)]` |
| `MinimaxBot(time_limit=2.0)` | 232.2 | 2.6 | 230.8 | `[4, 4, 4, 4, 4]` | `[(1, 2), (-2, 2)]` |
| `RandomBot` | 0.006 | 0.001 | 0.011 | n/a | n/a |

Iterative deepening terminates at depth ≤4 on this position — `last_depth` plateaus. `_deadline = time + time_limit × 2` (`minimax_bot.py:221`); per-call wall ≈ `time_limit` because search exhausts before deadline at depth 4. Returned move is a **2-tuple list** (pair_moves=True at `minimax_bot.py:198`).

### Shared-bot timing (TT persists; not realistic for tourney)

For reference (`/tmp/wave_a1/latency_minimax_heavy.py`): warm 200–236 ms, subsequent calls 1.2–2.9 ms because zobrist TT (`minimax_bot.py:96-98`) hits the cached position. Wave B harness MUST instantiate a fresh `MinimaxBot` per game (or call `_tt.clear()`) — otherwise same-position replays under-bench by ~150×.

### MCTSBot — not measured (weights missing; init fails per (d))

## (f) Build helper script

`scripts/build_vendor.sh` **NOT created**. MinimaxBot's only Python-side requirement is `mcts/pattern_table.py` (`build_arrays`), which is pure Python (`pattern_table.py:23` — no `cimport`, no `cdef`). Cython compile (`setup_puct.py`) is required ONLY for `MCTSBot` path (`mcts/tree.py:379 from mcts._puct_cy import puct_select`). Since MCTSBot is blocked on missing weights regardless, Cython build can be deferred to Wave B's MCTSBot enable step.

If Wave B activates MCTSBot, the build invocation is:

```bash
cd vendor/bots/krakenbot && \
  $REPO_ROOT/.venv/bin/python setup_puct.py build_ext --inplace
```

Document at that time; no script needed yet.

## Existing wrapper sanity

`hexo_rl/bots/krakenbot_bot.py` exercises `MinimaxBot` end-to-end via the §176 P38-pinned `BotProtocol` contract (`hexo_rl/bootstrap/bot_protocol.py:31-51` — `get_move(state, rust_board)`). Wrapper uses `_MockGame` duck-type (`krakenbot_bot.py:36-56`) to match `HexGame` interface (`game.py:28-122`), `_KPlayer` enum (`krakenbot_bot.py:33`) re-imported from vendor `Player`, and `_pending_move` cache (`krakenbot_bot.py:79-121`) to split pair-move returns across two `get_move()` calls.

`tests/test_krakenbot_bot.py` passes (1.77 s) — but it only tests path resolution, not move flow. End-to-end MinimaxBot move flow is validated only by the latency smoke above. Wave B will need an actual game-loop test.

## (g) Verdict

**`INTEGRABLE_NOW`** — for MinimaxBot + RandomBot path.

Empirical justification: submodule at pinned SHA `d9c5bfb`, `pattern_values.json` 12,688-B present, all deps in `.venv` except unused `pandas`, MinimaxBot & RandomBot import + init + first-move smoke PASS in fresh subprocesses, cold per-move latency 222–232 ms at `time_limit=0.1–2.0` (< 5 s gate), existing `hexo_rl/bots/krakenbot_bot.py` already wires `MinimaxBot` to `BotProtocol` and passes `tests/test_krakenbot_bot.py`.

**MCTSBot is NEEDS_WEIGHTS_DOWNLOAD** — gated separately:
- `training/resnet_results/best.pt` does NOT exist on disk; gitignored upstream (`.gitignore:8`), never committed, not an LFS pointer (no `git lfs` in repo metadata).
- No HF / S3 mirror URL surfaces in `README` / `requirements.txt` / `pyproject.toml`.
- Cython compile (`setup_puct.py`) required separately for `mcts._puct_cy`.
- If Wave B wants MCTSBot in the tourney, **operator must supply `best.pt`** — escalate to user with question "do you have a copy of the trained KrakenBot resnet weights, or should we contact the WolverinDEV/krakenbot maintainer?"

## Notes for Wave B

- Spec-compliant tourney harness: fresh `MinimaxBot` per game (or `bot._tt.clear()` + `bot._history.clear()` between games) — otherwise zobrist TT (`minimax_bot.py:96-98`) artificially deflates per-move latency by ~150× on repeated positions.
- MinimaxBot wall ≈ `time_limit` on early-game 4-stone positions; budget tourney games accordingly (e.g., `time_limit=0.1` × 30 plies × 2 bots ≈ 6 s/game).
- `KrakenBotBot.name() → "KrakenBot(t=<time_limit>)"` (`krakenbot_bot.py:127`) — encode dial in tourney leaderboard label.
- C++ `ai_cpp` extension (`cpp/ai_cpp.cpp`) is orphan; ignore unless Wave B explicitly needs it.
- `pandas` in `requirements.txt:3` is unused on bot runtime paths; don't add to venv unless tooling scripts pulled in.
