# HeXO Game Viewer

Standalone game-replay + play-against-model UI served from the existing
Flask dashboard. Spec: `docs/09_VIEWER_SPEC.md` (read first).

## Launch

The viewer is a route on the dashboard, not a separate process.

```bash
source .venv/bin/activate
make dashboard            # or: python -m hexo_rl.monitoring.web_dashboard
# then browse http://localhost:5050/viewer
```

If a training run is live the dashboard auto-discovers checkpoints; if no
run is active the viewer still works in replay-only mode (Play tab returns
HTTP 503 until a checkpoint passes the eval gate).

## What `/viewer` shows

- **Hex board** (left, 70%): replays a finished game move-by-move. Pointy-top
  axial coordinates, auto-centered on the stones. Last-placed stones get a
  ring overlay.
- **Threat overlay**: empty cells inside any 6-window where one player has
  ≥4 stones (FORCED) or 5 stones (CRITICAL). Computed live in Rust on every
  scrubber tick — not stored in the game record. See spec §3.
- **Scrubber + step buttons**: bottom bar. Slider min is `-1` (empty board)
  through last move. Click the value sparkline to seek.
- **Value sparkline** (below board): line chart of model value at each move,
  P0 winning at top, P1 at bottom. Empty + greyed if `value_trace` is null.
- **MCTS heat overlay** (toggle): per-cell opacity proportional to root
  visit count from `moves_detail.top_visits`. Toggle is hidden when no game
  has detail data.
- **Recent-games list** (right sidebar): last 20 games from the in-memory
  index, falls back to globbing `runs/<run_id>/games/*.json` for older
  games. Click to load; the deep `?n=` query param caps at 100.

## Loading a specific game by ID

```
GET /viewer/game/<game_id>
```

Returns the enriched record (game_complete fields + `positions[]` +
`data_capture_status`). Falls back to disk glob across all runs if the
game has been evicted from the 50-entry in-memory index.

## Known limitations (spec §10 deferred)

- `moves_detail` and `value_trace` are always `None` in current self-play
  game records. Capturing per-move MCTS distribution and root value
  requires Rust `game_runner/worker_loop.rs` changes to store the data
  before the tree is reset between moves. Until that lands, the value
  sparkline and MCTS-heat overlay show the deferred-status badge in the
  top-right of the sparkline panel.
- **Per-worker ID** in `game_complete` is set Python-side (pool layer).
  True per-worker attribution wants a Rust-side worker ID.
- **Play-mode gating on eval-gate passage**: play returns 503 only when
  no checkpoint is loaded. Gating on a passed eval gate (win-rate ≥ 55%
  vs SealBot) is not yet enforced — the viewer reloads its model on
  `eval_complete` with `anchor_promoted=True`, but earlier loads are
  served unconditionally.

## Architecture invariant

Nothing under `hexo_rl/viewer/` is imported by `train.py`, `pool.py`,
`trainer.py`, or any training path. Verify:

```bash
grep -rn "from hexo_rl.viewer\|import viewer" \
  scripts/ hexo_rl/training/ hexo_rl/selfplay/ hexo_rl/monitoring/events.py
# expect 0 lines
```

## Tests

```bash
source .venv/bin/activate
pytest tests/test_viewer.py -v
```

Threat-detection unit tests live under `engine/src/board/threats.rs`
(`cargo test -p engine threats`) — covered by `make test-rust`.
