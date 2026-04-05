# Game Viewer Specification — HeXO /viewer
# docs/09_VIEWER_SPEC.md

**Status:** Authoritative spec. Implementation complete as of 2026-04-03. Deferred items listed in §10.
**Scope:** Standalone game viewer and play-against-model interface.
**Depends on:** docs/08_DASHBOARD_SPEC.md (event schema, web_dashboard.py server)
**Last updated:** 2026-04-03

---

## 1. Architecture

The viewer is a separate route on the existing Flask server. It does not run
as a separate process and does not touch the training loop, replay buffer, or
MCTS workers.

```
browser /viewer
    │
    ├── GET /viewer                     → viewer.html
    ├── GET /viewer/game/<game_id>      → enriched game record (JSON)
    ├── GET /viewer/recent?n=20         → list of recent game metadata
    └── POST /viewer/play               → human move → model response (JSON)
         │
         └── hexo_rl/viewer/engine.py
              ├── board.get_threats()   ← new Rust export (read-only)
              └── OurModelBot           ← existing wrapper, MCTS inference
```

Critical invariant: nothing in hexo_rl/viewer/ is imported by train.py,
pool.py, trainer.py, or any training path. Verify:
```bash
grep -rn "from hexo_rl.viewer\|import viewer" \
  scripts/ hexo_rl/training/ hexo_rl/selfplay/ hexo_rl/monitoring/events.py
# Must return 0 lines
```

---

## 2. Game record enrichment

### 2.1 Changes to game_complete event

Extend the existing schema with two optional fields (docs/08_DASHBOARD_SPEC.md §2.3):

```python
{
    "event":        "game_complete",
    # ... all existing fields unchanged ...
    "moves_list":   list[str],        # existing

    # NEW — both optional, None if capture_game_detail=false
    "moves_detail": list[dict] | None,
    "value_trace":  list[float] | None,
}
```

moves_detail — one dict per move, parallel to moves_list:
```python
{
    "coord": str,          # axial e.g. "(2,-1)"
    "top_visits": [        # top-15 MCTS nodes by visit count
        {"coord": str, "visits": int, "prior": float},
        ...
    ]
}
```

value_trace — one float per move: value head output from current player's
perspective in [-1, 1] at the moment that move was chosen.

### 2.2 Capture in pool.py

After each MCTS search, before committing the move:

```python
top_visits = mcts_tree.get_top_visits(n=15)
value_est  = mcts_tree.root_value()

move_detail = {
    "coord": chosen_coord_str,
    "top_visits": [{"coord": c, "visits": v, "prior": p} for c, v, p in top_visits]
}
move_details_list.append(move_detail)
value_trace_list.append(value_est)
```

Include in game_complete emit. Gate behind config:
```yaml
monitoring:
  capture_game_detail: true
```

### 2.3 New Rust exports needed

Check if these already exist on MCTSTree. Add if missing:

**`mcts_tree.get_top_visits(n: int) -> list[tuple[str, int, float]]`**
Returns top-N children of root sorted by visit count: (coord_str, visits, prior).
Read-only. Called after search, before move commit. Nanoseconds.

**`mcts_tree.root_value() -> float`**
Value estimate at root from perspective of player to move. Already computed
during search — just expose it.

### 2.4 Performance budget

- get_top_visits(15): read 15 nodes from searched tree. O(15). Negligible.
- root_value(): single float read. Negligible.
- Memory: ~3KB/game × 3,000 games/hr = ~9MB/hr extra in event history. Fine.
- No additional MCTS search or NN inference triggered.

Run `make bench.full` before and after. If >1% regression, gate behind
`capture_game_detail: false` by default and document.

---

## 3. Threat detection — Rust

### 3.1 Algorithm

For Connect-6, a threat is a window of 6 consecutive cells along any hex axis
where one player has exactly N stones (N ≥ 3) and the remaining (6-N) cells
are empty (no opponent stones blocking the window).

The highlighted cells are the EMPTY cells within the window — the cells that
need to be filled. Do not highlight the existing stones.

Threat levels:
- N=5: one empty cell — player wins if they place there → CRITICAL
- N=4: two empty cells — player wins in one compound turn (places both) → FORCED
- N=3: three empty cells — needs two more turns → WARNING

Example line (zero-indexed): `[_, O, O, O, _, O, _, _]`
  Window [1..6] = [O,O,O,_,O,_]: N=4, empty cells at index 4 and 6
  → FORCED threat, highlight index 4 and 6 (the gaps)
  Window [2..7] = [O,O,_,O,_,_]: N=3, empty cells at index 4,6,7
  → WARNING threat, highlight index 4,6,7

A cell may appear in multiple windows. Precedence: CRITICAL > FORCED > WARNING.

### 3.2 Hex axes (axial coordinates)

Three axes:
- Axis 0: direction (1, 0)
- Axis 1: direction (0, 1)
- Axis 2: direction (1, -1)

For each axis, scan all lines through every occupied cell. For each line,
slide a window of width 6. Limit scan range to bounding box of occupied cells
plus 6 cells of margin in each direction.

### 3.3 Rust interface

New function in engine/src/board/ (threats.rs or inline):

```rust
pub struct ThreatCell {
    pub q: i32,
    pub r: i32,
    pub level: u8,   // 3=warning, 4=forced, 5=critical
    pub player: u8,  // 0 or 1
}

pub fn get_threats(board: &Board) -> Vec<ThreatCell>
```

PyO3 binding on Board:
```rust
#[pymethods]
impl Board {
    pub fn get_threats(&self) -> Vec<(i32, i32, u8, u8)> {
        // (q, r, level, player)
    }
}
```

This function is NEVER called from MCTS or training. Viewer only.

---

## 4. Viewer backend — hexo_rl/viewer/engine.py

```python
class ViewerEngine:
    def __init__(self, config: dict, checkpoint_path: str | None = None):
        self._model_bot: OurModelBot | None = None
        if checkpoint_path:
            self._model_bot = OurModelBot(config, checkpoint_path)

    def enrich_game(self, game_record: dict) -> dict:
        """
        Replay the game move-by-move, calling board.get_threats() at each
        position. Returns game_record with added 'positions' field.
        """
        board = Board()
        positions = []
        moves = game_record.get("moves_list", [])
        value_trace  = game_record.get("value_trace")  or [None] * len(moves)
        moves_detail = game_record.get("moves_detail") or [None] * len(moves)

        for i, coord_str in enumerate(moves):
            q, r = parse_axial(coord_str)
            board.place_stone(q, r)
            threats = board.get_threats()
            positions.append({
                "move_index":  i,
                "coord":       coord_str,
                "value_est":   value_trace[i],
                "top_visits":  moves_detail[i]["top_visits"] if moves_detail[i] else None,
                "threats": [
                    {"q": t[0], "r": t[1], "level": t[2], "player": t[3]}
                    for t in threats
                ],
            })
        enriched = dict(game_record)
        enriched["positions"] = positions
        return enriched

    def play_response(self, moves_so_far: list[str], human_moves: list[str]) -> dict:
        """
        Reconstruct board from moves_so_far + human_moves, run MCTS,
        return model's response.
        """
        # returns {"moves": [str,...], "value_est": float,
        #          "top_visits": [...], "threats": [...]}
```

---

## 5. Routes — additions to web_dashboard.py

```python
@app.route("/viewer")
def viewer_page():
    return send_from_directory("static", "viewer.html")

@app.route("/viewer/recent")
def viewer_recent():
    # Reads lightweight refs from _game_index (maxlen=50), not _event_history.
    n = min(int(request.args.get("n", 20)), 100)
    with _history_lock:
        refs = list(_game_index)
    return jsonify([{
        "game_id": r["game_id"], "winner": r["winner"],
        "moves": r["moves"], "ts": r["ts"],
    } for r in reversed(refs[-n:])])

@app.route("/viewer/game/<game_id>")
def viewer_game(game_id):
    # Loads full record from disk (runs/<run_id>/games/<game_id>.json).
    # Falls back to glob search for games evicted from the 50-entry index.
    path_str = _lookup_game_path(game_id)  # index → disk fallback
    if path_str is None:
        return jsonify({"error": "game not found"}), 404
    record = json.loads(Path(path_str).read_text())
    return jsonify(_viewer_engine.enrich_game(record))

@app.route("/viewer/play", methods=["POST"])
def viewer_play():
    if not _viewer_engine or not _viewer_engine._model_bot:
        return jsonify({"error": "no model loaded"}), 503
    data = request.get_json()
    result = _viewer_engine.play_response(
        data.get("moves_so_far", []),
        data.get("human_moves", [])
    )
    return jsonify(result)
```

Checkpoint loading at startup and on eval_complete — see spec section 5.1 below.

### 5.1 Checkpoint management

At WebDashboard.start():
```python
latest_ckpt = find_latest_checkpoint(config)
_viewer_engine = ViewerEngine(config, checkpoint_path=latest_ckpt)
```

In on_event, on eval_complete with gate_passed=True:
```python
_viewer_engine = ViewerEngine(config, find_latest_checkpoint(config))
```

Play mode returns 503 if no checkpoint has passed the gate yet.

---

## 6. Browser SPA — viewer.html

Single self-contained HTML. No build step.

### 6.1 Layout

Left panel (70%): hex board canvas
Right panel (30%): mode toggle, legend, position stats, value sparkline,
                   move list (replay) or model status (play)
Bottom bar: scrubber, step buttons, play/pause, overlay toggles

### 6.2 Hex board renderer

Pointy-top hexagons. Axial to pixel:
```javascript
function hexToPixel(q, r, size, offX, offY) {
    return [
        offX + size * Math.sqrt(3) * (q + r / 2),
        offY + size * 1.5 * r
    ];
}
```

Viewport: center on centroid of all stones. Auto-scale so all stones fit.
On each move change: render stones up to that index from the cached game record.

### 6.3 Threat overlay

Draw on EMPTY cells in positions[i].threats, NOT on stone cells.

```javascript
threats.forEach(({q, r, level, player}) => {
    const [cx, cy] = hexToPixel(q, r, size, offX, offY);
    const color = level >= 4 ? '#E24B4A' : '#EF9F27';
    ctx.beginPath();
    ctx.arc(cx, cy, size - 3, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = level === 5 ? 3 : 2;
    ctx.setLineDash(level >= 4 ? [] : [4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    if (level === 5) {
        // fill lightly for critical
        ctx.fillStyle = 'rgba(226,75,74,0.15)';
        ctx.fill();
    }
});
```

### 6.4 MCTS heat overlay

Normalize visit counts to [0,1]. Draw filled hex with proportional opacity.
Hide toggle if top_visits is null (old game records).

### 6.5 Value sparkline

120px tall canvas below the board. Line from move 0 to end.
Y=0 is center (even game), Y=top is P0 winning, Y=bottom is P1 winning.
Vertical line at current scrubber position. Click to seek.

### 6.6 Play mode — click-click-auto

- Human clicks hex 1 → semi-transparent stone rendered, waiting
- Human clicks hex 2 → both stones rendered, POST /viewer/play fires immediately
- "Model thinking..." shown while waiting
- Hex clicks disabled during model thinking
- Second click on same hex as first: ignored
- P1 opens with 1 stone (move 0): only one click needed

### 6.7 Game list

On load, GET /viewer/recent?n=20. Scrollable sidebar list. Click to load.
"← back to dashboard" link at top navigates to /.

---

## 7. Implementation files

Created:
- hexo_rl/viewer/__init__.py
- hexo_rl/viewer/engine.py
- hexo_rl/monitoring/static/viewer.html
- engine/src/board/threats.rs
- tests/test_viewer.py

Modified:
- hexo_rl/monitoring/web_dashboard.py — 4 new routes + engine init
- engine/src/lib.rs — export get_threats, get_top_visits, root_value
- configs/monitoring.yaml — monitoring.capture_game_detail: true

---

## 8. Tests

tests/test_viewer.py must include:

1. get_threats() on known 4-in-row: returns correct 2 empty cells as level=4
2. get_threats() on [_, O, O, O, _, O, _, _] line: highlights positions 4 and 6
   NOT positions 1,2,3,5 (the existing stones)
3. get_threats() on empty board: returns []
4. get_threats() ignores windows blocked by opponent stone
5. enrich_game() returns positions field with len == len(moves_list)
6. GET /viewer/recent returns 200
7. GET /viewer/game/<unknown_id> returns 404
8. GET /viewer/game/<known_id> returns enriched record with positions field
9. play_response() returns valid axial coord strings
10. game_complete events include moves_detail and value_trace fields
    (integration test, mark with @pytest.mark.integration, skip if no GPU)

---

## 9. Out of scope

- Persistence across runs (in-memory only, last 500 games)
- Multiplayer / spectating
- Full MCTS tree visualization
- Mobile layout

---

## 10. Deferred items

These require Rust-side changes and are deferred to a future sprint:

- **Per-move MCTS detail in game records**: `moves_detail` and `value_trace` fields
  in `game_complete` events are currently `None`. Capturing `get_top_visits()` and
  `root_value()` at each move requires changes to `game_runner.rs` to store MCTS
  results before the tree is reset.
- **Per-worker ID in game_complete**: `worker_id` is included but currently set from
  the Python pool layer. True per-worker attribution requires a Rust-side worker ID.
- **Play mode gating on eval gate passage**: play-against-model should only serve
  the model after it has passed at least one eval gate (win rate ≥ 55% vs SealBot).
  Currently serves from the latest checkpoint unconditionally.
