# D-STRIX S4 — Play-UI spec (hexo-play)

Status: SPEC ONLY. No HeXO source touched. Dispatcher: D-STRIX S4.

---

## 1. hero.did.science verdict — ZERO-BUILD does NOT apply

**Task named `hero.did.science`. That domain does not resolve** (`curl`/DNS: `Could not
resolve host`). Codebase grep found no reference to it either. The real site referenced
throughout `docs/` is **`hexo.did.science`** (source: `docs/04_bootstrap_strategy.md:30,91,154`,
`docs/07_PHASE4_SPRINT_LOG.md:285,1220`) — source repo `WolverinDEV/infhex-tic-tac-toe`
(GPL-3.0). Treating this as the intended target.

**Fetched `hexo.did.science` directly (WebFetch, live 2026-07-02):**

- Homepage nav: Rules / Sandbox / Match History / Leaderboard / Tournaments / Discord /
  Sign In. Framed as **multiplayer only** ("place your hexes on an infinite board",
  "align six in a row"). No AI/bot-opponent option surfaced on the homepage.
- `/games` (match archive): 91,055 games, 4.2M+ moves, human/guest accounts only. No
  indication of bot games or a way to launch one from this page.
- `/sandbox`: **does** support "hand either side to a bot" — a local, clockless board
  where you can play both sides yourself or delegate a side to a bot. This is the site's
  **own built-in bot**, not a pluggable external engine.
- `/rules`: no mention of bot API, external bot integration, uploading a custom
  model/AI, or `htttx`-style third-party bot connection. The site has no exposed hook
  for wiring in an arbitrary checkpoint.

**Verdict: NOT zero-build.** `hexo.did.science` lets a human play against *the site's own*
bot, not against an arbitrary HeXO checkpoint we host. There is no known integration path
(no bot-upload feature, no API surfaced in `/rules`) to get our checkpoint into that
sandbox bot slot. Separately, our own deploy path *to* that site is a different, unrelated,
still-deferred item (`docs/07_PHASE4_SPRINT_LOG.md:1220`, `docs/designs/encoding_migration_v8.md:459`
— P2 fragmentation hotfix blocks *our bot* joining *their* multiplayer pool, which is not
this task anyway). Community bot-API deploy target for tournament play is a third, separate
site (`https://explore.htttx.io/`, `docs/05_community_integration.md:14`) — also not a
human-play-vs-checkpoint surface.

Conclusion: full `hexo-play` spec below stands. This is an integration-note-free build spec.

---

## 2. What already exists in-repo (don't rebuild)

`docs/09_VIEWER_SPEC.md` documents an **existing** `/viewer` play mode on the training
dashboard (Flask, `hexo_rl/monitoring/web_dashboard.py`, `hexo_rl/viewer/engine.py`) with
a `POST /viewer/play` human-move → model-response endpoint and click-click-auto UX
(§6.6). It is explicitly **not** a standalone product: same process as the training
dashboard, in-memory only (last 500 games, no persistence — §9), no SQLite, gated on the
training loop being alive, and out of scope for multiplayer/persistence/deploy. `hexo-play`
is architecturally the same idea (click-click, MCTS/model response) but as an independent,
persistent, deployable product with a real DB and a real checkpoint/solver/sims control
surface — do not confuse the two, and do not import from `hexo_rl.viewer` (that module is
explicitly walled off from every training path per the viewer spec's own invariant check).

---

## 3. hexo-play — architecture

**Separate repo**, `hexo-play/`, NOT inside `hexo_rl`. Depends on `hexo_rl` as an installed
package (checkpoint loading, encoding registry, bot classes) — imports it, doesn't vendor it.

```
hexo-play/
  backend/
    main.py            # FastAPI app, routes
    engine_wrapper.py   # loads checkpoint -> DeployHeadBot [+ SolverBackupBot]
    db.py               # sqlite3 schema + helpers
    game_session.py      # in-memory per-game state (board, compound-turn cache)
  frontend/
    index.html          # single-page, vanilla JS, canvas hex board
    board.js            # axial<->pixel, click-click-confirm state machine
    app.js               # fetch calls to backend, render loop
  hexo_play.db          # sqlite, gitignored
  pyproject.toml         # depends on hexo_rl (path or git dep)
```

No build step, no framework on the frontend (matches `docs/09_VIEWER_SPEC.md` §6's own
"single self-contained HTML, no build step" choice — proven pattern in this codebase).

### Reference pattern — hexo-strix `serve` (SootyOwl/hexo-strix, MIT, independent project)

Fetched their README directly (github.com/SootyOwl/hexo-strix). Their `serve` command is
the closest external analogue:

```
uv run --no-sync hexo-a0 serve --config config.toml --checkpoint <ckpt>.pt --port 8765
```

5-line summary of their pattern: single CLI command takes a config + checkpoint + port;
spins up an HTTP "play-and-analyze" server with **selectable difficulty tiers**; records
every game to **SQLite**; exposes a **token-gated `/admin` page** alongside the play UI;
superseded an older pygame `watch` viewer. It's a different stack (Rust/GNN engine, not
ours) — reference the *shape* only (CLI flags: config+checkpoint+port; SQLite game log;
admin page), do not port code. `hexo-play` should mirror: `--config`, `--checkpoint`,
`--port` CLI flags on the FastAPI entrypoint, SQLite log, and a lightweight `/admin` (see
§7) — no auth system needed at "build in a day" scope; gate `/admin` behind a bearer token
read from an env var if exposed beyond localhost.

---

## 4. Checkpoint loading — real entry points (cite, don't reinvent)

All paths verified live in this repo (not the worktree copy under `.claude/worktrees/`).

1. **Load checkpoint → model**: `hexo_rl.eval.checkpoint_loader.load_model_with_encoding`
   (`hexo_rl/eval/checkpoint_loader.py:110`) — signature
   `load_model_with_encoding(ckpt_path: str | Path, device: torch.device) -> Tuple[HexTacToeNet, EncodingSpec, str]`.
   Handles v6/v6w25/v6_live2/v6tp/v8 dispatch and legacy key-prefix stripping. Returns
   `(model, spec, label)` — `label` drives bot dispatch, `spec` gives board_size / window
   geometry from the registry (`engine/src/encoding/registry.toml`).

2. **Wrap model in an inference engine**: `hexo_rl.eval.deploy_strength_eval._build_engine_for_model`
   (`hexo_rl/eval/deploy_strength_eval.py:233`) —
   `_build_engine_for_model(model, encoding_name, device) -> LocalInferenceEngine`. Thin: looks
   up the `EncodingSpec` and constructs `LocalInferenceEngine(model, device, encoding_spec=spec)`.
   `LocalInferenceEngine` itself: `hexo_rl/selfplay/inference.py:29`, ctor
   `__init__(self, model: HexTacToeNet, device, encoding_spec: Optional[EncodingSpec] = None)`.
   (`hexo-play`'s `engine_wrapper.py` can call `_build_engine_for_model` directly, or inline
   the two-line body — it's private (`_`-prefixed) but stable and exactly what every eval
   script in this repo uses.)

3. **The move-choosing bot — DeployHeadBot** (the multi-window decode head cited in the
   task): `hexo_rl/eval/deploy_strength_eval.py:108`. Ctor:
   ```python
   DeployHeadBot(
       engine: LocalInferenceEngine,
       knobs: Dict[str, Any],   # {gumbel_m, n_sims_full, c_visit, c_scale, c_puct}
       label: str,
       seed: int = 0,
       legal_set: bool = False,  # True = multi-window no-drop decode (§D-DECODE)
   )
   ```
   `knobs` is normally read from run config via `extract_deploy_knobs(cfg)`
   (same file, line 103) — for `hexo-play` these become **user-facing sliders** (see §6):
   `gumbel_m` (search breadth), `n_sims_full` (sims), `c_visit`/`c_scale`/`c_puct` (PUCT
   constants — fix these to the checkpoint's training config by default, expose only as an
   "advanced" collapsible). `legal_set=True` is the "multi-window / off-window-aware decode"
   toggle the task asks for — plumb it straight through as a checkbox.
   `get_move(state: GameState, rust_board: engine.Board) -> Tuple[int, int]` — this is the
   `BotProtocol` interface (`hexo_rl/bootstrap/bot_protocol.py:20`, abstract methods
   `get_move`, `name`, `reset` at lines 32/54/58).

4. **Solver-backup toggle** (Python, proven-mate-only override, per D-SOLVER):
   `hexo_rl.eval.solver_backup_bot.SolverBackupBot` (`hexo_rl/eval/solver_backup_bot.py:69`).
   Wraps a bot:
   ```python
   SolverBackupBot(
       inner: BotProtocol,          # e.g. the DeployHeadBot above
       depth: int = 6,               # DEFAULT_BACKUP_DEPTH
       win_threshold: int = 99_999_000,
       colony_max_clusters: int = 4,
       colony_max_coord: int = 60,
       window_half: Optional[int] = None,   # e.g. 9 for v6_live2_ls — restrict backup to in-window band
       probe_engine: str = "sealbot",        # or "native" (engine::tactics)
       node_budget: int = 200_000,
       cand_cap: int = 40,
   )
   ```
   Overrides the model's move only on a *proven* mate at the probe depth; flags (does not
   override) proven losses. `hexo-play` toggle: a checkbox "solver backup (tactical
   safety net)" that wraps the `DeployHeadBot` in a `SolverBackupBot` when checked. Default
   OFF (matches every existing default in this codebase — solver backup is opt-in
   everywhere it's wired, per `scripts/eval/run_a1_solver_backup.py`).

5. **Full wiring example already in this repo**: `scripts/eval/run_a1_solver_backup.py:56-70`
   (`_build_cand`) shows the exact compose-a-bot pattern `hexo-play`'s `engine_wrapper.py`
   should copy: `DeployHeadBot(engine, knobs, label=..., seed=..., legal_set=...)`, optionally
   wrapped in `SolverBackupBot(head, depth=..., window_half=..., probe_engine=...)`.

6. **Board / GameState plumbing** (also copy, don't reinvent):
   `hexo_rl/eval/deploy_strength_eval.py:240-273` (`_play_one_game`) shows the canonical
   per-move loop: `Board.with_encoding_name(encoding)`, `GameState.from_board(board)`,
   `board.legal_moves()`, `board.current_player`, `state.apply_move(board, q, r)`,
   `board.check_win()`, `board.winner()`. `hexo-play`'s `game_session.py` is this loop
   re-entrant across HTTP requests (one call per human move, one call per bot move)
   instead of a tight synchronous while-loop.

**Default encoding**: read from the checkpoint itself (`load_model_with_encoding` already
detects it — don't let the operator have to specify it by hand; only override via
`--encoding` flag for a checkpoint with ambiguous/missing embedded label).

---

## 5. Board UX — compound-turn click-click-confirm

Per `docs/rules/board-representation.md`: P1 opens with **1** stone (ply 0), then both
players alternate placing **2** stones per turn (`Board.moves_remaining` tracks phase: 2 =
about to place stone 1 of the turn, 1 = stone 2 — `state/core.rs:109`). A turn's two stones
are an **unordered pair** (`{A,B} == {B,A}`, commutative-XOR zobrist) and a turn can WIN on
its first stone (`apply_move` places one stone, no win check; win detection is a separate
per-stone call) — the UI must check-win after *each* placed stone, not just at
turn-confirm.

Frontend state machine (mirrors `docs/09_VIEWER_SPEC.md` §6.6, which already validates this
exact pattern in production for the training-dashboard viewer):

1. Human's turn, `moves_remaining == 2` (or `== 1` on ply 0 only): click hex → render as
   semi-transparent pending stone. Second click on the *same* hex: ignored (no-op).
2. Second click (different hex) → both stones committed **client-side** as a single
   compound move; hex board disables further clicks; `POST /game/{id}/move` fires with
   `{"stones": [[q1,r1],[q2,r2]]}` (single stone array on ply 0).
3. Backend applies both stones via `state.apply_move` in sequence, checking `board.check_win()`
   after each (a turn can end the game on stone 1 — don't apply stone 2 if stone 1 already
   won, and don't require the human to have picked one in that case: reject/ignore stone 2
   if stone 1 was already terminal).
4. If game not over: backend runs the bot (`DeployHeadBot` [`+ SolverBackupBot`]) for its
   full 2-stone turn, returns both stones + win status in the same response ("model
   thinking..." shown client-side while this blocks — sims count controls latency, see §6).
5. Render bot's 2 stones, re-enable clicks if game continues.

Reject (400) a `/move` POST that doesn't match `moves_remaining` for the human's stone
count, or that reuses an occupied cell — `apply_move` has no legality check baked in per
the board-representation doc, so the backend is the enforcement point.

### Viewport — pragmatic finite window on an infinite board

The board is genuinely unbounded (`HashMap<(q,r), Player>`, no fixed grid — board-representation.md
§"Internal storage"). The *model's* NN view window (K-cluster, 19×19 for v6/v6_live2/v7full,
25×25 for v8_canvas) is a completely separate concern from the *display* viewport — don't
conflate them. Spec for `hexo-play`'s display:

- Render a **finite canvas viewport** sized to fit all placed stones + a fixed margin (e.g.
  3 cells), auto-recentering on the centroid of all stones after each compound turn —
  same `hexToPixel` convention already used by the viewer (`docs/09_VIEWER_SPEC.md` §6.2:
  `offX + size*sqrt(3)*(q+r/2)`, `offY + size*1.5*r`).
- **Pan**: click-drag or arrow keys shifts `offX/offY` without changing game state — pure
  client-side. **Expand**: no explicit "expand" button needed if auto-recenter-to-fit is
  live; add a manual zoom-out (`scale *= 0.9`) for a human who wants to see beyond the
  auto-fit margin (e.g. planning a long-range play). This is strictly a rendering
  convenience — it has no bearing on what the model "sees" (that's governed by the
  registry's `cluster_window_size` / K-cluster windowing, already correctly handled
  server-side by `LocalInferenceEngine`).
- No hard cap needed on the viewport's coordinate range for a single sat-down game (a
  human game realistically won't exceed a few dozen cells of spread before someone wins);
  don't build virtualized/tiled canvas rendering — out of scope for a 1-day build.

---

## 6. Backend controls — sims slider + toggles

Exposed on a pre-game "New Game" form (not mid-game — fix knobs for the session, matches
`DeployHeadBot`'s stateless-per-game design and avoids a resize-mid-search class of bugs):

| Control | Wired to | Default |
|---|---|---|
| Checkpoint path (dropdown of files in a configured dir, or upload) | `load_model_with_encoding` | required |
| Sims slider (`n_sims_full`) | `DeployHeadBot` ctor `knobs["n_sims_full"]` | checkpoint's training config value if embedded, else 128 |
| Search breadth (`gumbel_m`) | `knobs["gumbel_m"]` | training config value, else 16 |
| Multi-window decode (checkbox) | `DeployHeadBot(legal_set=...)` | off (matches live-gate default) |
| Solver backup (checkbox) | wraps head in `SolverBackupBot(depth=6, probe_engine="sealbot")` | off |
| Human plays (O first / X first) | which side gets `moves_remaining` seed | O (matches community convention, §10 `docs/05_community_integration.md`) |

Advanced (collapsed by default): `c_visit`, `c_scale`, `c_puct`, backup `depth`,
`probe_engine` (`sealbot` vs `native`), `window_half` (leave at the checkpoint-encoding
default — `9` for `v6_live2_ls` per `solver_backup_bot.py:92-93` — don't let an operator
set this blind).

---

## 7. SQLite schema

```sql
CREATE TABLE games (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,           -- ISO8601
    ended_at      TEXT,
    checkpoint    TEXT NOT NULL,           -- path/label of loaded model
    encoding      TEXT NOT NULL,           -- e.g. v6_live2_ls
    human_side    TEXT NOT NULL,           -- 'p1' or 'p2'
    n_sims        INTEGER NOT NULL,
    gumbel_m      INTEGER NOT NULL,
    legal_set     INTEGER NOT NULL,        -- 0/1, multi-window decode toggle
    solver_backup INTEGER NOT NULL,        -- 0/1
    backup_depth  INTEGER,                 -- null if solver_backup=0
    config_json   TEXT NOT NULL,           -- full knob snapshot, incl. advanced overrides
    winner        TEXT,                    -- 'p1' | 'p2' | 'draw' | null (in progress)
    result_reason TEXT                     -- 'win' | 'resign' | 'abandoned'
);

CREATE TABLE moves (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id       INTEGER NOT NULL REFERENCES games(id),
    turn_index    INTEGER NOT NULL,        -- 0-based compound-turn number
    player        TEXT NOT NULL,           -- 'p1' | 'p2'
    stone1_q      INTEGER NOT NULL,
    stone1_r      INTEGER NOT NULL,
    stone2_q      INTEGER,                 -- null only for ply-0 single-stone opening
    stone2_r      INTEGER,
    solver_fired  TEXT,                    -- null | 'win_override' | 'loss_flagged'
    value_est     REAL,                    -- model's value head output pre-move, if bot's turn
    wall_ms       INTEGER,                 -- time spent producing this turn (bot turns)
    UNIQUE(game_id, turn_index)
);
```

`config_json` duplicates the typed columns deliberately — typed columns for querying
("all games with solver_backup=1"), JSON for exact replay (advanced overrides that don't
have their own column). No separate `config` table needed at this scope — one row per
game is enough; don't over-normalize for a 1-day build.

---

## 8. API endpoints

```
POST /games                  -> create game, load checkpoint+knobs, returns {game_id, opening_state}
GET  /games/{id}              -> full game record + move list (for reload/spectate)
GET  /games                    -> list recent games (paginated, for a simple index page)
POST /games/{id}/move          -> human compound move -> applies, runs bot if game continues,
                                   returns {stones_placed, winner|null, bot_stones|null,
                                   bot_value_est|null, solver_fired|null}
POST /games/{id}/resign         -> ends game, result_reason='resign'
GET  /checkpoints                -> list available checkpoint files (for the New Game dropdown)
GET  /admin                       -> (optional, bearer-token gated) recent games + basic stats,
                                     mirrors hexo-strix's admin-page pattern (§3)
```

No websockets needed — bot response is synchronous within `/move`'s request/response
(sims counts in the dozens-to-low-hundreds at `n_sims_full` complete in well under typical
HTTP timeout budgets; if a very high sims slider value risks a slow response, cap the
slider's max in the frontend rather than adding async polling — out of scope for 1 day).

---

## 9. Build-effort estimate

Sized for Claude Code to build in ~1 day:

- Backend skeleton + checkpoint loading + `DeployHeadBot`/`SolverBackupBot` wiring
  (mostly copy-adapt from `scripts/eval/run_a1_solver_backup.py` + `deploy_strength_eval.py`):
  **~2 hrs**.
- SQLite schema + `db.py` CRUD: **~1 hr**.
- FastAPI routes (§8) + `game_session.py` compound-turn state machine: **~2 hrs**.
- Frontend: hex canvas renderer (adapt `docs/09_VIEWER_SPEC.md` §6.2/§6.6 conventions,
  already proven), click-click-confirm, New Game form, admin page: **~2.5 hrs**.
- Manual smoke test (play a full game each side, verify compound-turn win-on-stone-1,
  verify solver-backup override fires + is logged, verify SQLite rows correct): **~1 hr**.

Total: **~8.5 hrs**, one day. Biggest real risk is sims-latency UX (a naive high sims
slider value makes `/move` block for seconds) — mitigate by capping the slider default
range rather than building async infrastructure.

---

## 10. Out of scope (explicitly, per task scope)

- Multiplayer / spectating / matchmaking.
- Any change to `hexo_rl` or `engine/` — `hexo-play` only imports/depends on `hexo_rl`.
- Auth beyond an optional bearer token on `/admin`.
- Tiled/virtualized infinite canvas rendering.
- Mobile layout.
- Deploying to or integrating with `hexo.did.science`, `explore.htttx.io`, or any
  community site — separate, unrelated efforts (see §1).

## REVIEW

**Verdict: PASS**

**hero.did.science / hexo.did.science verdict — independently re-verified, holds up:**
- `curl`/DNS: confirmed `hero.did.science` does not resolve
  (`Could not resolve host`) in a fresh check.
- Independently fetched `hexo.did.science`, `/sandbox`, and `/rules` live
  (separate WebFetch calls, this review). Results match the doc's claims
  point for point: homepage nav is Rules/Sandbox/Match History/Leaderboard/
  Tournaments/Discord/Sign In with no AI/bot/checkpoint mention;
  `/sandbox` offers "hand either side to a bot" (the site's own built-in
  bot) with no custom-model upload path; `/rules` has no bot API/external-
  integration/upload mention. The "NOT zero-build... no known integration
  path to get our checkpoint into that sandbox slot" conclusion is
  supported by the evidence the doc presents, and reproduces independently.

**Code cites spot-checked (5 of the doc's citations, all exact):**
- `hexo_rl/eval/checkpoint_loader.py:110` → `def load_model_with_encoding(
  ckpt_path: str | Path, device: torch.device) -> Tuple[HexTacToeNet,
  EncodingSpec, str]` — signature matches verbatim.
- `hexo_rl/eval/deploy_strength_eval.py:108` → `class DeployHeadBot
  (BotProtocol)`; ctor at line 118 matches the doc's shown signature
  exactly, including the `legal_set: bool = False` multi-window-decode
  flag and its docstring meaning.
- `hexo_rl/eval/deploy_strength_eval.py:233` → `_build_engine_for_model`
  confirmed at that line.
- `hexo_rl/eval/solver_backup_bot.py:69` → `class SolverBackupBot
  (BotProtocol)`; defaults confirmed exact against the module's named
  constants: `DEFAULT_BACKUP_DEPTH=6`, `WIN_THRESHOLD=99_999_000`,
  `DEFAULT_COLONY_MAX_COORD=60`, `colony_max_clusters=4`,
  `node_budget=200_000`, `cand_cap=40`. (Doc's shown signature omits the
  ctor's keyword-only `*` marker and the `solver_probe: Optional[ProbeFn]`
  DI parameter — harmless simplification for spec purposes, not an error.)
  The `window_half=9 for v6_live2_ls` comment is confirmed present
  in-file near the cited lines.
- `hexo_rl/selfplay/inference.py:29` → `class LocalInferenceEngine`; ctor
  `__init__(self, model: HexTacToeNet, device, encoding_spec:
  Optional[EncodingSpec] = None)` matches verbatim.
- `engine/src/board/state/core.rs:109` → `Board.moves_remaining: u8` field
  with doc comment "Starts at 1 on ply 0 (P1's single first move), then 2
  for every turn" — confirmed, matches the doc's compound-turn UX
  description exactly (file path in the doc omits the `board/` segment,
  a trivial path typo — `engine/src/board/state/core.rs`, not
  `state/core.rs`, but the cited line number and content are correct).

No corrections beyond the one trivial path-prefix typo noted above; the
architecture, API, and effort-estimate sections all rest on cites that
check out, and no fabricated or unverifiable claims were found.

## RED-TEAM

### Target: conclusion E — S4 build recommendation

**Is "operator plays against own checkpoints" documented anywhere, or
assumed?** Searched `docs/07_PHASE4_SPRINT_LOG.md`, `docs/09_VIEWER_SPEC.md`,
`docs/05_community_integration.md`, `CLAUDE.md` for any prior statement of
this goal ("play against", "own checkpoint", "human vs model" as a
standalone objective) — **zero hits.** `docs/09_VIEWER_SPEC.md`'s own
header states its scope plainly: **"Standalone game viewer and
play-against-model interface,"** status **"Implementation complete as of
2026-04-03."** The goal S4 is building an 8.5-hour new product for
already has a shipped, working implementation in this repo. S4 §2
acknowledges `/viewer` exists and says the two are "architecturally the
same idea" but distinguishes on persistence/deployability/real-DB — it
never establishes that the operator actually *needs* those properties;
that need is **assumed**, not sourced from any dispatcher brief text or
prior doc quoted anywhere in S4. This is the load-bearing gap: the
redundancy risk isn't hexo.did.science's bot slot (that's a different
bot entirely, correctly ruled irrelevant in S4 §1) — it's `/viewer`,
which the doc mentions but doesn't argue past.

**Steel-manning "hexo-play is redundant":** everything in S4 §6's control
surface (checkpoint path, sims slider, `DeployHeadBot`/`SolverBackupBot`
wiring) is a thin config layer over exactly the same
`load_model_with_encoding` → `LocalInferenceEngine` → bot-compose chain
`/viewer` already exercises in production. The *only* genuinely new
capabilities S4 adds over `/viewer` are: (a) persistence across process
restarts (SQLite vs in-memory 500-game ring buffer), (b) not being gated
on the training loop being alive, (c) a public-facing deploy target. None
of these three is named as a requirement anywhere — they're inferred from
the task's *title* ("hexo-play") rather than a stated operator need. If
the actual ask is "let me play my own checkpoints without babysitting a
training run," the **zero-build** answer is: extract `/viewer`'s Flask
route into a tiny standalone script that loads a checkpoint without a
live trainer (a *much* smaller diff than S4's 8.5-hour full rewrite,
since the click-click board UX, hex canvas math, and checkpoint-loading
pattern are all already implemented and proven). S4 never evaluates this
middle option — it jumps from "reuse `/viewer` as-is" (rejected) straight
to "build a whole separate persistent product with SQLite/admin/8 API
routes," skipping the smaller-diff option that would satisfy "play
against own checkpoint" without new persistence/deploy infrastructure.

**Scope-creep in the 1-day estimate:** the spec bundles several features
that don't serve the core ask and inflate the estimate risk: a checkpoint
upload/dropdown UI, an `/admin` bearer-token-gated stats page (mirrors
hexo-strix's `serve` pattern by analogy, not by demonstrated need), a
2-table SQLite schema with a `config_json` duplication column, viewport
pan/zoom/auto-recenter beyond simple auto-fit, and an "advanced"
collapsible knobs panel (`c_visit`/`c_scale`/`c_puct`/`window_half`)
exposed to an end user who, per §6's own advanced-knobs warning ("don't
let an operator set this blind"), probably shouldn't be touching them at
all. Each is individually small, but the doc's own effort table has
**zero slack**: 8.5 hrs summed across backend skeleton (2h), DB (1h),
routes+state-machine (2h), full frontend incl. canvas renderer + admin
page (2.5h), and a *single* combined 1-hour manual smoke test covering
compound-turn win-on-stone-1 edge cases, solver-backup-fires-and-logs
verification, and SQLite row correctness — three independent
correctness properties in one hour, first integration of a new
FastAPI+vanilla-JS stack, no debugging buffer. `docs/09_VIEWER_SPEC.md`'s
existing hex-canvas/click-click implementation (the thing S4 says to
"adapt," not reinvent) already required its own dedicated spec and
implementation pass in this codebase — reusing its *conventions* doesn't
guarantee reusing its *debugged state* when re-implemented framework-free
in a new repo. **A more realistic estimate is 1.5-2 slower-moving days**,
driven by the admin page + upload UI + full knob surface being genuine
adds beyond `/viewer`'s proven baseline, not by the core loop (which is
legitimately cheap, per the REVIEW's clean cite-check).

**Verdict on E: WEAKENED.** Not a kill — a real, separate, persistent
product may be worth building if the operator explicitly wants
deploy-beyond-localhost / survive-training-loop-death properties. But (1)
that requirement is assumed, never sourced, and a much cheaper
`/viewer`-derived standalone script would satisfy the literal stated goal
("play against own checkpoints") if persistence/public-deploy aren't
actually needed; (2) the 1-day estimate has no slack and bundles
speculative features (admin page, upload UI, full advanced-knobs
exposure) that should be cut or explicitly flagged as follow-on, not
priced into a "build in a day" number. Recommend the dispatcher confirm
the persistence/deploy requirement before greenlighting the full build,
and re-scope the MVP to drop `/admin` + checkpoint upload + advanced-knobs
UI (keep them as config-file-only escape hatches) if the 1-day budget is
meant to be a hard constraint.
