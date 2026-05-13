<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §80 — Eval Determinism Fix: Temperature + Random Openings (2026-04-12)

### Root cause (from §70)

`Evaluator.evaluate()` constructed `ModelPlayer` without a `temperature` arg, so it
defaulted to `temperature=0.0`. `get_policy(τ=0)` returns a one-hot argmax policy.
With no stochastic element anywhere in the eval loop, all 50 games in a SealBot
match were bit-identical. Bradley-Terry CIs were ±100K — completely uninformative.

### Fix

Three targeted changes, all in the formal eval pipeline only. Training path untouched.

**1. Temperature propagation** (`hexo_rl/eval/evaluator.py`)

`Evaluator.__init__` reads `eval_temperature` (default 0.5) from the `evaluation`
config sub-dict. `evaluate()` passes `temperature=self._eval_temperature` when
constructing `ModelPlayer`. `evaluate_vs_model()` does the same for the opponent
player so the best-checkpoint matchup is symmetric.

**2. Per-game seeding** (`hexo_rl/eval/evaluator.py`)

At the start of each game loop iteration `i`, the evaluator calls:
```python
np.random.seed(self._eval_seed_base + i)
random.seed(self._eval_seed_base + i)
```
This seeds both the temperature sampling in `ModelPlayer.get_move()` (uses
`np.random.choice`) and the random opening selection (uses `random.choice`).
Games are reproducible across eval runs AND distinct from each other.

**3. Random opening plies** (`hexo_rl/eval/evaluator.py`)

For the first `eval_random_opening_plies` plies (default 4 = 2 compound moves),
both players' moves are replaced by `random.choice(board.legal_moves())`. This
creates genuinely diverse starting positions before model play begins, ensuring
even early-training checkpoints (whose policy is near-uniform) produce varied games.

**Config keys added** (`configs/eval.yaml`, propagated in `eval_pipeline.py`):

```yaml
eval_temperature: 0.5            # >0 → sample from visit dist; 0 → old argmax
eval_random_opening_plies: 4     # random opening plies for both players
eval_seed_base: 42               # game i uses seed base+i
```

**Tests added** (`tests/test_model_player.py`): 4 CPU-only unit tests covering
temperature variance, opening-ply bypass, deterministic regression guard, and
E2E seeded-opening diversity.

**Backward compat:** `ModelPlayer.__init__` signature unchanged (`temperature`
arg already existed). `eval_diagnostic.py` bypasses `Evaluator` — unaffected.
Old argmax + no-opening behaviour restored via `eval_temperature: 0.0` and
`eval_random_opening_plies: 0`.

### Effect

Next eval run against SealBot will produce 50 distinct games with varied opening
positions. Bradley-Terry CIs will reflect real checkpoint discrimination.

---

