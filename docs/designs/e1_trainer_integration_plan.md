# E1 Trainer Integration — 65-bin Distributional Value Head — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a 65-bin distributional value head (two-hot outcome targets, cross-entropy loss, expectation-decode for search) into the production HeXO self-play trainer, selectable against the current scalar head by a config flag, so E1's paired scalar-vs-distributional self-play trajectory can run.

**Architecture:** The value head becomes head-type-switchable at `min_max_window_head` — the single shared per-window head both `forward()` (single-window training) and `aggregated_forward_K` (K-cluster inference) route through. In distributional mode it emits `(B|K, 65)` bin logits and a decoded scalar `E[softmax·linspace(-1,1,65)]`. **Training** consumes the per-row bin logits (CE vs two-hot outcome `z`); **search** consumes only the decoded scalar, so the Rust MCTS contract (`backup.rs:252 value: f32`) and the InferenceServer scalar seam (`inference_server.py:489`) are byte-for-byte unchanged. Because the trainer forwards single-window, there is **no argmin-cluster gradient routing** — `MinMaxPool.min` over the per-cluster decoded scalars already picks the worst cluster at inference, drop-in.

**Tech Stack:** PyTorch (`hexo_rl/model`, `hexo_rl/training`), Rust/PyO3 engine (unchanged), YAML variant configs, pytest via `.venv/bin/python -m pytest`.

## Global Constraints (INV-D1 / R5 guards — every task inherits these)

- **Value target = game OUTCOME `z` ONLY**, two-hot encoded, `soft_z_lambda = 0`. No teacher / TD-bootstrap / distill / solver value in ANY gradient. SealBot is a probe LABEL only. (INV-D1, `run3_d1_distributional_head.md` D1.3.)
- **One variable** across the E1 pair: the value head. Everything else — trunk, policy/aux heads, optimizer, LR, batch, seed, data stream, self-play sims (150), Gumbel m=16 — identical between the scalar arm and the distributional arm.
- **Search interface unchanged:** the distributional head MUST decode to a scalar expectation before any value crosses into Rust. `inference_server.py:489`/`505-512` and `engine/src/mcts/backup.rs` receive a scalar exactly as today. No Rust change in this plan.
- **Bins:** support = `torch.linspace(-1.0, 1.0, 65)`, bin width `2/64 = 1/32 = 0.03125`, fp32 end-to-end (fp16 two-hot mis-splits adjacent bins ~3%). Scalar support over outcome `[-1,+1]` — NOT moves-to-end, NOT margin.
- **State-dict compatibility:** existing scalar checkpoints must still load byte-exact when `value_head_type=scalar` (the default). The new `value_fc2_bins` layer exists only when `value_head_type=dist65`.
- Every strength claim carries: protocol + n + eff_n + per-side compute.

---

## Key scoping findings (read before implementing)

1. **Trainer forwards single-window.** `trainer.py:699 fwd_result = self.model(...)` → `HexTacToeNet.forward()`, value loss at `trainer.py:735`. The K-cluster multi-window path (`aggregated_forward_K`) is inference/eval/self-play-decision only. ⇒ the distributional **training loss is a plain per-row CE**; the design doc's "argmin cluster → CE on that cluster" (`run3_d1_distributional_head.md` D3.1) describes multi-window *scoring*, not the E1 training path. Do not build argmin-cluster gradient routing.
2. **Both value paths share `min_max_window_head`** (`network_min_max_head.py:44`) for the `has_pass_slot` single-window encodings (v6, v6w25, v7*, and the run3 lineage v6_live2_ls). Switch the head there once and both `forward()` and `aggregated_forward_K` inherit it. (The `v8` inline path at `network.py:807-815` is a separate branch, NOT the run3 lineage — out of scope.)
3. **`MinMaxPool.forward` (`pooling.py:95-121`) needs no change** for dist inference: feed it per-cluster *decoded expectations* as `per_cluster_values`; `.min(dim=1)` picks the worst cluster exactly as today. `value_logit = atanh(...)` becomes an unused proxy at inference (InferenceServer reads `value`, ignores the logit).
4. **No buffer/schema change.** `outcomes` (row-player-perspective `z ∈ [-1,1]`) and `value_target_valid` are already emitted (`engine/src/replay_buffer/sample.rs`, per `run3_d1_distributional_head.md` D1.2). Two-hot is built in Python from `outcomes`.
5. **Two-hot primitive already exists** at `scripts/headswap/targets.py` — PORT it into the production package (`hexo_rl/training/`); do not import `scripts/` from the trainer.

---

## File Structure

- **Create** `hexo_rl/training/binned_value.py` — support vector, `scalar_to_two_hot`, `decode_binned_value`, `binned_value_loss`. Single responsibility: the distributional-value primitives (ported + hardened from `scripts/headswap/targets.py`).
- **Modify** `hexo_rl/model/network_min_max_head.py` — `min_max_window_head` gains `value_head_type` + optional `value_fc2_bins`; returns decoded scalar + (v_logit | bin_logits).
- **Modify** `hexo_rl/model/network.py:558-560` — construct `value_fc2_bins` when `value_head_type=dist65`; thread `value_head_type` from ctor into both `forward()` and `aggregated_forward_K` head calls.
- **Modify** `hexo_rl/training/trainer.py:699-735` — select `binned_value_loss` vs `compute_value_loss` by head type; pass bin logits through.
- **Modify** `configs/model.yaml` + `hexo_rl/training/model_defaults.py:28-34` + `hexo_rl/bootstrap/pretrain_cli.py` — `value_head_type` (default `scalar`) + `n_value_bins` (default 65) plumbing.
- **Create** `configs/variants/e1_scalar.yaml`, `configs/variants/e1_dist65.yaml` — the paired E1 variants (identical but the head type + warm-start head path).
- **Create** `hexo_rl/training/warmstart_value_head.py` — load a HEADSWAP head `.pt` onto a fresh net's value head (interface-specified; T6, gated).
- **Create** `scripts/e1/run_pair.sh`, `scripts/e1/validate_ckpt.py` — paired launch + per-checkpoint 234-probe recognition-lag/ECE (interface-specified; T7/T8).
- **Create** `docs/designs/e1_metric_freeze.md` — the A2 frozen-metric artifact (T8; MUST exist before the 5k read).

---

## Task 1: Distributional-value primitives (`hexo_rl/training/binned_value.py`)

**Files:**
- Create: `hexo_rl/training/binned_value.py`
- Test: `hexo_rl/training/tests/test_binned_value.py`

**Interfaces:**
- Produces:
  - `VALUE_SUPPORT: torch.Tensor` — shape `(65,)`, `linspace(-1,1,65)`, fp32, registered on the module (CPU; callers `.to(device)`).
  - `scalar_to_two_hot(z: torch.Tensor, n_bins: int = 65) -> torch.Tensor` — `z (N,)∈[-1,1]` → `(N, 65)` fp32 two-hot rows summing to 1.
  - `decode_binned_value(bin_logits: torch.Tensor) -> torch.Tensor` — `(N, 65)` → `(N, 1)` `E[softmax·support]`, clamped `[-1,1]`.
  - `binned_value_loss(bin_logits, outcome, value_mask=None) -> torch.Tensor` — masked CE vs `two_hot(outcome)`; mirrors `compute_value_loss`'s mask semantics (`losses.py:75-102`): rows with `value_mask==0` excluded from numerator and denominator; empty → `zeros(())`.

- [ ] **Step 1: Write the failing tests**

```python
# hexo_rl/training/tests/test_binned_value.py
import torch
from hexo_rl.training.binned_value import (
    VALUE_SUPPORT, scalar_to_two_hot, decode_binned_value, binned_value_loss,
)

def test_support_shape_and_endpoints():
    assert VALUE_SUPPORT.shape == (65,)
    assert torch.allclose(VALUE_SUPPORT[0], torch.tensor(-1.0))
    assert torch.allclose(VALUE_SUPPORT[-1], torch.tensor(1.0))
    assert torch.allclose(VALUE_SUPPORT[32], torch.tensor(0.0), atol=1e-6)

def test_two_hot_rows_sum_to_one_and_place_on_bin_centers():
    z = torch.tensor([-1.0, 0.0, 1.0])
    th = scalar_to_two_hot(z)
    assert th.shape == (3, 65)
    assert torch.allclose(th.sum(dim=1), torch.ones(3))
    assert th[0, 0] == 1.0            # z=-1 → bin 0
    assert th[2, 64] == 1.0           # z=+1 → bin 64
    assert th[1, 32] == 1.0           # z=0 → bin 32 (exact center)

def test_two_hot_splits_between_adjacent_bins():
    z = torch.tensor([0.0 + 1.0 / 64.0])   # +half a bin above center
    th = scalar_to_two_hot(z)               # pos=(z+1)*32=32.5 → split 32/33
    assert torch.allclose(th[0, 32], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(th[0, 33], torch.tensor(0.5), atol=1e-6)

def test_decode_is_left_inverse_of_two_hot():
    z = torch.linspace(-1, 1, 21)
    logits = torch.log(scalar_to_two_hot(z).clamp_min(1e-9))
    dec = decode_binned_value(logits).squeeze(1)
    assert torch.allclose(dec, z, atol=1e-3)

def test_binned_loss_masks_out_invalid_rows():
    logits = torch.zeros(4, 65, requires_grad=True)
    outcome = torch.tensor([1.0, -1.0, 0.0, 1.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss = binned_value_loss(logits, outcome, value_mask=mask)
    assert loss.requires_grad and torch.isfinite(loss)
    # all-invalid → exact zero scalar, no NaN
    z0 = binned_value_loss(logits, outcome, value_mask=torch.zeros(4))
    assert float(z0) == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest hexo_rl/training/tests/test_binned_value.py -q`
Expected: FAIL — `ModuleNotFoundError: hexo_rl.training.binned_value`.

- [ ] **Step 3: Implement**

```python
# hexo_rl/training/binned_value.py
"""65-bin distributional value primitives (INV-D1: outcome-z only, fp32).

Ported and hardened from scripts/headswap/targets.py so the production trainer
never imports from scripts/. Support is a scalar outcome support over [-1, 1];
NOT moves-to-end / margin / discounted return. fp16 two-hot mis-splits adjacent
bins by ~3% of a bin — keep targets fp32.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

N_VALUE_BINS = 65
VALUE_SUPPORT = torch.linspace(-1.0, 1.0, N_VALUE_BINS)   # (65,), fp32


def scalar_to_two_hot(z: torch.Tensor, n_bins: int = N_VALUE_BINS) -> torch.Tensor:
    """z (N,) in [-1,1] → (N, n_bins) fp32 two-hot (MuZero/KataGo style)."""
    z = z.reshape(-1).to(torch.float32).clamp(-1.0, 1.0)
    scale = (n_bins - 1) / 2.0                      # 32 for 65 bins
    pos = (z + 1.0) * scale                         # [0, n_bins-1]
    lo = torch.floor(pos).to(torch.long).clamp(0, n_bins - 1)
    hi = (lo + 1).clamp(0, n_bins - 1)
    frac = (pos - lo.to(pos.dtype))
    out = torch.zeros(z.shape[0], n_bins, dtype=torch.float32, device=z.device)
    out.scatter_(1, lo.unsqueeze(1), (1.0 - frac).unsqueeze(1))
    out.scatter_add_(1, hi.unsqueeze(1), frac.unsqueeze(1))
    return out


def decode_binned_value(bin_logits: torch.Tensor) -> torch.Tensor:
    """(N, n_bins) logits → (N, 1) E[softmax·support], clamped [-1,1]."""
    probs = F.softmax(bin_logits, dim=-1)
    support = VALUE_SUPPORT.to(bin_logits.device, bin_logits.dtype)
    v = (probs * support).sum(dim=-1, keepdim=True)
    return v.clamp(-1.0, 1.0)


def binned_value_loss(
    bin_logits: torch.Tensor,
    outcome: torch.Tensor,
    value_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked cross-entropy of head softmax vs two-hot(outcome). fp32 targets.

    Mask semantics identical to compute_value_loss (losses.py:75-102):
    value_mask==0 rows excluded from numerator AND denominator; empty→zeros(())."""
    target = scalar_to_two_hot(outcome.reshape(-1))          # (N, 65) fp32
    logp = F.log_softmax(bin_logits.to(torch.float32), dim=-1)
    per_row = -(target * logp).sum(dim=-1)                   # (N,)
    if value_mask is None:
        return per_row.mean()
    mask = value_mask.reshape(-1).bool()
    kept = per_row[mask]
    if kept.numel() == 0:
        return torch.zeros((), device=per_row.device, dtype=per_row.dtype)
    return kept.mean()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest hexo_rl/training/tests/test_binned_value.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add hexo_rl/training/binned_value.py hexo_rl/training/tests/test_binned_value.py
git commit -m "feat(training): 65-bin distributional value primitives (two-hot, decode, CE loss)"
```

---

## Task 2: Head-type-switchable `min_max_window_head` + net construction

**Files:**
- Modify: `hexo_rl/model/network_min_max_head.py:44-95`
- Modify: `hexo_rl/model/network.py:558-560` (construct `value_fc2_bins`), ctor signature, and the two `min_max_window_head` call sites (`forward()` and `aggregated_forward_K` `network.py:956-964`)
- Test: `hexo_rl/model/tests/test_dist_value_head.py`

**Interfaces:**
- `min_max_window_head(..., value_head_type: str = "scalar", value_fc2_bins: nn.Linear | None = None)` returns `(log_policy, value, value_aux)`:
  - `value` `(N,1)`: scalar mode = `tanh(v_logit)`; dist mode = `decode_binned_value(bin_logits)`.
  - `value_aux` `(N,·)`: scalar mode = `v_logit (N,1)`; dist mode = `bin_logits (N,65)`.
- `HexTacToeNet(..., value_head_type="scalar", n_value_bins=65)`: builds `self.value_fc2_bins = nn.Linear(256, n_value_bins)` only for `dist65`; `self.value_head_type` stored; passed to both head call sites.

- [ ] **Step 1: Write the failing test**

```python
# hexo_rl/model/tests/test_dist_value_head.py
import torch
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.binned_value import decode_binned_value

def _mk(head_type):
    # Use the smallest registry encoding the test-suite already builds nets with;
    # mirror an existing network test's ctor call and add value_head_type=...
    return HexTacToeNet(encoding="v6_live2_ls", value_head_type=head_type).eval()

def test_scalar_head_unchanged_output_contract():
    net = _mk("scalar")
    assert not hasattr(net, "value_fc2_bins") or net.value_fc2_bins is None
    # forward returns (log_policy, value(B,1) in [-1,1], v_logit(B,1))

def test_dist_head_builds_bins_and_decodes_to_scalar():
    net = _mk("dist65")
    assert net.value_fc2_bins.out_features == 65
    x = torch.zeros(1, net.in_channels, net.board_size, net.board_size)
    log_policy, value, value_aux = net(x)
    assert value.shape == (1, 1) and (-1.0 <= float(value) <= 1.0)
    assert value_aux.shape == (1, 65)          # bin logits
    assert torch.allclose(value, decode_binned_value(value_aux), atol=1e-5)
```

(Adapt `_mk` to the exact `HexTacToeNet` ctor kwargs an existing `hexo_rl/model/tests/` net test uses — read one first; the added kwarg is `value_head_type`.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest hexo_rl/model/tests/test_dist_value_head.py -q`
Expected: FAIL — `TypeError: unexpected keyword 'value_head_type'`.

- [ ] **Step 3: Implement** — in `network_min_max_head.py`, after computing `v = F.relu(value_fc1(v))` (+ optional bias):

```python
    if value_head_type == "dist65":
        assert value_fc2_bins is not None
        bin_logits = value_fc2_bins(v)                    # (N, 65)
        value = decode_binned_value(bin_logits)           # (N, 1) in [-1,1]
        return log_policy, value, bin_logits
    v_logit = value_fc2(v)
    value = torch.tanh(v_logit)
    return log_policy, value, v_logit
```

(Add `from hexo_rl.training.binned_value import decode_binned_value` — or pass a decode fn to avoid a model→training import; prefer a local `import` inside the function to dodge a circular import if one exists. Add `value_head_type` + `value_fc2_bins` params per the interface.) In `network.py`: add ctor kwargs, build `self.value_fc2_bins` only for dist, and pass `value_head_type=self.value_head_type, value_fc2_bins=self.value_fc2_bins` at both `min_max_window_head(...)` call sites.

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest hexo_rl/model/tests/test_dist_value_head.py -q` → PASS. Also run the existing network suite to prove scalar byte-parity: `.venv/bin/python -m pytest hexo_rl/model/tests/ -q`.

- [ ] **Step 5: Commit**

```bash
git add hexo_rl/model/network_min_max_head.py hexo_rl/model/network.py hexo_rl/model/tests/test_dist_value_head.py
git commit -m "feat(model): head-type-switchable value head (dist65 bin logits + expectation decode)"
```

---

## Task 3: K-cluster inference decode parity (`aggregated_forward_K` + `MinMaxPool`)

**Files:**
- Modify: `hexo_rl/model/network.py:956-971` (feed decoded per-cluster scalars into `MinMaxPool`)
- Test: `hexo_rl/model/tests/test_dist_kcluster_decode.py`

**Interfaces:** consumes Task 2's dist `min_max_window_head`. `aggregated_forward_K` in dist mode must pass per-cluster **decoded** values `(1, K, 1)` as `per_cluster_values`; `MinMaxPool.min` then selects the worst cluster (unchanged). The returned `value` is the min decoded scalar — the drop-in search scalar. `MinMaxPool` itself is NOT modified.

- [ ] **Step 1: Write the failing test** — assert that for a dist net, `aggregated_forward_K` over K identical clusters returns a scalar `value` equal to the single-window `forward()` decode (min of identical = the value), and shape `(1,1)`; and that the value is in `[-1,1]`. (Mirror an existing `aggregated_forward_K` test's input construction — read `hexo_rl/model/tests/` for the K-cluster fixture.)

- [ ] **Step 2: Run to verify failure** (dist branch not wired into `aggregated_forward_K` → wrong shape / uses `v_logit`).

- [ ] **Step 3: Implement** — in the `min_max` branch of `aggregated_forward_K`, when `self.value_head_type == "dist65"`, `per_val` from `min_max_window_head` is already the decoded `(K,1)` scalar (Task 2), so `per_cluster_values=per_val.unsqueeze(0)` already carries decoded expectations — verify no code path re-tanh's it. If `MinMaxPool` is fed decoded values, `.min` + `atanh` proxy are correct; assert the returned `value` is the min decoded scalar. (Likely a 0–2 line guard; the main work is the test proving parity.)

- [ ] **Step 4: Run to verify pass** — new test + full `hexo_rl/model/tests/` green.

- [ ] **Step 5: Commit**

```bash
git add hexo_rl/model/network.py hexo_rl/model/tests/test_dist_kcluster_decode.py
git commit -m "feat(model): K-cluster min-pool consumes dist-decoded per-cluster scalars (search seam unchanged)"
```

---

## Task 4: Config / variant plumbing

**Files:**
- Modify: `configs/model.yaml` (add `value_head_type: scalar`, `n_value_bins: 65`)
- Modify: `hexo_rl/training/model_defaults.py:28-34` (add both keys to `MODEL_HPARAM_DEFAULTS`)
- Modify: `hexo_rl/bootstrap/pretrain_cli.py` (thread the two keys into the `HexTacToeNet` ctor call; validate `value_head_type ∈ {scalar, dist65}`)
- Test: `hexo_rl/training/tests/test_value_head_config.py`

**Interfaces:** `value_head_type` defaults to `scalar` everywhere (byte-parity); a variant yaml sets `model: {value_head_type: dist65}`. Produces: net built from config carries the requested head.

- [ ] **Step 1: Failing test** — load `configs/model.yaml` merged with a `{model: {value_head_type: dist65}}` override, build the net through the same builder `pretrain_cli`/`trainer` uses, assert `net.value_head_type == "dist65"` and `net.value_fc2_bins.out_features == 65`; and that an invalid value raises.
- [ ] **Step 2: Run → FAIL** (key unknown / not threaded).
- [ ] **Step 3: Implement** the three edits per the pattern at `pretrain_cli.py:244-283` (flag validation) and `model_defaults.py:28-34`.
- [ ] **Step 4: Run → PASS** + `.venv/bin/python -m pytest hexo_rl/training/tests/ -q`.
- [ ] **Step 5: Commit** `config(model): value_head_type + n_value_bins plumbing (default scalar)`.

---

## Task 5: Trainer wiring (select loss by head type)

**Files:**
- Modify: `hexo_rl/training/trainer.py:699-735`
- Test: `hexo_rl/training/tests/test_trainer_dist_loss.py`

**Interfaces:** consumes Tasks 1–4. At `trainer.py:735`, when `self.model`'s head is `dist65`, the third forward element is `(B,65)` bin logits → `value_loss = binned_value_loss(bin_logits, outcomes_t, value_mask=value_mask_t)`; else the current `compute_value_loss(v_logit, ...)`. `compute_total_loss` (`losses.py:242-276`) and the `total = policy_loss + value_loss` 1:1 weight are unchanged.

- [ ] **Step 1: Failing test** — build a tiny dist net, run one `train_step` over a synthetic batch (reuse an existing trainer test's batch fixture), assert the loss is finite, `requires_grad`, and that `value_fc2_bins.weight.grad` is populated after `backward()` while `value_fc2` (absent) is not built.
- [ ] **Step 2: Run → FAIL** (trainer still calls `compute_value_loss` on 65-wide logits → shape error).
- [ ] **Step 3: Implement** the head-type branch at `trainer.py:735` (and ensure the forward-unpack at `:699` tolerates a 65-wide third element).
- [ ] **Step 4: Run → PASS** + `.venv/bin/python -m pytest hexo_rl/training/tests/ -q` (scalar path regression-clean).
- [ ] **Step 5: Commit** `feat(training): route dist65 value loss (two-hot CE) in the train step`.

**Gate after Task 5:** run `make test` (Rust + Python, excludes integration) AND `-m integration` closeout (launch-path memory: `not slow and not integration` misses launch-abort bugs). A dist variant must survive a real short launch, not just unit tests.

---

## Task 6: Warm-start value-head loader  [INTERFACE-SPECIFIED — needs a read + operator decision]

**Files:** Create `hexo_rl/training/warmstart_value_head.py`; Test `.../tests/test_warmstart_value_head.py`.

**Decision gate (D-G §6, operator):** warm trunk = run2 **248k** vs **210k** vs **fresh**; and warm the value head from the converged HEADSWAP heads (`/home/timmy/headswap_safe/box_results/headswap/{ab,cd}/arm_*/head_*.pt`: dist ← arm-B, scalar ← arm-A) vs fresh-init. E1 spec §2 pre-registers **warm** (skips the fresh-head transient); build the loader regardless, keep a `--fresh-head` fallback.

**Interface:** `load_value_head(net, head_pt_path, head_type) -> None` maps a HEADSWAP head checkpoint's `value_fc1` + (`value_fc2` | `value_fc2_bins`) tensors onto `net`, asserting shape match; raises on a scalar↔dist mismatch. **Before writing steps:** read `hexo_rl/training/checkpoints.py:408 load_inference_model` for the existing state-dict-map pattern, and inspect one HEADSWAP `head_*.pt` key layout. Then write the TDD steps.

---

## Task 7: Per-checkpoint validation stage  [INTERFACE-SPECIFIED — reuse, needs a read]

**Files:** Create `scripts/e1/validate_ckpt.py`; reuse `scripts/valprobe/measure_recognition_lag.py`, `scripts/valprobe/value_health.py`, `reports/valprobe/probe_set_v1.jsonl` (234 positions).

**Interface:** `validate_ckpt(ckpt_path, arm, out_jsonl)` scores the 234-probe under the deploy decode (multi-window no-drop, dist-decoded scalar) and appends one row to a `value_health_series.jsonl` (the D-G §2 watcher schema — build this AS the watcher's value-health stage, A3): `{step, arm, recognition_lag_mean_v_on_losses, ece, mean_v_on_losses, decoded_auc, tail_mass_auc}`. **Before writing steps:** read the two valprobe scripts' function signatures + the `value_health_series.jsonl` schema. Primary metric = LEVEL (recognition-lag/ECE), AUC secondary (R4).

---

## Task 8: Paired launch harness + A2 metric-freeze artifact

**Files:** Create `configs/variants/e1_{scalar,dist65}.yaml`, `scripts/e1/run_pair.sh`, `docs/designs/e1_metric_freeze.md`.

**A2 (BINDING — must be committed BEFORE the 5k read):** freeze in `e1_metric_freeze.md`, verbatim numeric:
- **Gap metric:** `Δrecog(step) = recog_lag_scalar(step) − recog_lag_dist(step)` (mean decoded-v on the 234 matched lost positions; more-negative dist-v = earlier recognition = better, so a POSITIVE Δ favors dist) **plus** `ΔECE(step) = ECE_scalar(step) − ECE_dist(step)`.
- **Bootstrap:** cluster bootstrap by source game, 10,000 resamples, over the 234 probe positions → CI on the final-point gap.
- **REVIVE** (A1): positive slope of `Δrecog` across {5k,10k,20k,50k} (OLS slope > 0) AND final-point (50k) gap-CI excluding 0. Strict monotonicity dropped.
- **CONFIRM-DEMOTE:** slope ≈ 0 / CI straddles 0 through 50k → scalar head + recalibration + tail monitoring; dist head parked.
- Read to 50k before any CONFIRM-DEMOTE (a 5k/10k NULL is weak). No post-hoc metric edits.

**Variants:** `e1_scalar.yaml` and `e1_dist65.yaml` are byte-identical except `model.value_head_type` (+ warm-start head path). Both: CONFRES-resolved config lineage, `selfplay_stall_timeout_sec` explicit (watchdog armed, A3), promotion-gate process-isolation if landed (R4), self-play sims 150, Gumbel m=16, radius curriculum as run2, checkpoint+validate at {5k,10k,20k,50k}, NO mid-run resume. `run_pair.sh` launches both arms with identical seed/data.

- [ ] Commit variants + `run_pair.sh` + `e1_metric_freeze.md` together: `feat(e1): paired scalar/dist65 variants + launch harness + frozen A2 metric`.

---

## Open operator decisions (surface before execution)

1. **Warm vs fresh trunk** (T6 / D-G §6): 248k vs 210k vs fresh; warm-head vs fresh-head. E1 spec pre-registers warm-from-HEADSWAP-heads on 248k — confirm.
2. **Venue** (R3/A4): E1 on the ORIGINAL box immediately post-run2-stop (second box let expire per the D-F closeout). Do not rush integration to beat the box clock.
3. **Validation ↔ watcher convergence** (T7): build E1's validation stage as the D-G §2 external-eval watcher's value-health stage (shared `value_health_series.jsonl`), so E1 doubles as the run3 watcher rehearsal — confirm.

## Self-review

- **Spec coverage:** two-hot/CE/decode (T1) ✓ E1 spec §2 + card D1.2; head into net (T2) ✓; search-scalar seam unchanged (T2/T3) ✓ §5 "MCTS consumes a scalar"; config flag (T4) ✓; trainer wiring (T5) ✓ §6; warm-start (T6) ✓ §2/§7; 234-probe recog-lag/ECE validation (T7) ✓ §3 primary metric; paired grid 5k/10k/20k/50k + A2 freeze (T8) ✓ §2/§8. INV-D1 λ=0 outcome-only ✓ Global Constraints. Aux/moves-left head explicitly OUT (card D1.2). No Rust change ✓.
- **Placeholder scan:** T1–T5 carry complete code / exact edits. T6–T8 are explicitly INTERFACE-SPECIFIED with a named read-before-steps prerequisite (checkpoint loader / valprobe signatures) or an operator gate — flagged, not hidden. Detailed TDD steps for T6–T8 are written after those reads.
- **Type consistency:** `value_aux` third-element convention (v_logit | bin_logits) is uniform across T2/T3/T5; `binned_value_loss(bin_logits, outcome, value_mask)` signature matches its T1 definition and T5 call; `decode_binned_value` used identically in T1/T2.
