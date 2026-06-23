# Bucket B1 — Training Stats Audit
**Phase 1 · Auditor: sonnet · Date: 2026-06-23**

---

## Per-stat verdicts

### loss
- **Emit site:** `trainer.py:1019` (`log.info("train_step", ..., total_loss=result["loss"])`).  
- A: `loss.item()` — scalar from `compute_total_loss`. Formula correct.  
- B: Per-step scalar, no aggregation claim, no CI.  
- C: Not planner-dependent. Value is training loss, regime-agnostic.  
- D: No band.  
- E: Names what it measures.  
- F: Not duplicated.  
- **CORRECT**

---

### policy_loss
- **Emit site:** `trainer.py:1020`  
- A: `policy_loss.item()` — KL(target||model) or CE from `compute_policy_loss`. Formula correct.  
- B: Per-step, no CI.  
- C: Policy target is Gumbel completed-Q improved policy; cross-entropy against it is well-defined under Gumbel. No Axis C issue.  
- D: No band.  
- E: Valid construct — optimized quantity.  
- F: Not duplicated.  
- **CORRECT**

---

### value_loss
- **Emit site:** `trainer.py:1021`  
- A: `value_loss.item()` — BCE with logits over value head. Formula correct.  
- B: Per-step, no CI.  
- C: Regime-agnostic.  
- D: No band.  
- E: Valid — the actual value-head training loss.  
- F: `value_loss_main = result["value_loss"]` (trainer.py:990) — `value_loss_main` is a literal alias. CANONICAL here.  
- **CORRECT**

---

### value_loss_main
- **Emit site:** `trainer.py:1022`  
- A: `result["value_loss_main"] = result["value_loss"]` (trainer.py:990) — identical float.  
- B: Per-step, no CI.  
- C: N/A.  
- D: No band.  
- E: Named differently but is byte-identical to `value_loss`.  
- F: **FAIL** — `value_loss_main` is an exact alias of `value_loss`; zero marginal signal.  
- **REDUNDANT** → canonical: `value_loss`

---

### value_loss_uncertainty
- **Emit site:** `trainer.py:1023`  
- A: `uncertainty_weight * result["uncertainty_loss"] if use_uncertainty else 0.0` (trainer.py:991-992). Formula correct — weighted contribution.  
- B: Per-step, no CI.  
- C: N/A.  
- D: No band.  
- E: Valid — weighted uncertainty-head loss component.  
- F: Not duplicated.  
- **CORRECT**

---

### value_loss_aux
- **Emit site:** `trainer.py:1024`  
- A: `aux_weight * result["opp_reply_loss"] if use_aux else 0.0` (trainer.py:994-995). Formula correct.  
- B: Per-step, no CI.  
- C: N/A.  
- D: No band.  
- E: **FAIL** — named `value_loss_aux` but `opp_reply_loss` is a **policy-shaped** head (opponent reply prediction), not a value-head auxiliary. The code comment at trainer.py:986-988 explicitly warns: "opp_reply aux is a POLICY-shaped head and can dominate the composite". The name implies this belongs to value diagnostics; it does not. Axis E fail.  
- **BIASED** → fix: rename to `policy_aux_loss_weighted` or annotate clearly in the event schema; remove from any value-head dashboard grouping.

---

### value_loss_composite
- **Emit site:** `trainer.py:1025`  
- A: `value_loss_main + value_loss_uncertainty + value_loss_aux` (trainer.py:997-1001). Arithmetic correct.  
- B: Per-step, no CI.  
- C: N/A.  
- D: No band.  
- E: **FAIL** — same Axis E issue as `value_loss_aux`: the composite includes a policy-shaped head (`opp_reply_loss`). Described in code comment (trainer.py:986-988) as NOT a clean "value-head signal". The name implies value-side composite; it is actually "total loss minus pure-policy accounting". Misleading for diagnosis.  
- **BIASED** → fix: rename to `mixed_head_loss_total` or add explicit caveat in monitoring; document that this ≠ value-head health indicator.

---

### value_accuracy
- **Emit site:** `trainer.py:1027`  
- A: `(pred_win == target_win).float().mean()` where `pred_win = (v_logit > 0)`, `target_win = (outcomes_t > 0)` (trainer.py:915-917). Formula correct.  
- B: Per-step batch fraction, no CI.  
- C: N/A.  
- D: No band.  
- E: **FAIL** — includes draw rows (outcome = draw_value ≈ -0.5, so `target_win = 0`), ply-capped rows (`ply_cap_value` ∈ (-1,0)), and corpus rows with non-±1 targets. A perfect value head on decided games would be deflated by these non-decided rows. Code comment (trainer.py:177-184) explicitly documents this as the reason `value_accuracy_masked` was added. Raw `value_accuracy` is a biased downward estimator of decided-game winner-calling accuracy. Axis E fail.  
- **BIASED** → fix: dashboard should primary-display `value_accuracy_masked` (decided+supervised rows only); keep `value_accuracy` as legacy diagnostic with caveat label.

---

### value_accuracy_masked
- **Emit site:** `trainer.py:1028`  
- A: `_mean_over(correct, supervised & decided)` (trainer.py:241). `supervised` = `value_mask` (ply-cap rows excluded), `decided` = `|z| > 0.999` (trainer.py:228). Only decisive, value-mask-valid rows counted. Formula correct.  
- B: Per-step, no CI. Coverage depends on batch composition — NaN when no decided+supervised rows in batch (handled).  
- C: N/A.  
- D: No band.  
- E: Valid — this IS the clean "did value head correctly call winner" stat, as documented.  
- F: Partially overlaps `value_accuracy_corpus` + `value_accuracy_selfplay` but carries different masking.  
- **CORRECT**

---

### value_accuracy_corpus
- **Emit site:** `trainer.py:1029`  
- A: `_mean_over(correct, is_corpus)` (trainer.py:242). `is_corpus = rows[:n_pretrain]`. Formula correct. NOTE: per docstring (trainer.py:188-192), corpus includes §178 bot-corpus rows (folded into n_pretrain upstream) — not separable.  
- B: Per-step, no CI. NaN when n_pretrain=0.  
- C: N/A.  
- D: No band.  
- E: Valid — unmasked accuracy for corpus slice. Correctly documented.  
- F: Not duplicated.  
- **CORRECT**

---

### value_accuracy_selfplay
- **Emit site:** `trainer.py:1030`  
- A: `_mean_over(correct, is_selfplay)` (trainer.py:243). `is_selfplay = rows[n_pretrain:]`. Formula correct.  
- B: Per-step, no CI. NaN when entire batch is corpus.  
- C: N/A.  
- D: No band.  
- E: Valid — per-source decomposition.  
- F: Not duplicated.  
- **CORRECT**

---

### value_bce_corpus
- **Emit site:** `trainer.py:1031`  
- A: `_mean_over(per_row_bce, is_corpus & supervised)` (trainer.py:244). `per_row_bce = BCE_with_logits(logit, (z+1)/2, reduction='none')` (trainer.py:220-222). Formula matches `compute_value_loss` semantics per docstring.  
- B: Per-step, no CI. NaN when corpus supervised count = 0.  
- C: N/A.  
- D: No band.  
- E: Valid — per-source BCE diagnostic. Correctly weighted by `*_supervised` counts to reproduce `value_loss`.  
- F: Not duplicated.  
- **CORRECT**

---

### value_bce_selfplay
- **Emit site:** `trainer.py:1032`  
- A: `_mean_over(per_row_bce, is_selfplay & supervised)` (trainer.py:245). Formula correct.  
- B: Per-step, no CI. NaN when selfplay supervised count = 0.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT**

---

### value_rows_corpus
- **Emit site:** `trainer.py:1033`  
- A: `n_pre = max(0, min(int(n_pretrain), batch_n))` (trainer.py:253). Literal count of corpus rows in batch. Formula correct.  
- B: Count not subject to CI issues.  
- C: N/A.  
- D: No band.  
- E: Diagnostic count — valid.  
- F: Not duplicated at per-step level.  
- **CORRECT**

---

### value_rows_selfplay
- **Emit site:** `trainer.py:1034`  
- A: `batch_n - n_pre` (trainer.py:254). Correct complement.  
- B: Count.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Trivially recoverable as `n_rows_total - value_rows_corpus` but the combination is non-obvious due to masking. Keep.  
- **CORRECT**

---

### value_rows_masked
- **Emit site:** `trainer.py:1035`  
- A: `n_masked = int((supervised & decided).sum())` (trainer.py:241). Count of rows used in `value_accuracy_masked`. Formula correct.  
- B: Count.  
- C: N/A.  
- D: No band.  
- E: Valid denominator for `value_accuracy_masked`.  
- F: Not duplicated.  
- **CORRECT**

---

### value_rows_corpus_supervised
- **Emit site:** `trainer.py:1036`  
- A: `n_corpus_sup` from `_mean_over(per_row_bce, is_corpus & supervised)` (trainer.py:244). Count of corpus rows that pass value_mask. Correct.  
- B: Count.  
- C: N/A.  
- D: No band.  
- E: Valid — denominator for `value_bce_corpus`.  
- F: Not duplicated.  
- **CORRECT**

---

### value_rows_selfplay_supervised
- **Emit site:** `trainer.py:1037`  
- A: `n_selfplay_sup` from `_mean_over(per_row_bce, is_selfplay & supervised)` (trainer.py:245). Count of selfplay rows passing value_mask. Correct.  
- B: Count.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT**

---

### policy_entropy
- **Emit site:** `trainer.py:957`  
- A: `torch.special.entr(p_fp32).sum(dim=-1).mean()` (trainer.py:1080) — Shannon entropy H = -Σ π log π in nats. Formula correct.  
- B: Per-step batch mean, no CI.  
- C: This is MODEL output entropy on training batch, not MCTS visit distribution. Regime-agnostic. No Axis C issue.  
- D: **has_band=true.** Band: alert `< 1.0 nats`, warn `< 2.0 nats` (MonitoringConfig:35-36, terminal_dashboard.py:438-443). Actual banked values: min=2.141, max=2.889, mean=2.455 (n=76). Band direction correct (low entropy = collapse). Thresholds (1.0/2.0) provide collapse detection; actual run sits just above warn threshold. No evidence these thresholds were re-calibrated for Gumbel vs PUCT — but model policy entropy is planner-independent (same logits either way). Thresholds appear empirically derived from PUCT-era collapse observations. Minor concern but not falsifiably wrong for Gumbel.  
- E: Valid — model's policy spread.  
- F: Not duplicated.  
- **CORRECT**

---

### policy_entropy_pretrain
- **Emit site:** `trainer.py:958`  
- A: `torch.special.entr(p_fp32[:n_pretrain]).sum(dim=-1).mean()` (trainer.py:1087). Entropy over corpus rows only. Formula correct. NaN when n_pretrain=0.  
- B: Per-step, no CI.  
- C: N/A.  
- D: No band.  
- E: Valid per-source split.  
- F: Not duplicated.  
- **CORRECT**

---

### policy_entropy_selfplay
- **Emit site:** `trainer.py:959`  
- A: `torch.special.entr(p_fp32[n_pretrain:]).sum(dim=-1).mean()` (trainer.py:1088). Entropy over selfplay rows only. Formula correct. NaN when entire batch is corpus.  
- B: Per-step, no CI.  
- C: N/A.  
- D: **has_band=true.** Band effectively applied via `selfplay_model_entropy_batch` alias in dashboard (terminal_dashboard.py:542,552-557): collapse < 1.5 nats, warn < 2.0 nats. Same calibration comment as `policy_entropy`. Banked values show selfplay entropy ~2.13, sitting near the warn floor. Directionally correct.  
- E: Valid.  
- F: **FAIL** — `selfplay_model_entropy_batch` (trainer.py:960) is `= entropies["policy_entropy_selfplay"]`, an explicitly documented alias ("alias; drop 2026-05-28"). Both are emitted and the alias has not been dropped. Zero marginal signal.  
- **REDUNDANT** → canonical: `policy_entropy_selfplay`; `selfplay_model_entropy_batch` to drop.

---

### selfplay_model_entropy_batch
- **Emit site:** `trainer.py:960`  
- A: `entropies["policy_entropy_selfplay"]` — explicit alias (trainer.py:960 comment: "alias; drop 2026-05-28"). Formula identical to `policy_entropy_selfplay`.  
- B: N/A.  
- C: N/A.  
- D: **has_band=true.** Band here (dashboard reads `selfplay_model_entropy_batch`, not `policy_entropy_selfplay`). The alias carries the active band.  
- E: N/A.  
- F: **FAIL** — duplicate of `policy_entropy_selfplay`. Documented for deletion (2026-05-28 deadline passed; still present 2026-06-23).  
- **REDUNDANT** → canonical: `policy_entropy_selfplay`. Drop. Migrate dashboard band reference to `policy_entropy_selfplay`.

---

### policy_entropy_recent
- **Emit site:** `trainer.py:961`  
- A: `torch.special.entr(p_fp32[n_pretrain:n_pretrain+n_recent]).sum(dim=-1).mean()` (trainer.py:1100-1103). Recent-buffer slice entropy. NaN when n_recent=0. Formula correct.  
- B: Per-step, no CI. NaN when recent buffer absent — first-class signal.  
- C: N/A.  
- D: No band.  
- E: Valid — finer-grained split of selfplay buffer.  
- F: Not duplicated.  
- **CORRECT**

---

### policy_entropy_uniform_selfplay
- **Emit site:** `trainer.py:962`  
- A: `torch.special.entr(p_fp32[n_pretrain+n_recent:]).sum(dim=-1).mean()` (trainer.py:1104-1107). Uniform-selfplay slice entropy. When no recent buffer, falls back to `policy_entropy_selfplay` (trainer.py:1110). Formula correct.  
- B: Per-step, no CI.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT**

---

### policy_target_entropy
- **Emit site:** `trainer.py:963`  
- A: `-(_tgt * _tgt.clamp_min(1e-9).log()).sum(-1).mean()` over `policy_valid` rows (trainer.py:930-933). Shannon entropy of MCTS improved-policy targets. Formula correct.  
- B: Per-step, no CI.  
- C: Under Gumbel, the `policies_t` tensor holds completed-Q improved-policy targets (confirmed via `gumbel_search_py.py` and `completed_q.py`). Entropy of this distribution is meaningful and regime-correct. No Axis C issue.  
- D: No band.  
- E: Valid — measures how concentrated search-improved policy targets are.  
- F: Overlaps with `policy_target_entropy_fullsearch` (subset when all rows are full-search). At runtime, `policy_target_entropy` covers all `policy_valid` rows while `_fullsearch` covers only the full-search subset. Distinct when fastsearch rows are present. Keep.  
- **CORRECT**

---

### policy_target_entropy_fullsearch
- **Emit site:** `trainer.py:948` (via `compute_policy_target_metrics`)  
- A: `H_full_s / n_full` where `H = torch.special.entr(p).sum(dim=-1)` and mask = `pvb & fs` (trainer.py:127,134,150). Formula correct.  
- B: Per-step, no CI. NaN when no full-search rows.  
- C: Under Gumbel, full-search = positions with full MCTS budget (Gumbel-SH). The entropy of the improved policy for these positions is meaningful. No Axis C issue.  
- D: No band.  
- E: Valid — full-search policy quality split.  
- F: Not duplicated.  
- **CORRECT**

---

### policy_target_entropy_fastsearch
- **Emit site:** `trainer.py:948` (via `compute_policy_target_metrics`)  
- A: `H_fast_s / n_fast` where mask = `pvb & (~fs)` (trainer.py:134,151). Formula correct. NaN when no fastsearch rows.  
- B: Per-step, no CI.  
- C: Under Gumbel regime, fastsearch = positions with reduced MCTS budget. Entropy of improved policy for low-budget search is well-defined. No Axis C issue.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT**

---

### policy_target_kl_uniform_fullsearch
- **Emit site:** `trainer.py:948` (via `compute_policy_target_metrics`)  
- A: `KL_full_s / n_full` where `KL_uniform = log_N - H`, `log_N = ln(A)` (trainer.py:128-129,152). KL divergence from uniform to target policy, averaged over full-search rows. Formula correct (KL(target||uniform) = log_N - H(target)).  
- B: Per-step, no CI.  
- C: Valid under Gumbel.  
- D: No band.  
- E: Valid — concentration of full-search targets relative to uniform.  
- F: Trivially computable as `log_N - policy_target_entropy_fullsearch` but emitted directly for dashboard convenience.  
- **CORRECT**

---

### policy_target_kl_uniform_fastsearch
- **Emit site:** `trainer.py:948` (via `compute_policy_target_metrics`)  
- A: `KL_fast_s / n_fast` (trainer.py:153). Formula correct. NaN when no fastsearch rows.  
- B: Per-step, no CI.  
- C: Valid under Gumbel.  
- D: No band.  
- E: Valid.  
- F: Trivially `log_N - policy_target_entropy_fastsearch`. Keep.  
- **CORRECT**

---

### frac_fullsearch_in_batch
- **Emit site:** `trainer.py:948` (via `compute_policy_target_metrics`)  
- A: `float(n_full) / batch_n` where `n_full = sum(pvb & fs)`, `batch_n = pvb.numel()` (trainer.py:147,154). This computes (policy_valid & full_search) / total_rows.  
- B: Per-step fraction.  
- C: N/A.  
- D: No band.  
- E: N/A.  
- F: **FAIL** — `full_search_frac` (trainer.py:967) computes `(policy_valid & full_search_mask).float().mean()` (trainer.py:943) which equals `sum(pvb & fs) / batch_n` = same numerator and denominator as `frac_fullsearch_in_batch`. Empirically: both appear in log at same step with matching values (events log `frac_fullsearch_in_batch=0.879`, train log `full_search_frac=0.871` for different steps — different steps, not same event; but formula is identical). Zero marginal signal.  
- **REDUNDANT** → canonical: `full_search_frac` (emitted in primary `train_step` structlog); `frac_fullsearch_in_batch` is the events-log duplicate.

---

### n_rows_policy_loss
- **Emit site:** `trainer.py:948` (via `compute_policy_target_metrics`)  
- A: `n_full = int(round(mf.sum()))` = count of full-search valid policy rows (trainer.py:146,155). Formula correct.  
- B: Count.  
- C: N/A.  
- D: No band.  
- E: Valid — denominator for full-search policy metrics.  
- F: Not duplicated (different from `n_rows_total`).  
- **CORRECT**

---

### n_rows_total
- **Emit site:** `trainer.py:948` (via `compute_policy_target_metrics`)  
- A: `n_valid = int(round(pvf.sum()))` = count of all policy-valid rows (trainer.py:146,156). Formula correct.  
- B: Count.  
- C: N/A.  
- D: No band.  
- E: Valid — total policy-valid denominator.  
- F: Not duplicated.  
- **CORRECT**

---

### grad_norm
- **Emit site:** `trainer.py:964`  
- A: L2 grad norm from `fp16_backward_step`, after clipping to `grad_clip` (trainer.py:878-882). Value is the PRE-CLIP grad norm (or post-clip if clipped); standard convention. Formula correct.  
- B: Per-step, no CI. `aggregates: false`.  
- C: N/A.  
- D: **has_band=true.** Band: alert when `> 10.0` (MonitoringConfig:37, alert_rules.py:63). Banked values: min=0.86, max=2.88, mean=1.16. Threshold 10.0 is a widely-used "NaN precursor" heuristic. Not Gumbel-specific. Directionally correct. Threshold is conservative (no recent violations).  
- E: Valid.  
- F: Not duplicated at per-step level.  
- **CORRECT**

---

### lr
- **Emit site:** `trainer.py:966`  
- A: `self.optimizer.param_groups[0]["lr"]` (trainer.py:936). Actual LR from optimizer state. Formula correct.  
- B: Per-step, no CI. `aggregates: false`.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT**

---

### full_search_frac
- **Emit site:** `trainer.py:967`  
- A: `(policy_valid & full_search_mask_t).float().mean()` (trainer.py:942-945). When full_search_mask_t is None, = `policy_valid.float().mean()`. Formula correct.  
- B: Per-step, no CI.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Identical formula to `frac_fullsearch_in_batch` (see that entry). This is the CANONICAL form (emitted in the primary `train_step` structlog entry).  
- **CORRECT** (canonical; `frac_fullsearch_in_batch` is the redundant duplicate)

---

### fp16_scale
- **Emit site:** `trainer.py:1045`  
- A: `self.scaler.get_scale()` — GradScaler loss scale. Formula correct. `aggregates: false`.  
- B: Per-step.  
- C: N/A.  
- D: No band.  
- E: Valid — fp16 training stability diagnostic.  
- F: Not duplicated.  
- **CORRECT**

---

### opp_reply_loss
- **Emit site:** `trainer.py:1038`  
- A: `opp_reply_loss.item()` — the raw (unweighted) opp-reply head loss (trainer.py:1139). Formula correct.  
- B: Per-step, conditional (only when `use_aux=True`). Not in banked sample (recent log shows `aux_loss: None`).  
- C: N/A.  
- D: No band.  
- E: Valid — the raw auxiliary policy-shaped head loss before weighting.  
- F: `value_loss_aux = aux_weight * opp_reply_loss` (trainer.py:994). The weighted version is a derivable multiple. Some redundancy but different scale makes it non-trivially redundant.  
- **CORRECT**

---

### uncertainty_loss
- **Emit site:** `trainer.py:1039`  
- A: `unc_loss.item()` — raw heteroscedastic uncertainty loss from `compute_uncertainty_loss` (trainer.py:1143). Formula correct.  
- B: Per-step, conditional (`use_uncertainty=True`). Present in banked sample (uncertainty_loss values observed ~0.49-0.49).  
- C: N/A.  
- D: No band.  
- E: Valid — uncertainty head loss.  
- F: `value_loss_uncertainty = uncertainty_weight * uncertainty_loss`. Raw vs weighted — not trivially redundant (different scale).  
- **CORRECT**

---

### avg_sigma
- **Emit site:** `trainer.py:1144`  
- A: `sigma2.float().sqrt().mean()` (trainer.py:1142). Mean predicted standard deviation (sqrt of predicted variance) over the batch. Formula correct. Comment: "sqrt(predicted squared err) since Wave 4 4B-impl-5".  
- B: Per-step, no CI.  
- C: N/A.  
- D: No band.  
- E: Valid — uncertainty head calibration diagnostic.  
- F: Not duplicated.  
- **CORRECT**

---

### ownership_loss
- **Emit site:** `trainer.py:1040`  
- A: `own_loss.item()` — ownership head loss (trainer.py:1146-1147). Conditional (`use_ownership=True` and `own_loss is not None`). Formula correct.  
- B: Per-step, conditional.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT**

---

### threat_loss
- **Emit site:** `trainer.py:1041`  
- A: `thr_loss.item()` (trainer.py:1148-1149). Conditional (`use_threat=True`). Formula correct.  
- B: Per-step, conditional.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT**

---

### chain_loss
- **Emit site:** `trainer.py:1042`  
- A: `chain_loss.item()` (trainer.py:1150). Conditional (`use_chain=True`). Formula correct.  
- B: Per-step, conditional.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT**

---

### ply_index_loss
- **Emit site:** `trainer.py:1151`  
- A: `ply_index_loss.item()` (trainer.py:1152). Conditional (`use_ply_index=True`). Formula correct.  
- B: Per-step, conditional. Not present in banked sample.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT**

---

### aux_loss_rows
- **Emit site:** `trainer.py:1154`  
- A: `max(0, batch_n - n_pretrain)` (trainer.py:1154). Count of selfplay rows in batch (rows eligible for aux/ownership heads). Formula correct.  
- B: Count.  
- C: N/A.  
- D: No band.  
- E: Valid denominator for aux-head loss interpretation.  
- F: Nearly identical to `value_rows_selfplay = batch_n - n_pre`. At trainer.py:254: `value_rows_selfplay = batch_n - n_pre`. At trainer.py:1154: `max(0, batch_n - n_pretrain)` — same value (n_pre = max(0,min(n_pretrain,batch_n)), and max(0,...) guards edge). Effectively duplicate.  
- **REDUNDANT** → canonical: `value_rows_selfplay`

---

### per_source_grad_norm
- **Emit site:** `trainer.py:865`  
- A: `compute_per_source_grad_attribution` — L2 norm of gradient w.r.t. each source slice loss × parameter group (track_b_attribution.py:43-73). Formula correct: `√Σ g²` per slice-group combination.  
- B: Per-step, diagnostic-mode-only (`_track_b_grad_attribution` flag). Not in banked sample — not present in any banked log event.  
- C: N/A.  
- D: No band.  
- E: Valid — gradient attribution diagnostic.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: only emits when Track B attribution mode enabled; absent from banked sample — emit site confirmed at trainer.py:865)

---

### axis_q
- **Emit site:** `events.py:122` (via `emit_axis_distribution`)  
- A: Fraction of same-color Q-axis pairs among recent games (computed by `compute_axis_fractions`, called with `pool.recent_move_histories`). Formula correct.  
- B: Aggregates over last ≤100 games. Game count not deduped but these are distinct games by construction. No CI.  
- C: N/A.  
- D: **has_band=true.** Warn at max_frac ≥ 0.45, alert at ≥ 0.50 (events.py:90-91). **FAIL** — corpus baseline shows axis_r = 0.4526, axis_q = 0.4517, already above the 0.45 warn threshold (events_cdf24392b8414486a28424673f221575.jsonl, `axis_distribution_baseline_loaded` event). Warn fires on HEALTHY corpus-matching selfplay behavior. Band calibrated below the natural baseline → false alarms at healthy operating point. Axis D fail.  
- E: Valid construct — axis concentration is a real pattern indicator.  
- F: Partially overlaps `axis_r`, `axis_s`, `axis_max` but each captures distinct axis.  
- **BIASED** → fix: raise warn threshold above the corpus baseline max (~0.46-0.47 with margin); alert at 0.50+ remains reasonable as genuine over-concentration signal.

---

### axis_r
- **Emit site:** `events.py:122`  
- A: Same as axis_q but for R-axis. Formula correct.  
- B: Same comment.  
- C: N/A.  
- D: **has_band=true.** Same band as axis_q. **FAIL** — corpus baseline r=0.4526 > warn threshold 0.45. Axis D fail.  
- E: Valid.  
- F: Not duplicated (distinct axis from q, s).  
- **BIASED** → fix: same as axis_q.

---

### axis_s
- **Emit site:** `events.py:122`  
- A: S-axis fraction. Formula correct.  
- B: Same.  
- C: N/A.  
- D: **has_band=true.** Corpus baseline s=0.4479 < warn threshold 0.45 — S-axis does not false-alarm at baseline. However all three axes share the same threshold and the warn fires on max(q,r,s) ≥ 0.45; since max is driven by q or r (both ≥ 0.45 baseline), the composite check still warns on healthy baseline.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT** (per-axis S formula and individual value are fine; the composite `max_frac` warn trigger is what's biased, addressed by axis_q/axis_r fix).

---

### axis_max
- **Emit site:** `events.py:122`  
- A: `metrics["axis_max"]` — label of the max-fraction axis (trainer.py:87, computed in `compute_axis_fractions`). Qualitative label, not numeric stat.  
- B: N/A.  
- C: N/A.  
- D: No separate numeric band for axis_max label.  
- E: Valid — identifies which axis dominates.  
- F: Derivable from axis_q, axis_r, axis_s but useful as a label.  
- **CORRECT**

---

### early_game_entropy_mean
- **Emit site:** `events.py:188` (via `emit_training_events` → `early_game_probe.compute`)  
- A: `float(entropy_cpu.mean())` where `entropy = -(p_legal * log_p_legal * legal_mask).sum(-1)` over 10-position fixture (early_game_probe.py:278-285). Shannon entropy over legal moves, renormalized. Formula correct.  
- B: Fixed 10-position fixture — n_eff = 10 positions. Small but consistent (same fixture every call). No CI.  
- C: This probes the MODEL, not the MCTS planner. Regime-agnostic.  
- D: **has_band=true.** WARN when `> 4.5 nats` (early_game_probe.py:295, events.py:185). Direction: warn when entropy is HIGH (model too uncertain / overconfident exploration). Comment: "Bootstrap-v4 reads ~3.0-4.0 nat; §116 read ~5.4". Threshold 4.5 placed between bootstrap and high-entropy regime. Direction is INVERTED relative to the training stats bands (here HIGH = warn, there LOW = warn) — this is intentional and correct for this probe (early-game probe warns against remaining too diffuse). **However: probe FAILED in the banked run** (`early_game_probe_failed` logged; `early_game_entropy_mean=None` in `train_step_summary`). Emit site confirmed but absent from banked sample.  
- E: Valid construct — probes early-game policy concentration.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: probe failed in banked run; real-run behavior unobserved from banked sample)

---

### early_game_top1_mass_mean
- **Emit site:** `events.py:323` (`train_step_summary` structlog, `early_game_top1_mass_mean=...`)  
- A: `float(top1_cpu.mean())` where `top1 = (p_legal * legal_mask).max(dim=-1).values` (early_game_probe.py:279,287). Mean max legal-renorm probability over 10 positions. Formula correct.  
- B: n_eff=10 positions.  
- C: Regime-agnostic model probe.  
- D: No band.  
- E: Valid complement to entropy.  
- F: Partially redundant with `early_game_entropy_mean` (both measure concentration). Keep — they capture different aspects.  
- **CORRECT** (coverage gap: same probe failure as `early_game_entropy_mean`)

---

### colony_frac
- **Emit site:** `track_b_buffer_snapshot.py:92`  
- A: `col_frac = n_col / n_total` where `n_col = (classes == "colony").sum()` (track_b_buffer_snapshot.py:81,85). Position-class fraction over sampled positions. Formula correct.  
- B: Aggregates over sampled positions (sample size from `_frac_and_mean`). Not in banked sample (`buffer_position_class_snapshot` events absent — instrumentation_enabled=False in the logged run). Emit site confirmed.  
- C: N/A.  
- D: No band on the stat itself.  
- E: Valid — measures buffer composition by play style.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: not in banked sample)

---

### extension_frac
- **Emit site:** `track_b_buffer_snapshot.py:93`  
- A: `ext_frac = n_ext / n_total`. Formula correct.  
- B: Same coverage gap.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: `1 - colony_frac - neither_frac` but individual values add information.  
- **CORRECT** (coverage gap: not in banked sample)

---

### neither_frac
- **Emit site:** `track_b_buffer_snapshot.py:94`  
- A: `nei_frac = n_nei / n_total`. Formula correct. By construction `colony_frac + extension_frac + neither_frac = 1.0`.  
- B: Same coverage gap.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Derivable as `1 - colony_frac - extension_frac`. Minor redundancy but not zero-marginal-signal (partition confirmation).  
- **CORRECT** (coverage gap: not in banked sample)

---

### colony_mean_value_target
- **Emit site:** `track_b_buffer_snapshot.py:96`  
- A: `col_mean` from `_frac_and_mean(classes=="colony")` — mean `value_target` over colony-classified positions. NaN when no colony positions. Formula correct.  
- B: Mean over sample. No CI.  
- C: N/A.  
- D: No band.  
- E: Valid — colony positions' value targets indicate whether they're rewarded.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: not in banked sample)

---

### extension_mean_value_target
- **Emit site:** `track_b_buffer_snapshot.py:97`  
- A: Same pattern for extension class. Formula correct.  
- B: Mean over sample. No CI.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: not in banked sample)

---

### value_probe_decisive_mean
- **Emit site:** `step_coordinator.py:1321`  
- A: Emitted under event `value_probe_drift` with key `decisive_mean` (step_coordinator.py:1321), NOT `value_probe_decisive_mean`. The B1 inventory uses the conceptual stat name; actual emitted key differs. The computation: `vp["decisive_mean"]` from `value_probe.compute()` — mean predicted value on decisive fixture positions. Formula correct.  
- B: Fixed fixture probe; n_eff = fixture size (typically small). Not in banked sample (instrumentation_enabled=False; value_probe disabled by default).  
- C: N/A.  
- D: No band.  
- E: Valid — tracks value-head drift on fixed decisive fixture.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: not in banked sample; emitted key is `decisive_mean` not `value_probe_decisive_mean`)

---

### value_probe_decisive_std
- **Emit site:** `step_coordinator.py:1322`  
- A: Emitted as `decisive_std`. Formula correct (std of value predictions over decisive positions).  
- B: Same coverage gap.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: not in banked sample)

---

### value_probe_draw_mean
- **Emit site:** `step_coordinator.py:1323`  
- A: Emitted as `draw_mean`. Formula correct.  
- B: Same coverage gap.  
- C: N/A.  
- D: No band.  
- E: Valid — tracks value-head response on draw fixture positions.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: not in banked sample)

---

### value_probe_draw_std
- **Emit site:** `step_coordinator.py:1324`  
- A: Emitted as `draw_std`. Formula correct.  
- B: Same coverage gap.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: not in banked sample)

---

### draw_target_fraction
- **Emit site:** `step_coordinator.py:1367` (via `buffer_composition()`)  
- A: `draws_in_buf / size` where `draws_in_buf = outcome_in_range_count(_lo, _hi)` and `_lo, _hi` are derived from config `draw_value` and `ply_cap_value` (pool.py:568-581). PIPE-4 fix: band derived from live config, not stale hardcoded window. Formula correct.  
- B: Snapshot over full buffer. Not in banked sample.  
- C: N/A.  
- D: No band.  
- E: Valid — fraction of buffer positions that have draw/ply-cap targets.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: not in banked sample)

---

### colony_terminal_fraction
- **Emit site:** `step_coordinator.py:1368` (via `buffer_composition()`)  
- A: `tr["colony"] / total_games` where `tr = terminal_reason_counts()`, `total_games = max(1, sum(tr.values()))` (pool.py:593-595). Fraction of games that ended by colony rule. Formula correct.  
- B: Cumulative over all games since start. Not in banked sample.  
- C: N/A.  
- D: No band.  
- E: Valid — game termination reason tracking. This is a factual count, not the colony-attractor diagnostic which is a different construct.  
- F: Not duplicated at this level.  
- **CORRECT** (coverage gap: not in banked sample)

---

### six_terminal_fraction
- **Emit site:** `step_coordinator.py:1369`  
- A: `tr["six_in_a_row"] / total_games`. Formula correct.  
- B: Cumulative. Not in banked sample.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: not in banked sample)

---

### cap_terminal_fraction
- **Emit site:** `step_coordinator.py:1370`  
- A: `tr["ply_cap"] / total_games`. Formula correct.  
- B: Cumulative. Not in banked sample.  
- C: N/A.  
- D: No band.  
- E: Valid.  
- F: Not duplicated.  
- **CORRECT** (coverage gap: not in banked sample)

---

## Seeds re-derived

No PREREG §3 seeds map to B1 training stats by ownership. Seeds 1 and 2 (root_concentration/depth — B3 search), seeds 3 and 6 (entropy band and colony/bce-gap bands — B5 monitor), seeds 4 and 5 (wr_sealbot, §D-ARGMAX — B4 eval), seeds 7 and 8 (alt_spread NaN and forced_win_conversion — B5/B4) all live in other buckets.

**However, the policy_entropy band (seed 3 adjacent):** The PREREG §3 seed 3 says "entropy band ≤2.6 backwards — Axis D". The B1 `policy_entropy` has a band at alert=1.0, warn=2.0 nats. These are LOWER-bound thresholds (warn when LOW), which is the CORRECT direction for collapse detection. The "≤2.6 backwards" claim refers to a different surface (monitor/dashboard rendering of the combined entropy with an upper threshold interpreted as healthy ceiling). The B1 stats' own band (warn < 2.0) is directionally correct. Seed 3 outcome: the "backwards" claim does NOT apply to the band direction encoded in `MonitoringConfig` — it likely applies to a display/rendering artifact in the terminal dashboard where a high-entropy check is inverted. From trainer.py and monitoring/config.py, the band implementation is directionally sound. **CONFIRMED as stale if applied to B1 stats; seed 3 is correctly owned by B5 where the display logic may invert it.**

**Axis distribution band (axis_q/r/s):** Re-derived independently from source. Corpus baseline (from banked log): axis_r=0.4526 > warn=0.45. Confirms band Axis D failure: warn fires on healthy-baseline selfplay. This is a new finding in B1 scope (not in PREREG seeds list), arising from the axis distribution band calibration.
