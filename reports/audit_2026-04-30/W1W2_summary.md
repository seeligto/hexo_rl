# W1+W2 Aggregator Summary — Post-§131 Audit Fix Wave

**Date:** 2026-04-30  
**Aggregator:** post-§131 audit fix wave W1+W2  
**Base commit (pre-wave):** `f848054` (chore(repo): gitignore rules — W1B, landed just before W1A's first code fix)  
**Wave commits:** 19 total (f848054 through c358ba4)

---

## Commits landed (chronological)

| Hash | Subject | Author | Finding refs |
|---|---|---|---|
| `f848054` | chore(repo): add scratch-pattern gitignore rules from audit | Tom S | F1–F6, F10, F11 |
| `9b44650` | fix(smoke): force strict load + read in_channels from checkpoint | Tom S | A1-1 |
| `99cf6e7` | docs(audit): W1D Q49 Dirichlet × rotation RNG independence audit | Tom S | Q49 (read-only audit report) |
| `3ff3ffa` | fix(eval): slice 18→8 planes in windowing_diagnostic before forward | Tom S | A1-2 |
| `0dcbfc9` | chore: remove dead policy_projection module | Tom S | D1 |
| `acea9df` | chore: remove dead replay_poller | Tom S | D3 |
| `b531701` | chore: remove empty hexo_rl/api package | Tom S | D2 |
| `411f8bc` | chore: remove dead opening_book package | Tom S | D4 |
| `4439e65` | chore: remove dead BootstrapDataset wrapper | Tom S | D5 |
| `ab6ca16` | chore: remove dead LossResult dataclass | Tom S | D10 |
| `0d9c967` | chore: retire CorpusPipeline + HybridGameSource | Tom S | D15 |
| `6459d1f` | chore: remove dead regen_bootstrap_corpus script | Tom S | A1-19 |
| `073e595` | docs(claude): align CLAUDE.md with bootstrap-v6 + 8-plane | Tom S | D01, D02, D03 |
| `9ef3ed0` | docs(readme): fix tensor shape + plane description | Tom S | D21 |
| `c9bcfa1` | docs(open-q): refresh Q40 gating + add Q45 | Tom S | D22 |
| `622507c` | docs(arch): align 01_architecture.md with §131 8-plane | Tom S | D04–D10 |
| `5982a9c` | docs(roadmap): retire stale throughput table + bump anchors | Tom S | D11–D15 |
| `f5a54b5` | docs(rules): align rules/ + agent_context + tooling with §131 | Tom S | D16–D20, D24, D25 |
| `c358ba4` | docs(probe): refresh stale 18-plane comments in probe script | Tom S | A1-16, A1-17, D23 |

---

## Findings closed (cross-ref to _summary.md fix-pass items)

| Finding ID | Description (brief) | Closed by commit |
|---|---|---|
| A1-1 | smoke_w4_step1_rotation strict=False silent garbage load | `9b44650` |
| A1-2 | windowing_diagnostic 18-plane forward crash (no KEPT_PLANE_INDICES slice) | `3ff3ffa` |
| A1-16 | probe_threat_logits.py stale "18-plane model" docstring/comment | `c358ba4` |
| A1-17 | probe_threat_logits.py --zero-chain-planes help text stale | `c358ba4` |
| A1-19 | regen_bootstrap_corpus.py 18/24-plane assertion (dead post-§134) | `6459d1f` |
| D01 | CLAUDE.md phase line references bootstrap-v4 | `073e595` |
| D02 | CLAUDE.md threat-probe header says "18-plane model" | `073e595` |
| D03 | CLAUDE.md threshold note references §97/v4 | `073e595` |
| D04 | docs/01_architecture.md channel layout table 18-plane | `622507c` |
| D05 | docs/01_architecture.md STATE_STRIDE 6498 stale | `622507c` |
| D06 | docs/01_architecture.md encoding-split section stale | `622507c` |
| D07 | docs/01_architecture.md tensor shape (K,18,19,19) stale | `622507c` |
| D08 | docs/01_architecture.md HEXB v5 + 18-plane buffer shape | `622507c` |
| D09 | docs/01_architecture.md v4 legacy-compat note stale | `622507c` |
| D10 | docs/01_architecture.md stale perf numbers (should pointer to perf-targets.md) | `622507c` |
| D1 | policy_projection module dead (zero callers) | `0dcbfc9` |
| D11 | docs/02_roadmap.md pre-§128 throughput table (wrong metric) | `5982a9c` |
| D12 | docs/02_roadmap.md "18-plane input (§97)" ref | `5982a9c` |
| D13 | docs/02_roadmap.md "18-plane temporal tensors" refs | `5982a9c` |
| D14 | docs/02_roadmap.md "bootstrap-v4" | `5982a9c` |
| D15 | docs/02_roadmap.md pre-§128 exit criterion + CorpusPipeline | `5982a9c` / `0d9c967` |
| D16 | docs/rules/board-representation.md "18-plane temporal tensor" | `f5a54b5` |
| D17 | docs/rules/phase-4-architecture.md "Input: 18 planes (§97)" | `f5a54b5` |
| D18 | docs/rules/phase-4-architecture.md HEXB v5 reference | `f5a54b5` |
| D19 | docs/rules/build-commands.md "(18,19,19) tensors" | `f5a54b5` |
| D20 | docs/rules/build-commands.md "HEXB v5 save/load" | `f5a54b5` |
| D2 | hexo_rl/api package empty (dead) | `b531701` |
| D21 | README.md tensor shape (24×19×19) + 18-plane description | `9ef3ed0` |
| D22 | docs/06_OPEN_QUESTIONS.md Q40 gating stale; Q45 missing | `c9bcfa1` |
| D23 | probe_threat_logits.py schema v2 docstring (actually v6) | `c358ba4` |
| D24 | docs/00_agent_context.md tensor shape 18 | `f5a54b5` |
| D25 | docs/03_tooling.md torch.zeros/np.zeros 18-plane examples | `f5a54b5` |
| D3 | replay_poller.py GameReplayPoller dead (zero importers) | `acea9df` |
| D4 | opening_book package dead | `411f8bc` |
| D5 | BootstrapDataset wrapper dead | `4439e65` |
| F1–F6 | root-level scratch debris (18 files, ~278 KB) | `f848054` + W1B disk rm |
| F10, F11 | .gitignore missing *.bak + profile_epyc_pyspy.sh patterns | `f848054` |
| Q49 | Dirichlet × sym_idx RNG independence (read-only audit) | `99cf6e7` (verdict) |

**Total finding IDs closed: 39** (counting D01/D1 as same lane, A1-19/D15 both partially closed by separate commits)

---

## Findings still open (deferred to W3 or later)

| Finding ID | Description | Reason / Target |
|---|---|---|
| L1 | `InferenceServer._traced_model` fixed-batch jit.trace dynamic-shape risk (SE block) | Needs bench-gate skill; touches inference hot path. Deferred to W3 per _summary fix-pass order item 8. |
| D6, D7, D8 | KrakenBotBot / CommunityAPIBot / scraper.py adapters | Orchestrator KEEP binding — not deleted. |
| D9 | `submit_request_and_wait_rust` Rust dead code | Out of scope (Rust lane, not W1C). |
| D11–D14 (minor) | `docs/sweep_deployment.md` still references regen_bootstrap_corpus.py (lines 83, 109, 215) | Doc-drift only; W1C surfaced, not fixed. Trivial W3 item. |
| D11–D14 (minor) | `docs/rules/build-commands.md:146` still shows `opening_book/` in repo-layout diagram | Doc-drift; W1C surfaced. |
| D17 (shim drop) | `tower` property alias removal + shim retirement | Time-gated 2026-05-28; coordinated with D19. |
| Q41, Q52 | bootstrap-v6 vs v5 H2H + SealBot v6 anchor (200+150 game eval) | Blocks sustained run launch; no code fix in W1/W2 scope. W3 candidate. |
| Q43 | Rotation × eval pipeline measurement | Orchestrator decision pending (pre-sustained vs during). |
| Q44 | Laptop bench floor recalibration post-§131 | Should precede L1 fix; independent otherwise. |
| F9 | `.claude/skills/rsync-vast/` untracked | Orchestrator KEEP-LOCAL decision pending. |
| F8 | `docs/notes/*.md` untracked siblings | Orchestrator decision pending (git add or keep local). |
| Q47 | Pre-§131 checkpoint archive retention policy | Deferred to Phase 4.0 housekeeping pass. |
| Cross-cut | `docs/07_PHASE4_SPRINT_LOG.md:31` still says "Current: 18 planes" | W2C flagged; not in EXCLUDE, minor doc-drift. |
| Cross-cut | CI `.github/workflows/` not audited for stale 18-plane assumptions | No agent owned this lane. |
| Cross-cut | `configs/variants/` — 17 stale variant files (sweep_18ch etc.) | Config-curation lane; no agent claimed. |

---

## Test results

Final `make test` run post-all-wave commits (including Q49 propagation edit):

```
937 passed, 8 skipped, 2 deselected, 5 xfailed, 1 xpassed, 18 warnings in 140.41s (0:02:20)
```

Rust (engine): `138 passed; 0 failed; 0 ignored` + 29 integration tests — all green.  
Exit code: 0.

Note: `1 xpassed` is pre-existing (confirmed by W1A, W1C baseline counts). Not introduced by this wave.

---

## Q49 verdict propagation

**Verdict:** COUPLED-NEGLIGIBLE (read-only audit, no code change required)

W1D audit (`reports/audit_2026-04-30/W1D_q49_dirichlet_rotation.md`) found:
- `sym_idx` and Dirichlet draws share one `ThreadRng` (ChaCha12 CSPRNG) per worker thread
- Coupling is structural (shared stream) but not statistical — ChaCha12 PRF security bounds any sym_idx → Dirichlet correlation to ≤ 2^-128
- Information-theoretic: sym_idx ≤ 3.6 bits entropy vs ≥ 53 bits per Dirichlet component; no measurable leak
- Marsaglia-Tsang accept/reject further decorrelates

Q49 marked RESOLVED in `docs/06_OPEN_QUESTIONS.md` (Resolved table, row added after Q37).

**W3 status: UNBLOCKED**

---

## Cross-cutting issues observed

### probe_threat_logits.py EXCLUDE-list coordination (W2A + W2B + W2C)

Both W2A and W2B observed `test_no_stale_plane_refs` failing on `scripts/probe_threat_logits.py:605` ("legacy 24-plane fixture" comment in the `--zero-chain-planes` backward-compat path).

- **W2B** added `scripts/probe_threat_logits.py` to EXCLUDE (line 46 of `tests/test_no_stale_plane_refs.py`) but did NOT commit it as a separate conventional commit — folded into test run coordination.
- **W2C** subsequently edited `probe_threat_logits.py` to refresh the comment (commit `c358ba4`), which changes the string but preserves the legitimate "24-plane" literal at line 605 (print now reads "legacy 24-plane fixture"). The EXCLUDE entry at line 46 remains correct and necessary because the literal still exists.

Current state (verified):
- `tests/test_no_stale_plane_refs.py:46` — `'scripts/probe_threat_logits.py'` is present and committed.
- `scripts/probe_threat_logits.py:605` — "legacy 24-plane fixture" comment present (intentional backward-compat label).
- `make test` exits 0; `test_no_stale_plane_refs` passes.

No duplicate EXCLUDE entries. No uncommitted state. **No action required.**

W2C also noted an ordering-dependent flake in `test_no_stale_plane_refs` where a temp file written by another test survives into the scanner walk. This is pre-existing and not introduced by any wave commit. Recommend narrowing scanner to `git ls-files` in a future W3 test-hygiene pass.

---

## Sprint log §136 candidate entry (DRAFT — orchestrator decision)

```
### §136 — Post-§131 W1+W2 audit fix wave (2026-04-30)

**Scope:** 19 commits. Correctness fixes, dead-code sweep, full doc-drift
alignment for 8-plane / bootstrap-v6 era. Q49 RNG independence audit (COUPLED-NEGLIGIBLE).

**Correctness fixes (W1A):**
- `9b44650` fix(smoke): smoke_w4_step1_rotation force strict=True load +
  read in_channels from checkpoint (was silent garbage trunk weights with
  strict=False against bootstrap-v6).
- `3ff3ffa` fix(eval): windowing_diagnostic slice 18→8 via KEPT_PLANE_INDICES
  before forward (was RuntimeError: expected 8 channels, got 18 on every
  probe_windowing.py invocation).

**Repo hygiene (W1B):**
- `f848054` chore(repo): .gitignore scratch patterns (*.bak, debug_print*.py,
  test_pool*.py, print_*.py, parse_yaml.py, test_benchmark.py, test_game.py,
  profile_epyc_pyspy.sh).
- 18 root-level scratch files deleted, ~278 KB; cargo target/ (1.7 GB) +
  .torchinductor-cache-probe/ (81 MB) removed.

**Dead-code wave (W1C, 8 commits):**
- Deleted: policy_projection, replay_poller, hexo_rl/api, opening_book,
  BootstrapDataset, LossResult dataclass, CorpusPipeline + HybridGameSource,
  regen_bootstrap_corpus.py. Collateral: FUTURE_REFACTORS.md entry removed,
  push_corpus_preview.py docstring cleaned, setup.sh scaffolds updated,
  docs/06_CORPUS_DESIGN.md marked RETIRED.

**Q49 RNG audit (W1D, read-only):**
- `99cf6e7` Dirichlet × sym_idx share one ThreadRng (ChaCha12); coupling
  structural, not statistical. Correlation ≤ 2^-128 per PRF security.
  Verdict: COUPLED-NEGLIGIBLE. No remediation. W3 UNBLOCKED.
  Q49 marked RESOLVED in docs/06_OPEN_QUESTIONS.md.

**Doc alignment wave (W2A/W2B/W2C, 7 commits):**
- CLAUDE.md, README.md, docs/01_architecture.md, docs/02_roadmap.md,
  docs/rules/{board-representation,phase-4-architecture,build-commands}.md,
  docs/00_agent_context.md, docs/03_tooling.md, scripts/probe_threat_logits.py:
  all 18-plane / HEXB v5 / STATE_STRIDE 6498 / bootstrap-v4 references
  replaced with 8-plane / HEXB v6 / STATE_STRIDE 2888 / bootstrap-v6.
  Q40 gating updated; Q45 (subtree reuse cost-benefit re-derivation) added
  to active table.

**Test gate:** 937 py + 138 rs passed, exit 0.

**Open for W3:** L1 (jit.trace dynamic-shape), Q41/Q52 (v6 H2H + SealBot
anchor), Q43 (rotation×eval), Q44 (bench recalibration), config-curation
sweep (17 stale variants), CI YAML audit, tower-shim retirement (2026-05-28).
```
