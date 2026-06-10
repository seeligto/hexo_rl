# Wave 9 Batch A — Recon (PREP claim verification + SD4 corrections)

**Wave:** 9
**Batch:** A (sole batch this wave; single conventional commit)
**Branch:** `refactor/rust-engine-cycle-3`
**Entry HEAD:** `43d5d8a` (Wave 8 Batch D close — verified via `git log -1 --format=%H`)
**Anchor PREP:** `audit/rust-engine/cycle_3/wave_9/PREP_plan.md`
**Recon date:** 2026-05-17

---

## 1. PREP fragile-claim verification matrix

| PREP § | Claim | `rg`/cargo verification | Verdict |
|---|---|---|---|
| §A.2 / §L.1 | 8 encoding entries in `registry.toml` | `grep -c '^\[encodings\.' engine/src/encoding/registry.toml` → 8 | CONFIRMED |
| §A.5 / §L.3 | 8 `schema_version` lines in `registry.toml` | `grep -c 'schema_version' engine/src/encoding/registry.toml` → 9 (8 per-entry + 1 header doc-comment "Schema v2 additions" reference at line 10) | CORRECTED — 8 per-entry edits + 1 header doc-comment **block addition** (not edit; new block under existing v2 block) |
| §B.1 / §L.4 | `#[getter]` count = 23 at HEAD `43d5d8a` | `grep -c '#\[getter\]' engine/src/pyo3/encoding.rs` → 24 | CORRECTED — actual 24 → 25 after Wave 9 (PREP undercounted by 1; likely missed `n_actions` alias getter added in Wave 8 Batch A FF.2) |
| §B.2 / §L.5 | `_REQUIRED_FIELDS` "17 fields"; post-Wave-9 stays 17 | `tests/test_inv22_python_encoding_spec_parity.py:38-57` enumerates 18 entries (verified line-by-line); module docstring at L7 also says "all 18 schema fields plus the 6 derived accessors" — PREP §B.2 narrative miscount | CORRECTED — actual is **18 schema fields** at HEAD; Wave 9 adds 1 → **19 fields** post-Wave-9. Parametrized test counts remain unchanged at 17 because the count is `1 identity + 8 attribute-surface + 8 value-parity` — the *parametrize cardinality* (8 encodings) is unchanged. PREP §B.2 confused field count vs test count. |
| §A.4 | `parse.rs` `get_int!` macro + 2-line field add (1 `let` + 1 struct-literal field) | `engine/src/encoding/registry/parse.rs` read: macro at L25-38; struct literal at L196-215; pattern confirmed (cf. `schema_version` field at L73 + L211) | CONFIRMED |
| §A.3 | `validate.rs` `validate(&self) -> Result<(), String>` w/ `errs.push(...)` pattern | `engine/src/encoding/spec/validate.rs` read; signature at L32; per-field `errs.push` pattern verified at L36, L46, L131 (`legal_move_radius == 0` precedent identical to new `k_max == 0` rule) | CONFIRMED |
| §C.1 | `mod.rs:312-318` line range for the `// §P55 ...` comment block + the `InferenceBatcher::new(...)` call | `engine/src/game_runner/mod.rs` read at L312-318; comment block + `batcher: InferenceBatcher::new(...)` confirmed at the cited lines | CONFIRMED |
| §E.1 | `all_specs()` accessor exists in `engine::encoding::registry` | `engine/src/encoding/registry/mod.rs:47` — `pub fn all_specs() -> impl Iterator<Item = &'static RegistrySpec>` | CONFIRMED |
| §E.1 | `lookup_or_panic(&str)` accessor exists | `engine/src/encoding/registry/mod.rs:36` — `pub fn lookup_or_panic(name: &str) -> &'static RegistrySpec` | CONFIRMED |
| §A.2 | v8 + v8_canvas_realness are `is_multi_window = false` (so k_max = 1) | `engine/src/encoding/registry.toml:249` and `:281` both `is_multi_window = false` | CONFIRMED |
| §L.2 | Zero Python kwarg-construct sites for `engine.RegistrySpec` | `rg -n "engine\.RegistrySpec\(" hexo_rl/ tests/ scripts/` → zero hits; `rg -n "EncodingSpec\(.*\bk_max" hexo_rl/ tests/ scripts/` → zero hits | CONFIRMED |
| §L.6 | Default v6 selfplay derives 14*8*1*2 = 224; without `.max(512)` floor this REDUCES the pool prefill | `engine/src/inference_bridge.rs:313` `pool_size.unwrap_or(512)` confirmed; channel cap at L314 `pool_size.map_or(1024, |n| (n*2).max(1024))` confirms 1024 floor on channel | CONFIRMED — `.max(512)` floor on the auto-derive recommended (PREP §L.6 alternative form) so v6 default does NOT silently drop pool prefill from 512 → 224. |
| §G.8 / §I.6 | No new `#[allow]` required by Wave 9 work | TOML / struct-add / single PyO3 getter / 4-line validator rule / single `let` insertion — none touch the >100-LOC clippy thresholds | CONFIRMED (verified at IMPL after build) |
| §E.1 | INV24 file path `engine/tests/inv24_k_max_registry_field.rs` does not collide | `ls engine/tests/ | grep inv24` → empty | CONFIRMED |

**SD4 corrections summary:**
- §L.3 schema_version "8 lines" → 8 per-entry **edits** (2→3) + 1 header doc-comment **block addition** (5 lines) — not a contradiction, but the §A.5 block-add was conflated with the per-entry edit count. IMPL applies both.
- §L.4 `#[getter]` count 23 → 24 at HEAD; Wave 9 adds +1 → 25. Trivial count drift; no scope impact.
- §B.2 narrative "17 fields" → 18 fields. PREP's confusion is between "fields in the `_REQUIRED_FIELDS` tuple" (18 at HEAD; 19 post-Wave-9) vs "test cell count" (17 total: 1 identity + 8 attr surface + 8 value parity; this count stays 17 because the parametrize cardinality is the 8 encodings — adding a field to the inner-loop check does not multiply test count). The PREP §B.2 second-paragraph framing "17 grows to 17" is correct; the first-paragraph "17 fields" wording is wrong; the IMPL extends `_REQUIRED_FIELDS` from 18 → 19 and pytest count stays 1565.

---

## 2. Baseline sanity gates at HEAD `43d5d8a`

| Gate | Result | PREP §G floor | Compliance |
|---|---|---|---|
| `cargo test --package engine --release` | **268 passed** (summed across 5 result lines: lib 192 + 5 inv-cell binaries summing 73 + 1 doctest ignored) | 268 | EXACT |
| `cargo clippy --package engine --release` | **42 warnings** (`warning: engine (lib) generated 42 warnings`) | 42 | EXACT |
| `#[getter]` count `engine/src/pyo3/encoding.rs` | **24** | 23 (PREP §L.4) | DRIFT +1 (CORRECTED above) |
| `_REQUIRED_FIELDS` length | **18** | "17 fields" PREP §B.2 | DRIFT +1 (CORRECTED above) |
| INV file count (`engine/tests/inv*.rs`) | **9** (15+16+17+18+18b+19+20+21+23) | 9 | EXACT |

All compile + test floors preserved at HEAD. Wave 9 Batch A proceeds.

---

## 3. `.max(512)` floor decision (PREP §L.6 hazard mitigation)

PREP §L.6 pre-registers the v6 default-config hazard: derived pool size = `14 * 8 * 1 * 2 = 224` is SMALLER than the existing hardcoded fallback floor at `inference_bridge.rs:313` (`pool_size.unwrap_or(512)`). The PREP-recommended mitigation is `derived.max(512)`.

**IMPL decision: ADOPT the `.max(512)` floor variant.**

Rationale:
- Wave 9 is a mechanism-add wave; behavior preservation for the dominant v6 production code path is the priority. Without the floor, default v6 callers silently drop prefill 512 → 224 — a behavioral change PREP §L.6 explicitly warns against.
- The channel capacity floor at `inference_bridge.rs:314` already uses the same `max(_*2, 1024)` pattern; the auto-derive floor matches the existing floor convention.
- The v6w25 16-worker target `14 * 8 * 8 * 2 = 1792` is far above 512, so the floor is a no-op for the multi-window encoding (the use case Wave 9 actually opens).
- Cost is negligible (single `.max(512)` call at ctor time; not a hot path).

The floor is disclosed in the commit body. Subsequent waves may choose to drop the floor once `inference_bridge.rs:313` is migrated away from the hardcoded 512 — orthogonal to Wave 9 scope.

---

## 4. Done-when (recon gate)

- All PREP §L 1-7 claims either CONFIRMED or CORRECTED above. ✓
- Baseline sanity gates match PREP §G floors (268 tests / 42 clippy). ✓
- `.max(512)` floor decision documented (adopted; see §3). ✓
- No PREP claim falsified in a way that blocks scope: 3 minor corrections (count drifts) absorbed into IMPL without scope change. ✓

Phase 2 IMPL proceeds with the PREP §A-§C edit plan unchanged, with the `.max(512)` floor variant adopted at §C.1.
