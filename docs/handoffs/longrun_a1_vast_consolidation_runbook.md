# §D-LONGRUN-READY Phase A1 — Vast Consolidation Runbook

**Goal:** One clean repo at `/workspace/hexo_rl` with all Gumbel-verdict reports preserved,
branch `phase4.5/gumbelprep` checked out, engine built, `/root/hexo_rl` removed.

**Risk level:** DATA LOSS possible if reports/ are not preserved first. Follow ORDER EXACTLY.

---

## Pre-flight checklist

Before touching anything, confirm:

```bash
# Confirm both trees exist
ls /root/hexo_rl/ | head -5
ls /workspace/hexo_rl/ | head -5

# Confirm which branch each is on
git -C /root/hexo_rl branch --show-current
git -C /workspace/hexo_rl branch --show-current
```

---

## Step 1 — Preserve /root/hexo_rl/reports/ FIRST (data-loss-risk step)

The Gumbel verdict is NOT in git — it lives in `/root/hexo_rl/reports/`. Lose this → verdict gone.

```bash
# Verify the verdict files are there before rsync
ls /root/hexo_rl/reports/gumbelsims/
ls /root/hexo_rl/reports/thr_opt/
ls /root/hexo_rl/reports/p3_rr_agg/ 2>/dev/null || echo "check p3_rr*"
ls /root/hexo_rl/reports/ | grep p3_

# Rsync reports/ → /workspace (additive, no delete — safe)
rsync -av /root/hexo_rl/reports/ /workspace/hexo_rl/reports/

# Verify critical files landed in /workspace
ls /workspace/hexo_rl/reports/gumbelsims/SESSION_VERDICT.md
ls /workspace/hexo_rl/reports/p3_rr_agg/
ls /workspace/hexo_rl/reports/thr_opt/REPORT.md

echo "=== reports/ preserved ==="
```

**Laptop backup (run from laptop):** also pull reports/ to the laptop as a second copy:
```bash
# From laptop — update rsync-vast skill or run directly:
rsync -avz -e "ssh -p 13053 -i ~/.ssh/vast_hexo" \
  root@ssh6.vast.ai:/root/hexo_rl/reports/ \
  ~/Work/Hexo/hexo_rl/reports_vast_backup/
```

**Gate: DO NOT proceed until reports/ verified in /workspace AND laptop backup confirmed.**

---

## Step 2 — Commit any uncommitted /root working-tree state

```bash
cd /root/hexo_rl

# Check for uncommitted changes
git status

# If there are uncommitted changes, commit them:
git add -p   # review and stage carefully
git commit -m "wip: pre-consolidation state from /root"

# Confirm the 3 key commits are on origin/phase4.5/gumbelprep:
git log --oneline origin/phase4.5/gumbelprep | head -5
# Expected top commits: e614327, 2930a68, 63060eb

# Check that /root's HEAD is at or beyond e614327:
git log --oneline | head -5
```

If /root has commits NOT yet on origin (`git status` shows "ahead of origin/..."), proceed to Step 3 to bundle them. If /root is already synced to origin, skip the bundle sub-step.

---

## Step 3 — Transport /root commits to /workspace via git bundle

Vast cannot `git fetch` from GitHub (outbound fetch blocked). Use bundle+rsync.

```bash
# On vast — create bundle of commits not yet in /workspace
cd /root/hexo_rl
WORKSPACE_HEAD=$(git -C /workspace/hexo_rl rev-parse HEAD)
git bundle create /tmp/gumbelprep_delta.bundle ${WORKSPACE_HEAD}..HEAD

# Verify bundle is valid
git bundle verify /tmp/gumbelprep_delta.bundle

# Copy bundle into /workspace
cp /tmp/gumbelprep_delta.bundle /workspace/hexo_rl/

# In /workspace — fetch from bundle and merge
cd /workspace/hexo_rl
git fetch /workspace/hexo_rl/gumbelprep_delta.bundle 'refs/heads/*:refs/remotes/bundle/*'
git checkout -b phase4.5/gumbelprep 2>/dev/null || git checkout phase4.5/gumbelprep
git merge --ff-only remotes/bundle/phase4.5/gumbelprep

# Verify /workspace HEAD matches expected
git log --oneline | head -5
# Expected top: e614327 (or whichever is /root's HEAD)
```

If /root HEAD == origin HEAD (Step 2 showed nothing to push), skip the bundle and just:
```bash
cd /workspace/hexo_rl
git fetch origin
git checkout phase4.5/gumbelprep
git reset --hard origin/phase4.5/gumbelprep
```

---

## Step 4 — Rebuild engine in /workspace

```bash
cd /workspace/hexo_rl
source .venv/bin/activate

# Rebuild Rust engine (resolves v6_live2_ls: 4-plane, mw, k_max=8)
make build

# Smoke: verify encoding loads without error
python -c "from hexo_rl.encoding import lookup; e = lookup('v6_live2_ls'); print('planes:', e.in_channels, 'mw:', e.multi_window)"
# Expected: planes: 4, mw: True

# Smoke: verify round_robin import (the Arm-C lesson — pre-flight eval opponent)
python -c "from hammerhead import Bot; b = Bot(); print('hammerhead OK')"
# If this fails → rebuild hammerhead: cd hammerhead && maturin develop --release && cd ..

echo "=== engine rebuilt and smoke passed ==="
```

---

## Step 5 — Inventory /workspace unique artifacts before any removal

Before removing /root, document what is UNIQUE to /workspace that is NOT in /root:

```bash
# Key artifacts expected in /workspace (from §S178/S181 work):
ls /workspace/hexo_rl/checkpoints/ | grep -E "s181|v6_live2_rl|v6_live2$|golong"
ls /workspace/hexo_rl/data/ | grep -E "bootstrap_corpus"
sha256sum /workspace/hexo_rl/data/bootstrap_corpus_v6_live2.npz
# Expected corpus sha: 8f7115ab... (8300-game corpus for A2)

# If the corpus sha does NOT match 8f7115ab..., do NOT proceed — locate the corpus first.

# Also note v6_live2_rl and v6_live2 checkpoints for the archive:
ls -lh /workspace/hexo_rl/checkpoints/ | grep -v phase_b | head -20
```

**PRESERVE list** (do NOT remove from /workspace):
- `checkpoints/s181_fu1_5/` (or similar S181 checkpoints)
- `checkpoints/v6_live2_rl/` (v6_live2_rl training artifacts)
- `data/bootstrap_corpus_v6_live2.npz` (8300-game corpus, sha 8f7115ab)
- `reports/` (now synced from /root + /workspace combined)

---

## Step 6 — Verify /workspace is clean, complete, and builds

```bash
cd /workspace/hexo_rl
git status       # should show clean (or only expected untracked data/)
git log --oneline | head -5
make test        # should pass all tests (or explain known failures)
python -c "import hexo_rl; print('hexo_rl OK')"
```

**Gate: /workspace must:**
1. Be on `phase4.5/gumbelprep` at the same HEAD as /root
2. Pass `make build` (engine compiled)
3. Pass encoding + hammerhead smoke
4. Have `reports/gumbelsims/SESSION_VERDICT.md` present

**If ANY gate fails → KEEP BOTH TREES. Do not remove /root.**

---

## Step 7 — Remove /root (ONLY after Step 6 gate passed)

```bash
# Final sanity check
diff <(git -C /root/hexo_rl log --oneline | head -5) \
     <(git -C /workspace/hexo_rl log --oneline | head -5)
# Must be empty (identical)

# Remove /root
rm -rf /root/hexo_rl

echo "=== consolidation complete: /workspace/hexo_rl is the single source ==="
```

---

## Post-consolidation note

From this point forward, **`/workspace/hexo_rl` is the canonical path on vast**.
Update any scripts/tmux sessions that reference `/root/hexo_rl`.

Vast transport rule (already in memory): **update vast via rsync+bundle, never `git pull`**
(outbound fetch from vast to GitHub is blocked; push works, fetch dies).

---

## What comes next

→ A2: Re-pretrain bootstrap on the 8300-game corpus (`bootstrap_corpus_v6_live2.npz`)
   See `docs/handoffs/longrun_a2_bootstrap_pretrain_spec.md`

→ Phase B: m-gate training arms (after A2 produces the new bootstrap)
   Configs: `configs/variants/phase_b_mgate_m{8,16,32}.yaml`
