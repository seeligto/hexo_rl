---
name: hf-upload
description: >
  Use when uploading/removing HeXO artifacts (encoding-free corpus JSONL,
  bootstrap checkpoints, trained models) on Hugging Face Hub, or when the user
  mentions "upload to HF", "push corpus to huggingface", "publish bootstrap
  model", "update the HF readme/dataset card", "delete files from HF", "hf auth",
  "create HF dataset", or "create HF model repo". Covers token auth, `hf upload`
  CLI, `HfApi.upload_file` / `delete_file`, dataset/model cards, repo visibility,
  and the convention for which repo type HeXO artifacts belong to.
---

# Hugging Face upload

## Current HeXO repo layout (post 2026-06-04 JSONL migration)

- **`timmyburn/hexo-bootstrap-corpus`** (dataset, **public**) — the corpus is now
  shipped **encoding-free**: `hexo_human_corpus.jsonl` (raw axial move lists, one
  game per line) + `SCHEMA.md` + `dataset_metadata.json` + `README.md` dataset
  card. The old `bootstrap_corpus*.npz` tensors were **retired/deleted** — they
  baked one architecture's input planes into numpy and weren't portable.
- **`timmyburn/hexo-bootstrap-models`** (model, **private**) — `bootstrap_model.pt`
  only (currently the v6_live2 4-plane anchor, a weights-only `state_dict`) +
  `README.md` model card.

Encoding-specific NPZs are **no longer stored on HF** — `make install` rebuilds
them locally from the JSONL (see "Download / install side"). See
[[project_corpus_jsonl_install_migration]].

## Generate the artifacts before uploading

```bash
# Corpus → encoding-free JSONL bundle (jsonl + SCHEMA.md + dataset_metadata.json + README)
.venv/bin/python scripts/export_corpus_jsonl.py \
  --input data/corpus/raw_human --out /tmp/hexo_corpus_export
# Bootstrap model is just a checkpoint .pt (weights-only state_dict)
```

## Auth (one-time per machine)

```bash
# Token at https://huggingface.co/settings/tokens — type "Write".
.venv/bin/hf auth login          # stores ~/.cache/huggingface/token; "n" to git-cred
.venv/bin/hf auth whoami         # verify (should print: timmyburn)
```

Headless / CI alternative: `export HF_TOKEN=hf_xxx` (read from a secret store,
never hardcode). `huggingface_hub` picks it up automatically.

Keep the client current — the CLI is `hf` (the old `huggingface-cli` is
deprecated): `.venv/bin/pip install -U "huggingface_hub[cli]"` (≥ 1.17).

## Upload — CLI (preferred for one-off files)

`hf upload <repo_id> <local_path> <path_in_repo> --repo-type {dataset|model}`.
Repos already exist; `hf repo create <name> --type dataset --private` is only for
a brand-new repo.

```bash
REPO=timmyburn/hexo-bootstrap-corpus ; EXP=/tmp/hexo_corpus_export
for f in hexo_human_corpus.jsonl SCHEMA.md dataset_metadata.json README.md; do
  .venv/bin/hf upload "$REPO" "$EXP/$f" "$f" --repo-type dataset \
    --commit-message "Update $f"
done

# Bootstrap model — rename to the canonical bootstrap_model.pt in the repo
.venv/bin/hf upload timmyburn/hexo-bootstrap-models \
  checkpoints/bootstrap_model_v6_live2.pt bootstrap_model.pt --repo-type model \
  --commit-message "Replace bootstrap_model.pt with v6_live2 4-plane anchor"
```

Large files (>5 GB) auto-use hf-xet chunked upload (from `huggingface_hub[cli]`).
Do not pre-split. The JSONL corpus is only a few MB.

## Delete files / retire old artifacts

No CLI verb — use the Python API (idempotent per file):

```python
from huggingface_hub import HfApi
api = HfApi()
for f in ["bootstrap_corpus.npz", "bootstrap_corpus_v6w25.npz"]:
    api.delete_file(path_in_repo=f, repo_id="timmyburn/hexo-bootstrap-corpus",
                    repo_type="dataset", commit_message=f"Retire {f}")
```

List current contents first: `api.list_repo_files(repo_id, repo_type=...)`.
Deleting from a **public** repo is outward-facing — confirm scope with the user,
and watch ordering: don't delete a file `make install` still reads (e.g. the
corpus the trainer resolves) until the consumer has been migrated.

## Dataset / model card (README.md)

The repo `README.md` IS the card and needs YAML frontmatter. Dataset card:

```markdown
---
license: mit
pretty_name: Hexo Human Corpus (encoding-free)
task_categories: [other]
tags: [hex, hex-tac-toe, board-games, game-records]
size_categories: [1K<n<10K]
---
# ...schema, conventions, usage...
```

HF datasets default to CC-BY-4.0 — HeXO is **MIT**, so set `license: mit`
explicitly. Model card notes encoding (v6_live2 = 4-plane), architecture, the
corpus it was pretrained on, and format (weights-only state_dict).

## Download / install side (NOT future work — it's live)

`scripts/install.sh` step 9:
1. curl the model from the (private) model repo — `hf_download` **cache-skips** if
   `checkpoints/bootstrap_model.pt` already exists; a fresh box without auth gets
   a 401 (non-fatal). Provide the model out-of-band or set up auth on clean boxes.
2. curl `hexo_human_corpus.jsonl` from the public dataset repo, then **rebuild the
   encoding-specific NPZ locally**:
   `export_corpus_npz.py --from-jsonl data/hexo_human_corpus.jsonl --encoding $CORPUS_ENCODING`.
   The output path is the registry's `resolve_corpus_path(lookup(enc))` (e.g.
   v6_live2 → `data/bootstrap_corpus_v6_live2.npz`) — never the hardcoded
   `bootstrap_corpus.npz`, which is the 8-plane v6 path. `CORPUS_ENCODING` must
   match the model's input planes.

Verify the whole jumpstart path end-to-end against live HF (auth model pull +
rebuild at the resolved path + forward pass + install-path==trainer-path):

```bash
bash scripts/probe_install_jumpstart.sh        # PROBE PASS on v6_live2
# from a worktree without its own .venv: VENV=/path/to/repo/.venv bash scripts/...
```

## Pitfalls

- Repo visibility: dataset is **public**, model is **private** (intentional). New
  repos default public unless `--private` at create; flipping public is
  effectively a publish — confirm with the user first.
- Encoding coupling: the model's input-plane count and the rebuilt corpus
  encoding MUST match (v6_live2 = 4-plane). Uploading a model of a different
  encoding requires the matching `CORPUS_ENCODING` in install.
- `hf auth login --add-to-git-credential` lets small-file `git push` to HF work,
  but model/dataset pushes should go through `hf upload` (LFS/xet auto; raw
  `git push` without git-lfs rejects).
- **Never upload**: `checkpoints/` live training dir, `logs/`, `runs/`,
  `.torchinductor-cache/`, encoding-baked `*.npz` (retired — ship the JSONL).
