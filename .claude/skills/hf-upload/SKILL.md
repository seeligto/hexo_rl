---
name: hf-upload
description: >
  Use when uploading HeXO artifacts (corpus .npz, bootstrap checkpoints,
  trained models) to Hugging Face Hub, or when the user mentions "upload to
  HF", "push corpus to huggingface", "publish bootstrap model", "hf auth",
  "create HF dataset", or "create HF model repo". Covers token auth,
  `hf auth login`, `hf upload` CLI, `HfApi.upload_file` / `upload_folder`
  from huggingface_hub, and the convention for which repo type (dataset vs
  model) HeXO artifacts belong to.
---

# Hugging Face upload

## When to use

HeXO artifacts too large for git:
- `data/bootstrap_corpus*.npz` → **dataset repo** (`timmyburn/hexo-bootstrap-corpus`)
- `data/bootstrap_corpus_v3_human.npz` and successors → same dataset repo, versioned by filename
- `bootstrap_model.pt`, `archive/bootstrap_v2_*.pt` → **model repo** (`timmyburn/hexo-bootstrap-models`)
- Trained Phase-4 checkpoints (rare — usually stay local) → model repo

## Auth (one-time per machine)

```bash
# 1. Generate a token at https://huggingface.co/settings/tokens
#    - type "Write" (needed to upload)
#    - scope: restricted to the specific repos once they exist, or "All repos"
# 2. Log in (stores token at ~/.cache/huggingface/token)
.venv/bin/hf auth login
#    Paste token when prompted. Choose "n" for git credential integration
#    unless you want `git push` to large-file HF repos from CLI.
# 3. Verify
.venv/bin/hf auth whoami
```

Alternative — env var (CI / headless):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx   # read from secret store, never hardcode
```

`huggingface_hub` reads `HF_TOKEN` automatically; no login step needed.

## Upload — CLI (preferred for one-off large files)

```bash
# Create repo once (idempotent; --repo-type dataset for corpus, default is model)
.venv/bin/hf repo create hexo-bootstrap-corpus --type dataset --private
# Switch to public later: hf repo settings ... (or via web UI)

# Upload a single file
.venv/bin/hf upload \
  timmyburn/hexo-bootstrap-corpus \
  data/bootstrap_corpus_v3_human.npz \
  bootstrap_corpus_v3_human.npz \
  --repo-type dataset

# Upload a folder (e.g. all corpus variants at once)
.venv/bin/hf upload \
  timmyburn/hexo-bootstrap-corpus \
  data/ . \
  --repo-type dataset \
  --include "bootstrap_corpus*.npz"
```

Large files (>5 GB) automatically use hf-xet chunked upload — installed via
`huggingface_hub[cli]` extras. Do not pre-split.

## Upload — Python (for scripted/automated paths)

```python
from huggingface_hub import HfApi
api = HfApi()  # uses HF_TOKEN or ~/.cache/huggingface/token

api.create_repo(
    repo_id="timmyburn/hexo-bootstrap-corpus",
    repo_type="dataset",
    private=True,
    exist_ok=True,
)
api.upload_file(
    path_or_fileobj="data/bootstrap_corpus_v3_human.npz",
    path_in_repo="bootstrap_corpus_v3_human.npz",
    repo_id="timmyburn/hexo-bootstrap-corpus",
    repo_type="dataset",
    commit_message="add v3 human corpus (2026-04-22)",
)
```

## Download side (for install script — future work)

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="timmyburn/hexo-bootstrap-corpus",
    filename="bootstrap_corpus_v3_human.npz",
    repo_type="dataset",
    local_dir="data/",
    local_dir_use_symlinks=False,
)
```

CLI equivalent: `hf download timmyburn/hexo-bootstrap-corpus bootstrap_corpus_v3_human.npz --repo-type dataset --local-dir data/`.

## Conventions for HeXO

- **Dataset repo**: `hexo-bootstrap-corpus` — all `.npz` corpora; filename carries
  the version (`_v2`, `_v3_human`, `_18plane`). README.md in repo documents
  each file's source, row count, and generation date.
- **Model repo**: `hexo-bootstrap-models` — `.pt` bootstrap checkpoints + model
  card noting training config (18-plane, SE+aux, BCE value), corpus used,
  probe pass rates (C2/C3).
- **Never upload**: `checkpoints/` live training dir, `logs/`, `runs/`,
  `.torchinductor-cache/`.

## Pitfalls

- `hf auth login --add-to-git-credential` will let `git push` to HF work for
  small files, but HF model/dataset pushes should go through `hf upload` (uses
  LFS/xet automatically; raw `git push` to HF without `git lfs` will reject).
- Repo visibility defaults to **public** unless `--private` is passed at create
  time. Default to private until corpus content + license are confirmed OK.
- HF datasets inherit CC-BY-4.0 by default unless overridden in the repo's
  `README.md` YAML frontmatter (`license: mit`). HeXO uses MIT — set it
  explicitly in the dataset README.
