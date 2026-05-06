---
name: rsync-vast
description: >
  Pull files or directories from a vast.ai remote server over rsync+SSH.
  Trigger phrases: "rsync from vast", "pull from remote", "fetch from vast",
  "sync from server", "grab remote report", "download from vast.ai".
  Use whenever the user wants to copy benchmark reports, sweep results,
  checkpoints, or any other files from a rented vast.ai instance.
---

# rsync-vast — pull from remote server

Pulls files or directories from a vast.ai instance using the project SSH key.
Always uses: `~/.ssh/vast_hexo`, `IdentitiesOnly=yes`, `UserKnownHostsFile=~/.ssh/known_hosts_vast`, user=`root`.

## Input format

The user invokes this skill with three positional arguments:

```
/rsync-vast  HOST:PORT  REMOTE_PATH  LOCAL_PATH
```

| Argument | Example | Notes |
|---|---|---|
| `HOST:PORT` | `sshN.vast.ai:NNNNN` | vast.ai SSH host + port from instance page |
| `REMOTE_PATH` | `/workspace/hexo_rl/reports/sweeps/` | Absolute path on remote. Trailing `/` = contents only (rsync semantics) |
| `LOCAL_PATH` | `docs/notes/remote_reports/` | Relative to `~/Work/hexo_rl/`, or absolute. Trailing `/` = into directory |

Examples:
```
/rsync-vast sshN.vast.ai:NNNNN  /workspace/hexo_rl/reports/sweeps/some_report.md  docs/notes/remote_reports/
/rsync-vast sshN.vast.ai:NNNNN  /workspace/hexo_rl/reports/benchmarks/  reports/benchmarks/
/rsync-vast sshN.vast.ai:NNNNN  /workspace/hexo_rl/checkpoints/  ~/Work/hexo_rl/checkpoints/
```

If PORT is omitted from HOST (e.g. `sshN.vast.ai`), default to 22.

## Procedure

1. **Parse** HOST, PORT, REMOTE_PATH, LOCAL_PATH from the user's args.
   - If any argument is missing, ask the user for it — do not guess.
   - Strip surrounding quotes if present.

2. **Resolve** LOCAL_PATH: if relative, prepend `~/Work/hexo_rl/`. Ensure parent
   directory exists (`mkdir -p`).

3. **Run** the rsync command:

```bash
rsync -avz \
  -e "ssh -p PORT -i ~/.ssh/vast_hexo -o IdentitiesOnly=yes -o UserKnownHostsFile=~/.ssh/known_hosts_vast" \
  root@HOST:REMOTE_PATH \
  LOCAL_PATH
```

4. **Report** what was transferred (rsync output summary) and the final local path.
   If rsync exits non-zero, show the error and suggest checking: port number,
   host name, whether the instance is running, whether REMOTE_PATH exists on remote.

## Notes

- `~/.ssh/vast_hexo` is the project-specific key. Never substitute another key.
- `UserKnownHostsFile=~/.ssh/known_hosts_vast` keeps vast.ai host keys isolated
  from `~/.ssh/known_hosts` (different IP per rental, avoids MITM warnings).
- For large checkpoint directories, add `--progress` or `--info=progress2` to
  see transfer speed. For dry-run preview, add `-n`.
- If the transfer is for a sweep report or benchmark JSON, after rsync completes,
  ask the user if they want the file committed to the repo.
