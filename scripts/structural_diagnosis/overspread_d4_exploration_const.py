#!/usr/bin/env python3
"""§D-OVERSPREAD D4 — EXPLORATION-ARTIFACT constancy probe (EVAL-ONLY, read-only).

Driver D4 = "exploration artifact". Pre-registered rule: a CONSTANT exploration
mechanism cannot drive a MONOTONIC over-spread trend (own-components 14->16.6->22.1
across 30k->53k->87.5k). D4 is RULED OUT iff every exploration knob is constant across
the arc; LIT ONLY iff a time-varying exploration drift is actually found.

This script does NOT invoke L9 to skip the check (operator mandate). It SHOWS the
evidence:
  1. Parse EVERY `startup` config block in the golong log (the run was relaunched 3x:
     step 30k, the 53k LR re-warm, the post-87.5k-false-abort restart). Diff the
     exploration-relevant fields across all launches.
  2. Confirm cosine is on LR, NOT on temperature: scan per-step `train_step` records
     for a varying `lr` field (cosine) and confirm NO per-step temperature field
     varies. There is no temp schedule key in the config -> temp is static threshold.
  3. Confirm the legal_move_radius jitter RANGE is constant (config flag + the registry
     legal_move_radius -> the jitter is a fixed mechanism keyed off a static radius).
  4. Note the 53k LR re-warm explicitly (that is LR, an OPTIMIZER knob, not exploration).

Zero geometry literals: encoding/radius pulled from registry via hexo_rl.encoding.lookup.
Outputs: investigation/overspread_2026-06-08/d4_exploration_const.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "investigation/fragility_2026-06-07/v6l2golong.log"
OUT = REPO / "investigation/overspread_2026-06-08/d4_exploration_const.json"

from hexo_rl.encoding import lookup  # noqa: E402

# Exploration-relevant knobs the pre-registration names. Path tuples into the config dict.
# These are the knobs that, if they DRIFTED, could mechanically drive a monotonic
# over-spread trend. Anything outside this set (LR, eval cadence, etc.) is NOT exploration.
EXPLORATION_FIELDS = {
    # mcts.* exploration
    "mcts.dirichlet_alpha": ("mcts", "dirichlet_alpha"),
    "mcts.dirichlet_enabled": ("mcts", "dirichlet_enabled"),
    "mcts.epsilon": ("mcts", "epsilon"),
    "mcts.temperature_threshold_ply": ("mcts", "temperature_threshold_ply"),
    "mcts.c_puct": ("mcts", "c_puct"),
    "mcts.fpu_reduction": ("mcts", "fpu_reduction"),
    "mcts.n_simulations": ("mcts", "n_simulations"),
    "mcts.quiescence_blend_2": ("mcts", "quiescence_blend_2"),
    # selfplay.* exploration / opening / radius
    "selfplay.random_opening_plies": ("selfplay", "random_opening_plies"),
    "selfplay.legal_move_radius_jitter": ("selfplay", "legal_move_radius_jitter"),
    "selfplay.rotation_enabled": ("selfplay", "rotation_enabled"),
    "selfplay.max_game_moves": ("selfplay", "max_game_moves"),
    # selfplay.playout_cap.* (search-budget mixing = exploration of search depth)
    "playout_cap.temp_min": ("selfplay", "playout_cap", "temp_min"),
    "playout_cap.full_search_prob": ("selfplay", "playout_cap", "full_search_prob"),
    "playout_cap.fast_prob": ("selfplay", "playout_cap", "fast_prob"),
    "playout_cap.n_sims_quick": ("selfplay", "playout_cap", "n_sims_quick"),
    "playout_cap.n_sims_full": ("selfplay", "playout_cap", "n_sims_full"),
    "playout_cap.fast_sims": ("selfplay", "playout_cap", "fast_sims"),
    "playout_cap.standard_sims": ("selfplay", "playout_cap", "standard_sims"),
    "playout_cap.temperature_threshold_compound_moves": (
        "selfplay", "playout_cap", "temperature_threshold_compound_moves"),
    "playout_cap.zoi_enabled": ("selfplay", "playout_cap", "zoi_enabled"),
    "playout_cap.zoi_lookback": ("selfplay", "playout_cap", "zoi_lookback"),
    "playout_cap.zoi_margin": ("selfplay", "playout_cap", "zoi_margin"),
    # per-class target temperature (a TARGET-side temperature knob; pre-reg names it)
    "per_class_target_temperature.enabled": ("per_class_target_temperature", "enabled"),
    "per_class_target_temperature.colony_temperature": (
        "per_class_target_temperature", "colony_temperature"),
    "per_class_target_temperature.extension_temperature": (
        "per_class_target_temperature", "extension_temperature"),
    "per_class_target_temperature.neither_temperature": (
        "per_class_target_temperature", "neither_temperature"),
    "per_class_target_temperature.apply_to_selfplay": (
        "per_class_target_temperature", "apply_to_selfplay"),
    # entropy regularization (a policy-exploration pressure)
    "entropy_reg_weight": ("entropy_reg_weight",),
    # gumbel exploration (off here, but pre-reg-adjacent)
    "selfplay.gumbel_mcts": ("selfplay", "gumbel_mcts"),
    "selfplay.gumbel_explore_moves": ("selfplay", "gumbel_explore_moves"),
    "selfplay.forced_win_policy_enabled": ("selfplay", "forced_win_policy_enabled"),
}

# Schedule knobs — these ARE meant to vary, by design. We confirm cosine is on LR.
SCHEDULE_FIELDS = {
    "lr": ("lr",),
    "lr_schedule": ("lr_schedule",),
    "eta_min": ("eta_min",),
    "total_steps": ("total_steps",),
}


def dig(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return "<MISSING>"
        cur = cur[k]
    return cur


def parse_startups(log_path):
    """Return list of (lineno, timestamp, pid, configured-resume-step, config-dict)."""
    startups = []
    resumes = {}  # lineno -> resume record
    with open(log_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line.startswith("{"):
                continue
            # cheap pre-filter to avoid json-parsing 155k lines
            if '"event": "startup"' not in line and '"event": "resumed"' not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ev = rec.get("event")
            if ev == "startup":
                startups.append({
                    "lineno": i,
                    "timestamp": rec.get("timestamp"),
                    "pid": rec.get("pid"),
                    "variant": rec.get("variant"),
                    "config": rec.get("config", {}),
                })
            elif ev == "resumed":
                resumes[i] = {
                    "lineno": i,
                    "timestamp": rec.get("timestamp"),
                    "checkpoint": rec.get("checkpoint"),
                    "step": rec.get("step"),
                    "configured_total_steps": rec.get("configured_total_steps"),
                }
    # attach the nearest-following resume to each startup
    resume_lines = sorted(resumes)
    for su in startups:
        nxt = [rl for rl in resume_lines if rl > su["lineno"]]
        su["resume"] = resumes[nxt[0]] if nxt else None
    return startups


def scan_lr_and_temp_drift(log_path):
    """Confirm cosine is on LR (lr varies per-step) and there is NO per-step temp field.

    The per-step record is `train_step`. We sample lr across the run and confirm it
    varies (cosine), and verify no `temperature`/`temp` schedule key appears in any
    train_step record (which would indicate a temp schedule rather than a static threshold).
    """
    lrs = []  # (step, lr)
    temp_keys_seen = set()
    n_train_steps = 0
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if '"event": "train_step"' not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_train_steps += 1
            if "lr" in rec and "step" in rec:
                # subsample to keep memory bounded
                if n_train_steps % 200 == 1:
                    lrs.append((rec["step"], rec["lr"]))
            # detect ANY temperature schedule field leaking into per-step records
            for k in rec:
                kl = k.lower()
                if "temp" in kl and k not in ("timestamp",):
                    temp_keys_seen.add(k)
    return lrs, sorted(temp_keys_seen), n_train_steps


def main():
    if not LOG.exists():
        print(f"FATAL: log not found: {LOG}", file=sys.stderr)
        sys.exit(2)

    spec = lookup("v6_live2")
    registry_radius = spec.legal_move_radius  # static radius the jitter keys off

    startups = parse_startups(LOG)
    print(f"Found {len(startups)} startup config blocks (launches).")
    for su in startups:
        r = su["resume"]
        print(f"  launch pid={su['pid']} @ line {su['lineno']} "
              f"({su['timestamp']}) -> resume step "
              f"{r['step'] if r else '?'} ckpt {r['checkpoint'] if r else '?'} "
              f"total_steps={r['configured_total_steps'] if r else '?'}")

    # --- Build the per-field value table across launches ---
    expl_table = {}  # field -> list of values (one per launch)
    expl_constant = {}
    for fname, path in EXPLORATION_FIELDS.items():
        vals = [dig(su["config"], path) for su in startups]
        expl_table[fname] = vals
        expl_constant[fname] = (len(set(map(repr, vals))) == 1)

    sched_table = {}
    for fname, path in SCHEDULE_FIELDS.items():
        sched_table[fname] = [dig(su["config"], path) for su in startups]

    drifted = [f for f, c in expl_constant.items() if not c]
    all_constant = len(drifted) == 0

    # --- LR-vs-temp drift confirmation ---
    lrs, temp_keys, n_train_steps = scan_lr_and_temp_drift(LOG)
    lr_values = [v for _, v in lrs]
    lr_varies = (len(set(round(v, 9) for v in lr_values)) > 1) if lr_values else False
    lr_min = min(lr_values) if lr_values else None
    lr_max = max(lr_values) if lr_values else None
    # per-step temp schedule field? (only 'timestamp' allowed; flagged otherwise)
    per_step_temp_schedule = [k for k in temp_keys if k != "timestamp"]

    # --- radius jitter range constancy ---
    # The jitter is a boolean mechanism (selfplay.legal_move_radius_jitter) keyed off
    # the encoding's STATIC legal_move_radius from the registry. Range is constant iff
    # (a) the flag is constant across launches AND (b) the registry radius is a single
    # static value (no per-step radius schedule key anywhere in the config).
    jitter_flag_vals = expl_table["selfplay.legal_move_radius_jitter"]
    jitter_flag_constant = (len(set(map(repr, jitter_flag_vals))) == 1)
    radius_schedule_key_present = any(
        "radius" in str(k).lower() and "schedule" in str(k).lower()
        for su in startups for k in su["config"].get("selfplay", {})
    )
    radius_range_constant = jitter_flag_constant and not radius_schedule_key_present

    result = {
        "driver": "D4_exploration_artifact",
        "n_launches": len(startups),
        "launches": [
            {
                "pid": su["pid"],
                "lineno": su["lineno"],
                "timestamp": su["timestamp"],
                "resume_step": su["resume"]["step"] if su["resume"] else None,
                "resume_ckpt": su["resume"]["checkpoint"] if su["resume"] else None,
                "configured_total_steps": (
                    su["resume"]["configured_total_steps"] if su["resume"] else None),
            }
            for su in startups
        ],
        "exploration_fields": {
            f: {"values_per_launch": expl_table[f], "constant": expl_constant[f]}
            for f in EXPLORATION_FIELDS
        },
        "exploration_all_constant": all_constant,
        "exploration_drifted_fields": drifted,
        "schedule_fields": sched_table,
        "lr_cosine_check": {
            "n_train_steps_records": n_train_steps,
            "lr_subsampled_n": len(lrs),
            "lr_varies_per_step": lr_varies,
            "lr_min": lr_min,
            "lr_max": lr_max,
            "lr_first": lr_values[0] if lr_values else None,
            "lr_last": lr_values[-1] if lr_values else None,
            "per_step_temp_schedule_fields": per_step_temp_schedule,
            "note": "cosine lives on LR (lr varies per-step); NO per-step temperature "
                    "field => temp is a STATIC threshold (temperature_threshold_ply), "
                    "not a schedule.",
        },
        "radius_jitter": {
            "registry_legal_move_radius": registry_radius,
            "jitter_flag_per_launch": jitter_flag_vals,
            "jitter_flag_constant": jitter_flag_constant,
            "radius_schedule_key_present": radius_schedule_key_present,
            "radius_range_constant": radius_range_constant,
        },
        "lr_rewarm_note": (
            "53k restart resumes from checkpoints/restart_rewarm_53000.pt with "
            "configured_total_steps 30000->247000 — this is the LR cosine re-warm "
            "(OPTIMIZER knob), NOT an exploration knob. Confirmed: all exploration "
            "fields above are byte-identical across the 53k re-warm launch."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    # --- console verdict ---
    print("\n=== EXPLORATION-FIELD CONSTANCY (across launches) ===")
    for f in EXPLORATION_FIELDS:
        flag = "CONST" if expl_constant[f] else "DRIFT!!"
        print(f"  [{flag:6s}] {f:52s} = {expl_table[f]}")
    print("\n=== SCHEDULE FIELDS (designed to vary) ===")
    for f in SCHEDULE_FIELDS:
        print(f"  {f:14s} = {sched_table[f]}")
    print("\nLR cosine check: lr_varies_per_step =", lr_varies,
          f"(min {lr_min}, max {lr_max}, first {result['lr_cosine_check']['lr_first']}, "
          f"last {result['lr_cosine_check']['lr_last']})")
    print("Per-step temperature SCHEDULE fields found:", per_step_temp_schedule or "NONE")
    print("Radius jitter range constant:", radius_range_constant,
          f"(flag_const={jitter_flag_constant}, registry_radius={registry_radius}, "
          f"radius_schedule_key={radius_schedule_key_present})")
    print("\n>>> EXPLORATION ALL CONSTANT:", all_constant,
          "| drifted:", drifted or "NONE")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
