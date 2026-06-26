"""D-TACTICAL — plane-state → engine.Board reconstruction (Option-B + translate).

The d7 proven-loss bank stores 4-plane v6_live2 states, not move sequences, and
the Rust Board has no place-stone-ignore-turn API (it tracks turn cadence and
validates legality). To feed a plane-state to the deploy search / native tactical
search we replay the stones in a LEGAL cadence order:

  - cadence: P1 opens with 1 stone (ply 0), then players alternate 2 stones/turn.
  - the empty-board legal region is the hardcoded central 5x5 (radius-independent),
    so the opener must sit there. Clusters offset from window-center would fail.
  - FIX: per-cluster windowing makes the net translation-invariant, so we rigidly
    TRANSLATE the whole position to put a chosen P1 opener at (0,0). Relative
    geometry (all threats, the full game tree) is preserved; only absolute coords
    shift. We then place the rest under a generous radius override and reset the
    radius to the deploy default (5) so the search sees correct future legal moves.

Validation (V1): reconstructed board's stone set + current_player + moves_remaining
must byte-match the (translated) target state. Failures are dropped + LOGGED.

Plane layout (v6_live2, audited): 0=to-move stones, 1=opp stones,
2=moves_remaining==2 ? 1.0 : 0.0 (bcast), 3=ply%2 (bcast). (row,col)->(q,r):
q = col-9, r = row-9.
"""
from __future__ import annotations
import numpy as np
import engine

DEPLOY_RADIUS = 5
RECON_RADIUS = 18  # generous during placement; reset to DEPLOY_RADIUS after

HEX_AXES = [(1, 0), (0, 1), (1, -1)]


def _cells_from_plane(plane: np.ndarray):
    """Return list of (q,r) where plane>0.5. (row,col)->(q,r): q=col-9, r=row-9."""
    rs, cs = np.where(plane > 0.5)
    return [(int(c) - 9, int(r) - 9) for r, c in zip(rs, cs)]


def to_move_is_p1(n_total: int) -> bool:
    """Whose turn (absolute) after n_total stones placed, per the HTTT cadence.
    ply 0 = P1 (opener). ply>=1: turn (ply-1)//2; P2 if turn even else P1."""
    if n_total == 0:
        return True
    return ((n_total - 1) // 2) % 2 == 1


def decode_state(state: np.ndarray):
    """Decode a (4,19,19) v6_live2 state into absolute-player stone sets + turn.

    Returns dict: p1, p2 (sets of (q,r)), to_move (1/-1), mr (1/2), parity (0/1).
    """
    p0, p1pl, p2pl, p3 = state[0], state[1], state[2], state[3]
    tomove = _cells_from_plane(p0)
    opp = _cells_from_plane(p1pl)
    n = len(tomove) + len(opp)
    tm_is_p1 = to_move_is_p1(n)
    if tm_is_p1:
        p1, p2 = set(tomove), set(opp)
        to_move = 1
    else:
        p1, p2 = set(opp), set(tomove)
        to_move = -1
    mr = 2 if np.mean(p2pl) > 0.5 else 1  # mr==2 iff plane2 broadcast-1
    parity = int(round(float(np.mean(p3))))
    return {"p1": p1, "p2": p2, "to_move": to_move, "mr": mr, "parity": parity, "n": n}


def _turn_color_schedule(n_p1: int, n_p2: int):
    """The cadence color per placement slot: P1 once, then 2 P2, 2 P1, 2 P2, ...
    Returns a list of player ids (1/-1) of length n_p1+n_p2, or None if the counts
    are inconsistent with the cadence (cannot be a real position)."""
    sched = []
    if n_p1 == 0 and n_p2 == 0:
        return sched
    # opener
    sched.append(1)
    rem1, rem2 = n_p1 - 1, n_p2
    # alternate turns of 2: P2 turn, P1 turn, ...
    turn_owner = -1
    while rem1 > 0 or rem2 > 0:
        for _ in range(2):
            if turn_owner == -1 and rem2 > 0:
                sched.append(-1); rem2 -= 1
            elif turn_owner == 1 and rem1 > 0:
                sched.append(1); rem1 -= 1
            elif turn_owner == -1 and rem2 == 0:
                # partial P2 turn but no P2 left -> inconsistent unless we are at the very end
                break
            elif turn_owner == 1 and rem1 == 0:
                break
        turn_owner *= -1
        if rem1 < 0 or rem2 < 0:
            return None
    if rem1 != 0 or rem2 != 0:
        return None
    return sched


def reconstruct(state: np.ndarray, validate: bool = True):
    """Reconstruct an engine.Board from a (4,19,19) v6_live2 state.

    Returns (board, info) on success or (None, info) on failure. info carries the
    failure reason and the translation applied. The board has legal_move_radius
    reset to DEPLOY_RADIUS (5).
    """
    d = decode_state(state)
    p1, p2 = d["p1"], d["p2"]
    n_p1, n_p2 = len(p1), len(p2)
    info = {"n": d["n"], "to_move": d["to_move"], "mr": d["mr"],
            "n_p1": n_p1, "n_p2": n_p2, "reason": None, "shift": None}

    if n_p1 == 0:
        info["reason"] = "no_p1_stone"; return None, info
    sched = _turn_color_schedule(n_p1, n_p2)
    if sched is None:
        info["reason"] = "cadence_inconsistent"; return None, info

    # Translate so a chosen P1 opener lands at (0,0). Prefer a P1 stone already
    # near window-center to minimize the shift magnitude (keep within +/-70 for
    # SealBot's 140x140 array).
    opener = min(p1, key=lambda c: c[0] * c[0] + c[1] * c[1])
    sq, sr = opener
    info["shift"] = (-sq, -sr)
    tp1 = {(q - sq, r - sr) for (q, r) in p1}
    tp2 = {(q - sq, r - sr) for (q, r) in p2}
    opener_t = (0, 0)

    board = engine.Board.with_encoding_name("v6_live2_ls")
    board.override_legal_move_radius(RECON_RADIUS)

    # Greedy legal placement following the cadence color schedule. opener first.
    remaining = {1: set(tp1), -1: set(tp2)}
    remaining[1].discard(opener_t)
    try:
        board.apply_move(0, 0)  # P1 opener at origin (central 5x5 -> always legal)
    except Exception as e:
        info["reason"] = f"opener_illegal:{e}"; return None, info

    placed = {opener_t}
    for slot_color in sched[1:]:
        pool = remaining[slot_color]
        if not pool:
            info["reason"] = "schedule_pool_empty"; return None, info
        # pick the remaining stone of this color closest to any placed stone
        best = None; best_d = None
        for c in pool:
            dmin = min(_hexdist(c, pc) for pc in placed)
            if best_d is None or dmin < best_d:
                best_d = dmin; best = c
        try:
            board.apply_move(best[0], best[1])
        except Exception as e:
            info["reason"] = f"place_illegal:{best}:{e}"; return None, info
        pool.discard(best); placed.add(best)

    board.override_legal_move_radius(DEPLOY_RADIUS)

    if validate:
        ok, why = _validate(board, tp1, tp2, d["to_move"], d["mr"])
        if not ok:
            info["reason"] = f"validate:{why}"; return None, info
        info["reason"] = "ok"
    return board, info


def _hexdist(a, b):
    dq = a[0] - b[0]; dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _validate(board, tp1, tp2, want_to_move, want_mr):
    got = board.get_stones()
    g1 = {(q, r) for (q, r, p) in got if p == 1}
    g2 = {(q, r) for (q, r, p) in got if p == -1}
    if g1 != tp1:
        return False, f"p1_mismatch n_got={len(g1)} n_want={len(tp1)}"
    if g2 != tp2:
        return False, f"p2_mismatch n_got={len(g2)} n_want={len(tp2)}"
    if board.current_player != want_to_move:
        return False, f"to_move {board.current_player}!={want_to_move}"
    if board.moves_remaining != want_mr:
        return False, f"mr {board.moves_remaining}!={want_mr}"
    return True, "ok"


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/timmy/Work/Hexo/hexo_rl/data/distill_d7_train.npz"
    d = np.load(path)
    states = d["states"]
    ok = 0; reasons = {}
    shifts = []
    for i in range(len(states)):
        b, info = reconstruct(states[i].astype(np.float32))
        if b is not None:
            ok += 1
            shifts.append(max(abs(info["shift"][0]), abs(info["shift"][1])))
        else:
            reasons[info["reason"]] = reasons.get(info["reason"], 0) + 1
    n = len(states)
    print(f"V1 reconstruction: {ok}/{n} = {100*ok/n:.1f}% byte-exact")
    if shifts:
        import numpy as _np
        print(f"  shift magnitude: max={max(shifts)} median={int(_np.median(shifts))} "
              f">70(sealbot risk)={sum(s>70 for s in shifts)}")
    if reasons:
        print(f"  failures: {reasons}")
