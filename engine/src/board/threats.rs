/// Threat detection for the game viewer.
///
/// Scans all three hex axes for windows of 6 consecutive cells where one player
/// has N stones (N >= 3) and the remaining cells are empty (no opponent blocking).
///
/// The highlighted cells are the EMPTY cells within the window — the cells that
/// need to be filled. Occupied cells are never returned as threats.
///
/// Threat levels:
///   5 = CRITICAL (N=5, one empty cell — immediate win)
///   4 = FORCED   (N=4, two empty cells — wins in one compound turn)
///   3 = WARNING  (N=3, three empty cells — needs two more turns)
///
/// This function is NEVER called from MCTS or training. Viewer only.

use std::collections::HashMap;

use super::state::HEX_AXES;

const WIN_LEN: usize = 6;

/// A single threat cell: an empty cell within a threatening window.
#[derive(Debug, Clone, Copy)]
pub struct ThreatCell {
    pub q: i32,
    pub r: i32,
    pub level: u8,   // 3=warning, 4=forced, 5=critical
    pub player: u8,  // 0 or 1
}

/// Scan the board for threat cells.
///
/// `stones` maps (q, r) -> player (0 or 1).
pub fn get_threats(stones: &HashMap<(i32, i32), u8>) -> Vec<ThreatCell> {
    if stones.is_empty() {
        return Vec::new();
    }

    // Compute bounding box, extended by WIN_LEN in each direction.
    let mut min_q = i32::MAX;
    let mut max_q = i32::MIN;
    let mut min_r = i32::MAX;
    let mut max_r = i32::MIN;
    for &(q, r) in stones.keys() {
        if q < min_q { min_q = q; }
        if q > max_q { max_q = q; }
        if r < min_r { min_r = r; }
        if r > max_r { max_r = r; }
    }
    let margin = WIN_LEN as i32;
    min_q -= margin;
    max_q += margin;
    min_r -= margin;
    max_r += margin;

    // Track best threat level per (q, r, player).
    // Key: (q, r, player) -> best level seen.
    let mut best: HashMap<(i32, i32, u8), u8> = HashMap::new();

    for &(dq, dr) in &HEX_AXES {
        // For each axis direction, we need to scan all lines.
        // A "line" is defined by a starting point perpendicular to the direction.
        // We iterate over all relevant starting positions.

        // Determine the set of lines to scan. For direction (dq, dr), we need
        // a perpendicular coordinate to enumerate lines. We iterate over a grid
        // of starting points at the minimum extent along the direction.
        //
        // Strategy: collect all unique line identifiers. A line along (dq, dr) is
        // identified by the perpendicular component. For:
        //   (1, 0): lines differ by r
        //   (0, 1): lines differ by q
        //   (1,-1): lines differ by q+r (since moving along (1,-1) keeps q+r constant... no, q+r changes)
        //
        // Actually, for direction (dq, dr), two points are on the same line iff
        // their difference is a multiple of (dq, dr). We can parameterize:
        //   For (1,0): same line = same r. Scan q from min_q to max_q.
        //   For (0,1): same line = same q. Scan r from min_r to max_r.
        //   For (1,-1): same line = same (q+r). Scan: start at various (q,r) with q+r=const.

        if dq == 1 && dr == 0 {
            // Lines indexed by r. For each r, slide window over q.
            for r in min_r..=max_r {
                scan_line(stones, &mut best, min_q, max_q, r, r, dq, dr);
            }
        } else if dq == 0 && dr == 1 {
            // Lines indexed by q. For each q, slide window over r.
            for q in min_q..=max_q {
                scan_line(stones, &mut best, q, q, min_r, max_r, dq, dr);
            }
        } else {
            // (1, -1): lines indexed by q+r. For constant s = q+r,
            // q ranges and r = s - q.
            let min_s = min_q + min_r;
            let max_s = max_q + max_r;
            for s in min_s..=max_s {
                // Starting point: q = min_q, r = s - min_q (if in range)
                // We just need any starting point on the line; scan_line_general handles the rest.
                let start_q = min_q;
                let start_r = s - start_q;
                scan_line_general(stones, &mut best, start_q, start_r, dq, dr, max_q, max_r, min_q, min_r);
            }
        }
    }

    // Convert best map to result vec.
    best.into_iter()
        .map(|((q, r, player), level)| ThreatCell { q, r, level, player })
        .collect()
}

/// Scan a line along direction (dq, dr) starting at (start_q, start_r).
/// For axis (1,0): start_r is fixed, q varies from q_min to q_max.
/// For axis (0,1): start_q is fixed, r varies from r_min to r_max.
fn scan_line(
    stones: &HashMap<(i32, i32), u8>,
    best: &mut HashMap<(i32, i32, u8), u8>,
    start_q: i32,
    _end_q: i32,
    start_r: i32,
    _end_r: i32,
    dq: i32,
    dr: i32,
) {
    // Determine line length from bounding box.
    let (line_start_q, line_start_r, steps) = if dq == 1 && dr == 0 {
        (start_q, start_r, (_end_q - start_q + 1) as usize)
    } else {
        (start_q, start_r, (_end_r - start_r + 1) as usize)
    };

    if steps < WIN_LEN {
        return;
    }

    // Slide window of size WIN_LEN.
    for w in 0..=(steps - WIN_LEN) {
        let wq = line_start_q + (w as i32) * dq;
        let wr = line_start_r + (w as i32) * dr;

        check_window(stones, best, wq, wr, dq, dr);
    }
}

/// General line scanner for (1,-1) direction.
fn scan_line_general(
    stones: &HashMap<(i32, i32), u8>,
    best: &mut HashMap<(i32, i32, u8), u8>,
    start_q: i32,
    start_r: i32,
    dq: i32,
    dr: i32,
    max_q: i32,
    max_r: i32,
    min_q: i32,
    min_r: i32,
) {
    // Count how many steps we can take from start in direction (dq, dr)
    // while staying within bounds.
    let mut steps = 0usize;
    loop {
        let q = start_q + (steps as i32) * dq;
        let r = start_r + (steps as i32) * dr;
        if q < min_q || q > max_q || r < min_r || r > max_r {
            break;
        }
        steps += 1;
    }

    if steps < WIN_LEN {
        return;
    }

    for w in 0..=(steps - WIN_LEN) {
        let wq = start_q + (w as i32) * dq;
        let wr = start_r + (w as i32) * dr;
        check_window(stones, best, wq, wr, dq, dr);
    }
}

/// Check a single window of WIN_LEN cells starting at (wq, wr) in direction (dq, dr).
fn check_window(
    stones: &HashMap<(i32, i32), u8>,
    best: &mut HashMap<(i32, i32, u8), u8>,
    wq: i32,
    wr: i32,
    dq: i32,
    dr: i32,
) {
    // Count stones per player and collect empty cells.
    let mut p0_count = 0u8;
    let mut p1_count = 0u8;
    let mut empties: [(i32, i32); WIN_LEN] = [(0, 0); WIN_LEN];
    let mut n_empties = 0usize;

    for i in 0..WIN_LEN {
        let cq = wq + (i as i32) * dq;
        let cr = wr + (i as i32) * dr;
        match stones.get(&(cq, cr)) {
            Some(&0) => p0_count += 1,
            Some(&1) => p1_count += 1,
            None => {
                empties[n_empties] = (cq, cr);
                n_empties += 1;
            }
            _ => {}
        }
    }

    // Check if this window is a threat for player 0.
    if p1_count == 0 && p0_count >= 3 {
        let level = p0_count; // 3=warning, 4=forced, 5=critical
        for i in 0..n_empties {
            let (eq, er) = empties[i];
            let entry = best.entry((eq, er, 0)).or_insert(0);
            if level > *entry {
                *entry = level;
            }
        }
    }

    // Check if this window is a threat for player 1.
    if p0_count == 0 && p1_count >= 3 {
        let level = p1_count;
        for i in 0..n_empties {
            let (eq, er) = empties[i];
            let entry = best.entry((eq, er, 1)).or_insert(0);
            if level > *entry {
                *entry = level;
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_board_no_threats() {
        let stones: HashMap<(i32, i32), u8> = HashMap::new();
        assert!(get_threats(&stones).is_empty());
    }

    #[test]
    fn test_threat_forced_two_gaps() {
        // Line: O O O _ O along axis (1,0) at r=0
        // Stones at q=0,1,2,4 for player 1
        let mut stones = HashMap::new();
        stones.insert((0, 0), 1u8);
        stones.insert((1, 0), 1u8);
        stones.insert((2, 0), 1u8);
        stones.insert((4, 0), 1u8);

        let threats = get_threats(&stones);
        let forced: Vec<_> = threats.iter().filter(|t| t.level == 4 && t.player == 1).collect();

        // Window [0..5] = [O, O, O, _, O, _]: 4 stones, empties at (3,0) and (5,0)
        assert!(forced.iter().any(|t| t.q == 3 && t.r == 0),
            "Expected forced threat at (3,0), got: {:?}", forced);
        assert!(forced.iter().any(|t| t.q == 5 && t.r == 0),
            "Expected forced threat at (5,0), got: {:?}", forced);

        // Stone positions must NOT appear as threats.
        assert!(!forced.iter().any(|t| t.q == 0 && t.r == 0));
        assert!(!forced.iter().any(|t| t.q == 1 && t.r == 0));
        assert!(!forced.iter().any(|t| t.q == 2 && t.r == 0));
        assert!(!forced.iter().any(|t| t.q == 4 && t.r == 0));
    }

    #[test]
    fn test_threat_blocked_by_opponent() {
        // X O O O O _ _  along axis (1,0) at r=0
        // X (player 0) at q=0, O (player 1) at q=1,2,3,4
        let mut stones = HashMap::new();
        stones.insert((0, 0), 0u8); // X blocker
        stones.insert((1, 0), 1u8);
        stones.insert((2, 0), 1u8);
        stones.insert((3, 0), 1u8);
        stones.insert((4, 0), 1u8);

        let threats = get_threats(&stones);
        // Window [0..5] = [X, O, O, O, O, _] → blocked (has X)
        // Window [1..6] = [O, O, O, O, _, _] → 4 O's, forced, empties at (5,0) and (6,0)
        let forced: Vec<_> = threats.iter().filter(|t| t.level == 4 && t.player == 1).collect();
        assert!(forced.iter().any(|t| t.q == 5 && t.r == 0),
            "Expected forced at (5,0), got: {:?}", forced);
        assert!(forced.iter().any(|t| t.q == 6 && t.r == 0),
            "Expected forced at (6,0), got: {:?}", forced);
        // Cell 0 (the X) must NOT be a threat
        assert!(!forced.iter().any(|t| t.q == 0 && t.r == 0));
    }

    #[test]
    fn test_critical_threat_five_in_row() {
        // 5 stones in a row for player 0 along (1,0): q=0..4
        let mut stones = HashMap::new();
        for q in 0..5 {
            stones.insert((q, 0), 0u8);
        }
        let threats = get_threats(&stones);
        let critical: Vec<_> = threats.iter().filter(|t| t.level == 5 && t.player == 0).collect();
        // Window [0..5] = [O,O,O,O,O,_]: critical at (5,0)
        // Window [-1..4] = [_,O,O,O,O,O]: critical at (-1,0)
        assert!(critical.iter().any(|t| t.q == 5 && t.r == 0),
            "Expected critical at (5,0), got: {:?}", critical);
        assert!(critical.iter().any(|t| t.q == -1 && t.r == 0),
            "Expected critical at (-1,0), got: {:?}", critical);
    }

    #[test]
    fn test_warning_three_in_row() {
        // 3 stones in a row for player 0: q=0,1,2
        let mut stones = HashMap::new();
        stones.insert((0, 0), 0u8);
        stones.insert((1, 0), 0u8);
        stones.insert((2, 0), 0u8);
        let threats = get_threats(&stones);
        let warnings: Vec<_> = threats.iter().filter(|t| t.level == 3 && t.player == 0).collect();
        assert!(!warnings.is_empty(), "Expected warning threats for 3-in-a-row");
        // All warning cells must be empty
        for w in &warnings {
            assert!(!stones.contains_key(&(w.q, w.r)),
                "Threat cell ({},{}) is occupied!", w.q, w.r);
        }
    }

    #[test]
    fn test_ne_axis_threats() {
        // 4 stones along NE axis (0,1): (0,0),(0,1),(0,2),(0,3) for player 1
        let mut stones = HashMap::new();
        for r in 0..4 {
            stones.insert((0, r), 1u8);
        }
        let threats = get_threats(&stones);
        let forced: Vec<_> = threats.iter().filter(|t| t.level == 4 && t.player == 1).collect();
        assert!(!forced.is_empty(), "Expected forced threats along NE axis");
        // Empties should be at (0,4) and (0,5) or (0,-1) and (0,-2)
        for f in &forced {
            assert!(!stones.contains_key(&(f.q, f.r)),
                "Forced threat at ({},{}) is occupied!", f.q, f.r);
        }
    }

    #[test]
    fn test_nw_axis_threats() {
        // 4 stones along NW axis (1,-1): (0,0),(1,-1),(2,-2),(3,-3) for player 0
        let mut stones = HashMap::new();
        for i in 0..4 {
            stones.insert((i, -i), 0u8);
        }
        let threats = get_threats(&stones);
        let forced: Vec<_> = threats.iter().filter(|t| t.level == 4 && t.player == 0).collect();
        assert!(!forced.is_empty(), "Expected forced threats along NW axis, got: {:?}", threats);
        for f in &forced {
            assert!(!stones.contains_key(&(f.q, f.r)));
        }
    }

    #[test]
    fn test_both_players_threats() {
        // Player 0: 3-in-row at (0,0),(1,0),(2,0)
        // Player 1: 3-in-row at (0,5),(1,5),(2,5)
        let mut stones = HashMap::new();
        for q in 0..3 {
            stones.insert((q, 0), 0u8);
            stones.insert((q, 5), 1u8);
        }
        let threats = get_threats(&stones);
        let p0: Vec<_> = threats.iter().filter(|t| t.player == 0).collect();
        let p1: Vec<_> = threats.iter().filter(|t| t.player == 1).collect();
        assert!(!p0.is_empty(), "Player 0 should have threats");
        assert!(!p1.is_empty(), "Player 1 should have threats");
    }
}
