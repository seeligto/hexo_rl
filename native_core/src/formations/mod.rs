
use crate::board::{Board, Player, HEX_AXES};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Formation {
    Triangle,
    OpenThree,
    Rhombus,
    Arch,
    Trapezoid,
    Ladder,
    Bone,
}

pub struct FormationDetector;

impl FormationDetector {
    /// Returns true if the current player has a forced-win formation.
    /// For now, we implement a simplified version that checks for 
    /// well-known community patterns.
    pub fn has_forced_win(board: &Board, player: Player) -> bool {
        let target_cell = match player {
            Player::One => crate::board::Cell::P1,
            Player::Two => crate::board::Cell::P2,
        };

        for (&(q, r), &cell) in &board.cells {
            if cell != target_cell {
                continue;
            }

            for &(dq, dr) in &HEX_AXES {
                let forward = board.count_direction(q, r, dq, dr, target_cell);
                let backward = board.count_direction(q, r, -dq, -dr, target_cell);
                let total = 1 + forward + backward;

                if total >= 4 {
                    // Check if both ends are open.
                    // Start of the line is (q - (backward+1)*dq, r - (backward+1)*dr)
                    // End of the line is (q + (forward+1)*dq, r + (forward+1)*dr)
                    let head_q = q - (backward as i32 + 1) * dq;
                    let head_r = r - (backward as i32 + 1) * dr;
                    let tail_q = q + (forward as i32 + 1) * dq;
                    let tail_r = r + (forward as i32 + 1) * dr;

                    if board.get_cell(head_q, head_r) == crate::board::Cell::Empty
                        && board.get_cell(tail_q, tail_r) == crate::board::Cell::Empty
                    {
                        return true;
                    }
                }
            }
        }
        
        false
    }

    fn detect_open_three(_board: &Board, _player: Player) -> bool {
        // TODO: Implement actual geometric detection
        // For now, return false to avoid false positives until logic is solid
        false
    }

    fn detect_triangle(_board: &Board, _player: Player) -> bool {
        // TODO: Implement actual geometric detection
        false
    }
}
