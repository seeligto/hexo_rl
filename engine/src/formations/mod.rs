
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
    /// Returns true if the current player has a forced-win formation
    /// (open four — 4 in a row with both ends empty on a hex axis).
    /// Used for analysis and logging; NOT used in MCTS path.
    pub fn has_forced_win(board: &Board, player: Player) -> bool {
        let target_cell = match player {
            Player::One => crate::board::Cell::P1,
            Player::Two => crate::board::Cell::P2,
        };

        for (&(q, r), &cell) in board.cells.iter() {
            if cell != target_cell {
                continue;
            }

            for &(dq, dr) in &HEX_AXES {
                let forward = board.count_direction(q, r, dq, dr, target_cell);
                let backward = board.count_direction(q, r, -dq, -dr, target_cell);
                let total = 1 + forward + backward;

                if total >= 4 {
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
}
