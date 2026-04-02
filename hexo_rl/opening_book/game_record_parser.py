
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class GameTurn:
    turn_number: int
    q1: int; r1: int
    q2: int; r2: int
    threats: int = 0

@dataclass
class GameRecord:
    metadata: Dict[str, str]
    turns: List[GameTurn]

def parse_axial_record(text: str) -> GameRecord:
    """Parse a game record in the live axial notation format."""
    metadata = {}
    
    # Extract metadata
    meta_part = text.split(";")
    if len(meta_part) > 1 and "[" in meta_part[0]:
        meta_str = meta_part[0]
        kv_pairs = re.findall(r'(\w+)\[(.*?)\]', meta_str)
        for k, v in kv_pairs:
            metadata[k] = v
        turns_str = ";".join(meta_part[1:])
    else:
        turns_str = text

    # Extract turns
    # Format: N. [q,r][q,r]!?;
    turns = []
    turn_matches = re.finditer(r'(\d+)\.\s*\[(-?\d+),(-?\d+)\]\s*\[(-?\d+),(-?\d+)\]\s*(!*)', turns_str)
    
    for m in turn_matches:
        turns.append(GameTurn(
            turn_number=int(m.group(1)),
            q1=int(m.group(2)), r1=int(m.group(3)),
            q2=int(m.group(4)), r2=int(m.group(5)),
            threats=len(m.group(6))
        ))
        
    return GameRecord(metadata, turns)

def record_to_moves(record: GameRecord) -> List[Tuple[int, int]]:
    """Convert a GameRecord to a flat list of (q, r) moves."""
    # Cross at [0,0] is always implicit and first
    moves = [(0, 0)]
    for turn in record.turns:
        moves.append((turn.q1, turn.r1))
        moves.append((turn.q2, turn.r2))
    return moves
