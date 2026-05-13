"""INV4 (§176 §E) — cosine temp must pair with LEGAL_MOVE_RADIUS jitter.

§156 / L9: cosine temp is load-bearing for draw-collapse mitigation when
active; LEGAL_MOVE_RADIUS jitter is mandatory pairing partner (Q2 verdict).
Variants opting into cosine temp must also enable jitter.

Actual key paths (verified against configs/selfplay.yaml and all
configs/variants/*.yaml on 2026-05-13):

  cosine active:  selfplay.playout_cap.temperature_threshold_compound_moves > 0
                  (NO separate cosine.enabled flag; threshold=0 disables cosine)
  jitter:         selfplay.legal_move_radius_jitter (bool, must be truthy)

Effective config = deep-merge(configs/selfplay.yaml, variant) via
hexo_rl.utils.config.load_config — same merge logic the runtime uses.

Surfaces drift; does NOT auto-fix variants.
"""
from pathlib import Path

import pytest

from hexo_rl.utils.config import load_config


_BASE_CONFIG = Path(__file__).parent.parent.parent / "configs" / "selfplay.yaml"
_VARIANT_DIR = Path(__file__).parent.parent.parent / "configs" / "variants"

_VARIANT_PATHS = sorted(_VARIANT_DIR.glob("*.yaml"))


@pytest.mark.parametrize(
    "variant_path",
    _VARIANT_PATHS,
    ids=lambda p: p.name,
)
def test_cosine_temp_implies_jitter(variant_path: Path) -> None:
    """INV4: every variant with cosine temp active must also enable jitter.

    Cosine active = temperature_threshold_compound_moves > 0 in effective
    merged config. Jitter = selfplay.legal_move_radius_jitter truthy.
    """
    cfg = load_config(str(_BASE_CONFIG), str(variant_path))
    selfplay = cfg.get("selfplay", {}) or {}
    playout_cap = selfplay.get("playout_cap", {}) or {}
    threshold = playout_cap.get("temperature_threshold_compound_moves", 0)
    cosine_active = int(threshold) > 0

    if not cosine_active:
        pytest.skip(f"{variant_path.name}: cosine temp not active (threshold={threshold})")

    jitter = selfplay.get("legal_move_radius_jitter", False)
    assert jitter, (
        f"{variant_path.name}: cosine temp active (threshold={threshold}) "
        f"without legal_move_radius_jitter — violates §156/L9 mandatory pairing"
    )
