"""§176 P1 — capture v6 forward fixture.

Re-stamped under §S181 FU-2 A2 (multi-scale avg-pool value head, 2026-05-23) —
the architecture changed value_fc1 input dim from 2C to 5C, breaking the
prior fixture by design. The test's guarantee is "v6 forward output is
byte-stable post-A2"; re-run this generator on any future intentional
arch change and update the fixture in the same commit.
"""
import torch

from hexo_rl.model.network import HexTacToeNet


def main() -> None:
    torch.manual_seed(0)
    net = HexTacToeNet(encoding="v6").eval()
    x = torch.zeros(1, 8, 19, 19)
    x[0, 0, 9, 9] = 1.0  # one stone in center to break trivial symmetry
    with torch.no_grad():
        log_p, v, v_logit = net(x)
    torch.save(
        {"log_p": log_p, "v": v, "v_logit": v_logit, "x": x},
        "tests/fixtures/p1_v6_forward_baseline.pt",
    )
    print(f"saved fixture: log_p={tuple(log_p.shape)} v={tuple(v.shape)}")


if __name__ == "__main__":
    main()
