"""§176 P1 — capture pre-refactor v6 forward fixture.

Run ONCE at HEAD before the network.py refactor. Produces
tests/fixtures/p1_v6_forward_baseline.pt; loaded by
test_v6_forward_byte_parity_vs_baseline to guard against v6 drift
during the 9 encoding-dispatch retires.
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
