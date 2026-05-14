"""Regression test for C-001: KrakenBot parents[] path fix.

§176 P78d: krakenbot_bot.py moved hexo_rl/bootstrap/bots/ → hexo_rl/bots/,
which is one level shallower. The required parents[] index decreased
from 3 to 2.
"""
from pathlib import Path


def test_krakenbot_path_resolves_inside_repo():
    """_KRAKENBOT_ROOT must resolve to <repo_root>/vendor/bots/krakenbot, not outside."""
    # Derive repo root the same way the fixed code does.
    bots_file = Path(__file__).parent.parent / "hexo_rl" / "bots" / "krakenbot_bot.py"
    assert bots_file.exists(), f"krakenbot_bot.py not found at {bots_file}"

    # parents[2] from bots_file location:
    # [0] = hexo_rl/bots
    # [1] = hexo_rl
    # [2] = <repo root>
    expected_root = bots_file.parents[2] / "vendor" / "bots" / "krakenbot"
    wrong_root    = bots_file.parents[3] / "vendor" / "bots" / "krakenbot"

    # Confirm the wrong index goes outside the repo.
    repo_root = bots_file.parents[2]
    assert not str(wrong_root).startswith(str(repo_root)), (
        f"parents[3] should resolve OUTSIDE repo but got {wrong_root}"
    )

    # Confirm the module constant uses parents[2].
    import ast
    src = bots_file.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_KRAKENBOT_ROOT":
                    src_snippet = ast.get_source_segment(src, node.value) or ""
                    assert "parents[2]" in src_snippet, (
                        f"_KRAKENBOT_ROOT must use parents[2], got: {src_snippet}"
                    )
                    assert "parents[3]" not in src_snippet, (
                        f"_KRAKENBOT_ROOT must NOT use parents[3], got: {src_snippet}"
                    )
                    return
    raise AssertionError("_KRAKENBOT_ROOT assignment not found in krakenbot_bot.py")
