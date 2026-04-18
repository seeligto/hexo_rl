"""Regression test for C-001: KrakenBot parents[] path fix."""
from pathlib import Path


def test_krakenbot_path_resolves_inside_repo():
    """_KRAKENBOT_ROOT must resolve to <repo_root>/vendor/bots/krakenbot, not outside."""
    # Derive repo root the same way the fixed code does.
    bots_file = Path(__file__).parent.parent / "hexo_rl" / "bootstrap" / "bots" / "krakenbot_bot.py"
    assert bots_file.exists(), f"krakenbot_bot.py not found at {bots_file}"

    # parents[3] from bots_file location:
    # [0] = hexo_rl/bootstrap/bots
    # [1] = hexo_rl/bootstrap
    # [2] = hexo_rl
    # [3] = <repo root>
    expected_root = bots_file.parents[3] / "vendor" / "bots" / "krakenbot"
    wrong_root    = bots_file.parents[4] / "vendor" / "bots" / "krakenbot"

    # Confirm the wrong index goes outside the repo.
    repo_root = bots_file.parents[3]
    assert not str(wrong_root).startswith(str(repo_root)), (
        f"parents[4] should resolve OUTSIDE repo but got {wrong_root}"
    )

    # Confirm the module constant uses parents[3].
    import ast, textwrap
    src = bots_file.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_KRAKENBOT_ROOT":
                    src_snippet = ast.get_source_segment(src, node.value) or ""
                    assert "parents[3]" in src_snippet, (
                        f"_KRAKENBOT_ROOT must use parents[3], got: {src_snippet}"
                    )
                    assert "parents[4]" not in src_snippet, (
                        f"_KRAKENBOT_ROOT must NOT use parents[4], got: {src_snippet}"
                    )
                    return
    raise AssertionError("_KRAKENBOT_ROOT assignment not found in krakenbot_bot.py")
