"""§5 audit allowlist — must NOT flag these false-positive patterns."""
import textwrap
import pytest


@pytest.fixture
def audit_one_file(tmp_path):
    """Run §5 on a synthetic file; return findings filtered to that file.

    Files are placed under tmp_path/hexo_rl/ so _section_hardcode finds them
    (it searches repo_root/hexo_rl and repo_root/engine/src).
    """
    src_dir = tmp_path / "hexo_rl"
    src_dir.mkdir()

    def _run(filename: str, content: str):
        from hexo_rl.encoding.audit import _section_hardcode, AuditReport
        p = src_dir / filename
        p.write_text(textwrap.dedent(content).lstrip("\n"))
        report = AuditReport()
        _section_hardcode(report, repo_root=tmp_path)
        # Only return warn/error findings — info("no unjustified hits") is not a false positive.
        hits = [f for f in report.findings if f.section == "§5" and f.severity != "info"]
        return hits
    return _run


def test_skip_apply_move_coords(audit_one_file):
    """`apply_move(5, 0)` is a coordinate, not an encoding constant."""
    hits = audit_one_file("foo.rs", """
        fn body() {
            b.apply_move(5, 0).unwrap();
            b.apply_move(8, -2).unwrap();
            b.apply_move(0, 19).unwrap();
        }
    """)
    assert hits == [], f"false-positive: {[h.message for h in hits]}"


def test_skip_loop_bounds(audit_one_file):
    hits = audit_one_file("foo.rs", """
        fn body() {
            for _ in 0..5 {}
            for q in 0..=8 {}
            let _: Vec<_> = (0..19).collect();
        }
    """)
    assert hits == [], f"false-positive: {[h.message for h in hits]}"


def test_skip_test_module(audit_one_file):
    hits = audit_one_file("foo.rs", """
        pub fn prod_fn() -> usize { 19 }  // SHOULD flag
        #[cfg(test)]
        mod tests {
            fn helper() -> usize { 19 }  // should NOT flag
            #[test]
            fn t() { assert_eq!(helper(), 19); }  // should NOT flag
        }
    """)
    flagged = hits  # already filtered to §5
    assert len(flagged) == 1, f"expected 1 hit, got {len(flagged)}: {[h.message for h in flagged]}"


def test_skip_float_tolerance(audit_one_file):
    hits = audit_one_file("foo.rs", """
        fn body() {
            assert!((a - b).abs() < 1e-5);
            let eta = 1e-8;
        }
    """)
    assert hits == [], f"false-positive: {[h.message for h in hits]}"


def test_skip_tunables(audit_one_file):
    hits = audit_one_file("foo.py", """
        c_puct = 1.5
        dirichlet_alpha = 0.25
        temp_min = 0.05
        figsize = (10, 5)
    """)
    assert hits == [], f"false-positive: {[h.message for h in hits]}"


def test_skip_docstring_literals(audit_one_file):
    hits = audit_one_file("foo.py", '''
        def f():
            """
            Computes 19 things across 5 dimensions.
            """
            pass
    ''')
    assert hits == [], f"false-positive: {[h.message for h in hits]}"


def test_skip_trailing_comment(audit_one_file):
    hits = audit_one_file("foo.rs", """
        fn body() {
            let n = some_fn(); // see issue #19 for details
        }
    """)
    assert hits == [], f"false-positive: {[h.message for h in hits]}"


def test_flag_real_encoding_constant(audit_one_file):
    """True positive — should still flag."""
    # 361 is in _HARDCODE_TARGETS; a bare `361` in prod code must flag.
    hits = audit_one_file("foo.rs", """
        pub fn build_buffer() -> Vec<f32> {
            let policy_len: usize = 361;
            vec![0.0; policy_len]
        }
    """)
    # At least one warn/error finding should be emitted.
    assert len(hits) >= 1, f"missed true-positive: no §5 warn/error findings"
    assert "unjustified literal" in hits[0].message, f"unexpected message: {hits[0].message}"


def test_skip_range_explicit(audit_one_file):
    """0..19 range bound — strip before scanning."""
    hits = audit_one_file("foo.rs", """
        fn body() {
            for i in 0..19 {
                do_thing(i);
            }
            let v: Vec<i32> = (0..=25).collect();
        }
    """)
    assert hits == [], f"false-positive: {[h.message for h in hits]}"


def test_skip_canonical_define_line(audit_one_file):
    """Canonical-define lines are the source of truth — don't flag them."""
    hits = audit_one_file("foo.rs", """
        pub const BOARD_SIZE: usize = 19;
        pub const N_PLANES: usize = 8;
    """)
    assert hits == [], f"false-positive: {[h.message for h in hits]}"


def test_skip_set_row_col_calls(audit_one_file):
    """set_row_*/set_col_*/set_diag_* coordinate calls — skip."""
    hits = audit_one_file("foo.rs", """
        fn body() {
            set_row_e(&mut bb, 0, 0, 5);
            set_col_ne(&mut bb, 0, 5, 5);
            set_diag_se(&mut bb, -4, -5, 5);
        }
    """)
    assert hits == [], f"false-positive: {[h.message for h in hits]}"


def test_skip_cells_insert(audit_one_file):
    """cells.insert((q, r), ...) coordinate pair — skip."""
    hits = audit_one_file("foo.rs", """
        fn body() {
            board.cells.insert((5, 5), Cell::P2);
            board.cells.insert((5, 8), Cell::P1);
        }
    """)
    assert hits == [], f"false-positive: {[h.message for h in hits]}"


def test_skip_python_test_class(audit_one_file):
    """Python class Test* body should be skipped."""
    hits = audit_one_file("foo.py", """
        def prod_fn():
            return 19  # SHOULD flag

        class TestFoo:
            def test_it(self):
                assert something == 19  # should NOT flag
    """)
    # prod_fn line should flag, TestFoo body should not
    messages = [h.message for h in hits]
    assert len(hits) == 1, f"expected 1 hit, got {len(hits)}: {messages}"
