"""Pre-flight regression tests for HeXO refactor invariants (§176).

Each test in this directory pins one behavioral invariant catalogued in
reports/refactor_audit/00_MASTER_PLAN.md §E. The pins must exist BEFORE
the corresponding production refactor lands so the refactor can use them
as a guard rail (per docs/refactor-template.md pure-move discipline).

Adding a test here? Reference the INV# from master plan §E and the
proposal (P#) whose execute commit relies on it.
"""
