"""CONFRES resolution authority — one rule function per regime knob.

Submodules are imported lazily by consumers. This package ``__init__`` intentionally does
NOT eagerly import the submodules that touch ``hexo_rl.eval`` (radius/planner adoption in
later batches), which would form an import cycle via ``hexo_rl/eval/__init__.py`` (design §8,
finding N3). Import the specific rule module you need, e.g.
``from hexo_rl.config.resolve.bootstrap import resolve_bootstrap``.
"""
