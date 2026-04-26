"""hexo_rl.utils — utility submodules.

Intentionally empty so that ``from hexo_rl.utils.X import ...`` does NOT
trigger torch / numpy imports via sibling re-exports. The previous
``from hexo_rl.utils.device import best_device`` line pulled in torch at
package init, which broke the contract that ``hexo_rl.utils.cpu_budget``
must be importable BEFORE numpy / torch (it sets OMP_NUM_THREADS et al.).

Callers should always use the fully-qualified submodule path:
    from hexo_rl.utils.config import load_config
    from hexo_rl.utils.device import best_device
    from hexo_rl.utils.cpu_budget import apply_auto_thread_budget
"""
