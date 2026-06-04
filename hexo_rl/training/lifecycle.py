"""Training-loop subsystem lifecycle: model builds + monitor/probe/dashboard
boot and teardown.

Owns construction of the inference model, eval-side model, and a
``LoopSubsystems`` bundle covering GPU monitor, disk guard, dashboards,
probes, and TensorBoard writer. Anchor (best_model.pt) management lives
in training/anchor.py; eval pipeline construction lives in
eval/pipeline_setup.py.

Extracted from training/loop.py per §159 refactor. No behavior change.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.model_defaults import MODEL_HPARAM_DEFAULTS
from hexo_rl.monitoring.disk_guard import DiskGuard
from hexo_rl.monitoring.early_game_probe import EarlyGameProbe
from hexo_rl.monitoring.events import register_jsonl_sink, register_renderer
from hexo_rl.monitoring.gpu_monitor import GPUMonitor
from hexo_rl.monitoring.value_probe import ValueProbe
from hexo_rl.training.trainer import Trainer

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class InfModelArch:
    board_size: int
    res_blocks: int
    filters: int
    in_channels: int
    se_reduction_ratio: int
    input_channels: Any


def build_inference_model(
    trainer: Trainer,
    device: torch.device,
) -> tuple[torch.nn.Module, InfModelArch]:
    # ── Inference model — separate instance owned by InferenceServer ──────────
    from hexo_rl.encoding import resolve_from_config as _registry_resolve
    board_size         = _registry_resolve(trainer.config).trunk_size
    res_blocks         = int(trainer.config.get("res_blocks",         MODEL_HPARAM_DEFAULTS["res_blocks"]))
    filters            = int(trainer.config.get("filters",            MODEL_HPARAM_DEFAULTS["filters"]))
    in_channels        = int(trainer.config.get("in_channels",        MODEL_HPARAM_DEFAULTS["in_channels"]))
    se_reduction_ratio = int(trainer.config.get("se_reduction_ratio", MODEL_HPARAM_DEFAULTS["se_reduction_ratio"]))
    input_channels     = trainer.config.get("input_channels", None)

    _torch_compile_enabled = (
        trainer.config.get("torch_compile", False) and device.type == "cuda"
    )
    inf_model = HexTacToeNet(
        board_size=board_size,
        in_channels=in_channels,
        input_channels=input_channels,
        res_blocks=res_blocks,
        filters=filters,
        se_reduction_ratio=se_reduction_ratio,
    ).to(device)
    # §S181-AUDIT Wave 2 — route through trainer.inference_state_dict so the
    # InferenceServer reads EMA weights when EMA is enabled. At step 0 this
    # is identical to trainer.model's state (EMA was deep-copied from it).
    inf_model.load_state_dict(trainer.inference_state_dict())
    inf_model.eval()
    if _torch_compile_enabled:
        # inf_model runs in InferenceServer background thread. reduce-overhead's
        # cudagraph_trees uses C++ dynamic TLS (per-thread); cross-thread entry
        # asserts or deadlocks (configs/training.yaml:16 deadlock @ step 6002,
        # bench fix c26b9b4). Force mode=default here — kernel fusion only,
        # thread-safe. trainer.model keeps config-driven mode (main-thread only).
        _compile_mode = "default"
        try:
            inf_model = torch.compile(inf_model, mode=_compile_mode, fullgraph=False)
            log.info("torch_compile_inf_enabled", mode=_compile_mode)
        except Exception as exc:
            log.warning("torch_compile_inf_failed", mode=_compile_mode, error=str(exc))

    arch = InfModelArch(
        board_size=board_size,
        res_blocks=res_blocks,
        filters=filters,
        in_channels=in_channels,
        se_reduction_ratio=se_reduction_ratio,
        input_channels=input_channels,
    )
    return inf_model, arch


def cuda_warmup(
    inf_model: torch.nn.Module,
    device: torch.device,
    board_size: int,
) -> None:
    """Warm up CUDA kernels with a dummy forward pass.

    ``board_size`` here is the model trunk side (registry `trunk_size`,
    e.g. 19 for v6, 25 for v6w25/v8). Callers should pass either
    ``arch.board_size`` (post-`_propagate_encoding_into_config` it equals
    the trunk) or ``spec.trunk_size`` directly.
    """
    # ── CUDA warm-up ─────────────────────────────────────────────────────────
    # Force CUDA kernel compilation now (before workers start) so the first
    # inference call from a worker returns immediately instead of blocking for
    # 90-120s while PyTorch JIT-compiles kernels. Without this, the warmup
    # phase shows "games=0" for ~2 minutes on a cold start, which looks broken.
    if device.type == "cuda":
        log.info("cuda_warmup_start")
        _t_warmup = time.time()
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                # Warmup width must match what forward() expects as input.
                # Sweep variants (`_input_channels` set) index_select from the
                # wire width inside forward, so they need the wire width; every
                # other model (v6 → 8, v6tp → 10) takes exactly in_channels with
                # no internal slice. Using the v6 WIRE_CHANNELS constant fed an
                # 8-plane dummy into v6tp's 10-channel conv and crashed here.
                from hexo_rl.model.network import WIRE_CHANNELS as _WIRE_CH
                _base = getattr(inf_model, "_orig_mod", inf_model)
                if getattr(_base, "_input_channels", None) is not None:
                    _warmup_ch = _WIRE_CH
                else:
                    _warmup_ch = int(getattr(_base, "in_channels", _WIRE_CH))
                _dummy = torch.zeros(1, _warmup_ch, board_size, board_size, device=device)
                inf_model(_dummy)
        torch.cuda.synchronize()
        log.info("cuda_warmup_done", elapsed_sec=round(time.time() - _t_warmup, 1))


def cuda_stream_audit(
    config: dict[str, Any],
    device: torch.device,
) -> None:
    # ── CUDA stream audit (B4 perf probe) ─────────────────────────────────────
    # Logged from the main (training) thread context. InferenceServer logs its
    # own stream in run(). If both are on the same default stream pointer, no
    # copy/compute overlap is possible — the Q18 hypothesis.
    _diag = config.get("diagnostics") if isinstance(config.get("diagnostics"), dict) else {}
    if bool(_diag.get("perf_timing", False)) and device.type == "cuda":
        try:
            _cur = torch.cuda.current_stream(device)
            _def = torch.cuda.default_stream(device)
            log.info(
                "cuda_stream_audit",
                context="training_thread",
                current_stream_ptr=int(_cur.cuda_stream),
                default_stream_ptr=int(_def.cuda_stream),
                on_default_stream=_cur.cuda_stream == _def.cuda_stream,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("cuda_stream_audit_failed", context="training_thread", error=str(exc))


def build_eval_model(arch: InfModelArch, device: torch.device) -> HexTacToeNet:
    eval_model = HexTacToeNet(
        board_size=arch.board_size, res_blocks=arch.res_blocks, filters=arch.filters,
        in_channels=arch.in_channels, input_channels=arch.input_channels,
        se_reduction_ratio=arch.se_reduction_ratio,
    ).to(device)
    eval_model.eval()
    return eval_model


@dataclass
class LoopSubsystems:
    gpu_monitor: GPUMonitor
    disk_guard: DiskGuard
    early_game_probe: Optional[EarlyGameProbe]
    value_probe: Optional[ValueProbe]
    value_probe_interval: int
    composition_interval: int
    instrumentation_enabled: bool
    axis_baseline: dict
    tb_writer: Optional[Any]
    dashboards: list = field(default_factory=list)

    def teardown(self) -> None:
        self.gpu_monitor.stop()
        self.gpu_monitor.join(timeout=2.0)
        self.disk_guard.stop()
        for d in self.dashboards:
            try:
                d.stop()
            except Exception:
                pass


def build_subsystems(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
    run_id: str,
) -> LoopSubsystems:
    """Build and start GPU monitor, disk guard, probes, TB writer, dashboards."""
    # ── GPU monitor ───────────────────────────────────────────────────────────
    gpu_monitor = GPUMonitor(interval_sec=5)
    gpu_monitor.start()

    # ── Disk guard ────────────────────────────────────────────────────────────
    _dg_cfg = config.get("disk_guard", {})
    disk_guard = DiskGuard(
        watch_path=args.checkpoint_dir,
        interval_sec=float(_dg_cfg.get("interval_sec", 60.0)),
        warn_gb=float(_dg_cfg.get("warn_gb", 10.0)),
        fail_gb=float(_dg_cfg.get("fail_gb", 5.0)),
        keep_all=bool(_dg_cfg.get("keep_all", False)),
    )
    disk_guard.start()

    # ── Early-game policy-entropy probe (§115 monitoring signal) ──────────────
    # Fixed 10-position fixture. One forward pass per log_interval — rides on
    # the existing _emit_training_events cadence so probe cost is amortised.
    early_game_probe: Optional[EarlyGameProbe]
    try:
        from hexo_rl.encoding import normalize_encoding_name as _normalize_encoding_name
        _probe_encoding = _normalize_encoding_name(config.get("encoding"))
        early_game_probe = EarlyGameProbe(device=device, encoding_name=_probe_encoding)
        log.info(
            "early_game_probe_init",
            n_positions=early_game_probe.n_positions,
            plies=early_game_probe.plies,
            encoding=_probe_encoding,
        )
    except Exception as _egp_err:
        log.warning("early_game_probe_unavailable", error=str(_egp_err))
        early_game_probe = None

    # ── Phase B' Class-2 value-head drift probe ────────────────────────────────
    # Loaded only when instrumentation is on; absent fixture → silent skip.
    instr_cfg_main = config.get("instrumentation", {}) or {}
    instrumentation_enabled = bool(instr_cfg_main.get("enabled", False))
    value_probe_interval = int(instr_cfg_main.get("value_probe_interval", 250))
    composition_interval = int(instr_cfg_main.get("composition_interval", 500))
    value_probe: Optional[ValueProbe] = None
    if instrumentation_enabled:
        fixture_path = Path(instr_cfg_main.get(
            "value_probe_fixture", "fixtures/value_probe_50.npz",
        ))
        try:
            value_probe = ValueProbe(fixture_path=fixture_path, device=device)
            log.info(
                "value_probe_loaded",
                path=str(fixture_path),
                n=value_probe.n_positions,
                n_decisive=value_probe.n_decisive,
                n_draw=value_probe.n_draw,
                interval=value_probe_interval,
            )
        except Exception as exc:
            log.warning(
                "value_probe_load_failed",
                path=str(fixture_path),
                error=str(exc),
            )
            value_probe = None

    # ── Axis-distribution baseline (§axis_dist) ───────────────────────────────
    import json as _json
    from pathlib import Path as _Path
    _axis_baseline: dict[str, float] = {}
    _axis_baseline_path = _Path("reports/baselines/corpus_axis_distribution.json")
    if _axis_baseline_path.exists():
        try:
            _axis_baseline = _json.loads(_axis_baseline_path.read_text())
            log.info("axis_distribution_baseline_loaded", path=str(_axis_baseline_path), baseline=_axis_baseline)
        except Exception as _ab_err:
            log.warning("axis_distribution_baseline_load_failed", error=str(_ab_err))

    # ── TensorBoard writer (axis-distribution metrics) ────────────────────────
    from hexo_rl.monitoring.metrics_writer import MetricsWriter as _MetricsWriter
    _tb_writer: Optional[_MetricsWriter] = None
    try:
        _tb_log_dir = str(_Path(getattr(args, "log_dir", "logs") or "logs") / "tb" / run_id)
        _tb_writer = _MetricsWriter(_tb_log_dir)
        log.info("tensorboard_writer_init", log_dir=_tb_log_dir)
    except Exception as _tb_err:
        log.warning("tensorboard_writer_unavailable", error=str(_tb_err))

    # ── Events JSONL sink ────────────────────────────────────────────────────
    # Registered unconditionally so `make dashboard` works whether or not the
    # in-process dashboard is enabled. Out-of-process dashboards
    # (scripts/serve_dashboard.py via EventsTailer) tail this file.
    _events_jsonl_path = Path(getattr(args, "log_dir", "logs") or "logs") / f"events_{run_id}.jsonl"
    register_jsonl_sink(_events_jsonl_path)
    log.info("events_jsonl_sink", path=str(_events_jsonl_path))

    # ── Dashboard renderers ───────────────────────────────────────────────────
    dashboards: list = []
    mon_cfg = config.get("monitoring", {})
    if mon_cfg.get("enabled", True) and not args.no_dashboard:
        if mon_cfg.get("terminal_dashboard", True):
            from hexo_rl.monitoring.terminal_dashboard import TerminalDashboard
            td = TerminalDashboard(config)
            td.start()
            register_renderer(td)
            dashboards.append(td)
        if mon_cfg.get("web_dashboard", True) and not getattr(args, "no_web_dashboard", False):
            from hexo_rl.monitoring.web_dashboard import WebDashboard
            wd = WebDashboard(config)
            wd.start()
            register_renderer(wd)
            dashboards.append(wd)

    return LoopSubsystems(
        gpu_monitor=gpu_monitor,
        disk_guard=disk_guard,
        early_game_probe=early_game_probe,
        value_probe=value_probe,
        value_probe_interval=value_probe_interval,
        composition_interval=composition_interval,
        instrumentation_enabled=instrumentation_enabled,
        axis_baseline=_axis_baseline,
        tb_writer=_tb_writer,
        dashboards=dashboards,
    )
