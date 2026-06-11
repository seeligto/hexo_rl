<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

### §66 — C1 bug impact assessment (amendment)

The C1 bug affected only the reported loss scalar, not training dynamics. CE and KL against fixed targets share gradients, so the checkpoint was structurally valid. The Phase 4.0 runs are being restarted — but the trigger is the LR scheduler bug (§67), not C1. C1 is fixed and loss reporting is now correct for all future runs.

---

