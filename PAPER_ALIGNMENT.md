# Paper 7944 implementation alignment

The paper-registered implementation is separate from earlier local and CV20 experiments.

Implemented in this repository:

- `credit/paper_7944_protocol.py`: published workload counts, six-stage 1,536-pair search schedule, fixed certification catalogs, fresh sampling with replacement, and lifetime attempt indexing.
- `credit/paper_7944_model.py`: five source-group bootstrap role models, two 128-unit ReLU layers per masked head, AdamW training, validation-based early stopping, and isotonic calibration of the ensemble mean.
- `credit/v3/selection.py`: one frozen finalist, uniform registered target weights, capacity check, and pre-certification freeze in `switch_registered`.
- `credit/v3/inference.py`: paired PPI inference and the paper's per-attempt error allocation in `certify_registered`.
- `credit/v3/drift.py`: registered probe and source-support thresholds.
- `credit/paper_7944_runtime.py`: frozen-embedding cosine retrieval with role-scope filtering, exact `o200k_base` token checks, physical request IDs, and durable pre-certification snapshots.
- `credit/paper_7944_runner.py`: one-workload sequencing of model fit, 32-bank screening, 1,536 verified development pairs, fresh certification, and 2,000 final episodes, provided that real benchmark adapters and registered data are supplied.
- `credit/paper_7944_analysis.py`: the 10,000-resample paired source-group bootstrap for final comparisons.
- `credit/paper_7944_preflight.py` and `requirements-paper.txt`: pinned environment and task-manifest checks.

The registered switch no longer draws an extra certification pilot. It requires residual weights fixed on development data before the 200 new pairs per guard. The paper model requires an explicit frozen feature encoder; it cannot silently use the earlier signed-hash local feature encoding.

The older `python -m credit.v3` entry point is a local synthetic experiment, not the paper's three-workload run. The public ToolSandbox CV20 driver likewise uses a different population. Neither should be used to regenerate the paper's tables.

Still required for full experimental reproduction: the exact task IDs and source-family mapping; the paper's feature encoder and development-ranking implementation; real ToolSandbox, Retail, and Airline execution adapters using the frozen actor, simulator and embedding calls; complete candidate/incumbent and certification records; and a verifier mapping each paper table cell to those records. The registered runner and request ledger define and check these inputs, but this repository does not contain those measurements. No reported success rate or cost was synthesized from the protocol code.

The bundled `datasets/tau_original_credit_pool_v2/split_manifest.json` is historical input, not the paper split: its Retail/Airline test sets contain 30/9 task IDs, whereas the paper registers 70/25. The exact paper task IDs and source-group assignments are needed before a paper-aligned run can be executed.

The anonymous code snapshot linked in the PDF was checked on 2026-11-09. Its dataset directory contains only the same historical `tau_original_credit_pool_v2` pool; it does not supply the missing registered task/source-group manifest or raw paper execution records.

The local `vendor/tau-bench` checkout matches the registered commit. The pre-existing `vendor/ToolSandbox` checkout does not; a separate local archive of upstream commit `165848b9a78cead7ca7fe7c89c688b58e6501219` is available under `vendor/ToolSandbox-paper`. The archive is not bundled in the public repository.

