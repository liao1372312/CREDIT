# Paper 7944 implementation alignment

The paper-registered implementation is separate from earlier local and CV20 experiments.

Implemented in this repository:

- `credit/paper_7944_protocol.py`: published workload counts, six-stage 1,536-pair search schedule, fixed certification catalogs, fresh sampling with replacement, and lifetime attempt indexing.
- `credit/paper_7944_model.py`: five source-group bootstrap role models, two 128-unit ReLU layers per masked head, AdamW training, validation-based early stopping, and isotonic calibration of the ensemble mean.
- `credit/v3/selection.py`: one frozen finalist, uniform registered target weights, capacity check, and pre-certification freeze in `switch_registered`.
- `credit/v3/inference.py`: paired PPI inference and the paper's per-attempt error allocation in `certify_registered`.
- `credit/v3/drift.py`: registered probe and source-support thresholds.

The older `python -m credit.v3` entry point is a local synthetic experiment, not the paper's three-workload run. The public ToolSandbox CV20 driver likewise uses a different population. Neither should be used to regenerate the paper's tables.

Still required for full experimental reproduction: the exact 2,000-episode ToolSandbox, Retail, and Airline execution harnesses; frozen actor/simulator/embedding calls and benchmark revisions; complete candidate/incumbent and certification records; request-level cost ledgers; and a verifier mapping each paper table cell to those records. This repository does not contain those measurements. No reported success rate or cost was synthesized from the protocol code.
