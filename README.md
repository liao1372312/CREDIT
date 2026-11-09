# CREDIT

Research code and bundled data for *CREDIT: Retrieval-Aware Causal Memory Maintenance for Evolving Web Service Agents*.

## Repository contents

- `credit/v3/` — core CREDIT implementation: factorial exposure, outcome modeling, drift gating, bank selection, and switch certification.
- `credit/paper_7944_protocol.py` — registered paper settings and catalog sampling.
- `credit/` — experiment drivers and service adapters.
- `datasets/tau_original_credit_pool_v2/` — bundled Retail and Airline memory pools and split manifest.
- `configs/` — experiment configurations.
- `tests/` — code tests.

## Install and test

```sh
pip install -e ".[test]"
python -m pytest -q
```
