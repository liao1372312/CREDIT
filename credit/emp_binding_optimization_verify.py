"""Rebuild the saved calibration comparison and historical-only model fit."""

import argparse
import json
from pathlib import Path

from .emp_binding_mechanism import LocalMechanism
from .emp_binding_shared import score_predictions
from .emp_binding_v2_verify import load_model, read, verify_cv, verify_sources
from .emp_binding_verify import require
from .user_data import digest, jsonl


def verify(directory):
    protocol = verify_sources(directory)
    baseline, data = Path(protocol['baseline_directory']), Path(protocol['data_directory'])
    for path, key in [(baseline / 'trials.jsonl', 'source_trials_sha256'), (baseline / 'memories.json', 'source_memories_sha256'), (data / 'edges.jsonl', 'source_edges_sha256')]:
        require(digest(path) == protocol[key], 'source data changed')
    edges = {e['uid']: e for e in jsonl(data / 'edges.jsonl')}
    trials, pool = list(jsonl(baseline / 'trials.jsonl')), read(baseline, 'memories.json')
    training = [r for r in trials if r['stage'] == 'intervention_train']
    verify_cv(directory, training, edges, pool)
    model = load_model(directory, pool)
    weights = read(baseline, 'local_mechanism.json')['weights']; old = LocalMechanism(pool, weights)
    predictions, metrics = [], []
    for stage in ('validation', 'development_evaluation'):
        subset = [r for r in trials if r['stage'] == stage]
        for variant in ('original', 'shared_unrestricted_diagnostic', 'shared_with_support_gate'):
            def predict(row):
                return old.predict(edges[row['edge_uid']], row['exposure']) if variant == 'original' else model.predict(edges[row['edge_uid']], row['exposure'], abstain=variant == 'shared_with_support_gate')
            metrics.append({'partition': stage, 'variant': variant, **score_predictions(subset, predict)})
            predictions.extend({'partition': stage, 'variant': variant, 'edge_uid': r['edge_uid'], 'bank_uid': r['bank_uid'], 'exposure': r['exposure'], **predict(r)} for r in subset)
    require(predictions == read(directory, 'predictions.json'), 'saved predictions differ')
    summary = read(directory, 'summary.json')
    require(metrics == summary['comparison'], 'comparison metrics differ')
    require(summary['new_LLM_calls'] == 0 and not summary['independent_test_claim'], 'scope differs')
    report = {'sources_unchanged': True, 'train_only_CV_and_weights_reconstructed': True,
        'predictions_and_metrics_reconstructed': len(predictions), 'new_LLM_calls': 0,
        'reserved_workflows_executed': 0, 'verifier_sha256': digest(Path(__file__))}
    (directory / 'integrity.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/emp_binding_optimization_v3'))
    verify(parser.parse_args().results)
