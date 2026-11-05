"""Train-only grouped selection and honest support evaluation of a smaller model."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .emp_binding_mechanism import LocalMechanism
from .emp_binding_shared import score_predictions, tune_shared
from .user_data import digest, jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('results/emp_binding_pilot_v5'))
    parser.add_argument('--output', type=Path, default=Path('results/emp_binding_optimization_v1'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    old_protocol = json.loads((args.baseline / 'frozen_protocol.json').read_text())
    data = Path(old_protocol['data_directory'])
    edges = {e['uid']: e for e in jsonl(data / 'edges.jsonl')}
    trials = list(jsonl(args.baseline / 'trials.jsonl'))
    pool = json.loads((args.baseline / 'memories.json').read_text())
    args.output.mkdir(parents=True)
    protocol = {'source_trials_sha256': digest(args.baseline / 'trials.jsonl'),
        'source_memories_sha256': digest(args.baseline / 'memories.json'), 'source_edges_sha256': digest(data / 'edges.jsonl'),
        'baseline_directory': str(args.baseline.resolve()), 'data_directory': str(data.resolve()),
        'tuning': 'leave historical workflow out; exclude that workflows origin notes and all trials exposing them',
        'grid': [[.1, 0], [.1, .2], [1, 0], [1, .2], [10, 0], [10, .2]],
        'selection_metric': 'mean heldout workflow final NLL, on all eligible heldout exposures including unsupported',
        'training_estimand': 'uniform workflow then uniform distinct visible exposure',
        'heldout_development_data_used_for_tuning': False, 'new_LLM_calls': 0,
        'scope': 'optimization on previously examined development portfolio, not blind test or full CREDIT',
        'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in ['emp_binding_shared.py', 'emp_binding_optimization.py', 'emp_binding_mechanism.py', 'emp_binding.py', 'emp_binding_agent.py', 'user_data.py']}}
    (args.output / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2))
    (args.output / 'source').mkdir()
    for name in protocol['source_sha256']:
        (args.output / 'source' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    training = [r for r in trials if r['stage'] == 'intervention_train']
    model, info, tuning = tune_shared(training, edges, pool)
    weights = json.loads((args.baseline / 'local_mechanism.json').read_text())['weights']
    old = LocalMechanism(pool, weights)
    (args.output / 'model.json').write_text(json.dumps({'weights': model.weights.tolist(), 'mean': model.mean.tolist(),
        'scale': model.scale.tolist(), 'training_profiles': model.training_profiles, 'fit': info}, indent=2))
    (args.output / 'grouped_cv.json').write_text(json.dumps(tuning, indent=2))
    rows, metrics = [], []
    for stage in ['validation', 'development_evaluation']:
        subset = [r for r in trials if r['stage'] == stage]
        for variant in ['original', 'shared_unrestricted_diagnostic', 'shared_with_support_gate']:
            def predict(row):
                if variant == 'original':
                    return old.predict(edges[row['edge_uid']], row['exposure'])
                return model.predict(edges[row['edge_uid']], row['exposure'], abstain=variant == 'shared_with_support_gate')
            metrics.append({'partition': stage, 'variant': variant, **score_predictions(subset, predict)})
            for row in subset:
                rows.append({'partition': stage, 'variant': variant, 'edge_uid': row['edge_uid'],
                    'bank_uid': row['bank_uid'], 'exposure': row['exposure'], **predict(row)})
    (args.output / 'predictions.json').write_text(json.dumps(rows, indent=2))
    summary = {'fit': info, 'original_parameters': len(weights) * len(weights[0]), 'comparison': metrics,
               'new_LLM_calls': 0, 'independent_test_claim': False, 'reserved_workflows_executed': 0}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
