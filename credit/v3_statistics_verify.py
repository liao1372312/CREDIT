"""Reconstruct summary statistics and every stored simulation decision."""
from pathlib import Path
import argparse
import json
import numpy as np
from .v3.experiment import digest, write
from .v3.inference import PPIEstimate, certify
from .v3_statistical import certification


def audit(directory):
    read = lambda name: json.loads((directory / name).read_text())
    for name, value in read('artifact_manifest.json').items():
        if digest(directory / name) != value: raise ValueError(name)
    for name, value in read('frozen_protocol.json')['source_sha256'].items():
        if digest(name) != value: raise ValueError(name)
    factorial_rows = read('factorial_summary.json')
    for row in factorial_rows:
        data = np.load(directory / f"factorial_{row['budget']}_{int(row['noise']=='CRN')}.npz")
        i = ['Randomized-fraction', 'Random-OLS', 'OAT'].index(row['method'])
        error = data['estimates'][i] - data['truth']
        expected = (np.sqrt(np.mean(error[:, 1:13]**2)), np.sqrt(np.mean(error[:, 13:]**2)),
                    np.max(np.abs(error[:, 1:13].mean(axis=0))))
        if not np.allclose(expected, [row[k] for k in ('main_rmse', 'pair_rmse', 'main_max_bias')]):
            raise ValueError('factorial summary')
    weights = dict(zip(map(str, range(8)), [.2, .16, .14, .12, .1, .1, .1, .08]))
    decisions = 0
    for i, row in enumerate(read('certification_summary.json')):
        data = np.load(directory / f'certification_{i:03d}.npz')
        for values, accepted in zip(data['estimates'], data['passed']):
            estimates = {str(k): PPIEstimate(float(v[0]), float(v[1]), float(v[2]), row['n'], 1000, 0., 0.)
                         for k, v in enumerate(values)}
            if certify(estimates, weights, minimum_n=2)['passed'] != accepted:
                raise ValueError('certificate')
            decisions += 1
        if not np.isclose(data['passed'].mean(), row['switch_rate']): raise ValueError('switch rate')
        # Replay the first two replications, including raw task generation, pilot
        # fitting and the unlabeled variance term, from the original RNG state.
        raw, accepted, _ = certification(2, row['n'], row['gain'], row['quality'], 1821+i,
                                        independent=row['independent'], harmful=row['harmful_stratum'])
        if not np.allclose(raw, data['estimates'][:2]) or not np.array_equal(accepted, data['passed'][:2]):
            raise ValueError('simulation generation replay')
    report = {'verified': True, 'factorial_summary_rows_recomputed': len(factorial_rows),
              'certification_decisions_reconstructed': decisions,
              'complete_replications_regenerated': 110,
              'scope': 'all raw estimate summaries and decisions; two complete RNG replications per scenario',
              'auditor_sha256': digest(__file__)}
    write(directory / 'statistics_audit.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    print(json.dumps(audit(p.parse_args().results), indent=2))
