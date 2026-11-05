"""Reconstruct contract-disjoint support, balanced sampling and provenance."""

import argparse
import json
from pathlib import Path

from .emp_binding_v2_verify import verify_sources
from .emp_generalization_audit_v2 import contract_blocks, evaluate, load
from .user_data import digest


def verify(output):
    protocol = verify_sources(output)
    source = Path(protocol['source_directory'])
    verify_sources(source)
    for name, expected in protocol['input_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError('source observations changed')
    if protocol['all_data_weights_used'] or not protocol['heldout_origin_notes_and_exposure_rows_excluded']:
        raise ValueError('invalid fold provenance')
    rows, edges, memories = load(source)
    if protocol['blocks'] != contract_blocks(edges):
        raise ValueError('contract-block assignments differ')
    result = evaluate(rows, edges, memories)
    for name, actual in result.items():
        saved = json.loads((output / (name + '.json')).read_text())
        if saved != actual:
            raise ValueError('reconstruction differs: ' + name)
    report = {'sources_and_observations_unchanged': True,
        'workflow_and_public_contract_blocks_reconstructed': True,
        'physical_inputs_and_both_classes_reconstructed': True,
        'heldout_origin_memories_and_their_exposures_excluded': True,
        'unsupported_not_scored_as_rejection': True,
        'all_data_weights_used': False, 'weight_refitting_required': False,
        'reason': 'unseen contract alphabets force abstention for every parameter vector',
        'new_LLM_calls': 0, 'live_API_calls': 0, 'verifier_sha256': digest(Path(__file__))}
    (output / 'integrity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/emp_generalization_v2'))
    verify(parser.parse_args().results)
