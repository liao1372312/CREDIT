"""Run frozen CREDIT v3 protocol with cached decisions from a remote LLM.

This evaluates a constrained LLM actor against the finite local stateful API.
Each of 336 decision situations is queried once and cached; all training,
certification and test episodes use those fixed decisions. No user dataset or
natural service release is represented by this experiment.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from .llm_policy import install_policy, SYSTEM, contexts
from .llm_remote_policy_v2 import answer_letter
from .v3.experiment import digest, write


def read(path):
    return json.loads(Path(path).read_text())


def policy_guard(policy):
    manifest=read(policy/'model_manifest.json')
    protocol=read(policy/'frozen_protocol.json')
    scores=read(policy/'decision_scores.json')
    responses=[json.loads(line) for line in (policy/'responses.jsonl').read_text().splitlines() if line]
    inventory=list(contexts())
    if len(inventory)!=len(scores) or len(scores)!=len(responses) or len(scores)!=336:
        raise ValueError('incomplete LLM decision inventory')
    for path,key in [('frozen_protocol.json','frozen_protocol_sha256'),
                     ('decision_scores.json','decision_scores_sha256'),
                     ('responses.jsonl','responses_sha256')]:
        if digest(policy/path)!=manifest[key]:
            raise ValueError('remote decision artifact changed: '+path)
    for name,h in protocol['source_sha256'].items():
        if digest(name)!=h:
            raise ValueError('remote decision source changed: '+name)
    if sha256(SYSTEM.encode()).hexdigest()!=protocol['system_sha256']:
        raise ValueError('prompt system changed')
    for i,((key,prompt),response,score) in enumerate(zip(inventory,responses,scores)):
        if response['index']!=i or response['key']!=list(key) or score['key']!=list(key):
            raise ValueError('LLM decision context mismatch')
        if response['prompt_sha256']!=sha256(prompt.encode()).hexdigest():
            raise ValueError('LLM prompt changed')
        if score['response_sha256']!=response['response_sha256']:
            raise ValueError('LLM response checksum mismatch')
        if answer_letter(response['reply'])!=response['letter']:
            raise ValueError('LLM parsed answer mismatch')
        expected=[float(letter==response['letter']) for letter in 'ABCD']
        if score['probabilities']!=expected:
            raise ValueError('cached action mismatch')
    usage={key:sum(int(r['usage'].get(key,0) or 0) for r in responses)
           for key in ('prompt_tokens','completion_tokens','total_tokens')}
    if manifest['usage']!=usage:
        raise ValueError('reported token usage mismatch')
    return manifest,protocol


def fixed_protocol(policy,dataset):
    manifest,remote=policy_guard(policy)
    from .v3_online_data import SEEDS,CYCLES
    from .v3_online_experiment import SPECS,CAP
    source_files=['credit/v3_remote_llm_experiment.py','credit/v3_online_experiment.py',
                  'credit/v3_online_fast.py','credit/llm_policy.py','credit/llm_remote_policy_v2.py','credit/stateful.py',
                  'credit/v3/local.py','credit/v3_online_data.py']
    return {'scope':'full bounded CREDIT v3 re-training/selection/chronological test with cached remote LLM actor and synthetic local API',
            'paper_experiments_complete':False,'source_files_sha256':{name:digest(name) for name in source_files},
            'dataset':str(dataset.resolve()),'dataset_manifest_sha256':digest(dataset/'manifest.json'),
            'remote_policy':str(policy.resolve()),'remote_scores_sha256':manifest['decision_scores_sha256'],
            'remote_protocol_sha256':manifest['frozen_protocol_sha256'],
            'remote_model':manifest['model'],'remote_decisions':manifest['llm_context_forward_evaluations'],
            'seeds':list(SEEDS),'cycles':CYCLES,'specifications':[list(item) for item in SPECS],'cycle_call_cap':CAP,
            'primary':'CREDIT-Covered vs CREDIT-Sparse',
            'secondary':['CREDIT-VOI vs CREDIT-Random','CREDIT-VOI vs CREDIT-Covered',
                         'CREDIT-Covered vs Direct-Covered','CREDIT-Covered vs CREDIT-NoRevalidation'],
            'optimizer':'FastOnlineModel shared fixed-feature implementation for all training blocks',
            'actor':'336 temperature-zero one-hot A/B/C/D choices, cached before source fitting; choices not re-queried per episode',
            'test_policy':'all methods and seeds freeze before each block; score before ingestion',
            'evaluation_limit':'finite simulator and seen public templates; no natural drift, general chat agent or real service calls',
            'credential_recorded':False}


def run(policy,dataset,output):
    if output.exists():
        raise FileExistsError('fresh remote LLM experiment directory required')
    fixed=fixed_protocol(policy,dataset)
    output.mkdir(parents=True)
    write(output/'remote_frozen_protocol.json',fixed)
    install_policy(policy/'decision_scores.json')
    from . import v3_online_experiment as experiment
    from .v3_online_fast import fit_ensemble
    experiment.fit_ensemble=fit_ensemble
    experiment.run(dataset,output/'chronological')
    write(output/'summary.json',{'experiment':str((output/'chronological').resolve()),
          'remote_decisions':fixed['remote_decisions'],
          'llm_prompt_tokens':read(policy/'model_manifest.json')['usage'].get('prompt_tokens'),
          'llm_completion_tokens':read(policy/'model_manifest.json')['usage'].get('completion_tokens'),
          'chronological_summary':read(output/'chronological/summary.json'),
          'claim_limit':'same finite local API; LLM actor choices cached; no natural service/long-form LLM agent claims'})
    write(output/'remote_artifact_manifest.json',{
        'remote_frozen_protocol.json':digest(output/'remote_frozen_protocol.json'),
        'summary.json':digest(output/'summary.json'),
        'chronological/artifact_manifest.json':digest(output/'chronological/artifact_manifest.json'),
        'remote_policy/decision_scores.json':digest(policy/'decision_scores.json')})
    print('Completed remote LLM actor chronological experiment',flush=True)


def audit(policy,dataset,output):
    expected=fixed_protocol(policy,dataset)
    if read(output/'remote_frozen_protocol.json')!=expected:
        raise ValueError('remote experiment protocol changed')
    for name,h in read(output/'remote_artifact_manifest.json').items():
        path=policy/'decision_scores.json' if name.startswith('remote_policy/') else output/name
        if digest(path)!=h:
            raise ValueError('remote experiment artifact changed: '+name)
    install_policy(policy/'decision_scores.json')
    from .v3_online_verify import audit as audit_chronology
    audit_chronology(output/'chronological',streaming=False)
    chronology_audit=output/'chronological_audit.json'
    proof=read(chronology_audit)
    if not proof['verified'] or proof['test_episodes']!=172032:
        raise ValueError('remote chronology audit incomplete')
    write(output/'remote_execution_audit.json',{
        'verified':True,'remote_protocol_sha256':digest(output/'remote_frozen_protocol.json'),
        'remote_decision_scores_sha256':digest(policy/'decision_scores.json'),
        'chronological_audit_sha256':digest(chronology_audit),
        'limits':'canonical simulator reconstruction with cached LLM actor; optimizer not retrained by audit; no natural provider drift'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('datasets/credit_v3_online_v3'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit',action='store_true')
    a=p.parse_args()
    if a.audit:audit(a.policy,a.dataset,a.output)
    else:run(a.policy,a.dataset,a.output)
