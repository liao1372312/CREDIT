"""Stream a physical chronology audit under the frozen remote LLM actor."""
import argparse,json,time
from pathlib import Path

from .llm_policy import install_policy
from .v3.experiment import digest,write
from .v3_remote_llm_experiment import fixed_protocol,policy_guard,read
from .v3_online_verify import audit as audit_chronology


def wait(path,streaming):
    while streaming and not path.exists():time.sleep(2)
    if not path.exists():raise FileNotFoundError(path)


def verify(policy,dataset,output,streaming=False):
    policy_guard(policy)
    expected=fixed_protocol(policy,dataset)
    wait(output/'remote_frozen_protocol.json',streaming)
    if read(output/'remote_frozen_protocol.json')!=expected:
        raise ValueError('remote actor experiment protocol changed')
    install_policy(policy/'decision_scores.json')
    audit_chronology(output/'chronological',streaming=streaming)
    wait(output/'remote_artifact_manifest.json',streaming)
    for name,h in read(output/'remote_artifact_manifest.json').items():
        path=policy/'decision_scores.json' if name.startswith('remote_policy/') else output/name
        if digest(path)!=h:raise ValueError('remote experiment artifact changed: '+name)
    saved=read(output/'chronological_audit.json')
    if not saved['verified'] or saved['test_episodes']!=172032:
        raise ValueError('chronology audit incomplete')
    result={'verified':True,'remote_protocol_sha256':digest(output/'remote_frozen_protocol.json'),
            'remote_decision_scores_sha256':digest(policy/'decision_scores.json'),
            'chronological_audit_sha256':digest(output/'chronological_audit.json'),
            'auditor_source_sha256':digest(__file__),
            'limits':'reexecuted local API with fixed cached LLM actions; optimizer and vendor internals not retrained/audited; no natural release drift'}
    write(output/'remote_execution_audit.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('datasets/credit_v3_online_v3'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--streaming',action='store_true')
    a=p.parse_args();verify(a.policy,a.dataset,a.output,a.streaming)
