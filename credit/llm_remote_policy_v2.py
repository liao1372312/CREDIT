"""Cache remote LLM decisions for the repository's finite stateful actor.

The remote endpoint only chooses A/B/C/D. The local executable service, not
the model or response text, determines task success and state transitions.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .llm_policy import SYSTEM, contexts
from .stateful import ACTIONS


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    fields = {}
    for source in path.read_text().splitlines():
        line = source.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            raise ValueError('invalid .env assignment')
        key, value = line.split('=', 1)
        fields[key.strip()] = value.strip().strip('"').strip("'")
    for key in ('CREDIT_LLM_BASE_URL', 'CREDIT_LLM_MODEL'):
        if not fields.get(key):
            raise ValueError(key+' is required')
    uri = urlparse(fields['CREDIT_LLM_BASE_URL'])
    if uri.scheme != 'https' and not (uri.scheme == 'http' and uri.hostname in ('localhost','127.0.0.1','::1')):
        raise ValueError('HTTPS or loopback HTTP endpoint required')
    if uri.username or uri.password or uri.query or uri.fragment:
        raise ValueError('base URL must not contain embedded credentials or query parameters')
    timeout = int(fields.get('CREDIT_LLM_TIMEOUT_SECONDS') or '120')
    max_tokens = int(fields.get('CREDIT_LLM_MAX_OUTPUT_TOKENS') or '256')
    if not 1 <= timeout <= 600 or not 1 <= max_tokens <= 1024:
        raise ValueError('timeout or output token cap outside supported range')
    base = fields['CREDIT_LLM_BASE_URL'].rstrip('/')
    url = base if base.endswith('/chat/completions') else base+'/chat/completions'
    return {'url': url, 'api_key': fields.get('CREDIT_LLM_API_KEY',''),
            'model': fields['CREDIT_LLM_MODEL'], 'timeout': timeout, 'max_tokens': max_tokens,
            'base_url_sha256': sha256(base.encode()).hexdigest()}


def answer_letter(content):
    if isinstance(content, list):
        content = ''.join(str(x.get('text','')) for x in content if isinstance(x,dict))
    if not isinstance(content,str):
        raise ValueError('missing text content in chat response')
    # The constrained prompt requests one letter. Accept a short plain answer
    # or a final standalone answer line; do not infer an action from free prose.
    text = content.strip()
    direct = re.fullmatch(r'(?is)(?:["\'`*]{0,2})\s*([ABCD])\s*(?:[.):]|["\'`*]{1,2})?\s*(?:(?:standard|patched|repair|invalid)[.!]?)?',text)
    if direct:
        return direct.group(1).upper()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = re.fullmatch(r'(?i)(?:answer\s*[:：]\s*)?([ABCD])\s*[.!]?',lines[-1])
        if last:
            return last.group(1).upper()
    raise ValueError('LLM answer is not an unambiguous A/B/C/D choice')


def request_choice(config, prompt):
    body = json.dumps({'model': config['model'],
        'messages': [{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],
        'temperature':0,'max_tokens':config['max_tokens']},separators=(',',':')).encode()
    headers={'Content-Type':'application/json'}
    if config['api_key']:
        headers['Authorization']='Bearer '+config['api_key']
    request=Request(config['url'],body,headers,method='POST')
    last_error=None
    for attempt in range(3):
        try:
            with urlopen(request,timeout=config['timeout']) as response:
                raw=response.read()
            payload=json.loads(raw)
            first=payload['choices'][0]
            reply=first.get('message',{}).get('content')
            letter=answer_letter(reply)
            return {'letter':letter,'reply':reply if isinstance(reply,str) else str(reply),
                    'response_sha256':sha256(raw).hexdigest(),
                    'response_id':payload.get('id'),
                    'usage':payload.get('usage',{}),
                    'finish_reason':first.get('finish_reason')}
        except HTTPError as error:
            last_error=RuntimeError('LLM endpoint HTTP status '+str(error.code))
            if error.code not in (408,429,500,502,503,504):
                break
        except (URLError,TimeoutError) as error:
            last_error=RuntimeError('LLM endpoint connection failed: '+type(error).__name__)
        if attempt < 2:
            sleep(2**attempt)
    raise last_error


def run(env_path,output,limit=None):
    config=load_config(env_path)
    entries=list(contexts())
    if len(entries)!=336 or any(len(k)!=4 for k,_ in entries):
        raise ValueError('decision context inventory changed')
    wanted=len(entries) if limit is None else min(limit,len(entries))
    if wanted<1:
        raise ValueError('limit must be positive')
    output.mkdir(parents=True,exist_ok=True)
    source_files=['credit/llm_remote_policy_v2.py','credit/llm_policy.py','credit/stateful.py']
    protocol={'scope':'finite cached LLM actor decisions; not an end-to-end tool-using LLM or natural service drift',
        'inventory':len(entries),'model':config['model'],'base_url_sha256':config['base_url_sha256'],
        'credential_present':bool(config['api_key']),'temperature':0,'max_output_tokens':config['max_tokens'],
        'system_sha256':sha256(SYSTEM.encode()).hexdigest(),
        'source_sha256':{name:digest(name) for name in source_files},
        'prompt_sha256':[sha256(prompt.encode()).hexdigest() for _,prompt in entries],
        'actions':list(ACTIONS),'result_definition':'A/B/C/D cached one-hot decisions; local executable API supplies outcome labels',
        'error_policy':'failed or ambiguous response stops collection; no synthetic fallback'}
    protocol_path=output/'frozen_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:
            raise ValueError('remote LLM protocol changed; use a new output directory')
    else:
        protocol_path.write_text(json.dumps(protocol,indent=2))
    data_path=output/'responses.jsonl'
    rows=[]
    if data_path.exists():
        rows=[json.loads(line) for line in data_path.read_text().splitlines() if line]
    if len(rows)>wanted:
        raise ValueError('saved responses exceed this invocation limit')
    for i,row in enumerate(rows):
        if row['index']!=i or row['key']!=list(entries[i][0]) or row['prompt_sha256']!=protocol['prompt_sha256'][i]:
            raise ValueError('saved response order or prompt digest differs')
    with data_path.open('a') as stream:
        for i in range(len(rows),wanted):
            key,prompt=entries[i]
            response=request_choice(config,prompt)
            row={'index':i,'key':list(key),'prompt_sha256':protocol['prompt_sha256'][i],
                 **response}
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
            rows.append(row)
            if (i+1)%20==0 or i+1==wanted:
                print(json.dumps({'remote_decisions_collected':i+1,'planned':len(entries)}),flush=True)
    if len(rows)==len(entries):
        scores=[{'key':row['key'],'probabilities':[float(letter==row['letter']) for letter in 'ABCD'],
                 'prompt':prompt,'rendered_prompt_sha256':row['prompt_sha256'],
                 'response_sha256':row['response_sha256']}
                for row,(_,prompt) in zip(rows,entries)]
        scores_path=output/'decision_scores.json'
        scores_path.write_text(json.dumps(scores,ensure_ascii=False,indent=2))
        usage={k:sum(int(r['usage'].get(k,0) or 0) for r in rows) for k in ('prompt_tokens','completion_tokens','total_tokens')}
        manifest={'model':config['model'],'model_type':'remote OpenAI-compatible chat completions',
                  'policy':'temperature-zero one-hot A/B/C/D,336 finite contexts',
                  'llm_context_forward_evaluations':len(rows),'usage':usage,
                  'decision_scores_sha256':digest(scores_path),'responses_sha256':digest(data_path),
                  'frozen_protocol_sha256':digest(protocol_path),'adapter_source_sha256':digest('credit/llm_policy.py'),
                  'limits':['finite cached decisions','synthetic local API','no fresh per-task LLM calls','not natural provider drift']}
        (output/'model_manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps({'completed':True,'requests':len(rows),'usage':usage}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env',type=Path,default=Path('.env'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int)
    a=p.parse_args();run(a.env,a.output,a.limit)
