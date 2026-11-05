"""Remote cache safety: pinned protocol resumes without repeating a paid call."""
import json
from unittest.mock import patch

from credit.llm_remote_policy_v2 import run


class Reply:
    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False

    def read(self):
        return json.dumps({'id':'fixture','choices':[{'message':{'content':'A'},'finish_reason':'stop'}],
                           'usage':{'prompt_tokens':10,'completion_tokens':1,'total_tokens':11}}).encode()


def test_remote_cache_resumes_and_does_not_store_credential(tmp_path):
    secret='fixture-secret-that-must-not-be-saved'
    env=tmp_path/'.env'
    env.write_text('CREDIT_LLM_BASE_URL=https://example.invalid/v1\n'
                   f'CREDIT_LLM_API_KEY={secret}\nCREDIT_LLM_MODEL=fixture\n')
    output=tmp_path/'cache'
    calls=[]

    def fake(request, timeout):
        body=json.loads(request.data)
        assert body['temperature']==0
        assert body['messages'][0]['role']=='system'
        assert request.get_header('Authorization')=='Bearer '+secret
        calls.append(body)
        return Reply()

    with patch('credit.llm_remote_policy_v2.urlopen',side_effect=fake):
        run(env,output,limit=1)
        run(env,output,limit=1)
    assert len(calls)==1
    assert len((output/'responses.jsonl').read_text().splitlines())==1
    assert secret not in (output/'frozen_protocol.json').read_text()
    assert secret not in (output/'responses.jsonl').read_text()
