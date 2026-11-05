"""Summarize an audited remote-LLM-actor CREDIT v3 chronology."""
import argparse,csv
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .paired_statistics import summarize
from .v3.experiment import digest,write
from .v3_online_report import table
from .v3_remote_llm_experiment import read,policy_guard


def report(policy,experiment,reference,analysis):
    if analysis.exists():raise FileExistsError('fresh analysis directory required')
    guard=read(experiment/'remote_execution_audit.json')
    if not guard['verified'] or guard['remote_protocol_sha256']!=digest(experiment/'remote_frozen_protocol.json'):
        raise ValueError('remote execution audit missing or changed')
    if guard['remote_decision_scores_sha256']!=digest(policy/'decision_scores.json'):
        raise ValueError('remote decisions changed')
    if guard['chronological_audit_sha256']!=digest(experiment/'chronological_audit.json'):
        raise ValueError('chronological execution audit changed')
    chronology_proof=read(experiment/'chronological_audit.json')
    source=experiment/'chronological'
    if not chronology_proof['verified'] or chronology_proof['benchmark_manifest_sha256']!=digest(source/'artifact_manifest.json'):
        raise ValueError('chronological audit is not bound to this experiment')
    for name,h in read(experiment/'remote_artifact_manifest.json').items():
        path=policy/'decision_scores.json' if name.startswith('remote_policy/') else experiment/name
        if digest(path)!=h:raise ValueError('remote experiment artifact changed: '+name)
    manifest,remote_protocol=policy_guard(policy)
    prompt_lengths=[len(row['prompt'].encode()) for row in read(policy/'decision_scores.json')]
    response_rows=[json.loads(line) for line in (policy/'responses.jsonl').read_text().splitlines() if line]
    choices={role:dict(Counter(r['letter'] for r in response_rows if r['key'][0]==role))
             for role in sorted({r['key'][0] for r in response_rows})}
    for name,h in read(source/'artifact_manifest.json').items():
        if digest(source/name)!=h:raise ValueError('chronological source changed: '+name)
    rows=read(source/'raw_test_results.json')
    stored=read(source/'statistics.json')
    own=summarize(rows,'CREDIT-Covered',repeats=4000,random_seed=7013)
    if stored!={'main':own[0],'paired':own[1]}:raise ValueError('stored statistics differ')
    if len(rows)!=172032:raise ValueError('incomplete remote actor test coverage')
    methods=sorted({r['method'] for r in rows});seeds=sorted({r['seed'] for r in rows})
    cycle=[]
    for method in methods:
        for block in range(3):
            values=[np.mean([r['success'] for r in rows if r['method']==method and r['seed']==seed and r['cycle']==block]) for seed in seeds]
            cycle.append({'method':method,'cycle':block,'vsr':float(np.mean(values)),'seed_std':float(np.std(values,ddof=1))})
    expenses=sum([read(source/f'costs_cycle_{i}.json') for i in range(3)],[])
    costs=[]
    for method in methods:
        values=[sum(r['extra_api_calls'] for r in expenses if r['method']==method and r['seed']==seed) for seed in seeds]
        costs.append({'method':method,'mean_three_cycle_extra_calls':float(np.mean(values)),
                      'seed_std':float(np.std(values,ddof=1))})
    reference_manifest=reference/'artifact_manifest.json'
    reference_proof=read(reference.with_name(reference.name+'_audit.json'))
    if not reference_proof['verified'] or reference_proof['benchmark_manifest_sha256']!=digest(reference_manifest):
        raise ValueError('original actor audit is not bound to reference')
    if digest(reference/'statistics.json')!=read(reference_manifest)['statistics.json']:
        raise ValueError('original actor statistics changed')
    original=read(reference/'statistics.json')
    old={r['method']:r['vsr'] for r in original['main']}
    analysis.mkdir()
    write(analysis/'tables.json',{'remote_main':stored['main'],'remote_paired':stored['paired'],
                                 'remote_secondary':read(source/'secondary_statistics.json'),
                                 'cycle':cycle,'cost':costs,'actor_choices_by_role':choices,
                                 'original_actor_main':original['main']})
    for name,values in [('cycle',cycle),('cost',costs)]:
        with (analysis/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for method in methods:
        points=[r for r in cycle if r['method']==method]
        ax[0].plot([r['cycle'] for r in points],[100*r['vsr'] for r in points],marker='o',label=method)
    ax[0].set(xlabel='Chronological block',ylabel='Verified success (%)',xticks=[0,1,2])
    ax[0].legend(fontsize=7)
    for item in stored['main']:
        charge=next(r['mean_three_cycle_extra_calls'] for r in costs if r['method']==item['method'])
        ax[1].scatter(charge,100*item['vsr'],label=item['method'])
    ax[1].set(xlabel='Mean extra local API calls / seed',ylabel='Verified success (%)')
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(analysis/('remote_llm_v3.'+ext),dpi=180)
    plt.close(fig)
    primary=next(r for r in stored['paired'] if r['baseline']=='CREDIT-Sparse')
    rows_table=[]
    for item in stored['main']:
        cost=next(c['mean_three_cycle_extra_calls'] for c in costs if c['method']==item['method'])
        rows_table.append([item['method'],f"{100*item['vsr']:.3f}%",f"[{100*item['ci_low']:.3f}, {100*item['ci_high']:.3f}]",
                           f'{cost:,.0f}',f"{100*old[item['method']]:.3f}%"])
    markdown='# CREDIT v3 远程 LLM 决策实验\n\n'
    markdown+='## 范围与执行方式\n\n'
    markdown+=f"远程模型在 {manifest['llm_context_forward_evaluations']} 个固定决策情境中均返回了有效选项。已保存成功响应的供应商用量：提示 token {manifest['usage'].get('prompt_tokens',0):,}，生成 token {manifest['usage'].get('completion_tokens',0):,}。请求温度 0，模型从 A/B/C/D 四种请求模式选一个；缓存后各训练、认证和测试任务共同使用同一冻结 actor。服务状态、版本变化和验收由仓库的有限本地 API 执行。8 个种子、3 个连续块、7 种设置，共 172,032 次任务测试。\n\n"
    markdown+='实验从头重新收集 source 历史、训练模型、做目标干预、选择 bank 并逐块认证，不是只给旧 bank 换模型评分。所有方法使用同一组 LLM 决策、任务和调用上限；评分后才入库。\n\n'
    markdown+='固定提示的动作分布如下。缓存覆盖原有限状态 actor 的全部 336 个上下文，包括当前九记忆池可能用不到的组合；成功响应的 token 用量均已计入。\n\n'
    markdown+=table(['角色','A 标准','B 更新契约','C 修复','D 无效'],[
        [role]+[choices[role].get(letter,0) for letter in 'ABCD'] for role in sorted(choices)])+'\n\n'
    markdown+='## 成功率和费用\n\n'
    markdown+=table(['方法','LLM actor VSR','95% 区间 (%)','三周期额外本地调用','旧模拟 actor VSR'],rows_table)+'\n\n'
    markdown+=f"预声明的 Covered−Sparse：{100*primary['delta_vsr']:+.3f} 个百分点，bootstrap 95% 区间 [{100*primary['ci_low']:.3f}, {100*primary['ci_high']:.3f}]，8 种子配对符号翻转 p={primary['seed_sign_flip_p']:.5f}。该比较仍同时改变覆盖量、均衡设计和最低支持数。\n\n"
    markdown+=table(['次级比较','差值 (百分点)','Holm p'],[[r['contrast'],f"{100*r['delta_vsr']:+.3f}",f"{r['secondary_family_holm_p']:.5f}"] for r in read(source/'secondary_statistics.json')])+'\n\n'
    markdown+='旧模拟 actor 一列仅供定位：两次从头运行得到的模型、evidence 和部署 bank 也可能不同，不能把列差异单独解释为 LLM 的因果效果。`cycle.csv` 与 `cost.csv` 给出按块和按种子费用；额外本地调用与已记录的远程 token 分开核算。\n\n'
    markdown+='## 复核与限制\n\n'
    proof=read(experiment/'chronological_audit.json')
    markdown+=f"实际执行复核重建 {proof['physical_episodes']:,} 个本地 episode、{proof['physical_calls_with_fixtures']:,} 次本地调用及全部 {proof['test_episodes']:,} 次测试，检查模型验证/校准、完整策略和时序。远程决策的原始回复、解析结果、提示哈希和 token 用量另存，并绑定 manifest；审计不重新训练优化器，也不对供应商的内部生成作独立证明。\n\n"
    markdown+=f"本实验是固定有限情境上的真实 LLM 决策加合成 API 任务。{manifest['llm_context_forward_evaluations']} 个成功决策被缓存，没有在每个任务中即时生成，也不是长对话、真实 WebAPI 版本发布或一般语义泛化测试。用户提示长度为 {min(prompt_lengths)}–{max(prompt_lengths)} UTF-8 字节，共 {len(set(prompt_lengths))} 种；远程请求没有实施论文提出的等长预留槽提示干预，后续模拟器帧的槽位元数据不能替代该干预的证据。收集器可能重试，物理 HTTP 尝试次数及失败尝试的 token 用量未被记录，因此上述成功响应和 {manifest['usage'].get('total_tokens',0):,} token 仅是已记录的用量，不能视为总费用；另有一次保留在 v1 目录的开发请求。保存了回复文本和响应摘要，但没有完整 HTTP 响应体或供应商返回的模型身份，不能独立认证供应商内部生成。温度 0 的 one-hot 动作也与旧模拟 actor 不同，不能把两轮作为单独的模型优劣对照。论文章节所列自然漂移、未见 DAG、完整 ToolSandbox-M 等仍需单独评测。\n"
    report_path=Path('CREDIT_V3_REMOTE_LLM_RESULTS.md')
    report_path.write_text(markdown)
    write(analysis/'provenance.json',{'generator_sha256':digest(__file__),'report_sha256':digest(report_path),
                                    'remote_audit_sha256':digest(experiment/'remote_execution_audit.json'),
                                    'actor_manifest_sha256':digest(policy/'model_manifest.json'),
                                    'reference_audit_sha256':digest(reference.with_name(reference.name+'_audit.json'))})
    write(analysis/'analysis_manifest.json',{str(p.relative_to(analysis)):digest(p) for p in analysis.rglob('*') if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('results/credit_v3_online_completed_v1'))
    p.add_argument('--analysis',type=Path,required=True)
    a=p.parse_args();report(a.policy,a.experiment,a.reference,a.analysis)
