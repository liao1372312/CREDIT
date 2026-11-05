"""Audited raw tables and paired evidence for current-version admission."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .paired_statistics import summarize
from .v3.experiment import digest,write
from .v3_optimization_verify import read
from .v3_revalidation_experiment import SPECS

PRIMARY = ('CREDIT-Revalidated-K4','CREDIT-Probes-K4')
SECONDARY = (
    ('CREDIT-Revalidated-K4','CREDIT-Frozen-K4'),
    ('CREDIT-Revalidated-K4','Direct-Revalidated-K4'),
    ('CREDIT-Revalidated-K4','CREDIT-Revalidated-NoPairs-K4'),
    ('CREDIT-Revalidated-K4','CREDIT-NewCandidate-K4'),
    ('CREDIT-NewCandidate-K4','CREDIT-Probes-K4'),
    ('CREDIT-Revalidated-K6','Direct-Revalidated-K6'),
)


def generate(benchmark,output):
    if output.exists():raise FileExistsError('use a fresh analysis directory')
    audit_path = benchmark.with_name(benchmark.name+'_audit.json');audit = read(audit_path)
    if not audit['verified'] or digest(benchmark/'artifact_manifest.json')!=audit['benchmark_manifest_sha256']:
        raise ValueError('complete matching execution audit required')
    for name,value in read(benchmark/'artifact_manifest.json').items():
        if digest(benchmark/name)!=value:raise ValueError('changed experimental artifact')
    output.mkdir()
    protocol = read(benchmark/'frozen_protocol.json');seeds = protocol['seeds']
    rows,statistics = read(benchmark/'raw_test_results.json'),read(benchmark/'statistics.json')
    methods = [s[0] for s in SPECS]+['NoMem']
    summaries = {r['method']:r for r in statistics['main']}
    costs = {seed:read(benchmark/f'seed_{seed}/method_costs.json') for seed in seeds}
    table,seed_table,cost_table,admissions,probes,banks = [],[],[],[],[],[]
    for seed in seeds:
        sd = benchmark/f'seed_{seed}'
        for decision in read(sd/'adaptation/memory_decisions.json'):
            admissions.append({'seed':seed}|decision)
        for decision in read(sd/'adaptation/probe_decisions.json'):
            before,after = decision['before'],decision['after']
            probes.append({'seed':seed,'version':after['version'],'role':after['role'],
                'before':before['level'],'after':after['level'],'old_fixtures':before['fixtures'],
                'fresh_fixtures':after['fixtures'],'changed':after['changed'],'lower':after['lower'],'upper':after['upper'],
                'alpha':decision['alpha']})
        for name,bank in read(sd/'frozen_banks.json').items():banks.append({'seed':seed,'method':name,'bank':bank})
        hc,ac = read(sd/'compact_history/costs.json'),read(sd/'adaptation/costs.json')
        for name,policy,depth,direct,_ in SPECS:
            c = costs[seed][name]
            source = hc['api_calls']['factorial']+hc['api_calls']['replay_audit']+hc['fixtures'] if direct else hc['total_with_fixtures']
            cost_table.append({'seed':seed,'method':name,'source_history_and_probes':source,
                'fresh_role_probes':0 if policy=='frozen' else ac['role_probes'],
                'target_interventions':ac['selection_api_calls_including_common'] if policy in ('leases','new-only') else 0,
                'screening':c['api_calls'].get('direct-screen',0),'halving':c['api_calls'].get('halving',0),
                'pilot':c['api_calls'].get('pilot',0),'certification':c['api_calls'].get('certification',0),
                'total':c['selection_api_calls_including_common'],'allowance':c['call_allowance']})
    for method in methods:
        data = [r for r in rows if r['method']==method]
        rates,expense = [],[]
        for seed in seeds:
            sample = [r for r in data if r['seed']==seed]
            rate = float(np.mean([r['success'] for r in sample]));rates.append(rate)
            calls = costs[seed][method]['selection_api_calls_including_common'];expense.append(calls)
            seed_table.append({'seed':seed,'method':method,'vsr':rate,'extra_api_calls':calls,'tasks':len(sample)})
        table.append(summaries[method]|{'seed_std':float(np.std(rates,ddof=1)),
            'source_vsr':float(np.mean([r['success'] for r in data if r['epoch']=='source'])),
            'target_vsr':float(np.mean([r['success'] for r in data if r['epoch']=='target'])),
            'mean_extra_api_calls':float(np.mean(expense)),'api_call_std':float(np.std(expense,ddof=1)),
            'contract_invalid_calls_per_task':float(np.mean([r['contract_invalid_calls'] for r in data])),
            'mean_direct_evidence_exposures':float(np.mean([r['direct_evidence_exposures'] for r in data]))})
    def csv_file(name,values):
        with (output/name).open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    csv_file('method_statistics.csv',table);csv_file('seed_statistics.csv',seed_table);csv_file('cost_breakdown.csv',cost_table)
    if probes:csv_file('fresh_role_probes.csv',probes)
    write(output/'memory_admissions.json',admissions);write(output/'banks.json',banks)
    indexed = {method:{(r['seed'],r['task']):r for r in rows if r['method']==method} for method in methods}
    def contrast(a,b):
        subset = [r for r in rows if r['method'] in (a,b)]
        _,paired = summarize(subset,a,repeats=4000,random_seed=6011)
        result = paired[0]|{'method':a,'contrast':a+' vs '+b}
        x,y = indexed[a],indexed[b]
        if x.keys()!=y.keys():raise ValueError('paired test coverage mismatch')
        result.update(same_exposure_fraction=float(np.mean([x[k]['exposure']==y[k]['exposure'] for k in x])),
            same_outcome_fraction=float(np.mean([x[k]['success']==y[k]['success'] for k in x])),
            changed_seed_means=sum(bool(abs(np.mean([int(x[k]['success'])-int(y[k]['success']) for k in x if k[0]==seed]))>1e-12) for seed in seeds))
        return result
    primary = contrast(*PRIMARY);secondary = [contrast(*pair) for pair in SECONDARY]
    order = sorted(range(len(secondary)),key=lambda i:secondary[i]['seed_sign_flip_p']);running = 0.
    for rank,i in enumerate(order):
        running = max(running,min(1.,secondary[i]['seed_sign_flip_p']*(len(order)-rank)))
        secondary[i]['secondary_family_holm_p']=running
    csv_file('secondary_contrasts.csv',secondary)
    write(output/'contrasts.json',{'primary':primary,'secondary':secondary,
        'interpretation':'primary admission bundle includes the new candidate; new-only controls identify old-memory contribution'})
    lookup = {r['method']:r for r in table}
    effects = {}
    for a,b in (PRIMARY,*SECONDARY):
        x,y = lookup[a],lookup[b]
        effects[a+' vs '+b] = {'delta_vsr_points':100*(x['vsr']-y['vsr']),
            'relative_vsr_gain':x['vsr']/y['vsr']-1,
            'extra_call_reduction':1-x['mean_extra_api_calls']/y['mean_extra_api_calls']}
    write(output/'effects.json',effects)
    accepted = [d for d in admissions if d['admitted']]
    lease_counts = {}
    for d in accepted:
        key = d['stratum']+' / '+d['memory'];lease_counts[key]=lease_counts.get(key,0)+1
    role_counts = {}
    for p in probes:
        key = p['before']+'→'+p['after'];role_counts[key]=role_counts.get(key,0)+1
    coverage = {}
    for d in admissions:
        key = d['stratum']+' / '+d['memory'];coverage.setdefault(key,[]).append(d['contexts'])
    coverage_ranges = {key:{'min':min(v),'max':max(v),'below_minimum_12':sum(n<12 for n in v)} for key,v in coverage.items()}
    write(output/'adaptation_summary.json',{'admissions_by_family_memory':lease_counts,'fresh_role_transitions':role_counts,
        'context_coverage':coverage_ranges,
        'decision_count':len(admissions),'accepted_count':len(accepted),
        'old_memory_admissions':sum(d['memory']!='new_contract' for d in accepted)})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = methods[:-1];labels = ['Frozen K4','Probes K4','New only K4','Revalidated K4','No pairs K4','Direct K4','Revalidated K6','Direct K6']
    fig,axes = plt.subplots(1,2,figsize=(11,5))
    values = [100*lookup[n]['vsr'] for n in names]
    error = [[100*(lookup[n]['vsr']-lookup[n]['ci_low']) for n in names],
             [100*(lookup[n]['ci_high']-lookup[n]['vsr']) for n in names]]
    colors = ['#237aa4' if 'Revalidated' in n and not n.startswith('Direct') else '#8496a7' for n in names]
    axes[0].barh(labels,values,xerr=error,color=colors,capsize=3)
    axes[0].set_xlabel('Verified success (%)');axes[0].set_xlim(0,65)
    expenses = [lookup[n]['mean_extra_api_calls']/1000 for n in names]
    axes[1].barh(labels,expenses,color=colors);axes[1].set_xlabel('Extra physical API calls (thousands)')
    axes[1].set_xlim(0,max(expenses)*1.2)
    for ax,v in zip(axes,(values,expenses)):
        ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
        for i,value in enumerate(v):
            position = 100*lookup[names[i]]['ci_high']+1 if ax is axes[0] else value+1
            ax.text(position,i,f'{value:.2f}',va='center',fontsize=8)
    fig.suptitle('Fresh 8-seed synthetic evaluation; success intervals use paired bootstrap')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    text = ['# CREDIT v3 目标版本重验证实验','', '## 配置与原始汇总','',
        f'8 个新种子 {seeds}，9 个方法/设置，每方法每种子 2,048 个任务，共 {len(rows):,} 次实际测试。',
        '成功率误差为八种子样本标准差；全部方法使用相同配对任务，先冻结全部模型、许可和库，再评分。',
        '同轮方法统一 200,000 次额外 API 调用上限，报告实际支出；比旧轮 150,000 上限更大，不能直接比较跨轮均值。','',
        '| 方法 | VSR ± 种子标准差 | Source | Target | 平均额外 API 调用 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in table:text.append(f"| {r['method']} | {100*r['vsr']:.2f}% ± {100*r['seed_std']:.2f} 点 | {100*r['source_vsr']:.2f}% | {100*r['target_vsr']:.2f}% | {r['mean_extra_api_calls']:,.0f} |")
    text += ['', '费用含所用 source 历史/原探测、新探测、目标干预、筛选、halving、pilot 和认证；不含普通部署/最终测试。',
        'Direct 无模型、PPI 权重为零；不收取未使用的 source 验证/校准费用。Probes/Frozen 不收取未使用的目标干预费用。',
        '原始按种子结果及费用分项见 seed_statistics.csv、cost_breakdown.csv。','', '## 预声明配对统计','',
        f"唯一主比较 Revalidated K4 − Probes K4：{100*primary['delta_vsr']:+.3f} 点，95% bootstrap 区间 [{100*primary['ci_low']:+.3f}, {100*primary['ci_high']:+.3f}] 点；精确种子符号翻转 p={primary['seed_sign_flip_p']:.6f}。",
        '主比较衡量目标候选准入整个组合，包括新增候选；不能把全部差值归因于旧记忆重验证。以下六个次比较统一做 Holm 校正。','',
        '| 次比较 A − B | 差值（百分点） | 95% 区间 | 原始 p | 六比较 Holm p |',
        '| --- | ---: | ---: | ---: | ---: |']
    for c in secondary:text.append(f"| {c['contrast']} | {100*c['delta_vsr']:+.3f} | [{100*c['ci_low']:+.3f}, {100*c['ci_high']:+.3f}] | {c['seed_sign_flip_p']:.6f} | {c['secondary_family_holm_p']:.6f} |")
    text += ['', '区间按 seed/family 成对分层 bootstrap，p 来自八种子的 256 种精确符号翻转。区间为点比较 95%，次比较 p 经家族校正。','',
        '## 发现、解释与后续实验','',
        f"1. **观测：**主比较相对 VSR 改变量 {100*effects[PRIMARY[0]+' vs '+PRIMARY[1]]['relative_vsr_gain']:+.2f}%，仅 {primary['changed_seed_means']}/8 个种子的测试均值改变；曝光相同 {100*primary['same_exposure_fraction']:.2f}%。**解释：**许可保守，候选还须通过筛选和独立整库认证。**影响：**正的平均差值不代表优势确立。**下一步：**预先增加目标候选的独立干预覆盖，在新种子上确认。",
        f"2. **观测：**{len(admissions)} 个记忆/家族准入判定中 {len(accepted)} 个通过，其中旧记忆 {sum(d['memory']!='new_contract' for d in accepted)} 个；详见下面表。**解释：**原池没有当前目标 submit 绑定；repair_rule 只作用 transform/authorize。**影响：**新增候选与旧记忆重新准入必须分别报告。**下一步：**比较固定新候选与从执行反馈生成的候选。",
        f"3. **观测：**U 角色新探测转换 {role_counts}。**解释：**旧 96 fixtures 不进入新的区间，1024 个新 fixtures 可能确认差异，也可能仍不足。**影响：**重探测不保证恢复迁移。**下一步：**实现论文 Eq.VOI 的自适应 probe 排程，并与固定队列比较。",
        '4. **观测：**同 K 的 Direct 和显式 pair 消融见次比较表，保留所有平局。**解释：**检索深度、有限候选、证据覆盖和模型排序共同决定选库。**影响：**不能仅凭 NoMem 的低分证明 CREDIT 优于强基线，也不能把 NoPairs 当成完全没有交互。**下一步：**更大候选池、紧容量及单因素消融。','',
        '## 目标证据与许可','', '| 家族 / 记忆 | 获准种子数（最多 8） |', '| --- | ---: |']
    if lease_counts:
        for key,n in sorted(lease_counts.items()):text.append(f'| {key} | {n} |')
    else:text.append('| 无候选通过本轮准入 | 0 |')
    text += ['', '| 家族 / 候选 | 实际独立上下文范围 | 不足 12 的种子数 |', '| --- | ---: | ---: |']
    for key,v in sorted(coverage_ranges.items()):
        if key.endswith(' / new_contract') or key.endswith(' / lookup_key'):
            text.append(f"| {key} | {v['min']}–{v['max']} | {v['below_minimum_12']} |")
    old_contrast = next(c for c in secondary if c['baseline']=='CREDIT-NewCandidate-K4')
    if not any(d['memory']!='new_contract' for d in accepted):
        text += ['', f"本轮没有旧记忆获准，Full 与 NewCandidate K4 的曝光相同 {100*old_contrast['same_exposure_fraction']:.2f}%，成功率差 {100*old_contrast['delta_vsr']:+.3f} 点。",
                 '因此本轮无法证明旧记忆重验证有效；主比较中的任何收益应归于新增候选取得目标证据后的准入组合。']
    text += ['', '每个 target 家族 24 个基础上下文，三类公开模板各 8 个；每上下文一次随机符号因子设计。',
        '可选因子仅在部分设计中出现，故单个候选的实际独立上下文数可能低于 24；报告记录每项样本量和节点标签支持。',
        '按基础上下文聚合的 Banzhaf 单侧 Student-t 下界 >0，36 个可能假设分配 alpha=.05/36，至少 12 上下文且每个受限角色/版本至少 12 条可归因标签。',
        '这是近似的候选准入证据，作用于随机保留槽背景，不是每条记忆或任意部署库的保证。最终整库仍用新任务做独立配对认证。',
        '许可只匹配业务/版本、角色顺序与特征、拓扑边、初态；新版本、新形状或新初态不继承。D 分类保留，其 source head 及 PPI 权重仍被阻断。','',
        '## 执行审计','',
        f"全部通过：{audit['models']} 个模型，{audit['selections']} 次选择，{audit['certificates']} 个整库认证；重执行 {audit['physical_episodes']:,} 个 episode、{audit['physical_calls_with_fixtures']:,} 次 API 调用（含 fixtures），核验 {audit['reuse_references']:,} 条精确复用及全部 {audit['test_episodes']:,} 个测试。",
        '审计重建 source/target 历史、探测区间、许可、筛选、halving、认证、费用、每个测试事件及状态哈希；重算校准/NLL，不重新训练优化器。',
        '选择重建采用同一实现，不能当成独立算法实现证明。invalid-call 字段统计当前公开契约不接受的请求，不将其因果归咎于某条记忆。','',
        '## 论文对应与尚未完成项','',
        '新候选文本来自公开 submit 契约，执行沿用已有 new_contract 符号行为；没有修改 actor/API/verifier 或概率效果。所有方法共享九候选目录，新增候选无证据时一律隔离。',
        '这是有限符号适配器，不是 LLM 理解任意记忆文本，也不是自动自然语言记忆生成。目标标签不进入 source 模型拟合；当前准入扩展不把 D 角色变成可运输角色。',
        '目标适配已见全部三种最终测试拓扑，因此本轮没有未见拓扑泛化结论。原始用户数据保持原样，本轮是注入漂移的合成环境。',
        'probe 队列为固定排序 U 角色的探索批次；Beta 更新仅是记录，不构成 Eq.VOI 自适应调度。连续归档/压缩/重验证、大池搜索、真实服务/LLM及其它论文实验仍未完成。',
        '认证为每周期渐近 alpha=0.10，不宣称有限样本或生命周期保证；Direct 费用节省仅相对本轮每层 16 个筛选任务的指定配置。','',
        '## 复现','', '```sh',
        'python -m credit.v3_revalidation_data --output datasets/new_revalidation',
        'PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m credit.v3_revalidation_experiment --dataset datasets/new_revalidation --output results/new_revalidation',
        'PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m credit.v3_revalidation_verify --results results/new_revalidation',
        'python -m credit.v3_revalidation_report --benchmark results/new_revalidation --output results/new_revalidation_analysis',
        '```','']
    report = '\n'.join(text);(output/'report.md').write_text(report)
    canonical = Path('CREDIT_V3_REVALIDATION_RESULTS.md');canonical.write_text(report)
    write(output/'analysis_manifest.json',{'benchmark_manifest_sha256':digest(benchmark/'artifact_manifest.json'),
        'audit_sha256':digest(audit_path),'generator_sha256':digest(__file__),'canonical_report_sha256':digest(canonical),
        'artifacts':{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({'methods':table,'primary':primary,'secondary':secondary,'admissions':lease_counts}),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a = p.parse_args();generate(a.benchmark,a.output)
