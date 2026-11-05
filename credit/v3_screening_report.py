"""Audited tables, paired effects and standalone figures for screening study."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .paired_statistics import summarize
from .v3.drift import ProbeEvidence, Level, TransportGate
from .v3.experiment import digest, write
from .v3.local import memories
from .v3_screening_optimization import public_profiles
from .v3_optimization_verify import records


def generate(benchmark, output):
    if output.exists():
        raise FileExistsError('use a fresh analysis directory')
    audit_path = benchmark.with_name(benchmark.name+'_audit.json')
    audit = json.loads(audit_path.read_text())
    if not audit['verified']:
        raise ValueError('complete execution audit required')
    for name,value in json.loads((benchmark/'artifact_manifest.json').read_text()).items():
        if digest(benchmark/name) != value:
            raise ValueError('experimental artifact changed')
    output.mkdir()
    read = lambda p:json.loads(p.read_text())
    rows = read(benchmark/'raw_test_results.json')
    stats = read(benchmark/'statistics.json')
    primary = read(benchmark/'primary_statistics.json')['paired'][0]
    seeds = read(benchmark/'frozen_protocol.json')['seeds']
    methods = sorted({r['method'] for r in rows})
    table, costs, diagnostics = [], {}, []
    for seed in seeds:
        sd = benchmark/f'seed_{seed}'
        costs[seed] = read(sd/'method_costs.json')
        g = TransportGate([ProbeEvidence(**(r|{'level':Level(r['level'])}))
                           for r in read(sd/'compact_history/transfer_levels.json')])
        for method in methods:
            if method == 'NoMem':
                continue
            md = sd/method
            depth = 6 if method.endswith('K6') else 4
            # Profiles depend only on public templates and frozen gating.
            mixture = {f'{f}/{e}':.125 for f in ('books','travel','inventory','payments') for e in ('source','target')}
            profiles = public_profiles(mixture)
            from .v3.local import retriever
            exposure = lambda c,b:retriever(depth)(c,tuple(m.uid for m in g.candidates(c,[m for m in memories() if m.uid in b])))
            screen,half = read(md/'screening.json'),read(md/'halving.json')
            signatures = [tuple(exposure(c,b) for c in profiles) for b in screen['banks']]
            final = [tuple(exposure(c,b) for c in profiles) for b in half['finalists']]
            reuse,saved_calls = [],0
            if (md/'execution_reuse.json').exists():
                reuse = read(md/'execution_reuse.json')
                physical = list(records(md/'selection.jsonl.gz'))
                saved_calls = sum(physical[r['physical_record']]['episode']['calls'] for r in reuse)
            diagnostics.append({'seed':seed,'method':method,'screened_feasible':screen.get('feasible_banks',screen.get('evaluated')),
                'screened_distinct':screen.get('distinct_exposures'),
                'shortlist_size':len(signatures),'shortlist_distinct':len(set(signatures)),
                'finalists':len(final),'distinct_finalists':len(set(final)),
                'exact_execution_reuses':len(reuse),'avoided_physical_calls':saved_calls})
    for s in stats['main']:
        method = s['method']
        data = [r for r in rows if r['method'] == method]
        seed_rates = [np.mean([r['success'] for r in data if r['seed'] == seed]) for seed in seeds]
        expense = [costs[seed][method]['selection_api_calls_including_common'] for seed in seeds]
        table.append(s | {'seed_std':float(np.std(seed_rates,ddof=1)),
            'source_vsr':float(np.mean([r['success'] for r in data if r['epoch']=='source'])),
            'target_vsr':float(np.mean([r['success'] for r in data if r['epoch']=='target'])),
            'mean_extra_api_calls':float(np.mean(expense)), 'api_call_std':float(np.std(expense,ddof=1)),
            'mean_bank_tokens':float(np.mean([r['bank_tokens'] for r in data]))})
    def save_csv(name, values):
        with (output/name).open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(values[0]))
            writer.writeheader();writer.writerows(values)
    save_csv('method_statistics.csv',table)
    save_csv('paired_statistics.csv',stats['paired'])
    save_csv('selection_diagnostics.csv',diagnostics)
    indexed = {m:{(r['seed'],r['task']):r for r in rows if r['method']==m} for m in methods}
    primary_seed_differences = {str(seed):float(np.mean([
        indexed['CREDIT-Mixed-K4'][k]['success']-int(indexed['CREDIT-Unique-K4'][k]['success'])
        for k in indexed['CREDIT-Mixed-K4'] if k[0]==seed])) for seed in seeds}
    changed_seeds = sum(abs(v)>1e-12 for v in primary_seed_differences.values())
    comparisons = {}
    for a,b in [('CREDIT-Mixed-K4','CREDIT-Mixed-NoPairs-K4'),('CREDIT-Mixed-K4','CREDIT-Unique-K4'),
                ('CREDIT-Mixed-K6','Direct-Unique-K6')]:
        keys = indexed[a].keys()
        if keys != indexed[b].keys():
            raise ValueError('paired coverage mismatch')
        comparisons[a+' vs '+b] = {'tasks':len(keys),
            'same_exposure_fraction':float(np.mean([indexed[a][k]['exposure']==indexed[b][k]['exposure'] for k in keys])),
            'same_outcome_fraction':float(np.mean([indexed[a][k]['success']==indexed[b][k]['success'] for k in keys]))}
    k6rows = [r for r in rows if r['method'] in ('CREDIT-Mixed-K6','Direct-Unique-K6')]
    _,k6 = summarize(k6rows,'CREDIT-Mixed-K6',repeats=4000,random_seed=5003)
    # K6 is a secondary contrast: Holm adjustment includes it with all other secondary contrasts.
    secondary = [r for r in stats['paired'] if r['baseline'] not in ('CREDIT-Unique-K4','NoMem')]
    secondary.append(k6[0]|{'contrast':'CREDIT-Mixed-K6 vs Direct-Unique-K6'})
    order = sorted(range(len(secondary)),key=lambda i:secondary[i]['seed_sign_flip_p'])
    running = 0.
    for rank,i in enumerate(order):
        running = max(running,min(1.,secondary[i]['seed_sign_flip_p']*(len(order)-rank)))
        secondary[i]['secondary_family_holm_p'] = running
    write(output/'contrasts.json',{'primary':primary,'secondary':secondary,'exposure_checks':comparisons,
                                'primary_seed_differences':primary_seed_differences})
    lookup = {r['method']:r for r in table}
    def difference(a,b):
        x,y = lookup[a],lookup[b]
        return {'delta_vsr_points':100*(x['vsr']-y['vsr']),
            'relative_vsr_gain':x['vsr']/y['vsr']-1,
            'call_reduction':1-x['mean_extra_api_calls']/y['mean_extra_api_calls']}
    effects = {b:difference('CREDIT-Mixed-K4',b) for b in
               ('CREDIT-Unique-K4','CREDIT-Previous-K4','Direct-Unique-K4')}
    effects['K6 Direct'] = difference('CREDIT-Mixed-K6','Direct-Unique-K6')
    write(output/'effects.json',effects)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,4),gridspec_kw={'width_ratios':[5,2]})
    groups = [('CREDIT-Previous-K4','CREDIT-Unique-K4','CREDIT-Mixed-K4','CREDIT-Mixed-NoPairs-K4','Direct-Unique-K4'),
              ('CREDIT-Mixed-K6','Direct-Unique-K6')]
    labels = [('Previous','Unique','Mixed','No pairs','Direct'),('Mixed','Direct')]
    for ax,names,lab in zip(axes,groups,labels):
        values = [100*lookup[n]['vsr'] for n in names]
        error = [[100*(lookup[n]['vsr']-lookup[n]['ci_low']) for n in names],
                 [100*(lookup[n]['ci_high']-lookup[n]['vsr']) for n in names]]
        ax.bar(lab,values,yerr=error,capsize=3,color=['#8a9aaa' if 'Mixed' not in n else '#307ead' for n in names])
        for i,value in enumerate(values):
            ax.text(i,1,f'{value:.2f}%',ha='center',va='bottom',color='white',fontsize=9)
        ax.set_ylim(0,55);ax.set_title('K = 4' if len(names)>2 else 'K = 6')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Verified success (%)')
    fig.suptitle('Fresh 8-seed synthetic evaluation; bootstrap 95% intervals')
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(output/f'success.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    names = list(groups[0])+list(groups[1])
    cost_labels = ['Previous K4','Unique K4','Mixed K4','No pairs K4','Direct K4','Mixed K6','Direct K6']
    fig,ax = plt.subplots(figsize=(8,4))
    values = [lookup[n]['mean_extra_api_calls']/1000 for n in names]
    ax.barh(cost_labels,values,color=['#307ead' if 'Mixed' in n else '#8a9aaa' for n in names])
    for i,value in enumerate(values):
        ax.text(value+1,i,f'{value:.1f}',va='center',fontsize=9)
    ax.set_xlim(0,max(values)*1.15);ax.invert_yaxis()
    ax.set_xlabel('Mean extra physical API calls (thousands)')
    ax.set_title('Used history, probes, screening, halving and certification included')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(output/f'cost.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    old_diagnostics = [r for r in diagnostics if r['method']=='CREDIT-Previous-K4']
    new_diagnostics = [r for r in diagnostics if r['method']=='CREDIT-Mixed-K4']
    old_unique = np.mean([r['shortlist_distinct'] for r in old_diagnostics])
    old_duplicate_finalists = sum(r['distinct_finalists']<r['finalists'] for r in old_diagnostics)
    mean_reuses = np.mean([r['exact_execution_reuses'] for r in new_diagnostics])
    mean_saved_calls = np.mean([r['avoided_physical_calls'] for r in new_diagnostics])
    text = ['# CREDIT v3 候选筛选优化结果','',
        '## 配置与原始汇总','',
        f'八个新种子，8 个方法/设置，每方法每种子 2,048 个任务，共 {len(rows):,} 次实际测试。',
        '表中误差为八种子成功率的样本标准差；均值不是单个最好种子。API 费用含所用历史、探测、筛选及认证。','',
        '| 方法 | 成功率 ± 种子标准差 | Source | Target | 平均额外 API 调用 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in table:
        text.append(f"| {r['method']} | {100*r['vsr']:.2f}% ± {100*r['seed_std']:.2f} 点 | {100*r['source_vsr']:.2f}% | {100*r['target_vsr']:.2f}% | {r['mean_extra_api_calls']:,.0f} |")
    text += ['', '## 配对统计','',
        f"预注册唯一主比较：Mixed K4 − Unique K4 = {100*primary['delta_vsr']:+.2f} 点，95% bootstrap 区间 [{100*primary['ci_low']:+.2f}, {100*primary['ci_high']:+.2f}] 点，八种子精确符号翻转 p={primary['seed_sign_flip_p']:.6f}。",
        '这个主比较仅改变未运输层筛选估值：从零变为独立真实筛选任务的均值。其它搜索、复用、训练、pilot、认证配置相同。','',
        f'主比较仅 {changed_seeds}/8 个种子的测试均值发生变化；其余种子相同。不能把区间下界为零解释为显著正效应。','',
        '| Mixed K4 对照 | 差值（百分点） | 95% 区间 | 原始 p | Holm p（全部配对） |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in stats['paired']:
        text.append(f"| {r['baseline']} | {100*r['delta_vsr']:+.2f} | [{100*r['ci_low']:+.2f}, {100*r['ci_high']:+.2f}] | {r['seed_sign_flip_p']:.6f} | {r['holm_p']:.6f} |")
    text += ['', f"同 K6 比较：Mixed − Direct = {100*k6[0]['delta_vsr']:+.2f} 点，95% 区间 [{100*k6[0]['ci_low']:+.2f}, {100*k6[0]['ci_high']:+.2f}] 点，原始 p={k6[0]['seed_sign_flip_p']:.6f}。次比较家族校正见 contrasts.json。",'',
        '## 发现与解释','',
        f"1. 主比较的相对成功率变化为 {100*effects['CREDIT-Unique-K4']['relative_vsr_gain']:+.2f}%；与旧选择器相比为 {100*effects['CREDIT-Previous-K4']['relative_vsr_gain']:+.2f}%。正均值不自动代表优势确立，判断须同时看区间及 p。",
        '2. 旧模型对不通过运输检查的任务置零，但 travel/target 的 ready-submit 任务通过检查，仍有预测值；不能说旧版三个目标层全部为零。本轮 Unique 使用更保守的整层许可，三个 target 层均置零；Mixed 在这些层统一用真实执行。主比较因此是明确的整层零估值消融，对旧版则包含额外的许可聚合差异。未获准的模型迁移及 PPI 权重仍为零，没有放宽记忆漂移门。',
        '3. 新搜索枚举八条记忆的全部容量可行组合，并在全部公开三类模板/八个层上按有序曝光去重，保留最便宜代表。Unique 对 Previous 同时包含搜索、去重、pilot 缩小、复用和更保守的整层许可；冻结协议差异列表未详列最后这项，此处补充，不能单独归因于去重。',
        f'   旧 K4 shortlist 的八个库平均只有 {old_unique:.2f} 种曝光，{old_duplicate_finalists}/8 个种子的两个最终库曝光重复；新 K4 每次保留八种不同曝光。Mixed K4 平均复用 {mean_reuses:,.0f} 个准确任务/曝光记录，避免 {mean_saved_calls:,.0f} 次重复物理调用。',
        '4. 单个准确曝光的真实结果可服务多个等价库的配对估计，API 费用按实际物理调用计数。缓存条件包括完整任务/噪声/版本/初态及有序曝光；仅适用于本次隔离、确定重放的执行器。','',
        '## 曝光与消融核验','']
    for name,c in comparisons.items():
        text.append(f"- {name}：{c['tasks']:,} 个配对任务，曝光相同 {100*c['same_exposure_fraction']:.2f}%，结果相同 {100*c['same_outcome_fraction']:.2f}%。")
    text += ['', '无交互消融仍同时改变模型拟合，尚未证明显式交互能改善选库；相同结果须保留，不能通过更换有利测试任务制造优势。','',
        '## 执行审计与边界','',
        f"审计通过：{audit['models']} 个冻结模型、{audit['selections']} 个选择流程、{audit['certificates']} 个认证；重执行 {audit['physical_episodes']:,} 个实际 episode、核验 {audit['reuse_references']:,} 个复用引用及全部 {audit['test_episodes']:,} 个测试结果。",
        '审计重建历史、校准、验证 NLL、选择、费用和所有测试事件；不重新训练优化器，选择重建使用同一实现。',
        '测试分布延续旧合成环境但使用新种子，全部模型/库先冻结再评分。训练及执行机制未改；真实服务/LLM、较大记忆池和通用 DAG 尚未完成。',
        '本轮 Direct 使用相同去重候选的全枚举、每层 16 个真实筛选任务；调用节省仅相对这个明确配置，不代表 Direct 的最优费用前沿，也不能直接减去上轮不同种子的均值。额外费用不包括普通部署/测试调用。',
        '枚举方法只适用于小池；去重仅覆盖已声明公开模板。认证仍为每周期渐近 alpha=0.10，无有限样本或全生命周期保证。','',
        '## 复现','', '```sh',
        'PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m credit.v3_optimized_data --output datasets/new_screen --seeds 1009 1013 1019 1021 1031 1033 1039 1049',
        'PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m credit.v3_screening_optimization --dataset datasets/new_screen --output results/new_screen',
        'PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m credit.v3_screening_verify --results results/new_screen',
        'python -m credit.v3_screening_report --benchmark results/new_screen --output results/new_screen_analysis',
        '```','', '## 下一步','',
        '1. 核对未运输层候选排名稳定性及改变选库的种子比例；增加独立筛选样本/种子后在新冻结任务上确认，不用这轮测试选择参数。',
        '2. 在更紧容量/更大池上比较局部搜索与小池精确搜索上界，并把去重、pilot 缩小和执行复用分别消融。',
        '3. 补充目标版本记忆重验证，让经过直接证据核验的修复记忆有机会重新进入候选；再做真实可重放服务评估，不将本轮合成指标写成真实业务结果。','']
    report = '\n'.join(text)
    (output/'report.md').write_text(report)
    canonical = Path('CREDIT_V3_SCREENING_OPTIMIZATION_RESULTS.md')
    canonical.write_text(report)
    write(output/'analysis_manifest.json',{'benchmark_manifest_sha256':digest(benchmark/'artifact_manifest.json'),
        'audit_sha256':digest(audit_path),'generator_sha256':digest(__file__),
        'canonical_report_sha256':digest(canonical),
        'artifacts':{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({'methods':table,'primary':primary,'effects':effects}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    generate(a.benchmark,a.output)
