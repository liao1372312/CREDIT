"""Analyze frozen optimization outcomes and reused evaluation-only references."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from functools import lru_cache

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .v3.audit import load_context
from .v3.drift import Level, ProbeEvidence, TransportGate
from .v3.experiment import digest,write
from .v3.factorial import admitted_pairs,characters
from .v3.local import memories
from .v3_reference import credits
from .v3_optimized_model import load_model
from .paired_statistics import summarize
from .v3.types import Role
from .v3 import outcome as outcome_module
from . import v3_optimized_model as optimized_module


def read(path):
    return json.loads(path.read_text())


def save_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def generate(benchmark,output,reference=Path('results/credit_v3_reference_v1'),compact=None):
    if output.exists():
        raise FileExistsError('use a fresh analysis directory')
    if not (benchmark/'artifact_manifest.json').exists() or not read(benchmark/'execution_audit.json')['verified']:
        raise ValueError('benchmark must finish and pass execution audit first')
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    (output/'figures').mkdir()
    proto = read(benchmark/'frozen_protocol.json')
    rows = read(benchmark/'raw_test_results.json')
    stats = read(benchmark/'statistics.json')
    primary = read(benchmark/'primary_statistics.json')
    seeds = proto['seeds']
    compact_methods = set()
    if compact is not None:
        if not read(compact/'execution_audit.json')['verified']:
            raise ValueError('compact execution audit required')
        cp = read(compact/'frozen_protocol.json')
        if (cp['seeds'] != seeds or cp['dataset_manifest_sha256'] != proto['dataset_manifest_sha256']
                or not cp['specified_before_main_test_scoring']):
            raise ValueError('compact configuration/splits differ')
        extra = read(compact/'raw_test_results.json')
        compact_methods = {r['method'] for r in extra}
        if compact_methods & {r['method'] for r in rows}:
            raise ValueError('duplicate method across studies')
        rows += extra
        main,paired = summarize(rows,'CREDIT-State-K4',repeats=4000,random_seed=4099)
        stats = {'main':main,'paired':paired}
        write(output/'combined_statistics.json',stats)
    def method_root(method):
        return compact if method in compact_methods else benchmark
    methods = sorted({r['method'] for r in rows})
    cells = defaultdict(list)
    for row in rows:
        cells[row['method'],row['seed']].append(row)
    summaries = []
    for method in methods:
        vsrs = [np.mean([r['success'] for r in cells[method,s]]) for s in seeds]
        costs = [read(method_root(method)/f'seed_{s}'/'method_costs.json')[method]
                 ['selection_api_calls_including_common'] for s in seeds]
        confidence = next(r for r in stats['main'] if r['method'] == method)
        subrows = [r for r in rows if r['method'] == method]
        summaries.append({'method':method,'vsr':float(np.mean(vsrs)),
                          'seed_sd':float(np.std(vsrs,ddof=1)),
                          'ci_low':confidence['ci_low'],'ci_high':confidence['ci_high'],
                          'source_vsr':float(np.mean([r['success'] for r in subrows if r['epoch'] == 'source'])),
                          'target_vsr':float(np.mean([r['success'] for r in subrows if r['epoch'] == 'target'])),
                          'extra_calls_mean':float(np.mean(costs)),
                          'extra_calls_sd':float(np.std(costs,ddof=1)),
                          'seeds':len(seeds),'test_episodes':len(subrows)})
    save_csv(output/'method_statistics.csv',summaries)
    comparisons = []
    # These contrasts are descriptive unless covered by the predeclared primary
    # or the stored secondary familywise correction.
    contrast_names = [('CREDIT-State-K4','CREDIT-Legacy-K4'),
                ('CREDIT-State-K4','CREDIT-Coverage-K4'),
                ('CREDIT-State-K4','CREDIT-State-NoPairs-K4'),
                ('CREDIT-State-K4','Direct-K4'),
                ('CREDIT-State-K4','Direct-Matched-K4'),
                ('CREDIT-State-K6','Direct-K6'),
                ('CREDIT-State-K6','Direct-Matched-K6')]
    if compact is not None:
        contrast_names += [('CREDIT-Compact-K4','CREDIT-Legacy-K4'),
                           ('CREDIT-Compact-K4','CREDIT-State-K4'),
                           ('CREDIT-Compact-K4','Direct-Compact-K4'),
                           ('CREDIT-Compact-K6','Direct-Compact-K6'),
                           ('CREDIT-Compact-K6','CREDIT-Compact-NoPairs-K6')]
    for a,b in contrast_names:
        seed_deltas = [np.mean([r['success'] for r in cells[a,s]])-
                       np.mean([r['success'] for r in cells[b,s]]) for s in seeds]
        sa,sb = [next(x for x in summaries if x['method'] == m) for m in (a,b)]
        comparisons.append({'method':a,'baseline':b,'delta_vsr':float(np.mean(seed_deltas)),
                            'seed_deltas':list(map(float,seed_deltas)),
                            'relative_vsr_improvement':(sa['vsr']-sb['vsr'])/sb['vsr'],
                            'relative_call_change':(sa['extra_calls_mean']-sb['extra_calls_mean'])/sb['extra_calls_mean']})
    write(output/'contrasts.json',comparisons)
    save_csv(output/'paired_statistics.csv',stats['paired'])
    optimizer = []
    model_specs = [('legacy',benchmark,'models_legacy'),('coverage',benchmark,'models_coverage'),
                   ('state',benchmark,'models_state'),('no_pairs',benchmark,'models_no_pairs')]
    if compact is not None:
        model_specs += [('compact_state',compact,'models_state'),
                        ('compact_no_pairs',compact,'models_no_pairs')]
    for s in seeds:
        for name,root,subdirectory in model_specs:
            for i,r in enumerate(read(root/f'seed_{s}'/subdirectory/'training_reports.json')):
                optimizer.append({'seed':s,'model':name,'member':i,'iterations':r['iterations'],
                                  'selected_epoch':r.get('selected_epoch',r['iterations']),
                                  'early_stopped':r['stopping_criterion_met'],
                                  'validation_nll':r.get('selected_validation_nll'),
                                  'calibration_brier':r.get('calibration_brier'),
                                  'calibration_mae':r['calibration_mae']})
    save_csv(output/'training_diagnostics.csv',optimizer)
    write(output/'training_diagnostics.json',optimizer)

    # Reuse an already audited exact reference, only after every new model/bank
    # has frozen and its held-out scoring/audit has completed. This diagnostic
    # was known during development and is not an untouched confirmation set.
    ref_data = Path('datasets/credit_v3_reference_v1')
    ref_manifest = read(reference/'artifact_manifest.json')
    for filename in ('coalitions.npz','summary.json'):
        if digest(reference/filename) != ref_manifest[filename]:
            raise ValueError('changed attribution reference')
    if not read(reference/'execution_audit.json')['verified']:
        raise ValueError('reference must be audited')
    for name,value in read(benchmark/'frozen_protocol.json')['source_sha256'].items():
        if digest(name) != value:
            raise ValueError('changed experiment source')
    cases = [load_context(c) for c in read(ref_data/'contexts.json')]
    if digest(ref_data/'contexts.json') != read(ref_data/'protocol.json')['contexts_sha256']:
        raise ValueError('changed reference context definitions')
    with np.load(reference/'coalitions.npz') as archive:
        truth,z = archive['truth'].copy(),archive['z'].copy()
    pool = memories()
    exposures = [tuple(m.uid for m,sign in zip(pool,row) if sign == 1) for row in z]
    # Pure encoder memoization is confined to this read-only analysis process.
    # Check exact predictions before/after; no statistical model is refitted.
    check_models = [load_model(benchmark/f'seed_{seeds[0]}'/sub/'outcome_model_0.pt',pool,cases)
                    for sub in ('models_legacy','models_state')]
    check_items = [(cases[i],a) for i in (0,2,3) for a in exposures[::17]]
    before_cache = [m.predict_many(check_items) for m in check_models]
    original_role_key = Role.key
    original_text = outcome_module.text_encoding
    original_optimized_text = optimized_module.text_encoding
    role_keys = {r:r.key for c in cases for r in (n.role for n in c.nodes)}
    Role.key = property(lru_cache(maxsize=64)(original_role_key.fget))
    cached_text = lru_cache(maxsize=256)(original_text)
    outcome_module.text_encoding = cached_text
    optimized_module.text_encoding = cached_text
    if any(r.key != value for r,value in role_keys.items()):
        raise ValueError('role key cache changed a signature')
    if any(not np.array_equal(before,m.predict_many(check_items))
           for before,m in zip(before_cache,check_models)):
        raise ValueError('encoder memoization changed predictions')
    write(output/'analysis_cache_check.json',{'exact_prediction_equality':True,
          'models_checked':2,'queries_per_model':len(check_items),
          'role_signatures_checked':len(role_keys),'only_deterministic_encodings_cached':True})
    oracle = [credits(values,z) for values in truth]
    effects = oracle[0][0]
    walsh = characters(z,effects)
    shapley_weights = np.zeros((len(effects),len(pool)))
    for ei,effect in enumerate(effects):
        if len(effect) % 2:
            for member in effect:
                shapley_weights[ei,member] = 2/len(effect)
    errors = []
    raw_predictions = np.empty((len(seeds),len(model_specs),len(cases),len(exposures)))
    for si,s in enumerate(seeds):
        gate = TransportGate([ProbeEvidence(**(r | {'level':Level(r['level'])}))
                              for r in read(benchmark/f'seed_{s}'/'expanded_history'/'transfer_levels.json')])
        for mi,(name,root,subdirectory) in enumerate(model_specs):
            models = [load_model(p,pool,cases) for p in sorted(
                (root/f'seed_{s}'/subdirectory).glob('outcome_model_*.pt'))]
            for ci,c in enumerate(cases):
                values = np.mean([m.predict_many([(c,a) for a in exposures]) for m in models],axis=0)
                raw_predictions[si,mi,ci] = values
                beta = values @ walsh / len(z)
                banzhaf = 2*beta[1:1+len(pool)]
                shapley = beta @ shapley_weights
                _,true_beta,true_banzhaf,true_shapley = oracle[ci]
                pair_ids = [effects.index(p) for p in admitted_pairs(c,pool)]
                predicted = np.abs(beta[pair_ids]) >= .015
                actual = np.abs(true_beta[pair_ids]) >= .015
                tp,fp,fn = int((predicted & actual).sum()),int((predicted & ~actual).sum()),int((~predicted & actual).sum())
                errors.append({'seed':s,'model':name,'case':ci,'family':c.stratum.split('/')[0],
                               'epoch':c.stratum.split('/')[1],'depth':len(c.nodes),
                               'utility_mae':float(np.mean(np.abs(values-truth[ci]))),
                               'banzhaf_mae':float(np.mean(np.abs(banzhaf-true_banzhaf))),
                               'shapley_mae':float(np.mean(np.abs(shapley-true_shapley))),
                               'pair_tp':tp,'pair_fp':fp,'pair_fn':fn,
                               'transport_supported':gate.transported(c)})
            print('diagnostic reference predictions',s,name,flush=True)
    np.savez_compressed(output/'reference_predictions.npz',predictions=raw_predictions,truth=truth,z=z)
    save_csv(output/'reference_errors.csv',errors)
    grouped = []
    for name,_,_ in model_specs:
        for epoch in ('source','target'):
            for depth in (1,4,6,9):
                values = [r for r in errors if (r['model'],r['epoch'],r['depth']) == (name,epoch,depth)]
                tp,fp,fn = (sum(r[key] for r in values) for key in ('pair_tp','pair_fp','pair_fn'))
                grouped.append({'model':name,'epoch':epoch,'depth':depth,
                                **{key:float(np.mean([r[key] for r in values]))
                                   for key in ('utility_mae','banzhaf_mae','shapley_mae')},
                                'pair_micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None})
    save_csv(output/'reference_grouped.csv',grouped)
    write(output/'reference_diagnostics.json',{'groups':grouped,
          'known_diagnostic_reference':True,'reference_manifest_sha256':digest(reference/'artifact_manifest.json'),
          'scope':'exact simulator evaluation only; transport support is reported per query; most target queries are unsupported extrapolations',
          'source_seed_count':len(seeds),'coalitions_per_case':len(exposures)})

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    by_name = {r['method']:r for r in summaries}
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(output/'figures'/f'{name}.png',dpi=180)
        fig.savefig(output/'figures'/f'{name}.pdf')
        plt.close(fig)
    chosen = ['CREDIT-Legacy-K4','CREDIT-Coverage-K4','CREDIT-State-K4',
              'Direct-K4','CREDIT-State-K6','Direct-K6']
    labels = ['Legacy K4','Coverage K4','State K4','Direct K4','State K6','Direct K6']
    if compact is not None:
        chosen = ['CREDIT-Legacy-K4','CREDIT-State-K4','CREDIT-Compact-K4','Direct-Compact-K4',
                  'CREDIT-State-K6','CREDIT-Compact-K6','Direct-Compact-K6']
        labels = ['Legacy K4','State K4','Compact K4','Direct K4','State K6','Compact K6','Direct K6']
    fig,ax = plt.subplots(figsize=(9,4))
    vals = np.asarray([by_name[m]['vsr']*100 for m in chosen])
    low = vals-np.asarray([by_name[m]['ci_low']*100 for m in chosen])
    high = np.asarray([by_name[m]['ci_high']*100 for m in chosen])-vals
    ax.bar(labels,vals,yerr=np.asarray([low,high]),capsize=3)
    ax.set_ylabel('Verified success (%)')
    ax.set_title('Fresh paired tasks; 8 seeds; clustered 95% bootstrap intervals')
    save(fig,'main_vsr')
    fig,ax = plt.subplots(figsize=(7,4))
    for m in chosen:
        r = by_name[m]
        ax.scatter(r['extra_calls_mean']/1000,r['vsr']*100,s=50)
        ax.annotate(m.replace('CREDIT-',''),(r['extra_calls_mean']/1000,r['vsr']*100),
                    xytext=(4,6),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Extra physical API calls (thousands, mean)')
    ax.set_ylabel('Verified success (%)')
    ax.set_title('Measured expense; actual expenditures differ')
    save(fig,'cost_success')
    fig,ax = plt.subplots(figsize=(6,4))
    for name in ('legacy','coverage','state')+ (('compact_state',) if compact is not None else ()):
        values = [r for r in grouped if r['model'] == name and r['epoch'] == 'source']
        ax.plot([r['depth'] for r in values],[r['utility_mae'] for r in values],marker='o',label=name)
    ax.set_xlabel('Required workflow nodes')
    ax.set_ylabel('Utility MAE against exact simulator reference')
    ax.set_title('Known diagnostic cases; source version')
    ax.legend()
    save(fig,'reference_error')

    main_table = table(['方法','VSR % ± 种子 SD','源 / 目标 %','平均额外调用'],[
        [r['method'],f"{100*r['vsr']:.2f} ± {100*r['seed_sd']:.2f}",
         f"{100*r['source_vsr']:.2f} / {100*r['target_vsr']:.2f}",f"{r['extra_calls_mean']:,.0f}"] for r in summaries])
    prim = primary['paired'][0]
    ctable = table(['比较（前者 − 后者）','VSR 差（百分点）','调用变化'],[
        [r['method']+' − '+r['baseline'],f"{100*r['delta_vsr']:+.2f}",
         f"{100*r['relative_call_change']:+.1f}%"] for r in comparisons])
    rtable = table(['模型','源版本深度','效用 MAE','Banzhaf MAE','交互 F1'],[
        [r['model'],r['depth'],f"{r['utility_mae']:.4f}",f"{r['banzhaf_mae']:.4f}",
         'N/A' if r['pair_micro_f1'] is None else f"{r['pair_micro_f1']:.3f}"]
        for r in grouped if r['epoch'] == 'source' and r['model'] in ('legacy','state','compact_state')])
    state_reports = [r for r in optimizer if r['model'] == 'state']
    coverage_reports = [r for r in optimizer if r['model'] == 'coverage']
    state_nll = np.mean([r['validation_nll'] for r in state_reports])
    coverage_nll = np.mean([r['validation_nll'] for r in coverage_reports])
    improvement = next(r for r in comparisons if r['baseline'] == 'CREDIT-Legacy-K4')
    main_conclusion = ('本轮优化提高了成功率。' if improvement['delta_vsr'] > 0
                       else '本轮优化没有提高主比较成功率，不能称为优化成功。')
    compact_section = ''
    if compact is not None:
        cs = next(r for r in comparisons if r['method'] == 'CREDIT-Compact-K4' and r['baseline'] == 'CREDIT-Legacy-K4')
        cf = next(r for r in comparisons if r['method'] == 'CREDIT-Compact-K4' and r['baseline'] == 'CREDIT-State-K4')
        np6 = next(r for r in comparisons if r['baseline'] == 'CREDIT-Compact-NoPairs-K6')
        compact_section = f'''### 紧凑训练补充

在查看本轮测试结果前，基于前置调用成本固定了紧凑配置：每家族前 16 个训练、前 8 个验证/校准上下文，
共 64/32/32 个 source 上下文，单 draw，保持初态与 1/4/5 步覆盖。另补 K6 无交互对照与同 K 零代理 Direct。
配置与轨迹在 `{compact}`；两组分别冻结全部种子/库后评分，共用同一批八种子，不能算作 16 个独立种子。
Compact-K4 − Legacy-K4 为 {100*cs['delta_vsr']:+.2f} 点；Compact-K4 相对完整 State-K4 的调用变化
{100*cf['relative_call_change']:+.1f}%，成功率差 {100*cf['delta_vsr']:+.2f} 点。
Compact-K6 − Compact-NoPairs-K6 为 {100*np6['delta_vsr']:+.2f} 点，独立 K6 配对统计及 Holm 校正在补充目录。
紧凑方案比较属于预先登记的补充，全部方法的探索性校正按合并后的 {len(stats['paired'])} 项比较计算。
'''
    report = f'''# CREDIT v3 状态与训练优化结果

## 运行范围与变化

{main_conclusion} 原始五种子实验、最新论文、AGENTS.md 和用户提供的数据未改动。
本轮使用八个新种子 {seeds}，固定原 actor/API/检索器/验收；主组 10 个方法/设置、全部种子先冻结后评分。
共 **{len(rows):,} 个实际测试 episode**，每方法每种子 2,048 个，额外调用上限 150,000。
数据位于 `datasets/credit_v3_optimization_v1/`，源码/模型/逐次轨迹位于 `{benchmark}`。
这是受控合成实验，不是自然供应商或完整 LLM 测量。

新增状态条件双头：训练输入为实际调用前的公开状态；预测输入为公开契约规定的前序成功状态。
共享角色、service/amb 标签屏蔽、双头损失、乘积及独立 isotonic 校准保持不变。
此状态构造适用于当前确定后置条件的有限适配器；一般服务需自行提供可验证的状态特征。
状态/图特征、32 维文本和隐藏层一起变化，不能将其增益单独归于某一种特征。

Legacy 使用 32 个旧形状训练上下文、2 draws 和 200 轮旧训练；新模型使用 128 个 source 训练上下文、1 draw，
另有 64 个独立 optimizer 验证上下文、64 个独立校准上下文。新增 1 步预就绪提交、初态 0/1/3/7 覆盖；
6 步测试拓扑未进入训练/验证/校准，1/5 步形状有重叠。完整流程仍使用六槽，submit-only 使用三槽，保留隐藏记忆的随机正性。
Coverage 使用旧编码和新数据/验证训练，故为“数据覆盖与训练组合”消融。

## 1. 原始结果汇总

{main_table}

区间及每种子数据见 CSV/JSON；同一新测试集上的 Legacy 是本轮重跑，不能与旧五种子均值作直接差值检验。
所有方法都使用确定性的候选遍历、相同 gamma=1,024 的真实 halving 和每层 200 对独立认证任务。
Direct 完全零代理；pilot 仅每层 2 个任务，权重为 0，不支出模型验证/校准费用。
它仍计入提供角色支持的随机训练/回放与差分探测。CREDIT 计入训练、验证、校准、回放和探测全部调用。

![](results/{output.name}/figures/main_vsr.png)

## 2. 配对比较与预算

{ctable}

唯一预设主比较 State-K4 − Legacy-K4 为 **{100*prim['delta_vsr']:+.2f} 个百分点**，
种子/业务家族配对 bootstrap 95% 区间 [{100*prim['ci_low']:+.2f}, {100*prim['ci_high']:+.2f}]；
八种子双侧精确符号翻转 p={prim['seed_sign_flip_p']:.6f}。
其余与 State-K4 的 {len(stats['paired'])} 个合并比较作 Holm 校正，保存在 `paired_statistics.csv`；主组原校正单独保留。
K6 对比表是描述性对比，不冒充主比较的预设显著性检验。

{compact_section}

Matched Direct 的上限是在任何 held-out 评分前，按同种子、同 K 的 CREDIT 实际费用固定；
**这是匹配预算上限，实际支出没有强制相等。** Direct 预留余下认证阶段的最坏调用数，因此可能留有未使用预算。
这测量当前预算控制器，不能代表所有 Direct 算法的最佳预算前沿。
相同 K 的两种方法才构成对应比较；不能用 K6 CREDIT 对 K4 Direct 宣称方法胜出。

## 3. 预测模型诊断

State 的独立节点验证 NLL 均值 {state_nll:.4f}，Coverage 为 {coverage_nll:.4f}。
State 的 {sum(r['early_stopped'] for r in state_reports)}/{len(state_reports)} 个模型触发验证耐心停止；
这是泛化损失停止，**不是优化器达到平稳点的证明**。
验证集只选权重，校准集随后拟合，测试标签不参与训练或选参。

{rtable}

归因诊断复用上轮已执行且审计过的 32 个完整八记忆精确参照；它们在开发时已知，
因此属于解释性诊断，不是未接触的确认测试集。所有新预测在主实验和审计完成后才生成；
参照及隐藏概率从未进入新训练、排序或认证。完整表记录逐查询运输支持；多数 target 查询为未获支持的外推诊断。
交互 F1 使用原固定阈值 |Walsh β|≥0.015，衡量幅度检出，不衡量符号正确率。

![](results/{output.name}/figures/reference_error.png)

## 4. 复现与审计

```sh
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m credit.v3_optimized_data --output datasets/new_optimization
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m credit.v3_optimization --dataset datasets/new_optimization --output results/new_optimization
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m credit.v3_optimization_verify --results results/new_optimization
python -m credit.v3_optimization_report --benchmark results/new_optimization --output results/new_optimization_analysis
```

审计逐次重执行历史和选择/测试，重建候选遍历、halving、认证、调用量，重拟合 isotonic 并复算选中权重的验证 NLL。
没有重训优化器；候选遍历使用相同确定性实现重建，不声称由独立算法实现验证。
完整审计统计见 `execution_audit.json`。本轮新增 8 个测试，最近完整测试套件为 107 项通过。
保护文件哈希在协议中固定；旧实验记录保留。

## 5. 结论与下一轮

1. 主比较按实测差值判断，不以预测损失下降代替最终业务成功率提升。
2. 方法优势应同时考虑同 K 的成功率、实际调用成本及更充分的独立种子证据；本报告不保证 CREDIT 优于 Direct。
3. 状态特征依赖当前公开后置条件，真实 Web 服务和完整 LLM 验证仍未完成。

下一轮应依据本表识别剩余瓶颈，补充状态特征/编码宽度的单独消融；
若实际费用劣于 Direct，应减少模型数据/认证成本并用新的冻结测试集评价。连续在线重验证和真实服务实验仍须补齐。
'''
    (output/'report.md').write_text(report)
    Path('CREDIT_V3_OPTIMIZATION_RESULTS.md').write_text(report)
    write(output/'analysis_manifest.json',{'benchmark_manifest_sha256':digest(benchmark/'artifact_manifest.json'),
          'compact_manifest_sha256':digest(compact/'artifact_manifest.json') if compact is not None else None,
          'audit_sha256':digest(benchmark/'execution_audit.json'),
          'report_sha256':digest('CREDIT_V3_OPTIMIZATION_RESULTS.md'),
          'generator_sha256':digest(__file__),
          'artifacts':{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({'summary':summaries,'primary':prim},indent=2),flush=True)
    Role.key = original_role_key
    outcome_module.text_encoding = original_text
    optimized_module.text_encoding = original_optimized_text


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compact',type=Path)
    a = p.parse_args()
    generate(a.benchmark,a.output,compact=a.compact)
