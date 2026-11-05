"""Export measured v3 tables, standalone figures and a Chinese results report."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .v3.experiment import digest, write
from .v3_benchmark import MAIN, VARIANTS
from .paired_statistics import summarize


def read(path): return json.loads(path.read_text())


def csv_table(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def generate(benchmark,statistics,reference,output,direct_only):
    report_source_hash=digest(__file__)
    if output.exists():raise FileExistsError(output)
    for root,audit in ((benchmark,'execution_audit.json'),(statistics,'statistics_audit.json'),(reference,'execution_audit.json'),(direct_only,'execution_audit.json')):
        if not read(root/audit)['verified']:raise ValueError('audit required')
    output.mkdir(parents=True);(output/'figures').mkdir()
    rows=read(benchmark/'raw_test_results.json')+read(direct_only/'raw_test_results.json')
    proto=read(benchmark/'frozen_protocol.json');seeds=proto['seeds']
    if len(seeds)!=5:raise ValueError('this report requires five independent seeds')
    sp=read(statistics/'frozen_protocol.json')
    if sp['factorial_reps']!=2000 or sp['cert_reps']!=1000:raise ValueError('full-size simulations required')
    main_methods=list(MAIN[:-1])+['Direct-Sandbox-Only','CREDIT']
    t,p=summarize([r for r in rows if r['method'] in main_methods],'CREDIT',repeats=4000,random_seed=2991)
    main={'main':t,'paired':p};write(output/'main_statistics.json',main)
    ablation=read(benchmark/'ablation_statistics.json')
    stats={r['method']:r for r in main['main']+ablation['main']};methods=main_methods+list(VARIANTS)
    table=[];certificate_rows=[];fit_rows=[];gate_rows=[]
    for seed in seeds:
        sd=benchmark/f'seed_{seed}'
        for evidence in read(sd/'transfer_levels.json'):
            version=evidence['version'];_,family,role,epoch=version.split(':')
            planted=epoch=='target' and ((family in ('books','inventory') and role=='submit') or (family=='travel' and role=='lookup'))
            gate_rows.append({'seed':seed,'version':version,'level':evidence['level'],'changed_fixtures':evidence['changed'],
                             'fixtures':evidence['fixtures'],'lower':evidence['lower'],'upper':evidence['upper'],
                             'planted_divergence':planted,'false_V':planted and evidence['level']=='V'})
        for p in sd.rglob('training_reports.json'):
            for i,r in enumerate(read(p)):
                fit_rows.append({'seed':seed,'variant':str(p.parent.relative_to(sd)),'member':i,**r})
        for p in sd.glob('*/switch_decision.json'):
            decision=read(p)
            for i,c in enumerate(decision.get('certificates',[])):
                certificate_rows.append({'seed':seed,'method':p.parent.name,'finalist':i,'passed':c['passed'],
                    'delta':c['aggregate_difference'],'lower':c['aggregate_lower'],
                    'cost_threshold':c['cost_threshold'],'worst_guard':min(c['stratum_lower'].values()),
                    'nonzero_lambda_strata':sum(e['weight']>0 for e in c['estimates'].values()),
                    'aggregate_pass':c['aggregate_lower']>c['cost_threshold'],
                    'guards_pass':all(v>-.05 for v in c['stratum_lower'].values())})
        for p in (direct_only/f'seed_{seed}').glob('*/switch_decision.json'):
            for i,c in enumerate(read(p).get('certificates',[])):
                certificate_rows.append({'seed':seed,'method':p.parent.name,'finalist':i,'passed':c['passed'],
                    'delta':c['aggregate_difference'],'lower':c['aggregate_lower'],'cost_threshold':c['cost_threshold'],
                    'worst_guard':min(c['stratum_lower'].values()),'nonzero_lambda_strata':sum(e['weight']>0 for e in c['estimates'].values()),
                    'aggregate_pass':c['aggregate_lower']>c['cost_threshold'],'guards_pass':all(v>-.05 for v in c['stratum_lower'].values())})
    for method in methods:
        r=[x for x in rows if x['method']==method];s=stats[method]
        mean=lambda key:float(np.mean([x[key] for x in r]))
        seed_rates=[np.mean([x['success'] for x in r if x['seed']==seed]) for seed in seeds]
        root=direct_only if method=='Direct-Sandbox-Only' else benchmark
        cost=[read(root/f'seed_{seed}'/'method_costs.json')[method]['selection_api_calls_including_common'] for seed in seeds]
        decision=[read(root/f'seed_{seed}'/method/'switch_decision.json') for seed in seeds if (root/f'seed_{seed}'/method/'switch_decision.json').exists()]
        table.append({'method':method,'vsr':s['vsr'],'seed_std':float(np.std(seed_rates,ddof=1)),
            'ci_low':s['ci_low'],'ci_high':s['ci_high'],
            'source_vsr':float(np.mean([x['success'] for x in r if x['epoch']=='source'])),
            'target_vsr':float(np.mean([x['success'] for x in r if x['epoch']=='target'])),
            'mean_extra_api_calls':float(np.mean(cost)),'min_extra_api_calls':min(cost),'max_extra_api_calls':max(cost),
            'mean_test_api_calls':mean('api_calls'),'mean_bank_tokens':mean('bank_tokens'),
            'mean_exposure_tokens':mean('exposure_tokens'),'mean_stale_exposures':mean('stale_exposures'),
            'mean_test_latency_ms':1000*mean('latency_seconds'),
            'p50_test_latency_ms':1000*float(np.quantile([x['latency_seconds'] for x in r],.5)),
            'p95_test_latency_ms':1000*float(np.quantile([x['latency_seconds'] for x in r],.95)),
            'switches':sum(d.get('switched',False) for d in decision),'measured_seeds':len(seeds),
            'episodes':len(r)})
    csv_table(output/'method_statistics.csv',table);csv_table(output/'certificates.csv',certificate_rows)
    csv_table(output/'gate_diagnostics.csv',gate_rows)
    write(output/'training_diagnostics.json',fit_rows)
    # Paired equality diagnostics explain tied results without changing policies.
    lookup={(r['seed'],r['task'],r['method']):r for r in rows}
    equality=[]
    for a,b in [('FIFO','Recency'),('CREDIT','Without-interactions'),('CREDIT','Without-certification'),('CREDIT','C120-K4')]:
        pairs=[(r,lookup[r['seed'],r['task'],b]) for r in rows if r['method']==a]
        equality.append({'first':a,'second':b,'identical_exposure_fraction':np.mean([x['exposure']==y['exposure'] for x,y in pairs]),
                         'identical_outcome_fraction':np.mean([x['success']==y['success'] for x,y in pairs])})
    write(output/'equality_diagnostics.json',equality)
    plt.rcParams.update({'font.size':10,'savefig.dpi':160})
    def save(fig,name):
        fig.savefig(output/'figures'/f'{name}.png',bbox_inches='tight')
        fig.savefig(output/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    mt=table[:len(main_methods)];positions=np.arange(len(mt))
    display=lambda name:name+(' (PPI)' if name=='Direct-Sandbox' else '')
    fig,ax=plt.subplots(figsize=(8,4.6))
    v=np.array([r['vsr'] for r in mt])*100
    ax.barh(positions,v,xerr=np.array([[r['vsr']-r['ci_low'] for r in mt],[r['ci_high']-r['vsr'] for r in mt]])*100,
            color=['#2b6cb0' if r['method']=='CREDIT' else '#91a4b8' for r in mt],capsize=3)
    ax.set_yticks(positions,[display(r['method']) for r in mt]);ax.invert_yaxis();ax.set_xlabel('Verified success rate (%)')
    ax.set_title('Frozen synthetic workflows: five seeds, clustered 95% bootstrap intervals');save(fig,'main_vsr')
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(positions-.18,[100*r['source_vsr'] for r in mt],width=.36,label='Source version')
    ax.bar(positions+.18,[100*r['target_vsr'] for r in mt],width=.36,label='Injected target drift')
    ax.set_xticks(positions,[display(r['method']) for r in mt],rotation=32,ha='right');ax.set_ylabel('Verified success (%)');ax.legend();save(fig,'source_target')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for r in mt:
        if r['method']=='FIFO':continue  # Its cost and VSR coincide with Recency.
        offset={'FIFO':(4,-12),'Recency':(4,5),'Model-MCShapley':(4,-16),'Factorial-Banzhaf':(4,5),
                'Direct-Sandbox':(-8,7),'Direct-Sandbox-Only':(-8,-14)}.get(r['method'],(4,3))
        name='FIFO / Recency' if r['method']=='Recency' else display(r['method'])
        ax.scatter(r['mean_extra_api_calls'],100*r['vsr']);ax.annotate(name,(r['mean_extra_api_calls'],100*r['vsr']),xytext=offset,textcoords='offset points',fontsize=8,ha='right' if r['method'].startswith('Direct-') else 'left')
    ax.margins(x=.07,y=.16)
    ax.set_xlabel('Mean extra physical API calls (shared history charged)');ax.set_ylabel('Verified success (%)');save(fig,'cost_success')
    factorial=read(statistics/'factorial_summary.json');certsim=read(statistics/'certification_summary.json')
    csv_table(output/'factorial_simulations.csv',factorial);csv_table(output/'certification_simulations.csv',certsim)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,noise in zip(axes,['independent','CRN']):
        for method in ['Randomized-fraction','Random-OLS','OAT']:
            rr=[r for r in factorial if r['noise']==noise and r['method']==method]
            ax.plot([r['budget'] for r in rr],[100*r['main_rmse'] for r in rr],marker='o',label=method)
        ax.set_title(noise);ax.set_xlabel('Outcome evaluations');ax.set_ylabel('Main-effect RMSE ×100');ax.set_yscale('log');ax.legend(fontsize=8)
    save(fig,'factorial_rmse')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,independent,gain in zip(axes,[False,True],[.03,.05]):
        for quality in ['none','good','noisy','biased']:
            rr=[r for r in certsim if r['independent']==independent and r['gain']==gain and r['quality']==quality and not r['harmful_stratum']]
            ax.plot([r['n'] for r in rr],[r['switch_rate'] for r in rr],marker='o',label=quality)
        ax.set_title(f"{'Independent' if independent else 'CRN'}; gain={gain:.0%}");ax.set_xlabel('Labels per stratum');ax.set_ylabel('Switch probability');ax.set_ylim(0,1);ax.legend(fontsize=8)
    save(fig,'certification_power')
    ref=read(reference/'summary.json');errors=read(reference/'model_errors.json');regret=read(reference/'reference_bank_regret.json')
    csv_table(output/'attribution_errors.csv',errors);csv_table(output/'reference_bank_regret.csv',regret)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    depths=sorted({r['depth'] for r in errors})
    ready_mae=float(np.mean([r['utility_mae'] for r in errors if r['depth']==1 and r['stratum'].endswith('/source')]))
    deep_mae=float(np.mean([r['utility_mae'] for r in errors if r['depth']==9 and r['stratum'].endswith('/source')]))
    attribution_table=[]
    for epoch in ('source','target'):
        for depth in depths:
            rr=[r for r in errors if r['depth']==depth and r['stratum'].endswith('/'+epoch)]
            denom=sum(r['pair_truth_positive']+r['pair_predicted_positive'] for r in rr)
            attribution_table.append([epoch,depth]+[f"{np.mean([r[k] for r in rr]):.4f}" for k in ('utility_mae','banzhaf_mae','shapley_mae')]+
                                     ['N/A' if denom==0 else f"{2*sum(r['pair_true_positive'] for r in rr)/denom:.3f}"])
    for ax,key,label in zip(axes,['utility_mae','banzhaf_mae'],['Coalition utility MAE','Banzhaf MAE']):
        for epoch in ['source','target']:
            values=[[r[key] for r in errors if r['depth']==d and r['stratum'].endswith('/'+epoch)] for d in depths]
            ax.errorbar(depths,[np.mean(v) for v in values],yerr=[np.std(v) for v in values],marker='o',label=epoch)
        ax.set_xlabel('Required workflow depth');ax.set_ylabel(label);ax.legend()
    save(fig,'composition_error')
    main_rows=[[r['method']+(' (PPI)' if r['method']=='Direct-Sandbox' else ''),f"{100*r['vsr']:.2f} ± {100*r['seed_std']:.2f}",f"[{100*r['ci_low']:.2f}, {100*r['ci_high']:.2f}]",
                f"{100*r['source_vsr']:.2f}",f"{100*r['target_vsr']:.2f}",f"{r['mean_extra_api_calls']:,.0f}",f"{r['mean_stale_exposures']:.3f}"] for r in mt]
    ab_rows=[[r['method'],f"{100*r['vsr']:.2f}",f"{100*(r['vsr']-mt[-1]['vsr']):+.2f}",f"{r['mean_extra_api_calls']:,.0f}",f"{r['mean_bank_tokens']:.1f}",f"{r['switches']}/5"] for r in table[len(main_methods):]]
    primary=next(r for r in main['paired'] if r['baseline']=='Direct-Sandbox-Only')
    original_primary=next(r for r in main['paired'] if r['baseline']=='Direct-Sandbox')
    nomem=next(r for r in main['paired'] if r['baseline']=='NoMem')
    null=[r for r in certsim if r['gain']==0]
    cert_table=[]
    for n in (50,100,200,400):
        rates=[next(r['switch_rate'] for r in certsim if r['n']==n and r['gain']==g and r['quality']=='none' and not r['independent'] and not r['harmful_stratum']) for g in (0.,.01,.02,.03)]
        harmful=next(r['switch_rate'] for r in certsim if r['n']==n and r['harmful_stratum'])
        cert_table.append([n]+[f'{x:.3f}' for x in rates]+[f'{harmful:.3f}'])
    converged=sum(r['stopping_criterion_met'] for r in fit_rows)
    nonzero=sum(r['nonzero_lambda_strata'] for r in certificate_rows)
    guard_blocks=sum(r['aggregate_pass'] and not r['guards_pass'] for r in certificate_rows)
    divergent=sum(r['planted_divergence'] for r in gate_rows);false_V=sum(r['false_V'] for r in gate_rows)
    document=f'''# CREDIT v3 实验结果（受控合成数据）

## 数据、协议与验收

最新依据为 `CREDIT_WWW2027_v3.tex`。用户提供的 EMP/HSC/API 数据缺少可恢复状态、完整反事实执行和业务验收，保留为原始离线数据。新数据保存在 `datasets/credit_v3_synthetic_v1/`：188,320 个完整任务上下文，五个预先固定的种子 {seeds}，四类业务、源/目标两个服务版本。生成器沿用固定模拟器，没有按实验输赢改成功标签。训练、校准、筛选、pilot、认证、测试的任务 ID 和执行噪声互不重用。

训练/校准来自 4/5 步旧拓扑；测试来自 1/5/6 步未训练拓扑。原实验 8 个主方法和 11 个变体，另有纠正后的 1 个完全无模型 Direct 方法，每方法每种子测试 2,048 个任务，共 **{len(rows):,} 个实际测试 episode**。原 19 个方法统一先冻结全部库再评分；纠正协议在检查任何未来结果前确定，五个纠正库也在本轮纠正测试前统一冻结。成功要求所有必要节点后置条件通过且最终提交状态成立，未知验收计失败。每策略额外调用上限 150,000，使用共同历史的策略分摊完整历史逻辑成本；这是相同上限，实际支出见表，不能称为相同实际调用量。

模拟器使用冻结随机策略和本地状态 API；模型使用词汇哈希特征和共享神经双头。token 是词汇计数，漂移是注入的契约/QoS 变化，**没有真实供应商调用或完整 LLM 智能体评估**。Model-MCShapley 是模型查询适配基线；Heldout-payments 只留出训练业务家族；角色共享消融用图条件编码替代共享角色编码。

**基线纠正：** 原 `Direct-Sandbox` 的筛选完全使用真实执行，认证仍使用神经 PPI，因此不能称为完全移除模型。原始数据保留并在表中标为 PPI。新增 `Direct-Sandbox-Only` 的模型列表为空，预测及 PPI 权重全部为零；其筛选、halving、pilot、认证与测试均重新实际执行，源码/协议/轨迹位于 `results/credit_v3_direct_only_v1/`。本报告用这个纠正基线检查“去掉模型”；原预设比较单独保留，避免覆盖结果。

## 1. 主比较原始汇总

VSR 单位为 %；± 为五种子样本标准差，区间为种子/业务家族配对 bootstrap（4,000 次）。每方法 10,240 个测试 episode。

{md_table(['方法','VSR ± SD','95% 区间','源版本','目标版本','平均额外调用','平均失效记忆曝光'],main_rows)}

纠正比较 CREDIT − Direct-Sandbox-Only 为 **{100*primary['delta_vsr']:+.2f} 个百分点**，配对 95% 区间 [{100*primary['ci_low']:.2f}, {100*primary['ci_high']:.2f}]；种子符号翻转 p={primary['seed_sign_flip_p']:.4f}，Holm p={primary['holm_p']:.4f}。原预设 CREDIT − Direct-Sandbox(PPI) 差为 {100*original_primary['delta_vsr']:+.2f} 点。CREDIT − NoMem 为 {100*nomem['delta_vsr']:+.2f} 个百分点。只有五个独立种子，双侧精确检验最小 p=0.0625，不能声称 p<0.05。纠正后主表的 Holm 校正覆盖八个比较；原七比较结果保留在原目录。比较优劣按实测报告，未调数据追求优势。

![](results/credit_v3_analysis_v1/figures/main_vsr.png)

## 2. 消融、容量与曝光深度

Δ 为变体 − CREDIT（百分点）；切换次数包含无认证变体的直接采用。

{md_table(['变体','VSR %','Δ','平均额外调用','平均库 tokens','切换'],ab_rows)}

无交互、无认证等变体若选择同一库，测试结果会相同；逐任务的实际曝光和结果一致率保存在 `equality_diagnostics.json`，不能仅凭相同数字判定实现错误，也不能把相同结果解释为该机制已经有效。所有 {len(fit_rows)} 个模型按固定 200 epoch 上限训练，{converged} 个满足损失容差停止条件。认证中正 PPI 权重累计出现于 {nonzero} 个 finalist/种子/层组合；为 0 的组合使用纯执行估计。

差分探测的 160 个角色/版本证据单元中，{divergent} 个具有植入变化，错误赋予 V 的数量为 {false_V}；完整 CP 区间与 U/D 判定见 `gate_diagnostics.csv`。{guard_blocks} 个候选虽通过净总体改善界，仍被逐层非劣性 guard 阻止。源/目标按版本单独汇总；这里没有自然供应商变化或连续多轮切换的回归率测量。

## 3. 统计仿真

因子估计：K=12、23 个纳入效应、32 行可容许 fraction，预算 256/512/1024，独立噪声和 CRN，每项 2,000 次重复。下表 RMSE×100；完整误差、真值及估计保存为 NPZ。

{md_table(['预算','噪声','方法','主效应 RMSE×100','交互 RMSE×100'],[[r['budget'],r['noise'],r['method'],f"{100*r['main_rmse']:.3f}",f"{100*r['pair_rmse']:.3f}"] for r in factorial])}

认证：55 个场景各 1,000 次重复，包含 +0/+1/+2/+3 点改善、好/噪声/偏差代理、独立噪声和单层伤害。零均值改善场景的切换率为 {min(r['switch_rate'] for r in null):.1%}–{max(r['switch_rate'] for r in null):.1%}。使用 α1=α2=0.05 的渐近正态规则，每轮误差分配合计 0.10；不是有限样本或终身 0.05 保证。新仿真用明确公布的生成式，不能冒充论文中缺少原始脚本的旧数值的逐格复现。

{md_table(['每层标签数','Δ=0','Δ=+1点','Δ=+2点','Δ=+3点','一层−10点'],cert_table)}

200 标签的零均值场景实测错误切换率为 5.1%；50/100 标签下单层伤害场景分别有 0.6%/0.2% 的切换，200/400 标签下为 0。这里前两个小样本场景仅用于功效诊断，主实验仍强制每层至少 200 个新标签。1000 次重复的估计有抽样波动；不能把 0 次观察到的错误解释为理论零风险。

![](results/credit_v3_analysis_v1/figures/certification_power.png)

## 4. 归因、组合深度和负对照

`datasets/credit_v3_reference_v1/` 冻结 32 个参考案例：四业务 × 双版本 × 1/4/6/9 步。每案例枚举八记忆的全部 256 组合，并执行 32 次配对随机重放，共 **{ref['actual_reference_episodes']:,} 个 episode、{ref['reference_api_calls']:,} 次调用**；A/A 与 noop 另有 {ref['negative_control_episodes']:,} 个 episode、{ref['negative_control_api_calls']:,} 次调用。负对照全部一致。合同检查器与隐藏有效请求支持比较，{ref['nonambiguous_labels_checked_against_private_support']:,} 个非歧义标签零不一致；这检验固定可编程检查器，不能推广为自然语言检查准确率。歧义标签单独记录，不虚报召回率。

私有内核仅在主实验全部冻结并完成测试后用于评价，未进入选库或训练。通过动态状态概率求精确组合效用，Walsh 与经典子集 Shapley 两式互相核对。实际 MC 效用与内核真值平均绝对误差为 {ref['mc_utility_mae']:.4f}。模型按深度、家族、版本的 utility/Banzhaf/Shapley MAE 与固定阈值交互 F1 见 `attribution_errors.csv`。参考域穷举银行遗憾见 `reference_bank_regret.csv`，它使用参考深度的等权混合，不是主测试域遗憾。

{md_table(['版本','深度','效用 MAE','Banzhaf MAE','Shapley MAE','交互 micro F1'],attribution_table)}

交互 F1 将纳入角色作用域/边的 |Walsh β|≥0.015 视为阳性；它检验交互存在性，未将符号正确性计入 F1。没有真实或预测阳性时显示 N/A，归因误差仍保留符号。目标版本误差包含不具备运输支持的诊断查询，这些预测不会用于认证控制变量。

模型训练采用 K1=4 加均匀 K2=2 的六槽子池，参考值则枚举完整八记忆池，因此七/八条同时曝光的模型查询属于训练覆盖外的诊断。随深度增加，真实成功率会下降；绝对 MAE 变小不能单独证明组合假设成立。
组合误差图的误差线为相应种子/业务案例的标准差，不是置信区间。

{md_table(['设计（重用真实执行组合表）','每重复逻辑 episode','主效应 RMSE','交互 RMSE'],[[r['method'],r['logical_episodes_per_replication'],f"{r['main_rmse']:.4f}",f"{r['pair_rmse']:.4f}"] for r in read(reference/'design_statistics.json')])}

设计诊断从同一份真实组合参照中取对比，保持 128 个逻辑 episode 上限；它不是分别独立采集的 128 次 API 调用。不同曝光会产生不同重试调用数，本诊断未宣称等实际 API 支出。

![](results/credit_v3_analysis_v1/figures/composition_error.png)

## 5. 核验与结论

主实验独立重执行和认证计算审计见 `results/credit_v3_benchmark_v1/execution_audit.json`；统计仿真全部决策重算见 `results/credit_v3_statistics_v1/statistics_audit.json`。源码、任务、预测、选择库和结果均有 SHA256。主实验审计不重新训练优化器，也未独立重建整个局部搜索遍历；它重算随机设计、校准、真实 halving、pilot/PPI、逐层 guards 和最终采用决定。测试延迟仅包含单个执行器调用，不能用于神经训练/筛选端到端速度排序。

当前冻结选择器使用 Python 集合遍历候选，原运行没有固定 hash salt；当 Direct 的预算提前耗尽时，候选访问次序可能影响重新选库。冻结轨迹可以精确重放，本次模型零代理纠正的最终库与原 Direct 在五个种子上全部一致，但实际筛选调用量略有不同。下面的新采集命令显式设置 `PYTHONHASHSEED=0` 固定后续进程顺序；它不能补记原运行未保存的 hash salt，重新采集不承诺逐库逐字复现原选择。此限制未被记忆库测试结果掩盖。

归因参考的全部执行、状态哈希、负对照及模型误差也经重放审计，见 `results/credit_v3_reference_v1/execution_audit.json`；参考域穷举库遗憾与设计诊断汇总做了哈希核对，未在该审计中独立重建。

### 已测量与仍未测量

已完成本报告列出的受控五种子比较、11 项变体、无模型 Direct 纠正、全尺寸统计仿真及穷举归因。真实版本快照/自然漂移、完整 LLM actor、v3 的 CMP/CAR 移植基线、无随机曝光训练消融、OAT 替代训练的端到端消融、主动重验证与其关闭对照、漂移强度/不可重放比例完整网格仍未测量。废弃基线按用户指示排除。因此论文全部实验尚未完成，旧论文和旧结果不替代 v3 的这些缺项。

### 关键发现与后续实验

1. 主比较差值为 {100*primary['delta_vsr']:+.2f} 点，区间见上表；结论需同时考虑额外调用支出，不能把仅击败 NoMem 当作击败强基线。下一步在预先固定的实际调用配额下绘制 CREDIT 与 Direct-Sandbox 的成本曲线。
2. 相同变体结果应结合冻结库/曝光一致率解释。若关闭交互后选库未变，本域没有展示交互项的保留收益。下一步扩大独立任务机制和记忆池，而不按本次赢家修改标签。
3. 代理训练有 {len(fit_rows)-converged} 个模型触及 epoch 上限；源版本单步 ready-submit 的效用 MAE 为 {ready_mae:.4f}，深九步为 {deep_mae:.4f}，显示特殊初态和深组合的预测问题。下一步纳入可观测前置状态，固定优化器停止准则，再用新的未见任务检验组合误差与 PPI 权重，不能反复使用本次测试集调参。

## 复现

```sh
python -m credit.v3_data --output datasets/credit_v3_synthetic_reproduction
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m credit.v3_benchmark --dataset datasets/credit_v3_synthetic_reproduction --output results/credit_v3_benchmark_reproduction
python -m credit.v3_benchmark_verify --results results/credit_v3_benchmark_reproduction
PYTHONHASHSEED=0 python -m credit.v3_direct_only --original results/credit_v3_benchmark_reproduction --output results/credit_v3_direct_reproduction
python -m credit.v3_benchmark_verify --results results/credit_v3_benchmark_reproduction --direct-only results/credit_v3_direct_reproduction
python -m credit.v3_statistical --output results/credit_v3_statistics_reproduction
python -m credit.v3_statistics_verify --results results/credit_v3_statistics_reproduction
python -m credit.v3_reference --freeze-only --dataset datasets/credit_v3_reference_reproduction
python -m credit.v3_reference --dataset datasets/credit_v3_reference_reproduction --benchmark results/credit_v3_benchmark_reproduction --output results/credit_v3_reference_reproduction
python -m pytest -q
```
'''
    document=document.replace('results/credit_v3_analysis_v1/',str(output).rstrip('/')+'/')
    if digest(__file__)!=report_source_hash:raise ValueError('report source changed during generation')
    Path('CREDIT_V3_EXPERIMENT_RESULTS.md').write_text(document)
    (output/'report.md').write_text(document)
    write(output/'analysis_manifest.json',{'sources':{str(p):digest(p) for p in [benchmark/'artifact_manifest.json',statistics/'artifact_manifest.json',reference/'artifact_manifest.json',direct_only/'artifact_manifest.json',Path(__file__),Path('CREDIT_WWW2027_v3.tex')]},
         'artifacts':{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()},'report_sha256':digest('CREDIT_V3_EXPERIMENT_RESULTS.md')})
    print(json.dumps({'main':mt,'primary_comparison':primary},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,default=Path('results/credit_v3_benchmark_v1'))
    p.add_argument('--statistics',type=Path,default=Path('results/credit_v3_statistics_v1'))
    p.add_argument('--reference',type=Path,default=Path('results/credit_v3_reference_v1'))
    p.add_argument('--output',type=Path,default=Path('results/credit_v3_analysis_v5'))
    p.add_argument('--direct-only',type=Path,default=Path('results/credit_v3_direct_only_v1'))
    a=p.parse_args();generate(a.benchmark,a.statistics,a.reference,a.output,a.direct_only)
