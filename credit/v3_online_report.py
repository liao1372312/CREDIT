"""Generate audited chronological and priority tables and exportable plots."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .v3.experiment import digest, write
from .v3_optimization_verify import read


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def report(source,priority,output):
    if output.exists():
        raise FileExistsError('use a fresh analysis directory')
    audit = read(source.with_name(source.name+'_audit.json'))
    priority_audit = read(priority.with_name(priority.name+'_audit.json'))
    if not audit['verified'] or not priority_audit['verified']:
        raise ValueError('unverified experiment')
    if digest(source/'artifact_manifest.json') != audit['benchmark_manifest_sha256']:
        raise ValueError('online audit manifest anchor changed')
    if digest(priority/'manifest.json') != priority_audit['manifest_sha256']:
        raise ValueError('priority audit manifest anchor changed')
    for root,manifest in [(source,'artifact_manifest.json'),(priority,'manifest.json')]:
        for name,h in read(root/manifest).items():
            if digest(root/name)!=h:
                raise ValueError('artifact changed: '+name)
    output.mkdir()
    rows=read(source/'raw_test_results.json')
    methods=sorted({r['method'] for r in rows})
    seeds=sorted({r['seed'] for r in rows})
    cells=[]
    for method in methods:
        for cycle in range(3):
            for epoch in ('all','source','target'):
                seed_rates=[np.mean([r['success'] for r in rows if r['method']==method and r['seed']==seed
                            and r['cycle']==cycle and (epoch=='all' or r['epoch']==epoch)]) for seed in seeds]
                cells.append({'method':method,'cycle':cycle,'snapshot':epoch,'vsr':float(np.mean(seed_rates)),
                              'seed_std':float(np.std(seed_rates,ddof=1))})
    expenses=sum([read(source/f'costs_cycle_{i}.json') for i in range(3)],[])
    cost_by_method=defaultdict(list)
    for method in methods:
        if method=='NoMem':
            continue
        for seed in seeds:
            cost_by_method[method].append(sum(r['extra_api_calls'] for r in expenses if r['seed']==seed and r['method']==method))
    costs=[{'method':m,'mean_three_cycle_extra_calls':float(np.mean(v)),
            'seed_std':float(np.std(v,ddof=1)),'total_extra_calls':int(sum(v))} for m,v in cost_by_method.items()]
    detail=[]
    for seed in seeds:
        for cycle in range(3):
            for method in methods:
                md=source/f'seed_{seed}'/f'cycle_{cycle}'/method
                if not md.exists():
                    continue
                decision=read(md/'selection/switch_decision.json')
                admissions=read(md/'adaptation/admissions.json')
                schedule=read(md/'adaptation/schedule.json')
                store=read(md/'store_after_test.json')
                factorial_actions=sum(s.get('kind')=='factorial' or s.get('chosen',{}).get('kind')=='factorial'
                                      for s in schedule.get('steps',[]))
                differential_actions=len(read(md/'adaptation/probe_decisions.json'))
                detail.append({'seed':seed,'cycle':cycle,'method':method,
                    'certificates':len(decision['certificates']),
                    'successful_certificates':sum(bool(c['passed']) for c in decision['certificates']),
                    'admitted_cells':sum(bool(a['admitted']) for a in admissions),
                    'storage_counts':dict(Counter(e['state'] for e in store['entries'])),
                    'archived_memories':len(read(md/'archive_bounds.json')),
                    'switched':bool(decision['switched']),
                    'factorial_actions':factorial_actions,'differential_actions':differential_actions,
                    'executed_role_actions':factorial_actions+differential_actions})
    stats=read(source/'statistics.json')
    primary=next(r for r in stats['paired'] if r['baseline']=='CREDIT-Sparse')
    secondary=read(source/'secondary_statistics.json')
    priority_rows=read(priority/'summary.json');priority_cells=[]
    for method in sorted({r['method'] for r in priority_rows}):
        for budget in sorted({r['budget'] for r in priority_rows}):
            group=[r for r in priority_rows if r['method']==method and r['budget']==budget]
            priority_cells.append({'method':method,'budget':budget,'rmse':float(np.mean([r['rmse'] for r in group])),
                'seed_std':float(np.std([r['rmse'] for r in group],ddof=1)),
                'actual_calls':float(np.mean([r['actual_calls'] for r in group])),
                'selected_contexts':float(np.mean([r['contexts_selected'] for r in group]))})
    write(output/'tables.json',{'cycles':cells,'costs':costs,'mechanisms':detail,'priority':priority_cells,
                               'primary':primary,'secondary':secondary})
    for name,values in [('cycles',cells),('costs',costs),('priority',priority_cells)]:
        with (output/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    # Interrupted work is preserved; these are engineering overhead, outside
    # the algorithm budget of the completed, frozen continuation.
    prefix=Path(read(source/'frozen_protocol.json')['prefix_directory']);overhead=[]
    for sd in sorted(prefix.glob('seed_*/cycle_1/*')):
        sp,ap=sd/'selection/costs.json',sd/'adaptation/costs.json'
        if sp.exists():
            value=read(sp)['selection_api_calls_including_common'];scope='completed selection ledger;includes adaptation'
        elif ap.exists():
            value=read(ap)['total_with_fixtures'];scope='completed adaptation ledger only'
        else:
            continue
        overhead.append({'path':str(sd),'calls':value,'scope':scope})
    write(output/'interrupted_engineering_overhead.json',{'completed_ledger_calls':sum(r['calls'] for r in overhead),
          'records':overhead,'limits':'Lower bound if an interrupted unfinished ledger made additional calls. CPU and model fitting wall time were not instrumented.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for method in methods:
        values=[r for r in cells if r['method']==method and r['snapshot']=='all']
        axes[0].plot([r['cycle'] for r in values],[100*r['vsr'] for r in values],marker='o',label=method)
    axes[0].set(xlabel='Chronological block',ylabel='Verified success (%)',xticks=[0,1,2])
    axes[0].legend(fontsize=7)
    for method in sorted({r['method'] for r in priority_cells}):
        values=[r for r in priority_cells if r['method']==method]
        axes[1].errorbar([r['budget'] for r in values],[r['rmse'] for r in values],
                        yerr=[r['seed_std'] for r in values],marker='o',label=method)
    axes[1].set(xlabel='Physical probe call cap',ylabel='Coefficient RMSE (MC reference)')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(output/('chronology_priority.'+ext),dpi=180)
    plt.close(fig)
    main_rows=[]
    for m in stats['main']:
        cost=next((r['mean_three_cycle_extra_calls'] for r in costs if r['method']==m['method']),0)
        main_rows.append([m['method'],f"{100*m['vsr']:.3f}%",f"[{100*m['ci_low']:.3f}, {100*m['ci_high']:.3f}]",f'{cost:,.0f}'])
    text='# CREDIT v3 连续更新与上下文优先级实验结果\n\n'
    text+='## 1. 实验范围与原始证据\n\n'
    text+=f"8 个新种子、3 个评分块、7 个设置、每块每层 128 任务，共 {len(rows):,} 次真实本地执行器测试。四家族、source/target 两固定快照，九条记忆，三个公开模板均已在适配中出现。数据为自生成任务，用户数据未修改。\n\n"
    text+='主比较在运行前声明为 Covered−Sparse；它同时改变样本量、均衡 K2 和最低支持数，不能把差异单独归因于抽样设计。次级四项比较使用种子配对符号翻转检验并做 Holm 校正。bootstrap 区间重采样种子及家族；只有八个独立种子，不能将全部任务当作独立重复。\n\n'
    text+='## 2. 三周期总体结果\n\n'+table(['方法','平均成功率','bootstrap 95% 区间 (%)','每种子三周期额外调用'],main_rows)+'\n\n'
    text+=f"主比较 Covered−Sparse：{100*primary['delta_vsr']:+.3f} 个百分点，区间 [{100*primary['ci_low']:.3f}, {100*primary['ci_high']:.3f}]，种子配对 p={primary['seed_sign_flip_p']:.5f}。stats 中额外保存六比较 Holm 值，但预声明的单项主比较使用原始 p。\n\n"
    text+=table(['次级比较','差值 (百分点)','95% 区间','Holm p'],[[r['contrast'],f"{100*r['delta_vsr']:+.3f}",f"[{100*r['ci_low']:.3f}, {100*r['ci_high']:.3f}]",f"{r['secondary_family_holm_p']:.5f}"] for r in secondary])+'\n\n'
    text+='## 3. 按块与快照的结果\n\n'
    text+=table(['方法','块0 (%)','块1 (%)','块2 (%)'],[[m]+[f"{100*next(r['vsr'] for r in cells if r['method']==m and r['cycle']==i and r['snapshot']=='all'):.3f}" for i in range(3)] for m in methods])+'\n\n'
    text+='按快照的成功率及八种子标准差在 `cycles.csv`；费用在 `costs.csv`；准入、完整策略认证和存储状态在 `tables.json`。上限相同为每周期 300,000 次额外 API，实际消耗不同。source 历史首周期计费；Direct 不支付未使用的模型验证/校准。测试调用另计，不把适配费用和测试费用混为一项。\n\n'
    text+='## 4. 单独执行的上下文优先级\n\n'
    text+=table(['方法','调用上限','实际调用均值','探测上下文均值','RMSE ± 种子标准差'],[[r['method'],r['budget'],f"{r['actual_calls']:.1f}",f"{r['selected_contexts']:.1f}",f"{r['rmse']:.5f} ± {r['seed_std']:.5f}"] for r in priority_cells])+'\n\n'
    text+='四个新的 source 诊断种子，epsilon=.1；分歧/调用费用形成正概率队列，实际执行预算内的随机分数设计。这里是 Eq.probepriority 的非恒定系数变体：只汇总 unary/pair，不含原式的常数项，不能称为完整原式。所有探测选择完成后，才收集每上下文四次独立 full-cube 执行参考；参考有 Monte Carlo 误差，不是解析真值。未探测系数保留共同模型先验，报告全部 64 上下文的混合系数误差。它是单独组件诊断，未进入三周期主实验，不能据此宣称在线系统收益或总体优势。原始 Episode 的 randomized 字段保留旧 False 默认值；随机符号和条件概率已重构核验，诊断记录不参与训练。\n\n'
    text+='## 5. 复核与连续运行记录\n\n'
    online_counts={k:v for k,v in audit.items() if isinstance(v,int) and not isinstance(v,bool)}
    text+=f"在线复核计数：`{online_counts}`；优先级复核计数：`{priority_audit['counts']}`。复核重新执行 source、适配、认证和全部测试，重构曝光、状态、版本许可、费用及统计；校验训练 checkpoint、source 验证/校准和日志来源，没有重跑优化器。\n\n"
    text+='块0已完整评分后，原始运行因重复神经特征计算耗时停止。块0数据及初始模型逐字节复制；后续块使用相同参数结构、目标和优化器的去重特征计算。logit、损失和全部参数梯度通过容差等价测试；浮点训练路径不保证逐位相同。所有方法规格、种子、预算和超参数保持预声明设置。原始停止目录及部分块1记录保留。\n\n'
    text+=f"停止运行的已完成账本额外调用至少 {sum(r['calls'] for r in overhead):,} 次，作为工程重复消耗单独列入 `interrupted_engineering_overhead.json`，不计入完成实验的算法费用；未完成账本及 CPU 耗时未完整测量。\n\n"
    text+='## 6. 结论与尚未完成的论文评估\n\n'
    text+='本轮完成可复核的持续更新、角色探测执行、预算控制、候选直接再验证、完整 bank/gate/store 策略切换、逻辑冷存储和有证据的有限模板归档，并单独执行上下文排程。D/X 的 VOI 是乐观直接许可近似，不是完整后验知识梯度；CRN 不满足论文独立行噪声方差界；逐周期认证没有全生命周期或有限样本保证。\n\n'
    text+='真实服务漂移、LLM actor 全规模评估、大候选池和未见 DAG 仍未完成。逻辑冷存储没有 RAM 节省测量；归档只涵盖已知模板的精确曝光等价。原稿印刷仿真缺少原始生成脚本，当前新仿真不作为逐格复现。不能将本轮称为论文全部实验完成，也不能通过挑选种子或设置制造优势。\n'
    Path('CREDIT_V3_ONLINE_RESULTS.md').write_text(text)
    write(output/'analysis_manifest.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    write(output/'provenance.json',{'generator_sha256':digest(__file__),'source_manifest_sha256':digest(source/'artifact_manifest.json'),
        'priority_manifest_sha256':digest(priority/'manifest.json'),'audit_sha256':digest(source.with_name(source.name+'_audit.json')),
        'priority_audit_sha256':digest(priority.with_name(priority.name+'_audit.json')),'report_sha256':digest('CREDIT_V3_ONLINE_RESULTS.md')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--priority',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.source,a.priority,a.output)
