"""Report the full-coefficient priority diagnostic without changing older work."""
import argparse,csv
from itertools import product
from pathlib import Path

import numpy as np

from .v3.experiment import digest,write
from .v3_optimization_verify import read
from .v3_online_report import table


def report(source,output):
    audit_path=source.with_name(source.name+'_audit.json');audit=read(audit_path)
    if not audit['verified'] or audit['manifest_sha256']!=digest(source/'manifest.json'):
        raise ValueError('unverified full-coefficient experiment')
    for name,h in read(source/'manifest.json').items():
        if digest(source/name)!=h:raise ValueError('changed artifact: '+name)
    if output.exists():raise FileExistsError('use a fresh analysis directory')
    output.mkdir();rows=read(source/'summary.json');cells=[];contrasts=[]
    seeds=sorted({r['seed'] for r in rows})
    for budget in (512,1024,2048):
        for method in ('uniform','disagreement-per-call'):
            group=[r for r in rows if r['budget']==budget and r['method']==method]
            cells.append({'method':method,'budget':budget,'rmse':float(np.mean([r['rmse'] for r in group])),
                          'seed_std':float(np.std([r['rmse'] for r in group],ddof=1)),
                          'mean_actual_calls':float(np.mean([r['actual_calls'] for r in group])),
                          'mean_contexts_selected':float(np.mean([r['contexts_selected'] for r in group]))})
        delta=np.array([next(r['rmse'] for r in rows if r['seed']==s and r['budget']==budget and r['method']=='disagreement-per-call')
                        -next(r['rmse'] for r in rows if r['seed']==s and r['budget']==budget and r['method']=='uniform') for s in seeds])
        null=[np.mean(delta*np.asarray(signs)) for signs in product((-1,1),repeat=len(seeds))]
        contrasts.append({'budget':budget,'priority_minus_uniform_rmse':float(delta.mean()),
                          'seed_sign_flip_p':float(np.mean(np.abs(null)>=abs(delta.mean())-1e-12))})
    running=0.
    for rank,index in enumerate(sorted(range(3),key=lambda i:contrasts[i]['seed_sign_flip_p'])):
        running=max(running,min(1.,(3-rank)*contrasts[index]['seed_sign_flip_p']))
        contrasts[index]['exploratory_three_budget_holm_p']=running
    write(output/'tables.json',{'cells':cells,'contrasts':contrasts})
    with (output/'priority.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    for method in ('uniform','disagreement-per-call'):
        group=[r for r in cells if r['method']==method]
        ax.errorbar([r['budget'] for r in group],[r['rmse'] for r in group],
                    yerr=[r['seed_std'] for r in group],marker='o',label=method)
    ax.set(xlabel='Physical call cap',ylabel='Nonconstant coefficient RMSE (MC reference)')
    ax.legend();fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/('priority_full.'+ext),dpi=180)
    plt.close(fig)
    text='# CREDIT v3 含常数项的上下文优先级补充实验\n\n'
    text+='## 实现与冻结协议\n\n'
    text+='本实验补齐原稿 Eq.probepriority 在当前一、二阶系数集合上的常数项。候选全组合预测投影现在含常数、9 个 unary、36 个 pair 槽位；不适用的项填零。概率使用全部适用系数的 ensemble 方差除以重放调用上界，并混合 epsilon=.1 的均匀概率。实际选择时在剩余预算可执行集合上归一化，记录条件概率并执行随机符号设计。\n\n'
    text+='独立种子 1501/1511/1523/1531，每种子 64 个 source 上下文，预算 512/1024/2048。修正依据是原稿公式的完整性；没有改变已完成三周期主实验，旧非恒定系数诊断及其负结果保留。初始模型来自固定 source 模型，所有选择后才执行四重复 full-cube 参考；参考有 Monte Carlo 误差。\n\n'
    text+='## 结果\n\n'+table(['方法','上限','实际调用均值','上下文均值','RMSE ± 种子标准差'],[
        [r['method'],r['budget'],f"{r['mean_actual_calls']:.1f}",f"{r['mean_contexts_selected']:.1f}",f"{r['rmse']:.5f} ± {r['seed_std']:.5f}"] for r in cells])+'\n\n'
    text+=table(['上限','优先级−均匀 RMSE（负值较好）','三预算探索性 Holm p'],[
        [r['budget'],f"{r['priority_minus_uniform_rmse']:+.5f}",f"{r['exploratory_three_budget_holm_p']:.3f}"] for r in contrasts])+'\n\n'
    text+='512 上限的均值略有改善，1024/2048 的均值更差；四个独立种子不足以证明优势。不能将新旧不同种子的诊断差异归因于常数项修正。准确性指标保持非恒定归因系数 RMSE，未探测系数使用共同模型先验；优先级的常数项不被忽略，但不加入这个归因误差指标。\n\n'
    text+='## 核验与范围\n\n'
    text+=f"执行复核：`{audit['counts']}`，重构选择概率、预算、随机符号系数、事件与状态，以及所有执行参考。原始 Episode.randomized 仍为旧默认 False，实际随机分配已单独核验，诊断 episode 不用于训练。常数项方差另有针对性测试。\n\n"
    text+='这是完成公式的独立组件诊断，仍未接入主循环，不能宣称三周期系统已有上下文调度收益。没有真实服务、LLM 或自然漂移证据。原始结果、审计和导出图表分别在同名 experiment/audit/analysis 目录。\n'
    Path('CREDIT_V3_PRIORITY_FULL_RESULTS.md').write_text(text)
    write(output/'analysis_manifest.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    write(output/'provenance.json',{'generator_sha256':digest(__file__),'report_sha256':digest('CREDIT_V3_PRIORITY_FULL_RESULTS.md'),
                                  'audit_sha256':digest(audit_path),'manifest_sha256':digest(source/'manifest.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.source,a.output)
