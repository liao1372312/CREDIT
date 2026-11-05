"""Publish raw comparison tables and figures for the independently audited grid."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import budget_grid_v2 as grid
from .experiments import _write_csv


def report(root):
    audit = grid.read(root / 'execution_audit.json')
    if not audit['verified']:
        raise ValueError('audit required before reporting')
    comparisons = grid.csv_rows(root / 'comparison.csv')
    paired = grid.csv_rows(root / 'paired_statistics.csv')
    summary = grid.read(root / 'summary.json')
    anchor = summary['primary_grid_cell']
    match = lambda r: all(int(r[k]) == v for k,v in anchor.items())
    primary = 'CREDIT-finite-empirical'
    selected = [r for r in comparisons if match(r)]
    selected_p = [r for r in paired if match(r)]
    baseline = float(next(r['vsr'] for r in selected if r['method']=='NoMem'))
    for r in selected:
        r['delta_vs_NoMem'] = float(r['vsr'])-baseline
        r['relative_improvement_vs_NoMem'] = (float(r['vsr'])-baseline)/baseline if baseline else None
    _write_csv(root / 'anchor_comparison.csv',selected)
    _write_csv(root / 'anchor_paired_statistics.csv',selected_p)
    costs = grid.csv_rows(root / 'audited_costs.csv')
    cost_table = []
    for R in (16,64,256):
        rows = [r for r in costs if int(r['R'])==R]
        cost_table.append({'assigned_R':R,'extra_calls_min':min(int(r['actual_extra_calls']) for r in rows),
            'extra_calls_max':max(int(r['actual_extra_calls']) for r in rows),
            'mean_learning_observations':float(np.mean([int(r['learning_observations']) for r in rows])),
            'common_historical_API_calls':132224,'common_historical_decision_draws':86016})
    _write_csv(root / 'cost_summary.csv',cost_table)
    degree_table = []
    diag = defaultdict(list)
    for r in grid.csv_rows(root / 'forecast_diagnostics.csv'):
        if r['method']==primary:
            diag[int(r['d'])].append(float(r['absolute_error']))
    for d in (1,2,3):
        degree_table.append({'d':d,'mean_selected_bank_forecast_MAE':float(np.mean(diag[d])),
            'contexts':len(diag[d]),'scope':'different selected banks; entire pipeline, not fixed-exposure model accuracy'})
    _write_csv(root / 'degree_forecast_summary.csv',degree_table)
    bank_usage = defaultdict(list); lengths = defaultdict(list)
    for r in grid.csv_rows(root / 'raw_episodes.csv'):
        if r['method']==primary:
            bank_usage[int(r['C']),int(r['K'])].append((int(r['bank_tokens']),int(r['exposure_count'])))
            if match(r):
                lengths[r['topology']].append(int(r['observed_length']))
    active_table = [{'C':C,'K':K,'mean_bank_tokens':float(np.mean([v[0] for v in values])),
        'max_bank_tokens':max(v[0] for v in values),'mean_exposure_count':float(np.mean([v[1] for v in values])),
        'max_exposure_count':max(v[1] for v in values)} for (C,K),values in sorted(bank_usage.items())]
    _write_csv(root / 'budget_usage.csv',active_table)
    fig,axes = plt.subplots(1,3,figsize=(11.6,3.5))
    matrix = np.array([[float(next(r['vsr'] for r in comparisons if r['method']==primary
        and int(r['C'])==C and int(r['K'])==K and int(r['R'])==64 and int(r['d'])==2))
        for K in (4,8,16)] for C in (512,2048,8192)])
    plot = axes[0].imshow(matrix,vmin=0,vmax=1,cmap='Blues')
    axes[0].set_xticks(range(3),['4','8','16']);axes[0].set_yticks(range(3),['512','2048','8192'])
    axes[0].set_xlabel('K exposure cap');axes[0].set_ylabel('C stored tokens')
    axes[0].set_title('VSR: R=64, d=2')
    for i,j in np.ndindex(matrix.shape):
        axes[0].text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='black')
    for d in (1,2,3):
        values = [float(next(r['vsr'] for r in comparisons if r['method']==primary and int(r['C'])==2048
            and int(r['K'])==8 and int(r['R'])==R and int(r['d'])==d)) for R in (16,64,256)]
        axes[1].plot([16,64,256],values,'o-',label=f'd={d}')
    axes[1].set_xscale('log',base=2);axes[1].set_xticks([16,64,256],['16','64','256'])
    axes[1].set_xlabel('R total extra calls, audit included');axes[1].set_ylabel('VSR')
    axes[1].set_title('C=2048, K=8; common history charged');axes[1].legend()
    axes[2].bar([str(d['d']) for d in degree_table],[r['mean_selected_bank_forecast_MAE'] for r in degree_table],color='#346888')
    axes[2].set_xlabel('Maximum memory interaction order d');axes[2].set_ylabel('Selected-bank forecast MAE')
    axes[2].set_title('All grid cells; evaluation-only exact truth')
    fig.suptitle('Controlled finite grid; five seeds; not the full semantic CREDIT method',fontsize=10)
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(root/f'budget_sensitivity.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    lines=['# 受控预算与交互阶数网格实验','',
        '**81 个配置 × 5 个种子、9 种策略已执行并审计。范围是有限机制，尚非完整论文语义方法。**','',
        '## 冻结设置与成本','',
        '- C={512,2048,8192} 为固定 Qwen tokenizer 对真实历史计数记忆文本计数的存储上限；记忆池合计 6,064 tokens。',
        '- K={4,8,16} 限制实际检索曝光数量，记忆库和曝光 tokens 均不超过 C。',
        '- R={16,64,256} 限制新增服务调用，训练回放核验的全部调用也计入；复用前缀和核验重复不重复进入拟合。',
        '- d={1,2,3} 控制决策 logits 的记忆单项/二阶/三阶乘积项；历史状态特征另计。API 使用记录的有限分类分布。',
        '- 每个种子已有共同历史成本：132,224 次 API 调用及 86,016 次独立决策采样。低 R 不是总采集预算；这里只测该历史之后的增量预算。',
        '- 新执行种子为 101/103/107/109/113，分别使用历史种子 7/11/13/17/19 的记录；没有新 Qwen actor 推理。',
        '- 新工作流为就绪状态提交、先授权后变换、重复状态变更，实际长度分别为 1/4/9。已见家族和两个契约阶段的局部机制都在历史中；不是未见供应商/版本迁移。',
        '- 所有 405 个模型、全部策略库和 probe 前缀在第一个新未来结果前统一冻结。oracle 仅在全部未来评分后诊断使用。','',
        '| R 上限 | 实际新增调用范围 | 平均新增学习观察 |','| --- | ---: | ---: |']
    for r in cost_table:
        lines.append(f"| {r['assigned_R']} | {r['extra_calls_min']}–{r['extra_calls_max']} | {r['mean_learning_observations']:.2f} |")
    lines += ['', '## 固定报告点：C=2048、K=8、R=64、d=2','',
        '该报告点在配置中预先指定，不按测试成绩选最优；每策略为 5×96 个配对未来任务。', '',
        '| 方法 | VSR | 95% 种子/家族区间 | 相对 NoMem 差值 |','| --- | ---: | --- | ---: |']
    for r in selected:
        lines.append(f"| {r['method']} | {float(r['vsr']):.4f} | [{float(r['ci_low']):.4f}, {float(r['ci_high']):.4f}] | {r['delta_vs_NoMem']:+.4f} |")
    credit=float(next(r['vsr'] for r in selected if r['method']==primary))
    best=max((r for r in selected if r['method']!=primary),key=lambda r:float(r['vsr']))
    comparison=next(r for r in selected_p if r['baseline']==best['method'])
    lines += ['', '## 结果解释','',
        f"1. 报告点 CREDIT 有限经验版本 VSR={credit:.4f}，最佳比较策略 {best['method']} 为 {float(best['vsr']):.4f}；配对差值 {float(comparison['delta_vsr']):+.4f}，Holm p={float(comparison['holm_p']):.4f}。这不能支持完整语义方法的优势。",
        '2. 更大 C/K 可能不绑定已选库；实际利用率见 budget_usage.csv。相同结果保留，不能把不绑定预算说成独立机制消融。',
        '3. 本网格是在覆盖完整的高成本历史之后追加少量调用；R 的效果较小不能推断少量调用即可学习全部机制。',
        '4. 三阶模型最初在 300 次迭代下出现未收敛，v1 在任何新未来评分前停止；v2 提高上限至 2,000，全部条件模型通过收敛门槛。未根据新测试结果调整正则或选择库。','',
        '## 审计与统计','',
        f"- {audit['future_rows_reconstructed']:,} 条未来结果逐项重建，共 {audit['unique_counterfactuals_reexecuted']:,} 个不同任务/曝光组合实际重执行；重复行通过确定性缓存核对，未当作独立样本。",
        f"- 重建 {audit['conditional_kernel_rows_reconstructed']:,} 个条件概率行及 {audit['forecast_diagnostics_recomputed']:,} 个未来预测诊断；每个种子完整重算预设报告点的九种保留策略。",
        '- 2,000 次种子/家族配对 bootstrap；每个配置内对八个比较做 Holm。81 个配置整体探索未作跨网格显著性宣称。',
        '- 数值模型是局部多项式条件 NLL 拟合，非论文的完整四项神经联合损失，也没有统一迁移置信证书。',
        '- Model-LOO/Model-Shapley 使用拟合机制的离线反事实模型查询，不能称作同调用预算下的直接服务 LOO/MC-Shapley 或 CMP/CAR 原论文复现。','',
        '## 复现','', '```sh',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m credit.budget_grid_v2 --output results/budget_reproduction',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m credit.budget_grid_verify --results results/budget_reproduction',
        'python -m credit.budget_grid_report --results results/budget_reproduction','```','',
        '## 仍需完成','',
        '下一步是未见家族/版本的语义机制、按时间更新的主动 VOI 与自然供应商漂移，以及完整业务/副作用验收。它们不能由本有限网格替代。']
    Path('BUDGET_GRID_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return {'anchor_table':selected,'costs':cost_table,'degree_forecasts':degree_table,'best_comparator':best['method']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/budget_grid_v2'))
    print(json.dumps(report(parser.parse_args().results),indent=2))
