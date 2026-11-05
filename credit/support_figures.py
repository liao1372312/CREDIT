"""Plot audited support propagation and retention execution measurements."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    stress=Path('results/support_stress_v1');deployed=Path('results/partial_retention_v1')
    for root in (stress,deployed):
        if not json.loads((root/'execution_audit.json').read_text())['verified']:raise ValueError('audit required')
    table=json.loads((stress/'summary.json').read_text())['table']
    measured=json.loads((deployed/'summary.json').read_text())['table']
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.2))
    conditions=['complete','delete_unreachable_states','missing_25pct','missing_50pct','missing_75pct']
    labels=['Complete','Unreachable\ndeleted','25%\nmissing','50%\nmissing','75%\nmissing']
    for metric,label,color in [('coarse_transfer_coverage','All-input coverage','#999999'),
            ('reachable_transfer_coverage','Reachable-path coverage','#346888'),
            ('mean_unknown_mass','Unknown path mass','#aa6b34')]:
        values=[next(r[metric] for r in table if r['condition']==c) for c in conditions]
        axes[0].plot(range(len(conditions)),values,'o-',label=label,color=color)
    axes[0].set_xticks(range(len(labels)),labels);axes[0].set_ylim(-.05,1.05)
    axes[0].set_ylabel('Fraction / fitted path mass');axes[0].set_title('Controlled evidence deletion, fixed banks')
    axes[0].legend(fontsize=8)
    names=['complete','delete_unreachable_states','missing_25pct','missing_submit_role']
    x=np.arange(len(names));width=.37
    for method,offset,label,color in [('CREDIT-coarse-support',-width/2,'Coarse support','#a7b8c5'),
            ('CREDIT-reachable-support',width/2,'Reachable support','#346888')]:
        rows=[next(r for r in measured if r['condition']==c and r['method']==method) for c in names]
        values=np.array([r['vsr'] for r in rows]);lower=np.array([r['ci_low'] for r in rows]);upper=np.array([r['ci_high'] for r in rows])
        axes[1].bar(x+offset,values,width,label=label,color=color,
            yerr=[values-lower,upper-values],capsize=3)
    axes[1].set_xticks(x,['Complete','Unreachable\ndeleted','25%\nmissing','Submit\nmissing'])
    axes[1].set_ylim(0,.5);axes[1].set_ylabel('Verified success rate (paired seed/family CI95)')
    axes[1].set_title('5 new execution seeds; 3,840 episodes');axes[1].legend(fontsize=8)
    fig.suptitle('Finite support correction; model envelopes are not confidence certificates',fontsize=10)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(deployed/f'support_correction.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
