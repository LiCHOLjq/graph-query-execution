"""Optional local plotting dependency: matplotlib. No benchmark dependency."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(directory):
    root=Path(directory)
    data=json.loads((root/'summary.json').read_text())
    groups=sorted(set((r['n'],r['family'],r['correlated']) for r in data['aggregates']))
    lookup={(r['n'],r['family'],r['correlated'],r['method']):r for r in data['aggregates']}
    labels=[f"{'H' if f=='shared_hubs' else 'D'}{n}\n{'corr.' if c else 'shuffle'}" for n,f,c in groups]
    styles={'Restart':('#7b7b7b','o'),'GraphShare':('#d78d00','s'),'VectorResume':('#375fb1','^'),'CoExec':('#16836e','D'),'ColdSuffix':('#8550a5','v'),'ForwardGF':('#d25454','x')}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    for ax,metric,title in zip(axes,['mean_ms','graph_work','distances'],['(a) Mean query latency','(b) Graph primitives','(c) Distance evaluations']):
        for method in ['Restart','GraphShare','VectorResume','CoExec']:
            vals=[lookup[(*g,method)][metric]/lookup[(*g,'Restart')][metric] for g in groups]
            color,marker=styles[method]
            ax.plot(range(len(groups)),vals,color=color,marker=marker,label=method,linewidth=1.5,markersize=4)
        ax.set_xticks(range(len(groups)),labels,rotation=45,ha='right')
        ax.set_ylabel('Ratio to Restart (lower is better)'); ax.set_title(title)
        ax.grid(axis='y',alpha=.2); ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(root/'ablation.pdf'); fig.savefig(root/'ablation.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.3,3.5),constrained_layout=True)
    for method in ['Restart','GraphShare','VectorResume','ColdSuffix','ForwardGF']:
        vals=[lookup[(*g,method)]['mean_ms']/lookup[(*g,'CoExec')]['mean_ms'] for g in groups]
        color,marker=styles[method]
        ax.plot(range(len(groups)),vals,color=color,marker=marker,label=method,linewidth=1.5)
    ax.axhline(1,color='black',linestyle=':',linewidth=1)
    ax.set_yscale('log'); ax.set_ylabel('Baseline latency / CoExec latency')
    ax.set_xticks(range(len(groups)),labels); ax.grid(axis='y',alpha=.2)
    ax.legend(ncol=3,frameon=False,fontsize=8)
    ax.set_title('Exact-output baselines; values below 1 favor the baseline')
    fig.savefig(root/'baselines.pdf'); fig.savefig(root/'baselines.png',dpi=180); plt.close(fig)


if __name__=='__main__': main(sys.argv[1] if len(sys.argv)>1 else 'results/main')
