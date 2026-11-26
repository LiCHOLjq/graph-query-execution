"""Native matched comparisons and held-out work-counter cost diagnostics."""
from pathlib import Path
import json
import pandas as pd,numpy as np
from scipy.optimize import nnls
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;OUT=P/'results';PAPER=P.parent/'generated'; (PAPER/'figures').mkdir(parents=True,exist_ok=True)
def read(name):
    r=pd.read_csv(P/'native'/name)
    return r.groupby(['dataset','family','seed','n','d','m','h','k','anchor','anchor_degree','eligible','query_id','method'],as_index=False).median(numeric_only=True)
s=read('strong_synthetic_raw.csv');real=read('strong_real_raw.csv');base=read('synthetic_raw.csv');dim=read('dimension_control_raw.csv')
idx=['family','seed','n','h','k','anchor','anchor_degree','eligible','query_id']
wide=s.pivot(index=idx,columns='method',values='ms').reset_index();methods=['ForwardTopK','ReverseRecursiveTopK','LazyShared']
wide['winner']=wide[methods].idxmin(axis=1);wide['selectivity']=wide.eligible/(wide.n-1)
wide['lazy_over_best_materializer']=wide.LazyShared/wide[['ForwardTopK','ReverseRecursiveTopK']].min(axis=1)
wide['reverse_over_forward']=wide.ReverseRecursiveTopK/wide.ForwardTopK
wide.to_csv(OUT/'native_strong_query_medians.csv',index=False)
summary={'strong_queries':len(wide),'design_predicate_slots':len(wide.drop_duplicates(['family','seed','n','anchor','h'])),
         'winner_counts':wide.winner.value_counts().to_dict(),'selectivity_min_nonzero':float(wide.loc[wide.eligible>0,'selectivity'].min()),'selectivity_max':float(wide.selectivity.max()),
         'empty_configs':int((wide.eligible==0).sum()),'families':{}}
for fam,df in wide.groupby('family'):
    summary['families'][fam]={'queries':len(df),'winner_counts':df.winner.value_counts().to_dict(),'lazy_over_best_median':float(df.lazy_over_best_materializer.median()),
                              'lazy_over_forward_mean_ratio':float(df.LazyShared.mean()/df.ForwardTopK.mean()),'selectivity_min':float(df.selectivity.min()),'selectivity_max':float(df.selectivity.max()),
                              'anchor_degree_min':int(df.anchor_degree.min()),'anchor_degree_max':int(df.anchor_degree.max())}
wide.groupby(['n','family','h','k'])[methods].mean().to_csv(OUT/'native_strong_cell_means.csv')
wide.groupby(['seed','family'])[methods].mean().to_csv(OUT/'native_strong_per_seed.csv')
winner=wide[(wide.winner=='LazyShared')]
summary['lazy_winning_regimes']=winner.groupby(['n','h','k']).size().reset_index(name='queries').to_dict('records')
summary['lazy_wins_speedups']={'median':float((winner[['ForwardTopK','ReverseRecursiveTopK']].min(axis=1)/winner.LazyShared).median()),'max':float((winner[['ForwardTopK','ReverseRecursiveTopK']].min(axis=1)/winner.LazyShared).max())}
# Cluster all output-count settings of a predicate, conditional on generated graphs.
wide['cluster_seed']=wide.seed
wide.loc[(wide.family=='sparse')&(wide.anchor==wide.n-1),'cluster_seed']=-1
clusters=wide.groupby(['family','cluster_seed','n','anchor','h'])[methods].sum();summary['unique_structural_predicates']=len(clusters);rng=np.random.default_rng(20261023)
boot=[]
for _ in range(3000):
    z=clusters.iloc[rng.integers(0,len(clusters),len(clusters))].sum();boot.append([z.LazyShared/z.ForwardTopK,z.ReverseRecursiveTopK/z.ForwardTopK])
summary['conditional_predicate_intervals']=np.quantile(boot,[.025,.975],axis=0).T.tolist()
realwide=real.pivot(index='query_id',columns='method',values='ms')
summary['real']={m:{'mean_query_medians':float(realwide[m].mean()),'median':float(realwide[m].median()),'q25':float(realwide[m].quantile(.25)),'q75':float(realwide[m].quantile(.75)),'p95':float(realwide[m].quantile(.95))} for m in methods}
summary['real_winners']=realwide[methods].idxmin(axis=1).value_counts().to_dict()
# Resumability overhead with same work, matched full-sort main cohort.
bw=base.pivot(index='query_id',columns='method',values='ms')
summary['continuation_ratios']={label:{'median':float((bw[a]/bw[b]).median()),'q25':float((bw[a]/bw[b]).quantile(.25)),'q75':float((bw[a]/bw[b]).quantile(.75))} for label,a,b in [('forward','ForwardResume','ForwardGF'),('reverse','ReverseResume','ReverseRecursive')]}
dim.groupby(['d','family','h','method']).ms.agg(['mean','median']).to_csv(OUT/'native_dimension_summary.csv')
dw=dim.pivot(index=['d','family','h','seed','query_id'],columns='method',values='ms').reset_index()
summary['dimension']=[{'d':int(d),'lazy_wins_forward':int((z.LazyShared<z.ForwardGF).sum()),'configs':len(z),'lazy_forward_ratio_of_means':float(z.LazyShared.mean()/z.ForwardGF.mean())} for d,z in dw.groupby('d')]
# Cost models of strongest cohort: predictions charge partial-sort for materializers.
s['coord_work']=s.distances*s.d
s['ordering_work']=s.sort_items*np.log2(np.maximum(np.where(s.method=='LazyShared',s.sort_items,np.minimum(s.k,s.sort_items)),2))
features=['edge_examinations','state_entries','coord_work','ordering_work','root_checks'];scales=np.array([1000,1000,100000,100000,1000])
preds=[];fits=[]
for m,z in s.groupby('method'):
    X=np.column_stack([np.ones(len(z)),z[features].to_numpy()/scales]);y=z.ms.to_numpy();coef,_=nnls(X,y,maxiter=20000)
    fits.append({'method':m,'rank':int(np.linalg.matrix_rank(X)),'columns':6,'intercept_ms':coef[0],**{f:float(coef[i+1]) for i,f in enumerate(features)}})
    for seed in sorted(z.seed.unique()):
        train=z.seed.to_numpy()!=seed;beta,_=nnls(X[train],y[train],maxiter=20000)
        for (_,row),v in zip(z.loc[~train].iterrows(),X[~train]@beta):preds.append({'method':m,'query_id':int(row.query_id),'seed':int(row.seed),'actual':row.ms,'predicted':max(float(v),1e-9)})
pd.DataFrame(fits).to_csv(OUT/'native_cost_coefficients.csv',index=False)
pred=pd.DataFrame(preds);pred['residual']=pred.actual-pred.predicted;pred.to_csv(OUT/'native_cost_predictions.csv',index=False)
summary['cost_model']={m:{'mae_ms':float(abs(z.residual).mean()),'nmae':float(abs(z.residual).sum()/z.actual.sum()),'r2':float(1-(z.residual**2).sum()/((z.actual-z.actual.mean())**2).sum())} for m,z in pred.groupby('method')}
pp=pred.pivot(index='query_id',columns='method',values='predicted');aa=pred.pivot(index='query_id',columns='method',values='actual');choice=pp.idxmin(axis=1)
selected=np.array([aa.loc[q,m] for q,m in choice.items()]);best=aa.min(axis=1).to_numpy()
summary['counter_informed_selector']={'winner_accuracy':float((choice==aa.idxmin(axis=1)).mean()),'geometric_regret':float(np.exp(np.log(selected/best).mean())),'p95_regret':float(np.quantile(selected/best,.95)), 'warning':'Observed counters are an oracle input; this is not an online planner.'}
(OUT/'native_analysis.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,3,figsize=(8.2,3.0),sharey=True)
fams=['sparse','uniform','fanin','fanout','dense']
for ax,n in zip(axes,[1000,10000,100000]):
    z=wide[(wide.n==n)&(wide.k==10)&(wide.eligible>0)]
    mat=z.groupby(['family','h']).lazy_over_best_materializer.median().unstack().reindex(fams)
    im=ax.imshow(np.log10(mat.to_numpy()),vmin=-.6,vmax=3,cmap='RdYlBu_r',aspect='auto')
    for i in range(5):
        for j in range(3):
            value=mat.iloc[i,j];ax.text(j,i,f'{value:.2g}',ha='center',va='center',fontsize=8,color='white' if np.log10(value)>2 else 'black')
    ax.set(xticks=[0,1,2],xticklabels=['2','4','6'],yticks=range(5),yticklabels=fams,xlabel='Walk length',title=f'n = {n:,}')
fig.subplots_adjust(left=.1,right=.88,bottom=.2,top=.88,wspace=.13);cb=fig.colorbar(im,cax=fig.add_axes([.9,.2,.017,.66]));cb.set_label('log10 latency ratio',fontsize=8);fig.savefig(PAPER/'figures/native_native_boundary.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(7.3,2.6))
for m,z in pred.groupby('method'):
    axes[0].scatter(z.predicted,z.residual,s=5,alpha=.35,label=m)
axes[0].axhline(0,color='gray',lw=.7);axes[0].set(xlabel='Held-out predicted latency (ms)',ylabel='Residual (ms)');axes[0].legend(fontsize=6)
regret=np.sort(selected/best);axes[1].plot(regret,np.arange(1,len(regret)+1)/len(regret),color='#147d92');axes[1].set_xscale('log');axes[1].set_xticks([1,1.25,1.5,2],labels=['1','1.25','1.5','2']);axes[1].minorticks_off();axes[1].set(xlabel='Selected / best measured latency',ylabel='Fraction of queries');fig.tight_layout();fig.savefig(PAPER/'figures/native_native_model.pdf');plt.close(fig)
lines=[r'\begin{table}[tb]\centering\small',r'\caption{Strong native exact plans on the real cohort (80 queries). Statistics are over each query\textquotesingle s median of three repetitions; all times are milliseconds.}',r'\begin{tabular}{@{}lrrrr@{}}\toprule Plan & Mean & Median & [Q1,Q3] & p95\\\midrule']
for m in methods:
    r=summary['real'][m];lines.append(f"{m} & {r['mean_query_medians']:.4f} & {r['median']:.4f} & [{r['q25']:.4f},{r['q75']:.4f}] & {r['p95']:.4f}"+r'\\')
lines.append(r'\bottomrule\end{tabular}\label{tab:nativereal}\end{table}');(PAPER/'native_native_real_table.tex').write_text('\n'.join(lines))
print(json.dumps(summary,indent=2))
