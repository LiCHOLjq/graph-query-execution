"""Reanalyse immutable original data and independently separated sensitivity factors."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.optimize import nnls
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'native_comparisons/results';OUT.mkdir(exist_ok=True)
FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
raw=pd.read_csv(ROOT/'artifact/results/main/raw.csv')
inputs=json.loads((ROOT/'artifact/results/main/inputs.json').read_text())
digests={f"{r['family']}-n{r['n']}-c{int(r['correlated'])}-s{r['seed']}":r['graph_digest'] for r in inputs}
raw['predicate']=raw.case_id.map(digests)+':'+raw.anchor.astype(str)
raw.loc[raw.method.isin(['ColdSuffix','ForwardGF']),'prefix_recall']=np.nan
raw.loc[raw.eligible==0,'prefix_recall']=np.nan
group=['n','family','correlated','method']
agg=raw.groupby(group).latency_ms.agg(mean='mean',median='median',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75),p95=lambda x:x.quantile(.95)).reset_index()
agg.to_csv(OUT/'original_cell_latency.csv',index=False)
raw.groupby(group).prefix_recall.agg(mean='mean',nonempty_prefix_samples='count').reset_index().to_csv(OUT/'original_prefix_quality.csv',index=False,na_rep='NaN')
med=raw.groupby(['case_id','qid','predicate','n','family','correlated','seed','method']).latency_ms.median().unstack('method').reset_index()
ratios={}
for base in ['Restart','GraphShare','ColdSuffix','ForwardGF']:
    a=med.CoExec/med[base];ratios[base]={'slower':int((a>1).sum()),'n':len(a),'median':float(a.median()),'q25':float(a.quantile(.25)),'q75':float(a.quantile(.75)),'p95':float(a.quantile(.95))}
    med['CoExec_over_'+base]=a
med.to_csv(OUT/'original_query_medians.csv',index=False)
raw.groupby(['n','family','correlated','seed','method']).latency_ms.mean().unstack('method').to_csv(OUT/'original_per_seed.csv')
# Predicate clusters retain vector settings, repeated query IDs and timing repeats.
clusters=raw.groupby(['predicate','method']).latency_ms.sum().unstack('method')
rng=np.random.default_rng(20261023);boot=[]
for _ in range(5000):
    z=clusters.iloc[rng.integers(0,len(clusters),len(clusters))].sum()
    boot.append([z.CoExec/z.ColdSuffix,z.CoExec/z.ForwardGF,z.Restart/z.CoExec])
ci=np.quantile(boot,[.025,.975],axis=0).T.tolist()
summary={'original_rows':len(raw),'query_instances':len(med),'unique_predicates':len(clusters),'ratios':ratios,
         'conditional_predicate_bootstrap':{'CoExec/ColdSuffix':ci[0],'CoExec/ForwardGF':ci[1],'SplitFresh/SharedRetain':ci[2]},
         'bootstrap_scope':'Conditional on the 20 generated graphs; predicates within one graph remain dependent. Not a population-level confidence interval.',
         'empty_query_instances':int((raw.drop_duplicates(['case_id','qid']).eligible==0).sum()),
         'max_index_ms':max(r['index_ms'] for r in inputs)}
# Costs in ms, nonnegative least squares; leave one generator seed out.
features=['graph_entries','graph_edges','distances','ann_edges'];fits=[]
pred=[]
for method,df in raw.groupby('method'):
    X=np.column_stack([np.ones(len(df)),df[features].to_numpy()/1000]);y=df.latency_ms.to_numpy()
    coef,_=nnls(X,y,maxiter=5000)
    for seed in sorted(df.seed.unique()):
        train=df.seed.to_numpy()!=seed;beta,_=nnls(X[train],y[train],maxiter=5000)
        for ix,actual,estimate in zip(df.index[~train],y[~train],X[~train]@beta):pred.append({'row':int(ix),'method':method,'actual_ms':actual,'predicted_ms':estimate})
    fits.append({'method':method,'rank':int(np.linalg.matrix_rank(X)),'columns':X.shape[1],'intercept_ms':coef[0],**{f+'_ms_per_1000':coef[j+1] for j,f in enumerate(features)}})
pred=pd.DataFrame(pred);pred['residual_ms']=pred.actual_ms-pred.predicted_ms
pred.to_csv(OUT/'original_cost_residuals.csv',index=False);pd.DataFrame(fits).to_csv(OUT/'original_cost_coefficients.csv',index=False)
metrics=[]
for method,df in pred.groupby('method'):
    metrics.append({'method':method,'mae_ms':float(abs(df.residual_ms).mean()),'normalized_mae':float(abs(df.residual_ms).sum()/df.actual_ms.sum()),'r2':float(1-(df.residual_ms**2).sum()/((df.actual_ms-df.actual_ms.mean())**2).sum())})
summary['cost_model_leave_seed_out']=metrics
(OUT/'original_reanalysis.json').write_text(json.dumps(summary,indent=2))
fig,ax=plt.subplots(1,2,figsize=(7.4,2.7))
for axis,base in zip(ax,['ColdSuffix','ForwardGF']):
    for family,label,color in [('distributed','Distributed','#147d92'),('shared_hubs','Shared hubs','#d16b2e')]:
        values=np.sort(med.loc[med.family==family,'CoExec_over_'+base]);axis.plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=color)
    axis.axvline(1,color='gray',ls='--',lw=.8);axis.set(xlabel=f'Shared+Retain / {base}',ylabel='Fraction of query instances');axis.grid(alpha=.2)
ax[1].set_xscale('log');ax[0].legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'native_original_ratios.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(5.6,3.0))
for method,df in pred.groupby('method'):
    df=df.iloc[::15];ax.scatter(df.predicted_ms,df.residual_ms,s=6,alpha=.4,label=method)
ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='Held-out predicted latency (ms)',ylabel='Measured minus predicted (ms)');ax.legend(fontsize=6,ncol=3);fig.tight_layout();fig.savefig(FIG/'native_cost_residuals.pdf');plt.close(fig)
# Small complete latency tables with explicitly separate mean and p95 ratios.
cells=agg[['n','family','correlated']].drop_duplicates().sort_values(['n','family','correlated'])
lines=[r'\begin{table}[tb]\centering\scriptsize',r'\caption{Original Python experiment. Times are milliseconds. Median [IQR] is over timed executions; ratios use the indicated cell statistic. D/H denote distributed/shared-hub graphs and S/A shuffled/aligned vectors.}',r'\begin{tabular}{@{}rlrrrr@{}}\toprule',r'Cell & $n$/graph/vector & Shared+Retain & Cold reverse & Forward exact & Mean ratio$^{a}$\\\midrule']
tails=[r'\begin{table}[tb]\centering\small',r'\caption{Original ablation: cell means and p95 values, with both ratios labeled. The split graph stores retain their own states.}',r'\begin{tabular}{@{}rrrrrrr@{}}\toprule',r'Cell & Split+Fresh mean & Shared+Retain mean & Split p95 & Shared p95 & Mean ratio & p95 ratio\\\midrule']
for num,(_,c) in enumerate(cells.iterrows(),1):
    z=agg[(agg.n==c.n)&(agg.family==c.family)&(agg.correlated==c.correlated)].set_index('method')
    vals=[]
    for m in ['CoExec','ColdSuffix','ForwardGF']:
        r=z.loc[m];vals.append(f"{r['median']:.2f} [{r.q25:.2f}--{r.q75:.2f}]")
    label=f"{c.n}/{'D' if c.family=='distributed' else 'H'}/{'A' if c.correlated else 'S'}"
    lines.append(f"{num} & {label} & "+' & '.join(vals)+f" & {z.loc['CoExec','mean']/z.loc['ForwardGF','mean']:.1f}"+r'\\')
    a=z.loc['Restart'];b=z.loc['CoExec'];tails.append(f"{num} & {a['mean']:.2f} & {b['mean']:.2f} & {a.p95:.2f} & {b.p95:.2f} & {a['mean']/b['mean']:.2f} & {a.p95/b.p95:.2f}"+r'\\')
lines += [r'\bottomrule\end{tabular}',r'\par\footnotesize $^{a}$Shared+Retain divided by Forward exact; values above one favor Forward exact.',r'\label{tab:original}\end{table}']
tails += [r'\bottomrule\end{tabular}\label{tab:originaltail}\end{table}']
(OUT/'native_original_table.tex').write_text('\n'.join(lines));(OUT/'native_original_tail.tex').write_text('\n'.join(tails))
sens=pd.read_csv(OUT/'sensitivity/raw.csv');sens['setting']=sens.budget.astype(str)+'/'+sens.quantum.astype(str)
idx=['family','aligned','seed','qid','anchor','budget','quantum','repeat']
p=sens.pivot(index=idx,columns=['G','A','R'],values='ms')
effects=[]
for setting,df in sens.groupby(['budget','quantum']):
    data=p.loc[(slice(None),slice(None),slice(None),slice(None),slice(None),*setting,slice(None)),:]
    for factor,j in [('graph',0),('ANN',1),('repair',2)]:
        terms=[]
        for bits in [(g,a,r) for g in [0,1] for a in [0,1] for r in [0,1] if (g,a,r)[j]==0]:
            on=list(bits);on[j]=1;terms.extend((data[bits]/data[tuple(on)]).tolist())
        effects.append({'budget':setting[0],'quantum':setting[1],'factor':factor,'median_off_on':float(np.median(terms)),'q25':float(np.quantile(terms,.25)),'q75':float(np.quantile(terms,.75))})
pd.DataFrame(effects).to_csv(OUT/'sensitivity_factor_effects.csv',index=False)
sens.groupby(['budget','quantum','G','A','R'])[['ms','entries','edges','distances','ann_edges']].mean().to_csv(OUT/'sensitivity_cell_means.csv')
print(json.dumps(summary,indent=2));print(pd.DataFrame(effects).to_string(index=False))
