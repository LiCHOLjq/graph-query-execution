"""SENSITIVITY reanalysis of frozen NATIVE records. No timed evidence is overwritten."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sensitivity_controls/results'
OUT.mkdir(exist_ok=True)
METHODS=['ForwardTopK','ReverseRecursiveTopK','LazyShared']
STRATA=['empty','(0,1%)','[1%,10%)','[10%,50%)','[50%,90%)','[90%,99.98%)','[99.98%,100%]']
def stratum(s):
    if s==0:return STRATA[0]
    return STRATA[np.searchsorted([.01,.1,.5,.9,.9998],s,side='right')+1]
def grouped(df):
    return pd.crosstab(pd.Categorical(df.stratum,categories=STRATA),df.winner,dropna=False).reindex(columns=METHODS,fill_value=0)

def main():
    q=pd.read_csv(ROOT/'native_comparisons/results/native_strong_query_medians.csv')
    q['stratum']=q.selectivity.map(stratum)
    grouped(q).to_csv(OUT/'native_winners_by_selectivity.csv',index_label='stratum')
    q.groupby(['family','stratum'],observed=True)[METHODS].count().iloc[:,0].rename('configurations').to_csv(OUT/'native_family_stratum_counts.csv')
    family=q[q.eligible>0].groupby('family').agg(configurations=('query_id','size'),minimum=('selectivity','min'),median=('selectivity','median'),maximum=('selectivity','max'))
    family.to_csv(OUT/'native_family_selectivity.csv')
    p=pd.read_csv(ROOT/'native_comparisons/results/native_cost_predictions.csv')
    pred=p.pivot(index='query_id',columns='method',values='predicted').reindex(columns=METHODS)
    t=q.set_index('query_id').join(pred.idxmin(axis=1).rename('selected'))
    t['selected_ms']=[r[r.selected] for _,r in t.iterrows()]
    t['regret']=t.selected_ms/t[METHODS].min(axis=1)
    t['forward_regret']=t.ForwardTopK/t[METHODS].min(axis=1)
    t['correct']=t.selected.eq(t.winner)
    t['forward_correct']=t.winner.eq('ForwardTopK')
    t['broad_stratum']=np.where(t.selectivity==0,'empty',np.where(t.selectivity>=.9998,'saturated','nonempty unsaturated'))
    rows=[]
    for name,df in [('all',t)]+list(t.groupby('broad_stratum')):
        rows.append(dict(stratum=name,n=len(df),selector_accuracy=df.correct.mean(),selector_geomean_regret=np.exp(np.log(df.regret).mean()),selector_p95_regret=df.regret.quantile(.95),always_forward_accuracy=df.forward_correct.mean(),always_forward_geomean_regret=np.exp(np.log(df.forward_regret).mean()),always_forward_p95_regret=df.forward_regret.quantile(.95)))
    stats=pd.DataFrame(rows)
    stats.to_csv(OUT/'selector_strata.csv',index=False)
    t.to_csv(OUT/'selector_queries.csv')
    report=dict(native_configurations=len(q),native_winners=grouped(q).to_dict('index'),family_selectivity=family.to_dict('index'),selector=rows,nonforward_minimum_selectivity=float(q.loc[q.winner!='ForwardTopK','selectivity'].min()))
    (OUT/'selectivity_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    family_labels={'dense':'Dense','fanin':'Fan-in','fanout':'Fan-out','sparse':'Sparse','uniform':'Uniform'}
    rows_tex=[]
    for name in ['sparse','uniform','fanin','fanout','dense']:
        r=family.loc[name]
        rows_tex.append(f"{family_labels[name]} & {100*r.minimum:.4f} & {100*r['median']:.3f} & {100*r.maximum:.3f}"+r"\\")
    tex=r'''\begin{table}[tb]\centering\small
\caption{Selectivity (\%) in the original native grid, excluding the 30 explicit-empty configurations per family. Each row contains 270 configurations and repeats each predicate at three output counts. Selectivity is $|C_P|/(n-1)$.}
\begin{tabular}{@{}lrrr@{}}\toprule
Family & Minimum & Median & Maximum\\\midrule
'''+ '\n'.join(rows_tex)+r'''
\bottomrule\end{tabular}\label{tab:familyselectivity}\end{table}
'''
    (OUT/'sensitivity_family_selectivity.tex').write_text(tex,encoding='utf-8')
    rows_tex=[]
    names={'all':'All','empty':'Empty','nonempty unsaturated':'Nonempty, $s<99.98\\%$','saturated':'$s\\ge99.98\\%$'}
    for r in rows:
        rows_tex.append(f"{names[r['stratum']]} & {r['n']} & {100*r['selector_accuracy']:.1f} & {r['selector_geomean_regret']:.3f} & {100*r['always_forward_accuracy']:.1f} & {r['always_forward_geomean_regret']:.3f}"+r"\\")
    tex=r'''\begin{table}[tb]\centering\small
\caption{Frozen original-grid counter-informed selector, separated by selectivity $s$, compared with always choosing ForwardTopK. Acc. is winner accuracy (\%); regret is the geometric mean of selected/best measured latency. These are diagnostic predictions from observed counters.}
\begin{tabular}{@{}lrrrrr@{}}\toprule
& & \multicolumn{2}{c}{Counter-informed} & \multicolumn{2}{c}{Always forward}\\
Stratum & Queries & Acc. & Regret & Acc. & Regret\\\midrule
'''+ '\n'.join(rows_tex)+r'''
\bottomrule\end{tabular}\label{tab:selectorstrata}\end{table}
'''
    (OUT/'sensitivity_selector_strata.tex').write_text(tex,encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
