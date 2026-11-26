"""Rebuild the SENSITIVITY selectivity figure and winner tables from retained query medians."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_selectivity import ROOT,OUT,METHODS,STRATA,stratum,grouped

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--degree-csv',required=True)
    args=parser.parse_args()
    old=pd.read_csv(ROOT/'native_comparisons/results/native_strong_query_medians.csv')
    new=pd.read_csv(ROOT/args.degree_csv)
    cohorts=[('Original grid',old),('Degree sweep',new)]
    allrows=[]
    for name,df in cohorts:
        df['cohort']=name
        df['stratum']=df.selectivity.map(stratum)
        tab=grouped(df)
        tab.to_csv(OUT/(name.lower().replace(' ','_')+'_winners_by_selectivity.csv'))
        allrows.append(df)
    joined=pd.concat(allrows,ignore_index=True)
    joined.to_csv(OUT/'all_native_selectivity.csv',index=False)
    colors={'ForwardTopK':'#3576a8','ReverseRecursiveTopK':'#b56a20','LazyShared':'#278058'}
    markers=['o','^']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.2),gridspec_kw={'width_ratios':[.8,2.4,1.7]})
    for row,(name,df) in enumerate(cohorts):
        groups=[df[df.selectivity==0],df[(df.selectivity>0)&(df.selectivity<.9998)],df[df.selectivity>=.9998]]
        for col,(ax,sub) in enumerate(zip(axes[row],groups)):
            ax.axhline(1,color='#333333',lw=.8,ls='--',zorder=1)
            for method in METHODS:
                s=sub[sub.winner==method]
                xs=np.zeros(len(s)) if col==0 else s.selectivity*100
                ax.scatter(xs,s.lazy_over_best_materializer,s=11,alpha=.5,color=colors[method],marker=markers[row],linewidths=0,label=method,zorder=3)
            ax.set_yscale('log')
            ax.grid(axis='y',alpha=.15)
            ax.text(.04,.96,f'n = {len(sub)}',transform=ax.transAxes,va='top',fontsize=8)
            if col==0:
                ax.set_xlim(-.5,.5);ax.set_xticks([0],['0'])
                ax.set_ylabel(name+'\nLazy / best materializer')
            elif col==1:
                ax.set_xlim(-2,102)
                ax.set_xticks([0,25,50,75,100])
            else:
                ax.set_xlim(99.979,100.001)
                ax.set_xticks([99.98,99.99,100],['99.98','99.99','100'])
            if row==0:ax.set_title(['Empty','Nonempty: s < 99.98%','Near-universal: s ≥ 99.98%'][col],fontsize=9)
            if row==1:ax.set_xlabel('Selectivity (%)')
            if row==1 and col==1:
                wins=sub[sub.winner=='LazyShared']
                if len(wins):
                    first=wins.sort_values('selectivity').iloc[0]
                    ax.annotate(f"Lowest lazy win: {100*first.selectivity:.2f}%",xy=(100*first.selectivity,first.lazy_over_best_materializer),xytext=(5,.15),textcoords='data',fontsize=7,color='#1e6547',arrowprops=dict(arrowstyle='->',lw=.7,color='#1e6547'))
    # Empty-query ratios span a different scale; use a common range down each
    # column, while making the three stratum ranges independently visible.
    masks=[joined.selectivity==0,(joined.selectivity>0)&(joined.selectivity<.9998),joined.selectivity>=.9998]
    for col,mask in enumerate(masks):
        vals=joined.loc[mask,'lazy_over_best_materializer']
        lo=min(1,float(vals.min()))/1.6;hi=max(1,float(vals.max()))*2.2
        if col==1:lo=min(lo,.11)
        for ax in axes[:,col]:ax.set_ylim(lo,hi)
    handles=[plt.Line2D([0],[0],marker='o',ls='',color=colors[m],label=m,markersize=5) for m in METHODS]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,-.005),fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1],w_pad=.9,h_pad=1.1)
    fig.savefig(OUT/'sensitivity_selectivity.pdf',bbox_inches='tight')
    fig.savefig(OUT/'sensitivity_selectivity.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    names=[r'$s=0$',r'$0<s<1\%$',r'$1\%\le s<10\%$',r'$10\%\le s<50\%$',r'$50\%\le s<90\%$',r'$90\%\le s<99.98\%$',r'$99.98\%\le s\le100\%$']
    a,b=[grouped(df) for _,df in cohorts]
    body=[]
    for name,bin_name in zip(names,STRATA):
        vals=[int(v) for tab in [a,b] for v in tab.loc[bin_name,METHODS]]
        body.append(name+' & '+' & '.join(map(str,vals))+r'\\')
    body.append(r'\midrule Total & '+' & '.join(str(int(v)) for tab in [a,b] for v in tab.sum())+r'\\')
    tex=r'''\begin{table}[tb]\centering\small
\caption{Fastest exact plan by per-configuration median of three jointly timed repetitions, stratified by selectivity $s=|C_P|/(n-1)$. F/R/L denote ForwardTopK, ReverseRecursiveTopK and LazyShared. Counts are descriptive; repeated output counts share predicates.}
\begin{tabular}{@{}lrrrrrr@{}}\toprule
& \multicolumn{3}{c}{Original grid} & \multicolumn{3}{c}{Degree sweep}\\
Selectivity stratum & F & R & L & F & R & L\\\midrule
'''+ '\n'.join(body)+r'''
\bottomrule\end{tabular}\label{tab:winners}\end{table}
'''
    (OUT/'sensitivity_winner_table.tex').write_text(tex,encoding='utf-8')
    report={name:{'configurations':len(df),'nonforward_wins':int((df.winner!='ForwardTopK').sum()),'minimum_nonforward_selectivity':float(df.loc[df.winner!='ForwardTopK','selectivity'].min()),'minimum_lazy_selectivity':float(df.loc[df.winner=='LazyShared','selectivity'].min()),'band_1_to_90_percent':int(((df.selectivity>=.01)&(df.selectivity<.9)).sum())} for name,df in cohorts}
    (OUT/'combined_selectivity_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
