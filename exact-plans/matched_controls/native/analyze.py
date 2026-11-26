"""MATCHED M13 analysis. Read-only with respect to frozen protocol and raw measurements."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
SENSITIVITY=P.parents[1]/'sensitivity_controls'/'native'
METHODS=['ForwardTopK','ReverseRecursiveTopK','LazyShared']
KS=[1,2,3,5,10,100]
COUNTERS=['edge_examinations','state_entries','distances','sort_items','root_checks','pauses','state_bytes']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,x): (P/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
checks={}
def check(name,ok):
    checks[name]=bool(ok)
    if not ok: raise AssertionError(name)

raw=pd.read_csv(P/'k_sweep_raw.csv')
diag=pd.read_csv(P/'query_diagnostics.csv',dtype={x:str for x in ['eligible_fnv64','answer_fnv64','graph_fnv64','vectors_fnv64','query_fnv64']})
trace=pd.read_csv(P/'root_trace.csv')
proto=json.loads((P/'protocol.json').read_text())
env=json.loads((P/'environment.json').read_text())
integrity=json.loads((P/'measurement_integrity.json').read_text())
required_data={'k_sweep_raw.csv','query_diagnostics.csv','root_trace.csv','sensitivity_boundary_anchor_raw.csv'}
check('required_data_hashes',set(integrity['sha256'])==required_data and all((P/f).is_file() and sha(P/f)==h for f,h in integrity['sha256'].items()))
check('rows_540',len(raw)==540)
check('configurations_60',len(raw[['seed','k']].drop_duplicates())==60)
check('plan_cells_180',len(raw[['seed','k','method']].drop_duplicates())==180)
check('unique_raw_key',not raw.duplicated(['seed','k','method','rep']).any())
check('seeds_all_ten',set(raw.seed)==set(range(10)))
check('k_grid',set(raw.k)==set(KS))
check('methods_all_three',set(raw.method)==set(METHODS))
check('three_repeats',raw.groupby(['seed','k','method']).rep.apply(lambda x:set(x)=={0,1,2}).all())
check('exact_and_positive',(raw.exact==1).all() and (raw.ms>0).all())
check('fixed_parameters',(raw.n==100000).all() and (raw.d==64).all() and (raw.h==6).all() and (raw.m==1599984).all() and (raw.anchor_degree==8).all())
check('counters_deterministic',(raw.groupby(['seed','k','method'])[COUNTERS+['eligible','anchor']].nunique()==1).all().all())
check('oracle_diagnostics_60',len(diag)==60 and not diag.duplicated(['seed','k']).any())
check('predicate_query_graph_fixed_across_k',(diag.groupby('seed')[['eligible','anchor','eligible_fnv64','graph_fnv64','vectors_fnv64','query_fnv64','first_eligible_rank','first_root_edges','first_root_states']].nunique()==1).all().all())
for m in METHODS[:2]:
    check(m+'_graph_counters_fixed_across_k',(raw[raw.method==m].groupby('seed')[['edge_examinations','state_entries','root_checks','distances','sort_items']].nunique()==1).all().all())
lazy=raw[raw.method=='LazyShared'].merge(diag,on=['seed','k'],suffixes=('','_diag'),validate='many_to_one')
check('lazy_exact_kth_rank',(lazy.root_checks==lazy.kth_eligible_rank).all())
check('lazy_trace_edges',(lazy.edge_examinations==lazy.lazy_prefix_edges).all())
check('lazy_trace_states',(lazy.state_entries==lazy.lazy_prefix_states).all())
check('lazy_full_distance_sort',(lazy.distances==99999).all() and (lazy.sort_items==99999).all())
check('trace_unique_and_dense',not trace.duplicated(['seed','rank']).any() and trace.groupby('seed')['rank'].apply(lambda x:list(x)==list(range(1,len(x)+1))).all())
check('trace_increment_sums',trace.groupby('seed').apply(lambda x:np.array_equal(x.closure_new_edges.cumsum(),x.cumulative_edges) and np.array_equal(x.closure_new_states.cumsum(),x.cumulative_states),include_groups=False).all())
check('trace_hit_sums',trace.groupby('seed').apply(lambda x:np.array_equal(x.eligible.cumsum(),x.found),include_groups=False).all())
check('trace_final_100',(trace.groupby('seed').found.max()==100).all())
check('lazy_monotone_prefix',diag.sort_values('k').groupby('seed')[['kth_eligible_rank','lazy_prefix_edges','lazy_prefix_states']].apply(lambda x:(x.diff().dropna()>=0).all().all()).all())
hist=json.loads((P/'historical_reference.json').read_text())
check('strong_kernel_unchanged',sha(P/'strong_base.cpp')==hist['sensitivity_strong_base_sha256']==proto['source_sha256']['strong_base.cpp'])
check('frozen_measurement_source_hashes',all(sha(P/f)==h for f,h in proto['source_sha256'].items() if f!='freeze_protocol.py'))
# Only these two native binaries are intentionally omitted from public archives.
# Missing or changed sources/data are never accepted under this exception.
binary_names={'k_sweep','clock_probe'}
binary_verification={}
environment_ok=True
for f,h in env['hashes'].items():
    path=P/f
    if f in binary_names:
        status='not_distributed' if not path.is_file() else ('verified' if sha(path)==h else 'hash_mismatch')
        binary_verification[f]={'status':status,'recorded_sha256':h}
        environment_ok &= status!='hash_mismatch'
    else:
        environment_ok &= path.is_file() and sha(path)==h
check('environment_hashes',environment_ok and set(binary_verification)==binary_names)
check('all_exact_log','ALL_EXACT queries=60 plan_cells=180 timed=540' in (P/'run.log').read_text())
check('unit_tests','7776 exact comparisons' in (P/'tests.log').read_text() and '324 additional' in (P/'tests.log').read_text())
check('protocol_precedes_measurement',pd.Timestamp(proto['fixed_before_measurement_at_utc'])<pd.Timestamp((P/'STARTED').read_text().strip()))
check('cpu4_affinity',env['cpu_affinity']==[4])

# Historical endpoint invariants: no historical times are combined with MATCHED.
check('historical_subset_hash',sha(P/'sensitivity_boundary_anchor_raw.csv')==hist['retained_subset_sha256'])
sensitivity=pd.read_csv(P/'sensitivity_boundary_anchor_raw.csv')
r2cols=['n','d','m','h','anchor','anchor_degree','eligible']+COUNTERS
a=raw[raw.k.isin([1,10,100])].groupby(['seed','k','method'])[r2cols].first().sort_index()
b=sensitivity.groupby(['seed','k','method'])[r2cols].first().sort_index()
check('sensitivity_anchor_counter_membership_equivalence',a.equals(b))

mm=raw.groupby(['seed','k','method'],as_index=False)[['ms']+COUNTERS].median()
mm.to_csv(P/'method_medians.csv',index=False)
qm=mm.pivot(index=['seed','k'],columns='method',values='ms').reset_index()
qm=qm.merge(diag,on=['seed','k'],validate='one_to_one')
qm['best_materializer']=qm[METHODS[:2]].min(axis=1)
qm['lazy_over_best_materializer']=qm.LazyShared/qm.best_materializer
qm['lazy_over_forward']=qm.LazyShared/qm.ForwardTopK
qm['winner']=qm[METHODS].idxmin(axis=1)
qm.to_csv(P/'query_medians.csv',index=False)

def first_repeat_table(data,label):
    v=data[(data.k==1)&(data.rep==0)].copy()
    ans=[]
    for seed,g in v.groupby('seed'):
        by=g.set_index('method'); l=by.loc['LazyShared']; d=diag[(diag.seed==seed)&(diag.k==1)].iloc[0]
        med=data[(data.seed==seed)&(data.k==1)].groupby('method').ms.median()
        ans.append(dict(cohort=label,seed=int(seed),selectivity=float(l.eligible/(l.n-1)),eligible=int(l.eligible),
            forward_ms=float(by.loc['ForwardTopK','ms']),reverse_ms=float(by.loc['ReverseRecursiveTopK','ms']),lazy_ms=float(l.ms),
            lazy_over_forward=float(l.ms/by.loc['ForwardTopK','ms']),lazy_edges=int(l.edge_examinations),lazy_roots=int(l.root_checks),
            lazy_states=int(l.state_entries),lazy_distances=int(l.distances),lazy_sort_items=int(l.sort_items),
            first_eligible_rank=int(d.first_eligible_rank),first_root_edges=int(d.first_root_edges),first_root_states=int(d.first_root_states),
            first_rep_winner=str(by.ms.idxmin()),median_winner=str(med.idxmin()),
            forward_median_ms=float(med.ForwardTopK),reverse_median_ms=float(med.ReverseRecursiveTopK),lazy_median_ms=float(med.LazyShared),
            forward_edges=int(by.loc['ForwardTopK','edge_examinations']),reverse_edges=int(by.loc['ReverseRecursiveTopK','edge_examinations'])))
    z=pd.DataFrame(ans);z.to_csv(P/f'{label}_boundary_first_rep.csv',index=False)
    med=z.copy()
    for x in ['forward','reverse','lazy']:med[x+'_ms']=med[x+'_median_ms']
    med['lazy_over_forward']=med.lazy_ms/med.forward_ms
    med['latency_statistic']='median of three repetitions; counters constant across repetitions'
    med.to_csv(P/f'{label}_boundary_seed_counters.csv',index=False)
    data[(data.k==1)&(data.rep==0)].to_csv(P/f'{label}_k1_first_repetition_all_methods.csv',index=False)
    return z,med
old,old_med=first_repeat_table(sensitivity,'sensitivity');new,new_med=first_repeat_table(raw,'matched')
old_med[['seed','selectivity','lazy_edges','lazy_roots','first_eligible_rank','first_root_edges','lazy_over_forward','latency_statistic']].to_csv(P/'sensitivity_boundary_plot_data.csv',index=False)

rng=np.random.default_rng(20261025);idx=rng.integers(0,10,size=(10000,10))
matrix=qm.pivot(index='seed',columns='k',values='lazy_over_best_materializer').reindex(index=range(10),columns=KS).to_numpy()
boot=np.median(matrix[idx,:],axis=1)
relative=matrix/matrix[:,[0]];boot_relative=np.median(relative[idx,:],axis=1)
rows=[]
for j,k in enumerate(KS):
    s=qm[qm.k==k];lo,hi=np.quantile(boot[:,j],[.025,.975]);rlo,rhi=np.quantile(boot_relative[:,j],[.025,.975])
    rows.append(dict(k=k,seeds=10,forward_wins=int((s.winner=='ForwardTopK').sum()),reverse_wins=int((s.winner=='ReverseRecursiveTopK').sum()),lazy_wins=int((s.winner=='LazyShared').sum()),
        median_lazy_over_best=float(np.median(matrix[:,j])),ci95_low=float(lo),ci95_high=float(hi),
        median_ratio_relative_k1=float(np.median(relative[:,j])),relative_k1_ci95_low=float(rlo),relative_k1_ci95_high=float(rhi),
        median_forward_ms=float(s.ForwardTopK.median()),median_reverse_ms=float(s.ReverseRecursiveTopK.median()),median_lazy_ms=float(s.LazyShared.median()),
        lazy_roots_min=int(s.kth_eligible_rank.min()),lazy_roots_max=int(s.kth_eligible_rank.max()),
        lazy_edges_min=int(s.lazy_prefix_edges.min()),lazy_edges_max=int(s.lazy_prefix_edges.max())))
summary=pd.DataFrame(rows);summary.to_csv(P/'k_summary.csv',index=False)

# Source table data are retained; generated TeX contains no paper-side mutations.
lines=[r'\begin{tabular}{rrrrrl}',r'\toprule',r'$k$ & Forward & Reverse & Lazy & Roots & Lazy/best materializer (95\% CI)\\',r'\midrule']
for row in rows:
    lines.append(f"{row['k']} & {row['forward_wins']} & {row['reverse_wins']} & {row['lazy_wins']} & {row['lazy_roots_min']}--{row['lazy_roots_max']} & {row['median_lazy_over_best']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(P/'matched_k_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
lines=[r'\begin{tabular}{rrrrrr}',r'\toprule',r'Seed & Edges & Roots & Lazy ms & Forward ms & Lazy/forward\\',r'\midrule']
for row in old_med.itertuples():lines.append(f'{row.seed} & {row.lazy_edges:,} & {row.lazy_roots} & {row.lazy_ms:.2f} & {row.forward_ms:.2f} & {row.lazy_over_forward:.3f}'+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(P/'sensitivity_boundary_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

plt.rcParams.update({'font.size':10,'axes.titlesize':10,'axes.labelsize':10,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
ax[0].scatter(old_med.lazy_edges,old_med.lazy_over_forward,c='#245A81',s=30,zorder=3)
for r in old_med.itertuples():ax[0].annotate(str(r.seed),(r.lazy_edges,r.lazy_over_forward),xytext=(4,4),textcoords='offset points',fontsize=9)
ax[0].set_xscale('log');ax[0].set_xlabel('Lazy edge examinations (log scale)');ax[0].set_ylabel('Lazy / forward latency')
ax[0].set_title('(a) SENSITIVITY: $k=1$, all ten seeds')
ax[0].axhline(1,color='#555555',ls='--',lw=1)
for seed in range(10):ax[1].plot(KS,matrix[seed],color='#9AA8B1',alpha=.5,lw=.8,marker='o',markersize=2.5)
y=summary.median_lazy_over_best.to_numpy();low=summary.ci95_low.to_numpy();high=summary.ci95_high.to_numpy()
ax[1].errorbar(KS,y,yerr=np.stack([y-low,high-y]),color='#245A81',lw=1.6,marker='o',capsize=3,label='Paired median\nSeed-bootstrap 95% CI',zorder=5)
ax[1].axhline(1,color='#555555',ls='--',lw=1);ax[1].set_xscale('log');ax[1].set_xticks(KS,labels=[str(k) for k in KS]);ax[1].set_xlabel('Requested results, $k$ (log scale)');ax[1].set_ylabel('Lazy / best materializer latency');ax[1].set_title('(b) MATCHED: matched $k$ sweep, all seeds');ax[1].legend(loc='upper left',fontsize=9)
for a in ax:a.grid(axis='y',alpha=.2);a.spines[['top','right']].set_visible(False)
for ext in ['pdf','png']:fig.savefig(P/f'boundary_k_sweep.{ext}',dpi=220)
plt.close(fig)

analysis=dict(status='PASS',scope='Fixed MATCHED follow-up of a post-hoc SENSITIVITY uniform-topology cell; descriptive matched within-cohort comparisons, not a universal selectivity or k threshold',
    denominator=proto['denominators'],bootstrap=dict(replicates=10000,seed=20261025,unit='ten graph/predicate/query seeds; all k retained per resampled seed; percentiles of median paired ratios'),
    k_summary=rows,selectivity=dict(min=float(diag.selectivity.min()),max=float(diag.selectivity.max()),denominator=99999),
    sensitivity_k1_first_repeat=dict(all_seeds=10,lazy_edges_min=int(old.lazy_edges.min()),lazy_edges_max=int(old.lazy_edges.max()),edge_spread_ratio=float(old.lazy_edges.max()/old.lazy_edges.min()),
        roots_min=int(old.lazy_roots.min()),roots_max=int(old.lazy_roots.max()),first_eligible_rank_by_seed=dict(zip(old.seed.astype(str),old.first_eligible_rank.astype(int))),
        lazy_wins_first_repeat=int((old.first_rep_winner=='LazyShared').sum()),lazy_wins_median=int((old.median_winner=='LazyShared').sum())),
    matched_k1_first_repeat=dict(lazy_wins_first_repeat=int((new.first_rep_winner=='LazyShared').sum()),lazy_wins_median=int((new.median_winner=='LazyShared').sum())),
    graph_invariants=dict(strong_base_sha256=sha(P/'strong_base.cpp'),sensitivity_anchor_counter_equality=True,all_diagnostics_equal=True),
    interpretation=['The first eligible rank is 1 for nine seeds and 2 for seed 7, while reverse dependency-closure work varies greatly. Most edge-count variation therefore occurs even at the SAME first eligible rank.',
    'The cumulative kth eligible rank and the new dependency work of each tested root are distinct diagnostics. Neither is a free pre-query input, and no selectivity-only boundary follows.',
    'New k2/3/5 comprise 30 query configurations and 90 plan cells, not 90 independent query configurations.',
    'MATCHED and historical SENSITIVITY absolute times are never pooled. All MATCHED ratios match methods within the same seed/k/host cohort.'],
    binary_verification=binary_verification,checks=checks)
dump('analysis.json',analysis)
dump('audit_report.json',dict(status='PASS',checks=checks,check_count=len(checks),binary_verification=binary_verification,raw_sha256=sha(P/'k_sweep_raw.csv'),diagnostics_sha256=sha(P/'query_diagnostics.csv'),trace_sha256=sha(P/'root_trace.csv')))
report=['# MATCHED boundary k sweep','',
    'The fixed follow-up contains 60 query configurations, 180 plan cells, and 540 measured executions. The new k values contribute 30 configurations and 90 plan cells; k=1,10,100 are matched anchors measured again in this independent cohort. All ten seeds, including losses, enter every estimate.',
    '', 'Authorized-server SSH timed out. This cohort uses Ubuntu WSL on the recorded host with virtual CPU 4 affinity; frequency and exclusive physical-core ownership are not controlled. Timing uses the unchanged SENSITIVITY strong kernels. No historical absolute times are pooled.',
    '', '| k | Forward / reverse / lazy wins | Median lazy/best materializer | 95% seed-bootstrap CI | Lazy roots |', '|---:|---:|---:|---:|---:|']
for r in rows:report.append(f"| {r['k']} | {r['forward_wins']} / {r['reverse_wins']} / {r['lazy_wins']} | {r['median_lazy_over_best']:.4f} | [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | {r['lazy_roots_min']}–{r['lazy_roots_max']} |")
report+=['','The ratio is paired within each seed after taking the three-repetition median for each algorithm. Bootstrap percentile intervals use 10,000 resamples of the ten seeds, preserving the complete k vector. These finite-seed intervals are descriptive and conditional on this selected follow-up cell.',
    '',f"The historical SENSITIVITY k=1 first repetitions span {old.lazy_edges.min():,}–{old.lazy_edges.max():,} lazy edge examinations ({old.lazy_edges.max()/old.lazy_edges.min():.1f}×), at selectivity {diag.selectivity.min()*100:.4f}–{diag.selectivity.max()*100:.4f}%. First eligible rank is one in nine seeds and two in seed 7. Thus dependency-closure cost, not only first-hit rank, explains the variation at essentially fixed selectivity. The raw trace records each root's incremental edge/state work separately.",
    '', 'The first-repetition CSVs preserve every original single-run value and counter. The displayed boundary table and both figure panels use per-query three-repetition median latency; counters are constant across those repetitions. We recomputed the medians from raw rows rather than mixing repetition-level values. The left panel shows historical SENSITIVITY paired median latency ratios; the right panel shows MATCHED matched median ratios. Neither panel mixes absolute timing scales.',
    '', f"All {len(checks)} audit checks passed, including exact outputs, complete grid/repetitions, frozen source hashes, graph/query invariants, deterministic counters, root-trace equivalence, and SENSITIVITY endpoint counter equivalence. Every warmup and measured output was checked against independent flat-edge membership with scalar distance / identifier ranking. The unit suite includes 7,776 inherited and 324 new-k exact comparisons.",
    '', 'Artifact provenance: protocol.json was frozen before STARTED; environment.json stores compiler, platform, CPU affinity and hashes. run.log retains per-process resource measurements, tests.log retains exactness tests, and clock_probe.json retains clock resolution observations. Source and raw artifact SHA-256 values are retained in manifest.json.']
(P/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','checks':len(checks),'k_summary':rows},indent=2))
