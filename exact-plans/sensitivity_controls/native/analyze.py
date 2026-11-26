"""Audit and analyze the independently timed SENSITIVITY WSL native cohorts."""
from pathlib import Path
import hashlib,json,math,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
NATIVE=ROOT.parents[1]/'native_comparisons'/'native'
METHODS=['ForwardTopK','ReverseRecursiveTopK','LazyShared']
SIX=['ForwardGF','ForwardResume','ReverseRecursive','ReverseResume','LazyShared','LazySeparate']
BINS=['empty','(0,.01)','[.01,.1)','[.1,.5)','[.5,.9)','[.9,.9998)','[.9998,1]']
META=['family','seed','n','d','m','h','k','anchor','anchor_degree','eligible','query_id']
COUNTERS=['edge_examinations','state_entries','distances','sort_items','root_checks','pauses','state_bytes']
def stratum(x):
 if x==0:return BINS[0]
 return BINS[int(np.searchsorted([.01,.1,.5,.9,.9998],x,side='right'))+1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_audit(name,queries,methods):
 df=pd.read_csv(ROOT/name)
 assert len(df)==queries*len(methods)*3,(name,len(df))
 assert sorted(df.query_id.unique())==list(range(queries))
 assert set(df.method)==set(methods)
 assert set(df.rep)=={0,1,2}
 assert df.exact.eq(1).all() and np.isfinite(df.ms).all() and df.ms.ge(0).all()
 assert not df.duplicated(['query_id','method','rep']).any()
 assert df.groupby(['query_id','method']).size().eq(3).all()
 assert df.groupby(['query_id','method'])[COUNTERS].nunique().le(1).all().all()
 assert df.groupby('query_id')[META[:-1]].nunique().le(1).all().all()
 med=df.groupby(META+['method'],as_index=False)[['ms']+COUNTERS].median()
 assert med.ms.gt(0).all(),'a three-repetition median is below clock resolution'
 return df,med
def wide(med):
 out=med.pivot(index=META,columns='method',values='ms').reset_index()
 out['selectivity']=out.eligible/(out.n-1)
 out['stratum']=out.selectivity.map(stratum)
 return out
def ratio_ci(df,num,den,cluster_cols,seed=20261025,reps=5000):
 # Bootstrap structural predicates, keeping all k variants together.
 groups=[x for _,x in df.groupby(cluster_cols,sort=True)]
 sums=np.array([[g[num].sum(),g[den].sum(),len(g)] for g in groups])
 rng=np.random.default_rng(seed)
 draws=rng.integers(0,len(groups),(reps,len(groups)))
 totals=sums[draws].sum(axis=1)
 ratios=totals[:,0]/totals[:,1]
 query_ratios=df[num]/df[den]
 return {'ratio_of_means':float(df[num].mean()/df[den].mean()),'ci95':np.quantile(ratios,[.025,.975]).tolist(),'clusters':len(groups),'queries':len(df),'geomean_query_ratio':float(np.exp(np.log(query_ratios).mean())),'median_query_ratio':float(query_ratios.median()),'q25_query_ratio':float(query_ratios.quantile(.25)),'q75_query_ratio':float(query_ratios.quantile(.75))}

def main():
 assert (ROOT/'COMPLETE').exists(),'formal runs are incomplete'
 env=json.loads((ROOT/'environment.json').read_text())
 omitted_binaries=[]
 for name,expected in env['hashes'].items():
  if not (ROOT/name).exists() and name in ['degree_sweep','dimension_gaussian','dimension_origin','allocation_control']:
   omitted_binaries.append(name);continue
  assert sha(ROOT/name)==expected,('changed after measured',name)
 assert sha(ROOT/'strong_base.cpp')==sha(NATIVE/'strong_baseline.cpp')
 assert sha(ROOT/'landscape_base.cpp')==sha(NATIVE/'landscape.cpp')
 assert sha(ROOT/'dimension_origin.cpp')==sha(NATIVE/'dimension_control.cpp')
 raw,med=read_audit('degree_sweep_raw.csv',1440,METHODS)
 out=wide(med);out['degree']=out.family.str.removeprefix('degree_').astype(int)
 assert out.groupby(['n','degree','seed','h']).eligible.nunique().eq(1).all()
 assert out.m.eq(2*(out.n-1)*out.degree).all()
 assert set(out.degree)=={4,6,8,10,12,16,24,32}
 assert set(out.n)=={10000,100000} and set(out.seed)==set(range(10))
 assert set(out.h)=={2,4,6} and set(out.k)=={1,10,100}
 assert out.groupby(['n','degree','seed','h','k']).size().eq(1).all()
 out['best_materializer']=out[['ForwardTopK','ReverseRecursiveTopK']].min(axis=1)
 out['lazy_over_best_materializer']=out.LazyShared/out.best_materializer
 out['winner']=out[METHODS].idxmin(axis=1)
 sorted_times=np.sort(out[METHODS].to_numpy(),axis=1)
 out['winner_over_runner_up']=sorted_times[:,0]/sorted_times[:,1]
 out.to_csv(ROOT/'degree_query_medians.csv',index=False)
 med.to_csv(ROOT/'degree_method_medians.csv',index=False)
 win=[]
 for cohort,g in [('all',out)]+[(f'n={n}',x) for n,x in out.groupby('n')]:
  for b in BINS:
   z=g[g.stratum==b]
   rec={'cohort':cohort,'stratum':b,'queries':len(z),'selectivity_min':z.selectivity.min(),'selectivity_max':z.selectivity.max()}
   for m in METHODS:rec[m+'_wins']=int(z.winner.eq(m).sum())
   rec['lazy_over_best_ratio_of_means']=z.LazyShared.mean()/z.best_materializer.mean() if len(z) else None
   win.append(rec)
 pd.DataFrame(win).to_csv(ROOT/'degree_strata.csv',index=False)
 winner_summary={m:{'wins':int(out.winner.eq(m).sum()),'min_selectivity':float(out.loc[out.winner.eq(m),'selectivity'].min()) if out.winner.eq(m).any() else None,'max_selectivity':float(out.loc[out.winner.eq(m),'selectivity'].max()) if out.winner.eq(m).any() else None} for m in METHODS}
 for m in METHODS:
  clear=out[out.winner.eq(m)&out.winner_over_runner_up.le(.95)]
  winner_summary[m]['wins_with_at_least_5percent_margin']=len(clear)
  winner_summary[m]['min_selectivity_at_least_5percent_margin']=float(clear.selectivity.min()) if len(clear) else None
 boundary=out[out.n.eq(100000)&out.degree.eq(8)&out.h.eq(6)&out.k.eq(1)]
 boundary_stats={'selection':'Post-hoc descriptive summary of the full ten-seed cell containing the minimum-selectivity lazy win; no seed filtered by winner. Not an independently selected confirmatory hypothesis.','n':100000,'degree':8,'h':6,'k':1,'queries':len(boundary),'lazy_wins':int(boundary.winner.eq('LazyShared').sum()),'selectivity_min':float(boundary.selectivity.min()),'selectivity_max':float(boundary.selectivity.max()),'LazyShared_over_ForwardTopK':ratio_ci(boundary,'LazyShared','ForwardTopK',['seed']),'LazyShared_over_best_materializer':ratio_ci(boundary,'LazyShared','best_materializer',['seed']),'mean_query_medians_ms':{m:float(boundary[m].mean()) for m in METHODS}}
 (ROOT/'degree_boundary_cell.json').write_text(json.dumps(boundary_stats,indent=2),encoding='utf-8')
 # Endpoints reproduce the unchanged degree-4/32 NATIVE generators, despite different hardware.
 frozen_raw=pd.read_csv(NATIVE/'strong_synthetic_raw.csv')
 # The NATIVE explicit-empty query shares h=4,k=10 with an ordinary query, so
 # exclude its distinct anchor before matching the ordinary degree endpoints.
 frozen=frozen_raw[frozen_raw.anchor.ne(frozen_raw.n-1)].groupby(['family','seed','n','h','k','method'],as_index=False)[['eligible']+COUNTERS].median()
 endpoints=med[med.family.isin(['degree_4','degree_32'])].copy();endpoints['family']=endpoints.family.map({'degree_4':'uniform','degree_32':'dense'})
 joined=endpoints.merge(frozen,on=['family','seed','n','h','k','method'],suffixes=('_new','_r1'),validate='one_to_one')
 assert len(joined)==1080
 for c in ['eligible']+COUNTERS:assert joined[c+'_new'].eq(joined[c+'_r1']).all(),c
 dimension={};dimension_frames=[];topology_groups=0
 for kind in ['origin','gaussian']:
  d,dm=read_audit(f'dimension_{kind}_raw.csv',120,SIX)
  dw=wide(dm);dw['query_kind']=kind;dimension_frames.append(dw)
  for method in ['ForwardGF','ForwardResume']:
   z=dm[dm.method.eq(method)]
   group=z.groupby(['family','seed','h'])
   assert group[['eligible','edge_examinations','state_entries']].nunique().eq(1).all().all()
   topology_groups+=len(group)
  rows=[]
  for dim,g in dw.groupby('d'):
   rec={'d':int(dim),'queries':len(g),'LazyShared_over_ForwardGF_ratio_of_means':float(g.LazyShared.mean()/g.ForwardGF.mean())}
   rec.update({m+'_mean_ms':float(g[m].mean()) for m in SIX});rows.append(rec)
  dimension[kind]=rows
  dm.to_csv(ROOT/f'dimension_{kind}_method_medians.csv',index=False)
 pd.concat(dimension_frames).to_csv(ROOT/'dimension_query_medians.csv',index=False)
 # Gaussian and origin should have identical full membership work, while lazy probes may vary.
 origin=pd.read_csv(ROOT/'dimension_origin_method_medians.csv');gaussian=pd.read_csv(ROOT/'dimension_gaussian_method_medians.csv')
 dx=origin.merge(gaussian,on=['family','seed','n','d','h','k','method'],suffixes=('_origin','_gaussian'),validate='one_to_one')
 plain=dx[dx.method.isin(['ForwardGF','ForwardResume','ReverseRecursive','ReverseResume'])]
 for c in ['eligible','edge_examinations','state_entries','distances','sort_items','root_checks']:assert plain[c+'_origin'].eq(plain[c+'_gaussian']).all(),('dimension query differs membership',c)
 allocation_raw,am=read_audit('allocation_raw.csv',1500,['ForwardGF','ForwardReserved','ForwardResume'])
 aw=wide(am);aw['cluster_seed']=np.where(aw.family.eq('sparse')&aw.eligible.eq(0),-1,aw.seed)
 aw.to_csv(ROOT/'allocation_query_medians.csv',index=False)
 ac=am.groupby('query_id')[['eligible','edge_examinations','state_entries','distances','sort_items']].nunique()
 assert ac.eq(1).all().all()
 reserve=am[am.method.isin(['ForwardReserved','ForwardResume'])].groupby('query_id').state_bytes.nunique();assert reserve.eq(1).all()
 allocation={}
 for name,g in [('all',aw),('ForwardReserved_at_least_10us',aw[aw.ForwardReserved.ge(.01)])]+[(b,aw[aw.stratum.eq(b)]) for b in BINS if aw.stratum.eq(b).any()]:
  allocation[name]={num+'/'+den:ratio_ci(g,num,den,['family','n','cluster_seed','h','anchor']) for num,den in [('ForwardResume','ForwardGF'),('ForwardResume','ForwardReserved'),('ForwardReserved','ForwardGF')]}
 # Cohort metadata matches original query design exactly, independent of times.
 old=pd.read_csv(NATIVE/'synthetic_raw.csv');oldmeta=old[META].drop_duplicates().sort_values('query_id').reset_index(drop=True)
 newmeta=aw[META].sort_values('query_id').reset_index(drop=True);assert oldmeta.equals(newmeta)
 report={'status':'PASS','environment':'Local WSL; absolute times are not pooled with NATIVE Xeon data','timed_rows':len(raw)+2160*2+len(allocation_raw),'degree':{'queries':len(out),'wins':winner_summary,'intermediate_1_to_90_percent':int(((out.selectivity>=.01)&(out.selectivity<.9)).sum()),'intermediate_1_to_90_by_n':{str(n):int(((g.selectivity>=.01)&(g.selectivity<.9)).sum()) for n,g in out.groupby('n')},'endpoint_matches_NATIVE_counter_rows':len(joined)},'dimension':dimension,'dimension_topology_invariant_groups':topology_groups,'dimension_topology_violations':0,'allocation':allocation,'source_hashes':env['hashes'],'unshipped_binaries_not_rehashed':omitted_binaries,'csv_hashes':{p.name:sha(p) for p in ROOT.glob('*_raw.csv')},'memory':{}}
 report['timing_resolution']={'zero_single_repetitions':{p.name:int(pd.read_csv(p).ms.eq(0).sum()) for p in ROOT.glob('*_raw.csv')},'zero_query_medians':0,'allocation_ForwardReserved_medians_below_10us':int(aw.ForwardReserved.lt(.01).sum()),'handling':'Raw zeros retained, no imputation; every per-query/method median positive. Additional post-hoc resolution sensitivity uses ForwardReserved median >=10 microseconds, independent of observed winner.','probe':json.loads((ROOT/'clock_probe.json').read_text())}
 for log in ROOT.glob('*_run.log'):
  text=log.read_text();match=re.search(r'Maximum resident set size \(kbytes\): (\d+)',text)
  if match:report['memory'][log.name]={'rss_kib':int(match[1]),'rss_mib':int(match[1])/1024}
 (ROOT/'audit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 lines=['# Round-two native experiment report','','Status: PASS. All four measured cohorts completed; every warmup and timed output matched the flat-edge-scan membership oracle and exact ranking contract. There are 30,780 timed executions. All numbers below are from the new local WSL cohort on the i9-12900K, not the NATIVE Xeon server. Sources, seeds, protocol, raw records, logs and independent consistency audits are retained.','','## Degree sweep','','The fixed grid has 1,440 configurations and 480 graph-seed/predicate clusters. Degrees 4 and 32 overlap 360 NATIVE configurations, so combined cohort counts must not be interpreted as independent new predicates. A winner means the smallest median of three jointly timed repetitions; it is an observed per-configuration comparison, not a statistically established universal crossover.','','| Winner | Configurations | Minimum selectivity | Maximum selectivity | Wins by at least 5% over runner-up | Minimum selectivity for those wins |','|---|---:|---:|---:|---:|---:|']
 for m,v in winner_summary.items():lines.append(f"| {m} | {v['wins']} | {v['min_selectivity']} | {v['max_selectivity']} | {v['wins_with_at_least_5percent_margin']} | {v['min_selectivity_at_least_5percent_margin']} |")
 lines+=['','| Selectivity stratum | Configurations | ForwardTopK wins | ReverseRecursiveTopK wins | LazyShared wins |','|---|---:|---:|---:|---:|']
 for row in win:
  if row['cohort']=='all':lines.append(f"| {row['stratum']} | {row['queries']} | {row['ForwardTopK_wins']} | {row['ReverseRecursiveTopK_wins']} | {row['LazyShared_wins']} |")
 lines+=['',f"The 1%–90% band contains {report['degree']['intermediate_1_to_90_percent']} configurations, by n: {report['degree']['intermediate_1_to_90_by_n']}. All 1,080 method/configuration rows at degree-grid endpoints have identical eligibility and deterministic counters to the corresponding NATIVE runs. This is a cross-platform input/counter check, not a timing equivalence assumption.",'','Minimum-selectivity winning examples:']
 for m in METHODS:
  z=out[out.winner.eq(m)].sort_values(['selectivity','query_id']).head(1)
  if len(z):
   row=z.iloc[0];lines.append(f"- {m}: query {int(row.query_id)}, n={int(row.n)}, degree={int(row.degree)}, seed={int(row.seed)}, h={int(row.h)}, k={int(row.k)}, eligible={int(row.eligible)}, selectivity={row.selectivity:.9f}; Forward={row.ForwardTopK:.6f} ms, Reverse={row.ReverseRecursiveTopK:.6f} ms, Lazy={row.LazyShared:.6f} ms; winner/runner-up={row.winner_over_runner_up:.6f}.")
 b=boundary_stats['LazyShared_over_ForwardTopK']
 lines+=['',f"The full ten-seed cell n=100000, degree=8, h=6, k=1 has {boundary_stats['lazy_wins']} lazy wins and {10-boundary_stats['lazy_wins']} losses. Lazy/Forward ratio of mean query medians is {b['ratio_of_means']:.6f}, paired seed-bootstrap 95% CI [{b['ci95'][0]:.6f}, {b['ci95'][1]:.6f}]. All ten seeds are retained; the cell is a post-hoc descriptive illustration rather than an independently selected confirmatory test. This mixed outcome does not support a universal selectivity threshold. See degree_boundary_cell.json."]
 lines+=['','## Gaussian dimension control','','Both conditions use fixed topology across dimensions. The origin and Gaussian conditions are separate sequential timed cohorts on the same host, so the following is a sensitivity comparison with temporal machine variation still possible. Materializers in this control use the original full sort. Ratios are ratios of mean query medians over the 40 configurations at each dimension.','','| Dimension | Origin LazyShared/ForwardGF | Gaussian LazyShared/ForwardGF |','|---|---:|---:|']
 for a,b in zip(dimension['origin'],dimension['gaussian']):lines.append(f"| {a['d']} | {a['LazyShared_over_ForwardGF_ratio_of_means']:.6f} | {b['LazyShared_over_ForwardGF_ratio_of_means']:.6f} |")
 lines+=['',f"The audit checks {topology_groups} forward-method/family/seed/word groups across dimensions with zero topology/counter violations. Origin and Gaussian full-membership methods also have identical graph-work counters. Lazy probe work is allowed to change because the query-dependent rank order changes.",'','## Equal-reservation control','','All 1,500 original configurations were jointly timed under the three allocation conditions. The three methods have identical graph-work, distance and ranking counts; ForwardReserved and ForwardResume have identical selected-state capacities. The original 1.134 reported in NATIVE was a median of per-query ratios. Comparable medians and separate ratio-of-means estimates are given here. Confidence intervals resample 573 structural predicates, keeping k variants together and merging repeated sparse-empty predicates across seeds.','','| Comparison | Median per-query ratio | Q25–Q75 | Ratio of means | Cluster-bootstrap 95% CI for ratio of means |','|---|---:|---:|---:|---:|']
 for name,v in allocation['all'].items():lines.append(f"| {name} | {v['median_query_ratio']:.6f} | {v['q25_query_ratio']:.6f}–{v['q75_query_ratio']:.6f} | {v['ratio_of_means']:.6f} | [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |")
 lines+=['',f"Timing-resolution caveat: {report['timing_resolution']['zero_single_repetitions']} individual raw repetitions are zero; every three-repetition median is positive. Raw zeros are retained without imputation. Clock probe: {report['timing_resolution']['probe']}. The post-hoc resolution sensitivity retains configurations with ForwardReserved median >=10 microseconds (not selected by winning method):"]
 for name,v in allocation['ForwardReserved_at_least_10us'].items():lines.append(f"- {name}, {v['queries']} configurations: median ratio {v['median_query_ratio']:.6f}; ratio of means {v['ratio_of_means']:.6f}, 95% CI [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}].")
 lines+=['','These are implementation- and machine-specific controls; they do not estimate an intrinsic cost of resumability. Full per-stratum results and process RSS are in audit_report.json. RSS includes input data and oracle state and must not be confused with the selected per-request state_bytes model.','','## Files','','- `degree_query_medians.csv`: per-query strong-method medians, degree, selectivity, winner, and lazy/best-materializer ratio.','- `degree_strata.csv`: explicit empty/intermediate/saturated bins, both pooled within this cohort and split by n.','- `dimension_query_medians.csv`: both query-distribution conditions with original full-sort controls.','- `allocation_query_medians.csv`: all matched allocation conditions.','- `audit_report.json`: counts, invariants, hashes, detailed estimates and RSS.','- `protocol.json`, `environment.json`, `host_environment.json`, `run_local.sh`: design and reproducibility.','']
 (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k in ['status','timed_rows','degree','dimension','allocation','memory']},indent=2))
if __name__=='__main__':main()
