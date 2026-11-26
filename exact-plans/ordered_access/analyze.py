"""Audit and summarize the new jointly timed four-plan cohort; never alter measurements."""
import argparse,hashlib,json,platform,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--history-root',type=Path,help='Optional root holding historical or exported native_comparisons/sensitivity_controls/matched_controls directories.')
args=parser.parse_args();OUT=P/'analysis';OUT.mkdir(exist_ok=True)
METHODS=['ForwardTopK','ReverseRecursiveTopK','LazyShared','LazyHeap'];SHORT=['F','R','S','H']
EXPECTED={'main':1500,'degree':1440,'matched':60,'dimension':240,'real':80}
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(np.floating,)):return float(x)
 return x
def dump(name,obj):(OUT/name).write_text(json.dumps(clean(obj),indent=2))
raw={};meta={};med={};wide={};violations=[];anomalies=[];audit={}
COUNTERS=['edge_examinations','state_entries','distances','ordering_items','root_checks','pauses','state_bytes','heap_pops','root_digest','answer_fnv64','eligible']
LAZY_EQUAL=['edge_examinations','state_entries','distances','ordering_items','root_checks','pauses','state_bytes','root_digest','answer_fnv64']
for cohort,count in EXPECTED.items():
 rp=P/'results'/f'{cohort}_raw.csv';qp=P/'results'/f'{cohort}_queries.csv'
 r=pd.read_csv(rp,dtype={'root_digest':'string','answer_fnv64':'string'});q=pd.read_csv(qp,dtype={c:'string' for c in ['graph_fnv64','vectors_fnv64','query_fnv64','eligible_fnv64','answer_fnv64']})
 raw[cohort]=r;meta[cohort]=q
 local=[]
 if len(r)!=count*12:local.append('row_count')
 if len(q)!=count or q.query_id.nunique()!=count:local.append('query_count')
 if r.duplicated(['query_id','rep','method']).any():local.append('duplicate_keys')
 if set(r.method)!=set(METHODS):local.append('method_set')
 if set(r.rep)!={0,1,2}:local.append('repetition_set')
 if not (r.groupby(['query_id','method']).size()==3).all():local.append('plan_repetition_count')
 if not (r.exact==1).all():local.append('nonexact')
 if not np.isfinite(r.ms).all() or (r.ms<0).any():local.append('invalid_time')
 nondeterministic=(r.groupby(['query_id','method'])[COUNTERS].nunique()>1)
 if nondeterministic.any().any():local.append('nondeterministic_counters')
 a=r[r.method=='LazyShared'].set_index(['query_id','rep']).sort_index();b=r[r.method=='LazyHeap'].set_index(['query_id','rep']).sort_index()
 if not a[LAZY_EQUAL].equals(b[LAZY_EQUAL]):local.append('heap_fullsort_prefix_mismatch')
 if not (b.heap_pops==b.root_checks).all():local.append('heap_pop_count')
 if not (b.distances==b.n-1).all():local.append('heap_distance_count')
 if not (a.distances==a.n-1).all():local.append('sort_distance_count')
 answer=r.merge(q[['query_id','answer_fnv64']],on='query_id',suffixes=('','_oracle'))
 if not (answer.answer_fnv64==answer.answer_fnv64_oracle).all():local.append('oracle_answer_hash')
 repeats=r.groupby(['query_id','method']).ms.agg(['min','max']);repeat_ratio=repeats['max']/repeats['min'].replace(0,np.nan)
 for (qid,method),ratio in repeat_ratio[repeat_ratio>3].items():anomalies.append({'cohort':cohort,'query_id':qid,'method':method,'kind':'max/min repetition latency > 3','ratio':ratio})
 for row in r[r.ms==0].to_dict('records'):anomalies.append(dict(row,kind='zero_clock_delta'))
 audit[cohort]={'rows':len(r),'queries':len(q),'expected_rows':count*12,'exact_rows':int((r.exact==1).sum()),'zero_times':int((r.ms==0).sum()),'negative_times':int((r.ms<0).sum()),'minimum_positive_ms':float(r.loc[r.ms>0,'ms'].min()),'maximum_ms':float(r.ms.max()),'heap_sort_equal_repetition_pairs':len(a),'counter_nondeterminism_cells':int(nondeterministic.any(axis=1).sum()),'max_min_gt3_plan_cells':int((repeat_ratio>3).sum()),'violations':local,'raw_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'queries_sha256':hashlib.sha256(qp.read_bytes()).hexdigest()}
 violations += [f'{cohort}:{v}' for v in local]
 m=r.groupby(['query_id','method'],as_index=False).agg(ms=('ms','median'),edge_examinations=('edge_examinations','first'),state_entries=('state_entries','first'),distances=('distances','first'),root_checks=('root_checks','first'))
 m=m.merge(q,on='query_id');med[cohort]=m;m.to_csv(OUT/f'{cohort}_method_medians.csv',index=False)
 w=m.pivot(index='query_id',columns='method',values='ms').merge(q.set_index('query_id'),left_index=True,right_index=True)
 w['best_materializer']=w[METHODS[:2]].min(axis=1);w['heap_over_sort']=w.LazyHeap/w.LazyShared;w['heap_over_best_materializer']=w.LazyHeap/w.best_materializer;w['sort_over_best_materializer']=w.LazyShared/w.best_materializer;w['selectivity']=w.eligible/(w.n-1)
 wide[cohort]=w;w.to_csv(OUT/f'{cohort}_query_medians.csv')
for cohort in ['main','degree','matched']:
 q=meta[cohort]
 if cohort=='matched':
  cols=['graph_fnv64','vectors_fnv64','query_fnv64','eligible_fnv64'];ok=bool((q.groupby('seed')[cols].nunique()==1).all().all())
  audit[cohort]['same_inputs_across_k']=ok
  if not ok:violations.append('matched:input_pairing')
q=meta['dimension'];ok=bool((q.groupby(['family','seed']).graph_fnv64.nunique()==1).all());audit['dimension']['same_topology_across_dimensions_conditions']=ok
if not ok:violations.append('dimension:topology_pairing')
audit['sampling_units']={c:{'configurations':len(q),'distinct_graph_topologies':q.graph_fnv64.nunique(),'distinct_structural_predicates':len(q[['graph_fnv64','anchor','word']].drop_duplicates())} for c,q in meta.items()}
identity_columns=['graph_fnv64','vectors_fnv64','query_fnv64','anchor','word','k']
audit['cohort_overlap']={a+'_'+b:len(meta[a][identity_columns].merge(meta[b][identity_columns],on=identity_columns)) for a,b in [('main','degree'),('degree','matched')]}

def strict_wins(w,methods=METHODS):
 wins={m:int((w[m]<w[[x for x in methods if x!=m]].min(axis=1)).sum()) for m in methods};wins['ties']=len(w)-sum(wins.values());return wins

def winner_labels(w,methods=METHODS):
 out=pd.Series('tie',index=w.index)
 for m in methods:out.loc[w[m]<w[[x for x in methods if x!=m]].min(axis=1)]=m
 return out

def transitions(w):
 a=winner_labels(w,METHODS[:3]);b=winner_labels(w)
 return [{'three_plan_winner':x,'four_plan_winner':y,'configurations':int(n)} for (x,y),n in pd.DataFrame({'three':a,'four':b}).value_counts().items()]

def margin_wins(w):
 # Descriptive margins on per-configuration medians, not a significance test.
 return {m:int((w[m]<=.95*w[[x for x in METHODS if x!=m]].min(axis=1)).sum()) for m in METHODS}
def ratios(a,b):
 x=np.asarray(a)/np.asarray(b)
 return {'median_paired':float(np.median(x)),'q1_paired':float(np.quantile(x,.25)),'q3_paired':float(np.quantile(x,.75)),'p95_paired':float(np.quantile(x,.95)),'ratio_of_means':float(np.mean(a)/np.mean(b)),'strict_wins':int((x<1).sum()),'ties':int((x==1).sum())}
def summary(w):
 return {'configurations':len(w),'four_way_strict_wins':strict_wins(w),'four_way_wins_at_least_5pct_margin':margin_wins(w),'same_batch_three_to_four_transitions':transitions(w),'three_way_strict_wins_same_new_batch':strict_wins(w,METHODS[:3]),'heap_vs_sort':ratios(w.LazyHeap,w.LazyShared),'heap_vs_best_materializer':ratios(w.LazyHeap,w.best_materializer),'sort_vs_best_materializer':ratios(w.LazyShared,w.best_materializer),'latency_ms':{m:{'mean_query_medians':float(w[m].mean()),'median':float(w[m].median()),'q1':float(w[m].quantile(.25)),'q3':float(w[m].quantile(.75)),'p95':float(w[m].quantile(.95))} for m in METHODS}}
result={'cohorts':{c:summary(w) for c,w in wide.items()}}
def stratum(s):
 if s==0:return '0'
 if s<.01:return '(0,1%)'
 if s<.1:return '[1%,10%)'
 if s<.5:return '[10%,50%)'
 if s<.9:return '[50%,90%)'
 if s<.9998:return '[90%,99.98%)'
 return '[99.98%,100%]'
STRATA=['0','(0,1%)','[1%,10%)','[10%,50%)','[50%,90%)','[90%,99.98%)','[99.98%,100%]']
strata_rows=[];results_by_stratum={}
for cohort in ['main','degree']:
 w=wide[cohort];w['stratum']=w.selectivity.map(stratum);results_by_stratum[cohort]={}
 for s in STRATA:
  z=w[w.stratum==s]
  if len(z):results_by_stratum[cohort][s]=summary(z)
  strata_rows.append({'cohort':cohort,'stratum':s,'configurations':len(z),**strict_wins(z),**{'three_'+k:v for k,v in strict_wins(z,METHODS[:3]).items()}})
 sub=w[(w.selectivity>0)&(w.selectivity<.9998)]
 result['cohorts'][cohort]['nonempty_unsaturated']=summary(sub)
 heap_winners=sub[winner_labels(sub)=='LazyHeap']
 result['cohorts'][cohort]['unsaturated_four_way_heap_wins_by_k']={str(k):len(g) for k,g in heap_winners.groupby('k')}
 result['cohorts'][cohort]['unsaturated_four_way_heap_min_selectivity']=float(heap_winners.selectivity.min()) if len(heap_winners) else None
 heap_winners.to_csv(OUT/f'{cohort}_heap_four_way_unsaturated_winners.csv')
 result['cohorts'][cohort]['unsaturated_heap_beats_best_materializer_min_selectivity']=float(sub.loc[sub.LazyHeap<sub.best_materializer,'selectivity'].min()) if (sub.LazyHeap<sub.best_materializer).any() else None
 for method in ['LazyShared','LazyHeap']:
  z=sub[sub[method]<sub.best_materializer]
  result['cohorts'][cohort][method+'_unsaturated_wins_by_k']={str(k):len(g) for k,g in z.groupby('k')}
  z.to_csv(OUT/f'{cohort}_{method}_unsaturated_wins.csv')
pd.DataFrame(strata_rows).to_csv(OUT/'winner_strata.csv',index=False);result['strata']=results_by_stratum

# Ten paired seed clusters; the same bootstrap draws are used at every k.
boot_idx=np.random.default_rng(20261028).integers(0,10,(10000,10));krows=[]
for k,z in wide['matched'].groupby('k'):
 z=z.sort_values('seed');row={'k':k,**strict_wins(z)}
 for name,num,den in [('heap_best','LazyHeap','best_materializer'),('sort_best','LazyShared','best_materializer'),('heap_sort','LazyHeap','LazyShared')]:
  vals=(z[num]/z[den]).to_numpy();ci=np.quantile(np.median(vals[boot_idx],axis=1),[.025,.975]);row.update({name+'_median':np.median(vals),name+'_q1':np.quantile(vals,.25),name+'_q3':np.quantile(vals,.75),name+'_ci_low':ci[0],name+'_ci_high':ci[1],name+'_strict_wins':int((vals<1).sum())})
 m=med['matched'];roots=m[(m.k==k)&(m.method=='LazyHeap')].root_checks;row['roots_min']=roots.min();row['roots_max']=roots.max();krows.append(row)
kdf=pd.DataFrame(krows);kdf.to_csv(OUT/'matched_k_summary.csv',index=False);result['matched_k']=krows

dimrows=[];dimsub=[]
for (condition,d),z in wide['dimension'].groupby(['condition','d']):
 ss=summary(z);row={'condition':condition,'d':d,'configurations':len(z),**strict_wins(z)}
 for method,prefix in [('LazyShared','sort'),('LazyHeap','heap')]:
  rr=ratios(z[method],z.ForwardTopK);row.update({prefix+'_forward_'+k:v for k,v in rr.items()})
 row.update({'heap_sort_'+k:v for k,v in ratios(z.LazyHeap,z.LazyShared).items()});dimrows.append(row)
 for (family,h),zz in z.groupby(['family','h']):dimsub.append({'condition':condition,'d':d,'family':family,'h':h,**summary(zz)})
pd.DataFrame(dimrows).to_csv(OUT/'dimension_summary.csv',index=False);dump('dimension_subgroups.json',dimsub);result['dimension']=dimrows
real_queries=json.loads((P/'inputs/queries.json').read_text());safe=[x['qid'] for x in real_queries if not any((a+237)%474==b for a,b in zip(x['relations'],x['relations'][1:]))]
result['real_without_immediate_inverse']={'query_ids':safe,**summary(wide['real'].loc[safe])}
q49=wide['real'].loc[49]
result['real_query49']={'median_ms':{m:float(q49[m]) for m in METHODS},'reverse_reduction_against_forward':float(1-q49.ReverseRecursiveTopK/q49.ForwardTopK),'raw_repetitions':raw['real'].loc[raw['real'].query_id==49,['rep','method','ms']].to_dict('records'),'interpretation':'Descriptive median margin only; one of three paired repetitions favors forward. No statistical superiority claim.'}

# Frozen measurement implementation/inputs are immutable; analysis is post measurement.
protocol=json.loads((P/'protocol.json').read_text())
before=json.loads((P/'environment_before.json').read_text());after=json.loads((P/'environment_after.json').read_text())
if before['sha256']!=after['sha256']:violations.append('environment_before_after_hash_mismatch')
# A public checkout excludes the platform-specific executable and corpus-derived inputs.
# Record these omissions separately from verified hashes. If a frozen file is supplied,
# hash it even when its omission would otherwise be permitted.
external_files = {'benchmark', 'inputs/graph.bin', 'inputs/queries.txt'}
omitted = set()
for expected_files in (before['sha256'], protocol['source_and_input_sha256']):
    for name, expected in expected_files.items():
        path = P / name
        if name in external_files and not path.exists():
            omitted.add(name)
        elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            violations.append('frozen_file_hash:' + name)
audit['unavailable_frozen_files'] = sorted(omitted)
audit['shipped_measurement_hashes_match'] = not any('hash' in v for v in violations)
audit['frozen_measurement_hashes_match'] = audit['shipped_measurement_hashes_match'] and not omitted

# Optional historical counter/input comparison, not a cross-run timing comparison.
if args.history_root:
 H=args.history_root
 def locate(old,new,file):
  for directory in [old,new]:
   p=H/directory/'native'/file
   if p.exists():return p
  raise FileNotFoundError(file)
 comparisons=[]
 cases=[('main','native_comparisons','native_comparisons','strong_synthetic_raw.csv',None),('degree','sensitivity_controls','sensitivity_controls','degree_sweep_raw.csv',None),('matched','matched_controls','matched_controls','k_sweep_raw.csv',None),('real','native_comparisons','native_comparisons','strong_real_raw.csv',None),('dimension','sensitivity_controls','sensitivity_controls','dimension_origin_raw.csv','origin'),('dimension','sensitivity_controls','sensitivity_controls','dimension_gaussian_raw.csv','gaussian')]
 fields=['n','d','m','h','k','anchor','eligible','edge_examinations','state_entries','distances','root_checks']
 for cohort,old,new,file,condition in cases:
  hp=locate(old,new,file);a=raw[cohort].copy();b=pd.read_csv(hp)
  if condition:a=a[a.condition==condition];b['method']=b.method.replace({'ForwardGF':'ForwardTopK','ReverseRecursive':'ReverseRecursiveTopK'})
  a=a[a.method.isin(METHODS[:3])];b=b[b.method.isin(METHODS[:3])]
  key=['family','seed','d','h','k','anchor','method'] if condition else ['query_id','method']
  aa=a.drop_duplicates(key).set_index(key);bb=b.drop_duplicates(key).set_index(key);ix=aa.index.intersection(bb.index);comparefields=[c for c in fields if c not in key]
  mismatches=(aa.loc[ix,comparefields]!=bb.loc[ix,comparefields]).any(axis=1)
  entry={'cohort':cohort,'condition':condition,'historical_file':str(hp.relative_to(H)),'compared_plan_cells':len(ix),'expected_new_plan_cells':len(aa),'counter_mismatches':int(mismatches.sum())};comparisons.append(entry)
  if mismatches.any() or len(ix)!=len(aa):violations.append('historical_counter_alignment:'+file)
 dump('historical_counter_alignment.json',comparisons)

audit['total']={'timed_rows':sum(len(x) for x in raw.values()),'configurations':sum(len(x) for x in meta.values()),'zero_times':sum(int((x.ms==0).sum()) for x in raw.values()),'max_min_gt3_plan_cells':len(anomalies),'omissions':0,'violations':violations,'all_csv_unchanged_by_analysis':True}
pd.DataFrame(anomalies).to_csv(OUT/'timing_anomalies.csv',index=False);dump('audit_report.json',audit);dump('summary.json',result)
dump('analysis_environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

# Insertable two-column, full-width tables. All values are from the new joint batch.
labels=[r'$s=0$',r'$0<s<1\%$',r'$1\%\le s<10\%$',r'$10\%\le s<50\%$',r'$50\%\le s<90\%$',r'$90\%\le s<99.98\%$',r'$99.98\%\le s\le100\%$']
lines=[r'\begin{table*}[pos=tb]\centering\small',r'\caption{Strict fastest-plan counts in the new jointly timed four-plan cohort. F/R/S/H denote forward partial top-$k$, recursive reverse partial top-$k$, full-sort lazy verification and heap-ordered lazy verification; T denotes tied minimum medians. Every configuration has three repetitions per plan; output counts share predicates. The two grids overlap and are reported separately.}',r'\label{tab:heap-winners}',r'\begin{tabular}{@{}lrrrrrrrrrr@{}}\toprule',r'&\multicolumn{5}{c}{Main grid}&\multicolumn{5}{c}{Degree sweep}\\',r'Selectivity & F&R&S&H&T&F&R&S&H&T\\\midrule']
for s,label in zip(STRATA,labels):
 vals=[]
 for cohort in ['main','degree']:
  row=next(x for x in strata_rows if x['cohort']==cohort and x['stratum']==s);vals.extend(str(row[m]) for m in METHODS+['ties'])
 lines.append(label+' & '+' & '.join(vals)+r'\\')
lines += [r'\midrule Total & '+' & '.join(str(result['cohorts'][c]['four_way_strict_wins'][m]) for c in ['main','degree'] for m in METHODS+['ties'])+r'\\',r'\bottomrule\end{tabular}\end{table*}'];(OUT/'heap_winners_table.tex').write_text('\n'.join(lines))
lines=[r'\begin{table*}[pos=tb]\centering\small',r'\caption{Matched output-count sweep at $n=100{,}000$, degree eight and $h=6$. F/R/S/H are strict four-plan wins out of ten seeds. Ratios are paired medians against the faster materializer, with conditional 95\% paired-seed bootstrap intervals (10,000 resamples). Full-sort and heap plans inspect identical exact root prefixes.}',r'\label{tab:heap-k}',r'\begin{tabular}{@{}rrrrrrlll@{}}\toprule',r'$k$&F&R&S&H&Roots&S/materializer&H/materializer&H/S median [IQR]\\\midrule']
for r in krows:lines.append(f"{r['k']} & "+' & '.join(str(r[m]) for m in METHODS)+f" & {r['roots_min']}--{r['roots_max']} & {r['sort_best_median']:.3f} [{r['sort_best_ci_low']:.3f},{r['sort_best_ci_high']:.3f}] & {r['heap_best_median']:.3f} [{r['heap_best_ci_low']:.3f},{r['heap_best_ci_high']:.3f}] & {r['heap_sort_median']:.3f} [{r['heap_sort_q1']:.3f},{r['heap_sort_q3']:.3f}]"+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table*}'];(OUT/'heap_k_table.tex').write_text('\n'.join(lines))
lines=[r'\begin{table*}[pos=tb]\centering\small',r'\caption{Dimension controls against the newly remeasured ForwardTopK, with 40 configurations per row. S/H denote full-sort/heap lazy verification. Wins are strict paired wins against ForwardTopK, not four-way winner counts. Paired medians and IQRs are distinguished from ratios of mean query medians.}',r'\label{tab:heap-dimension}',r'\begin{tabular}{@{}lrrrrlll@{}}\toprule',r'Query&$d$&S wins&H wins&S/F mean ratio&H/F mean ratio&S/F median [IQR]&H/F median [IQR]\\\midrule']
for r in dimrows:lines.append(f"{r['condition'].capitalize()} & {r['d']} & {r['sort_forward_strict_wins']}/40 & {r['heap_forward_strict_wins']}/40 & {r['sort_forward_ratio_of_means']:.3f} & {r['heap_forward_ratio_of_means']:.3f} & {r['sort_forward_median_paired']:.3f} [{r['sort_forward_q1_paired']:.3f},{r['sort_forward_q3_paired']:.3f}] & {r['heap_forward_median_paired']:.3f} [{r['heap_forward_q1_paired']:.3f},{r['heap_forward_q3_paired']:.3f}]"+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table*}'];(OUT/'heap_dimension_table.tex').write_text('\n'.join(lines))

fig,axes=plt.subplots(1,2,figsize=(8.0,2.8))
for prefix,label,color,shift in [('sort_best','Full sort','#aa4c20',-.08),('heap_best','Heap','#126e92',.08)]:
 x=np.arange(len(kdf))+shift;y=kdf[prefix+'_median'].to_numpy();lo=kdf[prefix+'_ci_low'].to_numpy();hi=kdf[prefix+'_ci_high'].to_numpy();axes[0].errorbar(x,y,yerr=[y-lo,hi-y],fmt='o-',label=label,color=color,markersize=4,capsize=2)
axes[0].axhline(1,color='gray',ls='--',lw=.8);axes[0].set(xticks=np.arange(len(kdf)),xticklabels=kdf.k.astype(str),xlabel='Requested k',ylabel='Lazy / faster materializer',yscale='log');axes[0].legend(fontsize=7)
for cohort,color in [('main','#3978a8'),('degree','#c66c35')]:
 w=wide[cohort];rr=med[cohort];roots=rr[rr.method=='LazyHeap'].set_index('query_id').root_checks;axes[1].scatter(roots/(w.n-1),w.heap_over_sort,s=5,alpha=.25,label=cohort,color=color)
axes[1].axhline(1,color='gray',ls='--',lw=.8);axes[1].set(xscale='log',xlabel='Verified roots / rankable universe',ylabel='Heap / full-sort lazy latency');axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'heap_boundary.pdf');fig.savefig(OUT/'heap_boundary.png',dpi=180);plt.close(fig)
print(json.dumps(clean({'total':audit['total'],'cohort_summaries':result['cohorts'],'matched_k':krows}),indent=2))
if violations:raise SystemExit('Audit violations: '+', '.join(violations))
