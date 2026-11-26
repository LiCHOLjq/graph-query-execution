"""MATCHED: postanalysis of frozen SENSITIVITY timings; no index execution or new timing."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from provenance_checks import verify_benchmark_source
from collections import defaultdict, Counter
from statistics import mean, median
import csv, hashlib, json, platform, sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
REAL=ROOT/'sensitivity_controls'/'real'
NATIVE=ROOT/'sensitivity_controls'/'native'
EFS=(64,256,1024,4096)
METRICS=('ms','prefix_ms','callbacks','proof_edges','proof_entries','proof_memo_hits','forward_edges','prefix_distances','total_distances','recall','prefix_recall','repaired','exception')

def read(path):
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

raw=read(REAL/'filtered_raw.csv')
by=defaultdict(list)
for r in raw:by[(int(r['qid']),r['method'],int(r['budget']))].append(r)
assert len(raw)==2400 and all(len(v)==3 for v in by.values())
query=[]
for (qid,method,ef),rs in sorted(by.items()):
    assert {int(r['rep']) for r in rs}=={0,1,2}
    for x in ('callbacks','proof_edges','proof_entries','proof_memo_hits','repaired','exception','prefix_recall','recall'):
        assert len({r[x] for r in rs})==1,(qid,method,ef,x)
    z={'qid':qid,'method':method,'ef':ef,'repetitions':len(rs),'eligible':int(rs[0]['eligible']),'truth_k':int(rs[0]['truth_k'])}
    z.update({x:mean(float(r[x]) for r in rs) for x in METRICS})
    assert all(float(r['ms'])>=float(r['prefix_ms'])>0 for r in rs)
    z['post_prefix_ms']=z['ms']-z['prefix_ms']
    for x in ('prefix_ms','ms'):
        z[x+'_us_per_callback']=1000*z[x]/z['callbacks'] if z['callbacks'] else None
    query.append(z)
write('query_mean_normalization.csv',query)
ids={r['qid'] for r in query};assert len(ids)==80
common=ids.copy()
for ef in EFS:
    for method in ('InFilter','PostFilter'):
        common&={r['qid'] for r in query if r['method']==method and r['ef']==ef and r['exception']==0}
assert len(common)==56
audit=json.loads((REAL/'audit.json').read_text())
for name in ('benchmark_filtered.py','filtered_raw.csv','analysis.json'):
    assert verify_benchmark_source(REAL/name,audit['hashes'][name]) if name=='benchmark_filtered.py' else sha(REAL/name)==audit['hashes'][name]
assert ids-common==set(audit['insufficient_cardinality_query_ids'])
assert sum(int(r['exception']) for r in raw)==audit['infilter_expected_exceptions']==288
old=json.loads((REAL/'analysis.json').read_text())['cohorts']['All']
oldmap={(r['method'],r['ef']):r for r in old['sensitivity_policies']}
summary=[];comparisons=[]
for cohort,selected in (('all80',ids),('common_success56',common)):
    heap=[r for r in query if r['qid'] in selected and r['method']=='ForwardTopKHeap']
    heap_mean=mean(r['ms'] for r in heap)
    for ef in EFS:
        pair={}
        for method in ('PostFilter','InFilter'):
            rs=[r for r in query if r['qid'] in selected and r['method']==method and r['ef']==ef]
            assert len(rs)==len(selected) and all(r['callbacks']>0 for r in rs)
            z={'cohort':cohort,'method':method,'ef':ef,'queries':len(rs),'raw_repetitions':3*len(rs)}
            for x in METRICS:z['mean_'+x]=mean(r[x] for r in rs)
            z['mean_post_prefix_ms']=mean(r['post_prefix_ms'] for r in rs)
            for x in ('prefix_ms','ms'):
                z[x+'_us_per_callback_ratio_of_means']=1000*z['mean_'+x]/z['mean_callbacks']
                z[x+'_us_per_callback_mean_of_query_ratios']=mean(r[x+'_us_per_callback'] for r in rs)
            z['proof_entries_per_callback_ratio_of_means']=z['mean_proof_entries']/z['mean_callbacks']
            z['policy_ratio_of_means_to_heap']=z['mean_ms']/heap_mean
            z['policy_mean_of_query_ratios_to_heap']=mean(r['ms']/next(h['ms'] for h in heap if h['qid']==r['qid']) for r in rs)
            z['prefix_failed_queries']=sum(r['exception']>0 for r in rs)
            z['prefix_successful_queries']=sum(r['exception']==0 for r in rs)
            z['prefix_successful_only_recall']=mean(r['prefix_recall'] for r in rs if r['exception']==0)
            z['prefix_recall_failures_as_empty']=z['mean_prefix_recall']
            assert all(r['prefix_recall']==0 for r in rs if r['exception'])
            if cohort=='all80':
                o=oldmap[(method,ef)]
                for newkey,oldkey in (('mean_ms','mean_ms'),('mean_prefix_ms','prefix_mean_ms'),('mean_callbacks','mean_callbacks'),('mean_proof_entries','mean_proof_entries'),('prefix_recall_failures_as_empty','prefix_recall_failure_as_empty'),('policy_ratio_of_means_to_heap','mean_ratio_to_heap')):
                    assert abs(z[newkey]-o[oldkey])<1e-9,(method,ef,newkey)
            summary.append(z);pair[method]=z
        a,b=pair['InFilter'],pair['PostFilter']
        comparisons.append({'cohort':cohort,'ef':ef,'queries':len(selected),
          'callbacks_in_over_post':a['mean_callbacks']/b['mean_callbacks'],
          'proof_entries_in_over_post':a['mean_proof_entries']/b['mean_proof_entries'],
          'prefix_time_in_over_post_ratio_of_means':a['mean_prefix_ms']/b['mean_prefix_ms'],
          'policy_time_in_over_post_ratio_of_means':a['mean_ms']/b['mean_ms'],
          'normalized_prefix_in_over_post_ratio_of_ratios':a['prefix_ms_us_per_callback_ratio_of_means']/b['prefix_ms_us_per_callback_ratio_of_means'],
          'normalized_policy_in_over_post_ratio_of_ratios':a['ms_us_per_callback_ratio_of_means']/b['ms_us_per_callback_ratio_of_means'],
          'in_over_heap_policy_ratio_of_means':a['policy_ratio_of_means_to_heap'],
          'post_over_heap_policy_ratio_of_means':b['policy_ratio_of_means_to_heap']})
write('callback_normalization_summary.csv',summary);write('integration_ratios.csv',comparisons)

# Independent reconstruction of the existing native winner counts from raw times.
native=read(NATIVE/'degree_sweep_raw.csv');nb=defaultdict(list)
for r in native:nb[(int(r['query_id']),r['method'])].append(r)
qids={q for q,m in nb};assert len(qids)==1440 and len(native)==12960
assert all(len(v)==3 for v in nb.values())
nr=[]
for q in sorted(qids):
    r=nb[(q,'LazyShared')][0]
    z={x:int(r[x]) for x in ('query_id','seed','n','h','k','eligible','anchor_degree')}
    z['selectivity']=z['eligible']/(z['n']-1)
    for method in ('ForwardTopK','LazyShared','ReverseRecursiveTopK'):z[method]=median(float(x['ms']) for x in nb[(q,method)])
    vals={m:z[m] for m in ('ForwardTopK','LazyShared','ReverseRecursiveTopK')}
    assert len(set(vals.values()))==3
    z['winner']=min(vals,key=vals.get);nr.append(z)
saved={int(r['query_id']):r for r in read(NATIVE/'degree_query_medians.csv')}
for r in nr:
    assert r['winner']==saved[r['query_id']]['winner']
    for method in ('ForwardTopK','LazyShared','ReverseRecursiveTopK'):assert abs(r[method]-float(saved[r['query_id']][method]))<1e-12
    assert abs(r['selectivity']-float(saved[r['query_id']]['selectivity']))<1e-12
wins=Counter(r['winner'] for r in nr);lazy=[r for r in nr if r['winner']=='LazyShared'];sub=[r for r in lazy if r['selectivity']<.9998]
assert wins=={'ForwardTopK':1188,'ReverseRecursiveTopK':18,'LazyShared':234}
assert len(sub)==37 and {r['k'] for r in sub}=={1}
low=[r for r in lazy if .5<=r['selectivity']<.9]
assert len(low)==8 and {(r['n'],r['anchor_degree'],r['h'],r['k']) for r in low}=={(100000,8,6,1)}
write('native_recomputed_query_medians.csv',nr);write('native_subsaturation_lazy_wins.csv',sub)
native_summary={'configurations':len(nr),'raw_repetitions':len(native),'wins':dict(wins),'sub_99_98_percent_lazy_wins':len(sub),
 'sub_99_98_percent_lazy_winning_k':sorted({r['k'] for r in sub}),'lazy_wins_by_selectivity':{'[0.5,0.9)':len(low),'[0.9,0.9998)':sum(.9<=r['selectivity']<.9998 for r in lazy),'[0.9998,1]':sum(r['selectivity']>=.9998 for r in lazy)},
 'low_bin_unique_cells':[list(v) for v in sorted({(r['n'],r['anchor_degree'],r['h'],r['k']) for r in low})],
 'min_lazy_winning_selectivity':min(r['selectivity'] for r in lazy),
 'interpretation':'Invariant of the retained uniform degree-grid cohort, not a universal rule for every graph, topology, query, or k.'}

sources=[Path(__file__),REAL/'benchmark_filtered.py',REAL/'filtered_raw.csv',REAL/'analysis.json',REAL/'audit.json',NATIVE/'degree_sweep_raw.csv',NATIVE/'degree_query_medians.csv']
report={'status':'PASS','date':'2026-10-25','analysis_type':'Local postanalysis of existing frozen SENSITIVITY data; no new timings, ANN calls, or server use.',
 'host':platform.node(),'python':sys.version,'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
 'cohorts':{'all80':sorted(ids),'common_success56':sorted(common),'failed24':sorted(ids-common)},
 'raw_rows':len(raw),'query_mean_rows':len(query),'repetition_count':3,
 'aggregation':'Arithmetic means of per-query arithmetic means over three repetitions. No pooling of repetitions as independent queries.',
 'normalization':{'ratio_of_means':'1000 * mean_q(time_ms_q) / mean_q(callbacks_q), microseconds per recorded top-level callback. Equivalent to callback-weighted mean of per-query amortizations.',
 'mean_of_query_ratios':'mean_q(1000 * time_ms_q / callbacks_q); each query has equal weight. Both are reported and must not be interchanged.'},
 'timing_semantics':{'prefix_ms':'Elapsed outer-start to prefix construction: setup, set_ef, Proof construction, conversion, HNSW search, callback/proof work and exact reranking. Failure handling also included. No predicate-only sub-timer exists.',
 'ms':'Same prefix plus conditional forward materialization and heap selection (with distance-cache reuse), or minimal post-prefix acceptance overhead. It ends before diagnostic-dictionary construction, final local destruction, and caller validation/serialization.',
 'callbacks':'Number of top-level Proof.allow invocations inside or after ANN. Does not count recursive Proof.check entries, memo hits or forward repair work.',
 'proof_entries':'Fresh uncached (vertex,depth) checks, including recursive calls/base states; not equivalent to callbacks or uniform-cost operations.',
 'not_identifiable':'Pure per-callback predicate time, Python transition cost, GIL overhead, and separate ANN traversal time cannot be identified from these fields. Normalized times are amortizations, not causal overhead estimates or equal-work comparisons.'},
 'native_invariants':native_summary,'summary':summary,'integration_comparisons':comparisons}
(OUT/'callback_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf8')

for cohort,filename,label,title,recall_note in [
 ('all80','callback_table.tex','tab:matched-callbacks','All 80 queries','Recall counts failed API prefixes as empty.'),
 ('common_success56','callback_success56_table.tex','tab:matched-callbacks-success','The same 56 successful-call queries at every ef','Recall concerns successful prefixes on the common cohort.')]:
    tex=[r'\begin{table}[htbp]',r'\centering',r'\small',r'\setlength{\tabcolsep}{3pt}',
     r'\caption{'+title+r'. Calls and proof entries are query means. Time/call is $1000\bar T/\bar C$ in $\mu$s/callback, using per-query mean milliseconds $T$ and callback counts $C$. Prefix time includes setup, ANN search, proof work and reranking; policy time also includes conditional repair. These are amortizations, not isolated callback costs. '+recall_note+r'}',
     r'\label{'+label+r'}',r'\begin{tabular}{llrrrrr}',r'\toprule',
     r'Integration & ef & Calls & Entries & Prefix/call & Policy/call & Prefix recall \\',r'\midrule']
    for r in [s for s in summary if s['cohort']==cohort]:
        tex.append(f"{'Post' if r['method']=='PostFilter' else 'In'} & {r['ef']} & {r['mean_callbacks']:.1f} & {r['mean_proof_entries']:.1f} & {r['prefix_ms_us_per_callback_ratio_of_means']:.3f} & {r['ms_us_per_callback_ratio_of_means']:.3f} & {r['prefix_recall_failures_as_empty']:.5f} " + r'\\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUT/filename).write_text('\n'.join(tex)+'\n',encoding='utf8')
print(json.dumps({'native':native_summary,'comparisons':comparisons,'all80_summary':[r for r in summary if r['cohort']=='all80']},indent=2))
