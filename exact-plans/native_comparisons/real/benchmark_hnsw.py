"""Measured HNSW prefixes, matched exact baselines, and actual exact repair.

HNSW 0.8 public API restarts each search. Only proof and explicit score caches
are shared; this experiment does not claim a retained HNSW traversal frontier.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import csv,gc,hashlib,json,platform,random,sys,time,statistics
from pathlib import Path
import numpy as np
import hnswlib
import psutil

psutil.Process().cpu_affinity([4])

P=Path(__file__).resolve().parent
data=np.load(P/'fb15k237_text.npz');X=data['vectors'];Q=data['query_vectors'];triples=data['triples'];N,D=X.shape
queries=json.loads((P/'queries.json').read_text());out={};rev={}
for a,b,r in triples:
 a,b,r=int(a),int(b),int(r)
 out.setdefault((a,r),[]).append(b);rev.setdefault((b,r),[]).append(a)
for adj in [out,rev]:
 for v in adj.values():v.sort()

class Proof:
 def __init__(self,a,word):self.a=a;self.word=word;self.memo={};self.entries=0;self.edges=0;self.hits=0
 def check(self,v,i=None):
  if i is None:i=len(self.word)
  key=(v,i)
  if key in self.memo:self.hits+=1;return self.memo[key]
  self.entries+=1
  if i==0:ans=v==self.a
  else:
   ans=False
   for u in rev.get((v,self.word[i-1]),[]):
    self.edges+=1
    if self.check(u,i-1):ans=True;break
  self.memo[key]=ans;return ans

def forward(q):
 layer={q['anchor']};edges=0
 for r in q['relations']:
  nxt=set()
  for v in layer:
   nb=out.get((v,r),());edges+=len(nb);nxt.update(nb)
  layer=nxt
 return layer-{q['anchor']},edges

def oracle_scan(q):
 layer={q['anchor']}
 for r in q['relations']:
  relevant=triples[triples[:,2]==r]
  # Independent truth: edge-table selection, not the adjacency evaluator.
  layer={int(b) for a,b,_ in relevant if int(a) in layer}
 return layer-{q['anchor']}

def score(v,qvec,cache):
 if v not in cache:
  diff=X[v]-qvec;cache[v]=float(np.dot(diff,diff))
 return cache[v]

def rank(vs,qvec,cache,k):return sorted(vs,key=lambda v:(score(v,qvec,cache),v))[:k]
def elapsed(t):return (time.perf_counter_ns()-t)/1e6

baseline_rss=psutil.Process().memory_info().rss
index=hnswlib.Index(space='l2',dim=D)
index.init_index(max_elements=N,ef_construction=200,M=16,random_seed=20261023)
index.set_num_threads(1)
t=time.perf_counter();index.add_items(X.astype(np.float32),np.arange(N),num_threads=1);build_seconds=time.perf_counter()-t
oldcwd=os.getcwd();os.chdir(P);index.save_index('hnsw_index.bin');os.chdir(oldcwd)
index_rss=psutil.Process().memory_info().rss
truths=[]
for q,v in zip(queries,Q):
 truth=oracle_scan(q);fast,_=forward(q)
 assert fast==truth
 truths.append(rank(truth,v,{},q['k']))

BUDGETS=[64,256,1024,4096]
rows=[];exactrows=[]
# One warmup over each code path, outside formal timings.
for q,v in zip(queries[:3],Q[:3]):
 Proof(q['anchor'],q['relations']).check(0);index.set_ef(64);index.knn_query(v.astype(np.float32),k=64,num_threads=1);forward(q)
rng=random.Random(20261023);gc_before=gc.isenabled();gc.disable()
for rep in range(-1,3):
 order=list(range(len(queries)));rng.shuffle(order)
 for qi in order:
  q,v=queries[qi],Q[qi];truth=truths[qi];K=len(truth)
  methods=['ForwardGF','ColdMemo'];rng.shuffle(methods)
  for method in methods:
   t=time.perf_counter_ns();cache={}
   if method=='ForwardGF':eligible,edges=forward(q);entries=0;states=0
   else:
    proof=Proof(q['anchor'],q['relations']);eligible={x for x in range(N) if x!=q['anchor'] and proof.check(x)}
    edges,entries,states=proof.edges,proof.entries,len(proof.memo)
   answer=rank(eligible,v,cache,q['k']);ms=elapsed(t)
   assert answer==truth,(method,qi)
   if rep>=0:exactrows.append(dict(rep=rep,qid=qi,h=q['h'],mode=q['mode'],method=method,ms=ms,eligible=len(eligible),edges=edges,entries=entries,states=states,distances=len(cache),answer=json.dumps(answer)))
  budgets=BUDGETS.copy();rng.shuffle(budgets)
  for L in budgets:
   index.set_ef(max(64,L))
   t=time.perf_counter_ns();proof=Proof(q['anchor'],q['relations']);cache={}
   ids,_=index.knn_query(v.astype(np.float32),k=L,num_threads=1)
   hits=[int(x) for x in ids[0] if int(x)!=q['anchor'] and proof.check(int(x))]
   answer=rank(hits,v,cache,q['k']);prefix_ms=elapsed(t)
   entries,edges,states,dists=proof.entries,proof.edges,len(proof.memo),len(cache)
   recall=len(set(answer)&set(truth))/K if K else None
   # Actual executed forward repair reuses explicit prefix score cache.
   t=time.perf_counter_ns();all_eligible,repair_edges=forward(q);repaired=rank(all_eligible,v,cache,q['k']);repair_ms=elapsed(t)
   assert repaired==truth
   if rep>=0:rows.append(dict(rep=rep,qid=qi,h=q['h'],mode=q['mode'],budget=L,ef=max(64,L),prefix_ms=prefix_ms,repair_ms=repair_ms,recall=recall,returned=len(answer),truth_k=K,eligible=len(all_eligible),underfill=int(len(answer)<q['k']),edges=edges,entries=entries,states=states,distances=dists,repair_edges=repair_edges,repair_new_distances=len(cache)-dists,answer=json.dumps(answer),repair_answer=json.dumps(repaired)))
 print('completed repetition',rep,flush=True)
if gc_before:gc.enable()
(P/'truth.json').write_text(json.dumps(truths))
for name,rr in [('hnsw_raw.csv',rows),('exact_raw.csv',exactrows)]:
 with (P/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def mean(xs):return statistics.mean(xs)
def stats(xs):return dict(mean=mean(xs),median=statistics.median(xs),p95=float(np.quantile(xs,.95)),iqr=float(np.quantile(xs,.75)-np.quantile(xs,.25)))
summary=[];mixed=[]
fg={qi:mean([r['ms'] for r in exactrows if r['qid']==qi and r['method']=='ForwardGF']) for qi in range(len(queries))}
cm={qi:mean([r['ms'] for r in exactrows if r['qid']==qi and r['method']=='ColdMemo']) for qi in range(len(queries))}
for L in BUDGETS:
 rr=[r for r in rows if r['budget']==L];nonnull=[r for r in rr if r['recall'] is not None]
 A=mean(r['prefix_ms'] for r in rr);B=mean(r['repair_ms'] for r in rr)
 summary.append(dict(budget=L,prefix_ms=stats([r['prefix_ms'] for r in rr]),repair_ms=stats([r['repair_ms'] for r in rr]),mean_recall=mean(r['recall'] for r in nonnull),underfill_rate=mean(r['underfill'] for r in rr),underfill_policy_ms=mean(r['prefix_ms']+r['underfill']*r['repair_ms'] for r in rr),underfill_policy_recall=mean(1 if r['underfill'] else r['recall'] for r in nonnull),forward_break_even_probability=(mean(fg.values())-A)/B,cold_break_even_probability=(mean(cm.values())-A)/B))
 for probability in np.linspace(0,1,11):
  # Analytical expectation under an independent Bernoulli fallback decision,
  # built from measured per-query prefix and actual incremental repair costs.
  mixed.append(dict(budget=L,fallback_probability=float(probability),expected_ms=mean(r['prefix_ms']+probability*r['repair_ms'] for r in rr),expected_recall=mean(r['recall']+probability*(1-r['recall']) for r in nonnull),forward_ms=mean(fg.values()),cold_memo_ms=mean(cm.values())))
with (P/'mixed_fallback.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(mixed[0]));w.writeheader();w.writerows(mixed)
report=dict(vertices=N,dimension=D,query_count=len(queries),repetitions=3,prefix_executions=len(rows),baseline_executions=len(exactrows),hnsw_version='0.8.0',M=16,ef_construction=200,seed=20261023,index_build_seconds=build_seconds,index_file_bytes=(P/'hnsw_index.bin').stat().st_size,rss_before_index=baseline_rss,rss_after_index=index_rss,python=sys.version,platform=platform.platform(),processor=platform.processor(),logical_cpu=psutil.cpu_count(),physical_cpu=psutil.cpu_count(logical=False),memory_bytes=psutil.virtual_memory().total,cpu_frequency_mhz=psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,affinity=psutil.Process().cpu_affinity(),gc_during_timing=False,openmp_threads=1,empty_truth_queries=sum(not t for t in truths),eligible_counts=[next(r['eligible'] for r in exactrows if r['qid']==q) for q in range(len(queries))],exact_baselines={m:stats([r['ms'] for r in exactrows if r['method']==m]) for m in ['ForwardGF','ColdMemo']},budgets=summary,limitations=['Windows local machine timings; do not compare absolute latency with Linux C++ results','HNSW public API searches restart; no frontier-retention claim','HNSW is float32, exact re-ranking is float64 dot(delta,delta); ID breaks exact distance ties','single HNSW index seed; graph query random seed fixed','TFIDF-SVD is corpus-derived lexical embedding; not pretrained semantic embedding','independent fallback sweep is measured-cost interpolation, not a calibrated recall controller','underfill acceptance does not certify recall'],hashes={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['benchmark_hnsw.py','hnsw_raw.csv','exact_raw.csv','mixed_fallback.csv','hnsw_index.bin']})
(P/'hnsw_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
