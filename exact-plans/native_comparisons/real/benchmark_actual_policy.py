"""End-to-end underfill policy: actual branch, early return, one outer timer.

This is a separate matched run; prior component estimates remain unchanged.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import csv,gc,hashlib,heapq,json,random,statistics,time,platform,sys
from pathlib import Path
import numpy as np
import psutil,hnswlib
P=Path(__file__).resolve().parent
psutil.Process().cpu_affinity([4])
data=np.load(P/'fb15k237_text.npz');X=data['vectors'];Q=data['query_vectors'];E=data['triples'];N,D=X.shape
queries=json.loads((P/'queries.json').read_text());truth=json.loads((P/'truth.json').read_text());out={};rev={}
for a,b,r in E:
 a,b,r=int(a),int(b),int(r);out.setdefault((a,r),[]).append(b);rev.setdefault((b,r),[]).append(a)
for adj in [out,rev]:
 for vals in adj.values():vals.sort()
index=hnswlib.Index(space='l2',dim=D)
old=os.getcwd();os.chdir(P);index.load_index('hnsw_index.bin');os.chdir(old);index.set_num_threads(1)

class Proof:
 def __init__(self,a,word):self.a=a;self.word=word;self.memo={}
 def check(self,v,i=None):
  if i is None:i=len(self.word)
  key=(v,i)
  if key in self.memo:return self.memo[key]
  if i==0:ans=v==self.a
  else:ans=any(self.check(u,i-1) for u in rev.get((v,self.word[i-1]),()))
  self.memo[key]=ans;return ans

def forward(q):
 layer={q['anchor']}
 for r in q['relations']:
  nxt=set()
  for v in layer:nxt.update(out.get((v,r),()))
  layer=nxt
 return layer-{q['anchor']}

def rank(vs,qvec,cache,k,method='sort'):
 def key(v):
  if v not in cache:delta=X[v]-qvec;cache[v]=float(np.dot(delta,delta))
  return cache[v],v
 if method=='heap':return heapq.nsmallest(k,vs,key=key)
 if method=='partition':
  vs=list(vs)
  if len(vs)<=k:return sorted(vs,key=key)
  distances=np.fromiter((key(v)[0] for v in vs),dtype=np.float64,count=len(vs))
  kth=float(distances[np.argpartition(distances,k-1)[k-1]])
  selected=[v for v,d in zip(vs,distances) if d<=kth]
  return sorted(selected,key=key)[:k]
 return sorted(vs,key=key)[:k]

def policy(q,qvec,L):
 index.set_ef(max(64,L));proof=Proof(q['anchor'],q['relations']);cache={}
 ids,_=index.knn_query(qvec.astype(np.float32),k=L,num_threads=1)
 hits=[int(x) for x in ids[0] if int(x)!=q['anchor'] and proof.check(int(x))]
 answer=rank(hits,qvec,cache,q['k'])
 if len(answer)>=q['k']:
  return answer,False
 # This branch is never entered by an accepted prefix.
 return rank(forward(q),qvec,cache,q['k']),True

# Independent edge-table truth check before timing.
eligible=[]
for qi,(q,vec) in enumerate(zip(queries,Q)):
 layer={q['anchor']}
 for r in q['relations']:layer={int(b) for a,b,_ in E[E[:,2]==r] if int(a) in layer}
 layer.discard(q['anchor']);eligible.append(layer)
 assert rank(layer,vec,{},q['k'])==truth[qi]

rows=[];rng=random.Random(20261023);gc_enabled=gc.isenabled();gc.disable()
methods=['ForwardSort','ForwardTopKHeap','ForwardTopKPartition',64,256,1024,4096]
for rep in range(-1,3):
 order=list(range(80));rng.shuffle(order)
 for qi in order:
  q,vec=queries[qi],Q[qi];ordered=methods.copy();rng.shuffle(ordered)
  for method in ordered:
   t=time.perf_counter_ns()
   if isinstance(method,int):answer,repaired=policy(q,vec,method)
   else:
    mode={'ForwardSort':'sort','ForwardTopKHeap':'heap','ForwardTopKPartition':'partition'}[method]
    answer=rank(forward(q),vec,{},q['k'],mode);repaired=False
   ms=(time.perf_counter_ns()-t)/1e6
   assert set(answer)<=eligible[qi] and len(set(answer))==len(answer)
   if repaired or isinstance(method,str):assert answer==truth[qi]
   else:assert len(answer)==q['k']
   recall=len(set(answer)&set(truth[qi]))/len(truth[qi]) if truth[qi] else None
   if rep>=0:rows.append(dict(rep=rep,qid=qi,h=q['h'],mode=q['mode'],method='HNSWUnderfill' if isinstance(method,int) else method,budget=method if isinstance(method,int) else 0,ms=ms,repaired=int(repaired),recall=recall,answer=json.dumps(answer)))
 print('completed repetition',rep,flush=True)
if gc_enabled:gc.enable()
with (P/'actual_policy.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(xs):return statistics.mean(xs)
def stats(xs):return dict(mean=mean(xs),median=statistics.median(xs),p95=float(np.quantile(xs,.95)),iqr=float(np.quantile(xs,.75)-np.quantile(xs,.25)))
baselines={m:stats([r['ms'] for r in rows if r['method']==m]) for m in methods if isinstance(m,str)}
strongest=min(baselines,key=lambda m:baselines[m]['mean']);summary=[];rng=np.random.default_rng(20261023)
for L in [64,256,1024,4096]:
 rr=[r for r in rows if r['budget']==L];a=np.array([mean(r['ms'] for r in rr if r['qid']==q) for q in range(80)]);b=np.array([mean(r['ms'] for r in rows if r['qid']==q and r['method']==strongest) for q in range(80)])
 ix=rng.integers(0,80,(2000,80));ratios=a[ix].mean(1)/b[ix].mean(1)
 summary.append(dict(budget=L,ms=stats([r['ms'] for r in rr]),recall=mean(r['recall'] for r in rr if r['recall'] is not None),fallback_rate=mean(r['repaired'] for r in rr),early_returns=sum(not r['repaired'] for r in rr),mean_ratio_to_strongest=float(a.mean()/b.mean()),paired_bootstrap95ci=np.quantile(ratios,[.025,.975]).tolist(),faster_queries=int((a<b).sum())))
report=dict(timing='Actual executed early-return branch with one outer end-to-end timer; prior component reconstructions preserved separately',queries=80,warmup_passes=1,repetitions=3,policy_executions=960,exact_baseline_executions=720,platform=platform.platform(),python=sys.version,cpu_affinity=psutil.Process().cpu_affinity(),gc_during_timing=False,hnsw_threads=1,baselines=baselines,strongest_mean_baseline=strongest,budgets=summary,audit='All repaired and exact-baseline answers match independent edge-table oracle; every prefix answer is legal and of size k',source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),csv_hash=hashlib.sha256((P/'actual_policy.csv').read_bytes()).hexdigest(),index_hash=hashlib.sha256((P/'hnsw_index.bin').read_bytes()).hexdigest(),warnings=['Top-k controls share float64 scalar dot distance kernel; argpartition preserves all distance ties before ID tie-breaking','Original HNSW component experiment retained graph-counter instrumentation; this follow-up omits counters to measure actual control-flow overhead','Comparison with previous component sums is descriptive across separate matched runs, not a paired control'] )
(P/'actual_policy_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
