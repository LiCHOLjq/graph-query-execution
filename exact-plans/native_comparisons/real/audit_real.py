"""Reconstruct truth from the raw edge table and audit retained raw records."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
data=np.load(P/'fb15k237_text.npz');X=data['vectors'];Q=data['query_vectors'];E=data['triples']
queries=json.loads((P/'queries.json').read_text());truth=json.loads((P/'truth.json').read_text())
sets=[]
for qi,(q,vec) in enumerate(zip(queries,Q)):
 layer={q['anchor']}
 for r in q['relations']:layer={int(b) for a,b,_ in E[E[:,2]==r] if int(a) in layer}
 layer.discard(q['anchor']);sets.append(layer)
 def key(v):delta=X[v]-vec;return float(np.dot(delta,delta)),v
 assert sorted(layer,key=key)[:q['k']]==truth[qi]
raw=list(csv.DictReader((P/'hnsw_raw.csv').open()));exact=list(csv.DictReader((P/'exact_raw.csv').open()))
assert len(raw)==960 and len(exact)==480
for r in raw:
 qi=int(r['qid']);a=json.loads(r['answer']);b=json.loads(r['repair_answer']);k=len(truth[qi])
 assert set(a)<=sets[qi] and len(a)==int(r['returned']) and len(a)==len(set(a))
 assert b==truth[qi] and int(r['eligible'])==len(sets[qi])
 assert int(r['underfill'])==int(len(a)<queries[qi]['k'])
 assert abs(float(r['recall'])-len(set(a)&set(truth[qi]))/k)<1e-12
for r in exact:assert json.loads(r['answer'])==truth[int(r['qid'])]
report=json.loads((P/'hnsw_summary.json').read_text())
for fn,digest in report['hashes'].items():assert hashlib.sha256((P/fn).read_bytes()).hexdigest()==digest,fn
for s in report['budgets']:
 rr=[r for r in raw if int(r['budget'])==s['budget']]
 mean=lambda f:sum(f(r) for r in rr)/len(rr)
 assert abs(s['mean_recall']-mean(lambda r:float(r['recall'])))<1e-12
 assert abs(s['underfill_policy_ms']-mean(lambda r:float(r['prefix_ms'])+int(r['underfill'])*float(r['repair_ms'])))<1e-12
 assert abs(s['underfill_policy_recall']-mean(lambda r:1 if int(r['underfill']) else float(r['recall'])))<1e-12
out=dict(status='PASS',independent_edge_scan_queries=len(queries),legal_prefixes=len(raw),exact_repairs=len(raw),exact_baselines=len(exact),hashes_verified=len(report['hashes']),empty_truth_queries=sum(not t for t in truth),underfill_acceptance_misses=sorted(set(int(r['qid']) for r in raw if int(r['underfill'])==0 and float(r['recall'])<1)))
if (P/'actual_policy.csv').exists():
 actual=list(csv.DictReader((P/'actual_policy.csv').open()));actual_summary=json.loads((P/'actual_policy_summary.json').read_text())
 assert len(actual)==1680
 assert hashlib.sha256((P/'actual_policy.csv').read_bytes()).hexdigest()==actual_summary['csv_hash']
 assert hashlib.sha256((P/'benchmark_actual_policy.py').read_bytes()).hexdigest()==actual_summary['source_hash']
 for r in actual:
  qi=int(r['qid']);answer=json.loads(r['answer'])
  assert set(answer)<=sets[qi] and len(answer)==len(set(answer))
  assert abs(float(r['recall'])-len(set(answer)&set(truth[qi]))/len(truth[qi]))<1e-12
  if r['method']!='HNSWUnderfill' or int(r['repaired']):assert answer==truth[qi]
  else:assert len(answer)==queries[qi]['k']
 out['actual_policy_executions_verified']=sum(r['method']=='HNSWUnderfill' for r in actual)
 out['actual_policy_baselines_verified']=sum(r['method']!='HNSWUnderfill' for r in actual)
(P/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
