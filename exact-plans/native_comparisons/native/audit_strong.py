import csv,json,statistics,hashlib
from pathlib import Path
from collections import defaultdict
P=Path(__file__).resolve().parent
methods=['ForwardTopK','ReverseRecursiveTopK','LazyShared'];report={}
for name,expected in [('strong_real_raw.csv',80),('strong_synthetic_raw.csv',1500)]:
 rows=list(csv.DictReader((P/name).open()));groups=defaultdict(list)
 for r in rows:groups[int(r['query_id']),r['method']].append(r)
 qids=sorted({q for q,m in groups});violations=[]
 if len(qids)!=expected:violations.append('wrong_query_count')
 med={}
 for q in qids:
  for m in methods:
   rs=groups[q,m]
   if len(rs)!=3 or {r['rep'] for r in rs}!={'0','1','2'}:violations.append([q,m,'reps'])
   if any(r['exact']!='1' for r in rs):violations.append([q,m,'exact'])
   for key in ['edge_examinations','state_entries','distances','eligible']:
    if len({r[key] for r in rs})!=1:violations.append([q,m,key])
   if rs:med[q,m]=statistics.median(float(r['ms']) for r in rs)
 meta={int(r['query_id']):r for r in rows};out={'rows':len(rows),'queries':len(qids),'exact_rows':sum(r['exact']=='1' for r in rows),'violations':violations,'sha256':hashlib.sha256((P/name).read_bytes()).hexdigest(),'methods':{},'families':{}}
 def summary(qs):
  return {'queries':len(qs),'lazy_beats_forward':sum(med[q,'LazyShared']<med[q,'ForwardTopK'] for q in qs),'lazy_beats_both_materializers':sum(med[q,'LazyShared']<min(med[q,'ForwardTopK'],med[q,'ReverseRecursiveTopK']) for q in qs),'means_ms':{m:statistics.mean(med[q,m] for q in qs) for m in methods},'median_ratio_lazy_to_best_materializer':statistics.median(med[q,'LazyShared']/min(med[q,'ForwardTopK'],med[q,'ReverseRecursiveTopK']) for q in qs),'winner_counts':{m:sum(min(methods,key=lambda x:med[q,x])==m for q in qs) for m in methods}}
 valid=[q for q in qids if all((q,m) in med for m in methods)];out['overall']=summary(valid)
 for fam in sorted({meta[q]['family'] for q in valid}):out['families'][fam]=summary([q for q in valid if meta[q]['family']==fam])
 report[name]=out
(P/'strong_audit_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if any(x['violations'] for x in report.values()):raise SystemExit(1)
