"""Audit measured outputs without treating k values or repetitions as independent predicates."""
import argparse,csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fresh',action='store_true',help='Validate a new v2 sweep: all 100 large empty LazySeparate controls omitted (26,700 rows). Default validates the retained merged sweep: 77 omissions (26,769 rows).')
parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent,help='Directory containing measured CSVs; reports are written here.')
args=parser.parse_args()
ROOT=args.input_dir.resolve()
expected_queries={'synthetic_raw.csv':1500,'real_raw.csv':80,'dimension_control_raw.csv':120}
expected_rows={'synthetic_raw.csv':26700 if args.fresh else 26769,'real_raw.csv':1440,'dimension_control_raw.csv':2160}
expected_omitted={(q,'LazySeparate') for q in range(509 if args.fresh else 739,1500,10)}
METHODS=['ForwardGF','ForwardResume','ReverseRecursive','ReverseResume','LazyShared','LazySeparate']
report={}
for name in ['real_raw.csv','synthetic_raw.csv','dimension_control_raw.csv']:
 p=ROOT/name
 if not p.exists():continue
 rows=list(csv.DictReader(p.open()));groups=defaultdict(list)
 for row in rows:groups[(int(row['query_id']),row['method'])].append(row)
 violations=[];qids=sorted({q for q,m in groups});med={}
 if len(rows)!=expected_rows[name]:violations.append(['row_count',len(rows),expected_rows[name]])
 if len(qids)!=expected_queries[name]:violations.append(['query_count',len(qids),expected_queries[name]])
 for q in qids:
  for m in METHODS:
   rs=groups[q,m]
   if not rs and m=='LazySeparate' and name=='synthetic_raw.csv':
    basis=groups[q,'ForwardGF'][0]
    if not (int(basis['n'])>1000 and int(basis['eligible'])==0 and int(basis['anchor'])==int(basis['n'])-1 and (q,m) in expected_omitted):violations.append([q,m,'unplanned_omission'])
    continue
   if len(rs)!=3 or {r['rep'] for r in rs}!={'0','1','2'}:violations.append([q,m,'repetitions'])
   if any(r['exact']!='1' for r in rs):violations.append([q,m,'exact'])
   for field in ['edge_examinations','state_entries','distances','root_checks','eligible']:
    if len({r[field] for r in rs})!=1:violations.append([q,m,'counter_nondeterminism',field])
   if rs:med[q,m]=statistics.median(float(r['ms']) for r in rs)
  if all(groups[q,m] for m in METHODS):
   first={m:groups[q,m][0] for m in METHODS}
   for a,b in [('ForwardGF','ForwardResume'),('ReverseRecursive','ReverseResume')]:
    for field in ['edge_examinations','state_entries','distances']:
     if first[a][field]!=first[b][field]:violations.append([q,a,b,field])
   for field in ['edge_examinations','state_entries']:
    if int(first['LazyShared'][field])>int(first['LazySeparate'][field]):violations.append([q,'lazy_sharing',field])
   if first['LazyShared']['root_checks']!=first['LazySeparate']['root_checks']:violations.append([q,'lazy_stop'])
 meta={int(r['query_id']):r for r in rows}
 omitted={(q,m) for q in qids for m in METHODS if not groups[q,m]}
 planned=expected_omitted if name=='synthetic_raw.csv' else set()
 if omitted!=planned:violations.append(['omission_set_mismatch','actual',len(omitted),'expected',len(planned)])
 info={'audit_mode':'fresh' if args.fresh else 'retained','expected_rows':expected_rows[name],'expected_omitted_query_methods':len(planned),'rows':len(rows),'queries':len(qids),'exact_rows':sum(r['exact']=='1' for r in rows),'violations':violations,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'latency_aggregation':'median of 3 paired repetitions per query, then statistics over queries','methods':{},'omitted_query_methods':[{'query_id':q,'method':m,'reason':'bounded experiment: separate-store control omitted for explicit-empty n>1000 cases under selected audit mode'} for q,m in sorted(omitted)]}
 for m in METHODS:
  vals=[med[q,m] for q in qids if (q,m) in med and (q,'ForwardGF') in med]
  ratios=[med[q,m]/med[q,'ForwardGF'] for q in qids if (q,m) in med and (q,'ForwardGF') in med]
  info['methods'][m]={'mean_of_query_medians_ms':statistics.mean(vals),'median_ratio_to_forward':statistics.median(ratios),'wins_vs_forward':sum(r<1 for r in ratios)}
 info['selectivity_range']=[min(int(r['eligible'])/(int(r['n'])-1) for r in rows),max(int(r['eligible'])/(int(r['n'])-1) for r in rows)]
 info['strata']={}
 for family in sorted({r['family'] for r in rows}):
  qs=[q for q in qids if meta[q]['family']==family and (q,'LazyShared') in med and (q,'ForwardGF') in med]
  info['strata'][family]={'queries':len(qs),'lazy_wins':sum(med[q,'LazyShared']<med[q,'ForwardGF'] for q in qs),'ratio_of_mean_medians_lazy_to_forward':sum(med[q,'LazyShared'] for q in qs)/sum(med[q,'ForwardGF'] for q in qs)}
 if name=='dimension_control_raw.csv':
  strata=defaultdict(set)
  for r in rows:
   key=(r['family'],r['seed'],r['h'],r['method'])
   value=(r['m'],r['eligible'],r['anchor_degree'])
   if r['method'] not in ['LazyShared','LazySeparate']:value+=(r['edge_examinations'],r['state_entries'])
   strata[key].add(value)
  info['paired_dimension_invariant_groups']=len(strata)
  info['paired_dimension_invariant_violations']=sum(len(v)!=1 for v in strata.values())
  if info['paired_dimension_invariant_violations']:violations.append('dimension_pairing')
 report[name]=info
if not report:raise SystemExit('No measured CSVs found in input directory.')
(ROOT/('audit_report_fresh.json' if args.fresh else 'audit_report.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if any(v['violations'] for v in report.values()):raise SystemExit(1)
