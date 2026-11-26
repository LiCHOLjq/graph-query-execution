"""Independent postmeasurement CSV, source and paired-query audit."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
from summarize import percentile


def main(directory):
    d=Path(directory); root=Path(__file__).resolve().parent
    config=json.loads((d/'config.json').read_text())
    env=json.loads((d/'environment.json').read_text())
    sources={name:hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in env['source_sha256'].items()}
    queries={(q['case_id'],str(q['qid'])):q for q in map(json.loads,(d/'queries.jsonl').read_text().splitlines())}
    rows=list(csv.DictReader((d/'raw.csv').open()))
    violations=[]; unique=set(); grouped={}
    for r in rows:
        key=(r['case_id'],r['qid'],r['repeat'],r['method'])
        if key in unique: violations.append(['duplicate row',key])
        unique.add(key)
        q=queries[(r['case_id'],r['qid'])]
        if json.loads(r['ids'])!=q['oracle_ids']: violations.append(['oracle mismatch',key])
        if int(r['eligible'])!=q['eligible']: violations.append(['eligible mismatch',key])
        if r['method']=='CoExec':
            n=int(r['n'])
            if int(r['distances'])>n: violations.append(['score once bound',key])
            if int(r['graph_entries'])>(len(config['relations'])+1)*n: violations.append(['node entry bound',key])
            if int(r['graph_edges'])>len(config['relations'])*n*6: violations.append(['adjacency bound',key])
            if int(r['ann_edges'])>10*n: violations.append(['navigation edge bound',key])
        grouped.setdefault((r['case_id'],r['qid']),{}).setdefault(r['method'],[]).append(r)
    for key,plans in grouped.items():
        if sorted(plans)!=sorted(config['methods']): violations.append(['missing plan',key])
        for method,rr in plans.items():
            if len(rr)!=config['repeats']: violations.append(['missing repetition',key,method])
            for metric in ['graph_work','graph_edges','graph_entries','distances','ann_edges','ann_expansions','ids','eligible']:
                if len({r[metric] for r in rr})!=1: violations.append(['unstable work/output',key,method,metric])
    comparison=[]
    for group in [None]+sorted(set((r['n'],r['family'],r['correlated']) for r in rows)):
        matching=[plans for plans in grouped.values() if group is None or tuple(plans['CoExec'][0][k] for k in ['n','family','correlated'])==group]
        for baseline in [m for m in config['methods'] if m!='CoExec']:
            ratios=[st.median(float(r['latency_ms']) for r in p['CoExec'])/st.median(float(r['latency_ms']) for r in p[baseline]) for p in matching]
            comparison.append(dict(group='all' if group is None else group,baseline=baseline,queries=len(ratios),
                                   coexec_slower_fraction=sum(x>1 for x in ratios)/len(ratios),
                                   median_coexec_over_baseline=st.median(ratios),p95_coexec_over_baseline=percentile(ratios,.95),
                                   max_coexec_over_baseline=max(ratios)))
    report=dict(source_hashes_match=sources,rows=len(rows),unique_rows=len(unique),queries=len(queries),
                per_query_median_over_repetitions=True,violations=violations,paired_latency=comparison,
                note='Postmeasurement audit; timing source was not modified. Work counters are stable across repetitions.')
    (d/'audit_report.json').write_text(json.dumps(report,indent=2))
    if not all(sources.values()) or violations: raise AssertionError(report)
    print(json.dumps(dict(sources=sources,rows=len(rows),queries=len(queries),violations=violations,all_query_comparisons=comparison[:5]),indent=2))


if __name__=='__main__': main(sys.argv[1] if len(sys.argv)>1 else 'results/main')
