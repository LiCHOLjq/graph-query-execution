import csv
import json
from pathlib import Path
import random
import statistics as st


def percentile(values,p):
    s=sorted(values); pos=(len(s)-1)*p; lo=int(pos)
    return s[lo]+(s[min(lo+1,len(s)-1)]-s[lo])*(pos-lo)


def summarize(directory):
    d=Path(directory)
    rows=list(csv.DictReader((d/'raw.csv').open()))
    numeric=['n','seed','qid','repeat','latency_ms','graph_work','graph_edges','graph_entries','distances','ann_edges','ann_expansions','eligible','score_ms','suffix_entries_peak','score_entries_peak','prefix_recall']
    for row in rows:
        for k in numeric: row[k]=float(row[k])
    groups={}
    for row in rows: groups.setdefault((int(row['n']),row['family'],int(row['correlated']),row['method']),[]).append(row)
    out=[]
    for (n,f,c,m),rr in sorted(groups.items()):
        out.append(dict(n=n,family=f,correlated=c,method=m,rows=len(rr),mean_ms=st.mean(r['latency_ms'] for r in rr),
                        median_ms=st.median(r['latency_ms'] for r in rr),p95_ms=percentile([r['latency_ms'] for r in rr],.95),
                        graph_work=st.mean(r['graph_work'] for r in rr),graph_edges=st.mean(r['graph_edges'] for r in rr),
                        distances=st.mean(r['distances'] for r in rr),score_ms=st.mean(r['score_ms'] for r in rr),
                        suffix_entries_peak=st.mean(r['suffix_entries_peak'] for r in rr),score_entries_peak=st.mean(r['score_entries_peak'] for r in rr),
                        eligible_mean=st.mean(r['eligible'] for r in rr),prefix_recall=st.mean(r['prefix_recall'] for r in rr)))
    pairs={}
    for r in rows: pairs.setdefault((r['case_id'],r['qid'],r['repeat']),{})[r['method']]=r
    comparisons=[]
    for n,f,c in sorted(set((int(r['n']),r['family'],int(r['correlated'])) for r in rows)):
        for baseline in ('Restart','GraphShare','VectorResume','ColdSuffix','ForwardGF'):
            selected=[p for p in pairs.values() if int(p['CoExec']['n'])==n and p['CoExec']['family']==f and int(p['CoExec']['correlated'])==c]
            by_seed={}
            for p in selected: by_seed.setdefault(int(p['CoExec']['seed']),[]).append(p)
            seed_totals=[(sum(p[baseline]['latency_ms'] for p in pp),sum(p['CoExec']['latency_ms'] for p in pp)) for pp in by_seed.values()]
            rng=random.Random(8461); boots=[]
            for _ in range(5000):
                draw=[rng.choice(seed_totals) for _ in seed_totals]
                boots.append(sum(x for x,y in draw)/sum(y for x,y in draw))
            comparisons.append(dict(n=n,family=f,correlated=c,baseline=baseline,
                                    ratio_of_total_latency=sum(x for x,y in seed_totals)/sum(y for x,y in seed_totals),
                                    bootstrap_seed_cluster_ci95=[percentile(boots,.025),percentile(boots,.975)],
                                    graph_reduction=1-sum(p['CoExec']['graph_work'] for p in selected)/sum(p[baseline]['graph_work'] for p in selected),
                                    distance_reduction=1-sum(p['CoExec']['distances'] for p in selected)/sum(p[baseline]['distances'] for p in selected)))
    summary=dict(aggregates=out,comparisons=comparisons,
                 uncertainty='5000 paired cluster bootstrap draws over five generator seeds; repetitions and queries stay together, exploratory small-cluster CI',
                 row_count=len(rows),query_count=len(set((r['case_id'],r['qid']) for r in rows)))
    (d/'summary.json').write_text(json.dumps(summary,indent=2))
    for filename,data in [('aggregate.csv',out),('comparisons.csv',comparisons)]:
        with (d/filename).open('w',newline='') as fh:
            writer=csv.DictWriter(fh,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    print(json.dumps(dict(rows=len(rows),queries=summary['query_count'],comparisons=len(comparisons))))
    return summary


if __name__=='__main__':
    import sys
    summarize(sys.argv[1] if len(sys.argv)>1 else 'results/main')
