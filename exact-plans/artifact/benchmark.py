"""Run measured, fully repaired experiments using only the Python standard library."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from engine import execute, exact_oracle, METHODS
from workloads import make_case,make_queries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/main')
    parser.add_argument('--sizes',default='600,1200')
    parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--queries',type=int,default=12)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    config=dict(sizes=list(map(int,args.sizes.split(','))),seeds=list(range(args.seeds)),queries=args.queries,
                repeats=args.repeats,dimension=32,relations=[0,1,0,1],k=10,stages=[8,24,64],graph_quantum=256,
                methods=list(METHODS),family=['shared_hubs','distributed'],correlated=[True,False],
                timer='perf_counter_ns outer; setup, query hash, execution, repair, major state cleanup included',
                score_timer='nested scalar squared-L2 math.fsum only; do not add to outer latency',
                exclusions=['input generation','index construction','oracle','serialization','result CSV writes'])
    (output/'config.json').write_text(json.dumps(config,indent=2))
    environment=dict(utc_started=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
                     processor=platform.processor(),cpu_count=os.cpu_count(),hostname=platform.node(),
                     affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                     machine=platform.machine(),dependencies='Python standard library only',
                     source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    if Path('/proc/cpuinfo').exists():
        environment['cpu_model']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None)
    (output/'environment.json').write_text(json.dumps(environment,indent=2))
    rows=[]; manifest=[]
    started=time.time()
    with (output/'raw.csv').open('w',newline='') as raw, (output/'queries.jsonl').open('w') as queries_out:
        writer=None
        for n in config['sizes']:
            for family in config['family']:
                for correlated in config['correlated']:
                    for seed in config['seeds']:
                        case=make_case(n,family,correlated,seed)
                        case_id=f'{family}-n{n}-c{int(correlated)}-s{seed}'
                        manifest.append({key:case[key] for key in ['n','family','correlated','seed','index_ms','graph_digest','vector_digest','index_digest']})
                        for qid,anchor,q in make_queries(case,args.queries):
                            truth,eligible=exact_oracle(case['forward'],anchor,case['relations'],case['vectors'],q,10)
                            queries_out.write(json.dumps(dict(case_id=case_id,qid=qid,anchor=anchor,query=q,oracle_ids=truth,eligible=len(eligible)))+'\n')
                            # Warm every method once per query; all methods receive the same inputs.
                            for method in METHODS:
                                warm=execute(case,q,anchor,method)
                                if warm['ids']!=truth: raise AssertionError(('warm',case_id,qid,method))
                            for repeat in range(args.repeats):
                                order=list(METHODS)
                                random.Random(700000+n+seed*1000+qid*10+repeat).shuffle(order)
                                for ordinal,method in enumerate(order):
                                    result=execute(case,q,anchor,method)
                                    if result['ids']!=truth: raise AssertionError((case_id,qid,method,result['ids'],truth))
                                    prefix=result.pop('prefix_ids')
                                    row=dict(case_id=case_id,n=n,family=family,correlated=int(correlated),seed=seed,qid=qid,
                                             anchor=anchor,repeat=repeat,order=ordinal,method=method,
                                             correct=1,recall=1.0,prefix_recall=(len(set(prefix)&set(truth))/len(truth) if truth else 1.0),
                                             **result)
                                    row['ids']=json.dumps(row['ids'],separators=(',',':'))
                                    if writer is None:
                                        writer=csv.DictWriter(raw,fieldnames=list(row)); writer.writeheader()
                                    writer.writerow(row); rows.append(row)
                        raw.flush(); queries_out.flush()
                        print(f'{case_id}: {len(rows)} rows, elapsed {time.time()-started:.1f}s',flush=True)
    (output/'inputs.json').write_text(json.dumps(manifest,indent=2))
    # Paired work identities, including each timing repetition.
    grouped={}
    for row in rows:
        grouped.setdefault((row['case_id'],row['qid'],row['repeat']),{})[row['method']]=row
    violations=[]
    for key,d in grouped.items():
        for name,condition in [
            ('shared graph equals cold suffix',d['CoExec']['graph_work']==d['ColdSuffix']['graph_work']==d['GraphShare']['graph_work']),
            ('graph saving',d['CoExec']['graph_work']<=d['Restart']['graph_work']),
            ('distance saving',d['CoExec']['distances']<=d['Restart']['distances']),
            ('distance once',d['CoExec']['distances']<=d['CoExec']['n']),
            ('cross ablation graph',d['Restart']['graph_work']==d['VectorResume']['graph_work']),
            ('cross ablation score',d['CoExec']['distances']==d['VectorResume']['distances'])]:
            if not condition: violations.append([key,name])
    (output/'verification.json').write_text(json.dumps(dict(rows=len(rows),paired_groups=len(grouped),exact_matches=len(rows),violations=violations,
                                                             elapsed_seconds=time.time()-started,completed=True),indent=2))
    if violations: raise AssertionError(violations[:5])
    from summarize import summarize
    summarize(output)


if __name__=='__main__': main()
