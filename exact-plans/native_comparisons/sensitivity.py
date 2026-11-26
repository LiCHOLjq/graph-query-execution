"""Fresh three-factor sensitivity; imports frozen engine unchanged."""
from pathlib import Path
import sys,csv,json,time,random,gc,platform,os,hashlib,heapq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"artifact"))
from engine import Stats,SuffixEngine,Materializer,Scorer,BestFirst,exact_oracle
from workloads import make_case,make_queries

def run(case,q,anchor,g,a,r,stages,quantum):
    tick=time.perf_counter_ns();stat=Stats();identity=(case["graph_digest"],anchor)
    eager=SuffixEngine(case["reverse"],anchor,case["relations"],identity,stat)
    lazy=eager if g else SuffixEngine(case["reverse"],anchor,case["relations"],identity,stat)
    mat=Materializer(eager);ann=None
    for b in stages:
        if ann is None or not a:ann=BestFirst(case["index"],Scorer(case["vectors"],q,stat),identity)
        ann.advance_expansions(b)
        for v in ann.candidates(b):
            if v!=anchor:lazy.finish(lazy.task(v))
        mat.advance(quantum)
    mat.advance(1<<60)
    scorer=ann.scorer if r else Scorer(case["vectors"],q,stat)
    ids=[v for _,v in heapq.nsmallest(10,((scorer.score(v),v) for v in sorted(mat.eligible)))]
    eager.nodes.clear();lazy.nodes.clear();ann.heap.clear();ann.visited.clear()
    ann.scorer.cache.clear();scorer.cache.clear();mat.eligible.clear()
    elapsed=(time.perf_counter_ns()-tick)/1e6
    return ids,elapsed,stat.graph_entries,stat.graph_edges,stat.distances,stat.ann_edges

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    env={"python":sys.version,"platform":platform.platform(),"affinity":sorted(os.sched_getaffinity(0)),
         "gc_enabled":gc.isenabled(),"gc_threshold":gc.get_threshold(),
         "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for p in ["/proc/meminfo","/sys/devices/system/cpu/smt/active","/sys/devices/system/cpu/intel_pstate/no_turbo","/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"]:
        try:env[p]=Path(p).read_text()
        except OSError:env[p]="unavailable"
    (out/"environment.json").write_text(json.dumps(env,indent=2))
    config={"n":600,"seeds":list(range(10)),"queries":2,"repeats":3,"schedules":[[8,24,64],[16,48,128]],"quanta":[0,256,1024],
            "factors":["graph_store_shared","ANN_frontier_retained","repair_scores_reused"],"gc":"default enabled",
            "interpretation":"ANN retention necessarily retains scores used internally; repair reuse is independently toggled."}
    (out/"config.json").write_text(json.dumps(config,indent=2))
    count=0
    with (out/"raw.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["family","aligned","seed","qid","anchor","budget","quantum","G","A","R","repeat","ms","entries","edges","distances","ann_edges"])
        for fam in ["distributed","shared_hubs"]:
          for corr in [False,True]:
            for seed in range(10):
              case=make_case(600,fam,corr,seed)
              for qid,anchor,q in make_queries(case,2):
                truth,_=exact_oracle(case["forward"],anchor,case["relations"],case["vectors"],q,10)
                for budget,stages in enumerate(config["schedules"]):
                  for quantum in config["quanta"]:
                    for rep in range(3):
                      cells=[(g,a,r) for g in [0,1] for a in [0,1] for r in [0,1]]
                      random.Random(seed*10000+qid*100+rep).shuffle(cells)
                      for g,a,r in cells:
                        ids,*values=run(case,q,anchor,g,a,r,stages,quantum)
                        assert ids==truth,(fam,corr,seed,qid,g,a,r)
                        w.writerow([fam,int(corr),seed,qid,anchor,budget,quantum,g,a,r,rep,*values]);count+=1
              f.flush();print(fam,corr,seed,count,flush=True)
    (out/"verification.json").write_text(json.dumps({"rows":count,"exact_matches":count,"completed":True}))
if __name__=="__main__":main()

