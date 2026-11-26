"""Deterministic generated workloads; no production-data claim."""
import hashlib
import pickle
import random
import time
from engine import squared_l2


def make_case(n, family, correlated, seed, dimension=32):
    rng = random.Random(seed)
    groups = 12
    buckets = [[v for v in range(n) if v % groups == g] for g in range(groups)]
    reverse = [[[] for _ in range(n)] for _ in range(2)]
    for relation in range(2):
        for v in range(n):
            pool = buckets[v % groups]
            if family == 'shared_hubs':
                pool = pool[:max(8,len(pool)//4)]
            predecessors = rng.sample(pool, min(6, len(pool)))
            if rng.random() < 0.12:
                predecessors[-1] = rng.randrange(n)
            reverse[relation][v] = sorted(set(predecessors))
    forward = [[[] for _ in range(n)] for _ in range(2)]
    for relation in range(2):
        for v, predecessors in enumerate(reverse[relation]):
            for u in predecessors:
                forward[relation][u].append(v)
    centers = [[rng.gauss(0, 2) for _ in range(dimension)] for _ in range(groups)]
    vectors = [tuple(centers[v % groups][j] + rng.gauss(0,0.45) for j in range(dimension)) for v in range(n)]
    if not correlated:
        rng.shuffle(vectors)
    start = time.perf_counter_ns()
    ordered = sorted(range(n), key=lambda v:(vectors[v][0],v))
    index = [[] for _ in range(n)]
    for position, v in enumerate(ordered):
        candidates = ordered[max(0,position-12):position] + ordered[position+1:position+13]
        nearest = sorted((squared_l2(vectors[v],vectors[u]),u) for u in candidates)[:8]
        # Ring edges guarantee graph connectivity; approximate neighborhoods are explicit.
        index[v] = sorted(set([u for _,u in nearest] + [ordered[(position-1)%n], ordered[(position+1)%n]]))
    index_ms = (time.perf_counter_ns()-start)/1e6
    return dict(n=n,family=family,correlated=correlated,seed=seed,dimension=dimension,
                relations=[0,1,0,1],reverse=reverse,forward=forward,vectors=vectors,index=index,
                index_ms=index_ms, graph_digest=hashlib.sha256(pickle.dumps(reverse)).hexdigest(),
                vector_digest=hashlib.sha256(pickle.dumps(vectors)).hexdigest(),
                index_digest=hashlib.sha256(pickle.dumps(index)).hexdigest())


def make_queries(case, count):
    rng = random.Random(91000+case['seed'])
    # Topological anchor convention, independent of answers: first eight IDs per group.
    anchors = list(range(min(96,case['n'])))
    result=[]
    for qid in range(count):
        anchor = rng.choice(anchors)
        query = tuple(x+rng.gauss(0,0.25) for x in case['vectors'][anchor])
        result.append((qid,anchor,query))
    return result
