"""SENSITIVITY matched post-filter/in-filter HNSW policies and exact controls.

The input artifacts are frozen under native_comparisons/real. In-filter searches ask
for k=10 at ef=L; post-filter searches ask for L candidates at the same ef.
An ef setting is NOT a common bound on visited vertices or predicate calls.
No exact cardinality, bitmap, or oracle answer is exposed to either policy.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import csv, gc, hashlib, heapq, importlib.metadata, json, platform, random, sys, time
from pathlib import Path
import hnswlib
import numpy as np
import psutil

P = Path(__file__).resolve().parent
SOURCE = P.parents[1] / 'native_comparisons' / 'real'
BUDGETS = [64, 256, 1024, 4096]
if __name__ == '__main__':
    psutil.Process().cpu_affinity([4])
data = np.load(SOURCE / 'fb15k237_text.npz')
X, Q, E = data['vectors'], data['query_vectors'], data['triples']
N, D = X.shape
queries = json.loads((SOURCE / 'queries.json').read_text())
saved_truth = json.loads((SOURCE / 'truth.json').read_text())
out, rev = {}, {}
for a, b, r in E:
    a, b, r = int(a), int(b), int(r)
    out.setdefault((a, r), []).append(b)
    rev.setdefault((b, r), []).append(a)
for adj in [out, rev]:
    for values in adj.values():
        values.sort()
index = hnswlib.Index(space='l2', dim=D)
# hnswlib's narrow Windows filename handling cannot consume this Unicode path.
prior = os.getcwd()
os.chdir(SOURCE)
index.load_index('hnsw_index.bin')
os.chdir(prior)
index.set_num_threads(1)


class Proof:
    def __init__(self, anchor, word):
        self.anchor, self.word, self.memo = anchor, word, {}
        self.entries = self.edges = self.hits = self.callbacks = 0

    def check(self, vertex, depth=None):
        if depth is None:
            depth = len(self.word)
        key = (vertex, depth)
        if key in self.memo:
            self.hits += 1
            return self.memo[key]
        self.entries += 1
        if depth == 0:
            answer = vertex == self.anchor
        else:
            answer = False
            for parent in rev.get((vertex, self.word[depth - 1]), ()):
                self.edges += 1
                if self.check(parent, depth - 1):
                    answer = True
                    break
        self.memo[key] = answer
        return answer

    def allow(self, label):
        self.callbacks += 1
        vertex = int(label)
        return vertex != self.anchor and self.check(vertex)


def forward(query):
    layer, edges = {query['anchor']}, 0
    for relation in query['relations']:
        nxt = set()
        for vertex in layer:
            neighbors = out.get((vertex, relation), ())
            edges += len(neighbors)
            nxt.update(neighbors)
        layer = nxt
    return layer - {query['anchor']}, edges


def rank(vertices, vector, cache, k, heap=False):
    def key(vertex):
        if vertex not in cache:
            delta = X[vertex] - vector
            cache[vertex] = float(np.dot(delta, delta))
        return cache[vertex], vertex
    if heap:
        return heapq.nsmallest(k, vertices, key=key)
    return sorted(vertices, key=key)[:k]


def execute(query, vector, method, budget):
    started = time.perf_counter_ns()
    cache, repaired, exception, error_text, repair_edges = {}, False, False, '', 0
    if method.startswith('Forward'):
        eligible, edges = forward(query)
        answer = rank(eligible, vector, cache, query['k'], method == 'ForwardTopKHeap')
        ms = (time.perf_counter_ns() - started) / 1e6
        return dict(ms=ms, prefix_ms=ms, answer=answer, prefix_answer=answer,
                    repaired=0, exception=0, error_text='', callbacks=0,
                    proof_edges=0, proof_entries=0, proof_memo_hits=0,
                    forward_edges=edges, prefix_distances=len(cache),
                    total_distances=len(cache))
    index.set_ef(budget)
    proof = Proof(query['anchor'], query['relations'])
    if method == 'InFilter':
        try:
            labels, _ = index.knn_query(vector.astype(np.float32), k=query['k'],
                                        num_threads=1, filter=proof.allow)
            hits = [int(label) for label in labels[0]]
        except RuntimeError as error:
            # The binding cannot return an array of k labels when fewer are
            # admissible. Do not infer that cause from a free truth count.
            error_text = str(error)
            if 'Cannot return the results in a contiguous 2D array' not in error_text:
                raise
            exception, hits = True, []
    else:
        labels, _ = index.knn_query(vector.astype(np.float32), k=budget, num_threads=1)
        hits = [int(label) for label in labels[0] if proof.allow(label)]
    prefix_answer = rank(hits, vector, cache, query['k'])
    prefix_ms = (time.perf_counter_ns() - started) / 1e6
    prefix_distances = len(cache)
    if exception or len(prefix_answer) < query['k']:
        repaired = True
        eligible, repair_edges = forward(query)
        answer = rank(eligible, vector, cache, query['k'], heap=True)
    else:
        answer = prefix_answer
    ms = (time.perf_counter_ns() - started) / 1e6
    return dict(ms=ms, prefix_ms=prefix_ms, answer=answer,
                prefix_answer=prefix_answer, repaired=int(repaired),
                exception=int(exception), error_text=error_text,
                callbacks=proof.callbacks, proof_edges=proof.edges,
                proof_entries=proof.entries, proof_memo_hits=proof.hits,
                forward_edges=repair_edges, prefix_distances=prefix_distances,
                total_distances=len(cache))


def run():
    # Oracle built by scanning relation-specific edge tables, not via adjacency
    # or the Proof/forward implementations used inside the timed paths.
    eligible, truths = [], []
    for qi, (query, vector) in enumerate(zip(queries, Q)):
        layer = {query['anchor']}
        for relation in query['relations']:
            layer = {int(b) for a, b, _ in E[E[:, 2] == relation] if int(a) in layer}
        layer.discard(query['anchor'])
        eligible.append(layer)
        truth = rank(layer, vector, {}, query['k'])
        assert truth == saved_truth[qi]
        truths.append(truth)
    methods = [('ForwardSort', 0), ('ForwardTopKHeap', 0)]
    methods += [(method, budget) for method in ['PostFilter', 'InFilter'] for budget in BUDGETS]
    rng = random.Random(20261025)
    rows, enabled = [], gc.isenabled()
    gc.disable()
    try:
        for rep in range(-1, 3):
            order = list(range(len(queries)))
            rng.shuffle(order)
            for qi in order:
                query, vector = queries[qi], Q[qi]
                method_order = methods.copy()
                rng.shuffle(method_order)
                for method, budget in method_order:
                    result = execute(query, vector, method, budget)
                    answer, prefix = result.pop('answer'), result.pop('prefix_answer')
                    for output in [answer, prefix]:
                        assert len(output) == len(set(output))
                        assert set(output) <= eligible[qi]
                        assert output == rank(output, vector, {}, query['k'])
                    if result['repaired'] or method.startswith('Forward'):
                        assert answer == truths[qi]
                    else:
                        assert len(answer) == query['k']
                    if result['exception']:
                        assert prefix == []
                    denominator = len(truths[qi])
                    result.update(answer=json.dumps(answer), prefix_answer=json.dumps(prefix),
                                  recall=len(set(answer) & set(truths[qi])) / denominator,
                                  prefix_recall=len(set(prefix) & set(truths[qi])) / denominator,
                                  returned=len(answer), prefix_returned=len(prefix))
                    if rep >= 0:
                        rows.append(dict(rep=rep, qid=qi, h=query['h'], mode=query['mode'],
                                         method=method, budget=budget, eligible=len(eligible[qi]),
                                         truth_k=denominator, **result))
            print('completed repetition', rep, flush=True)
    finally:
        if enabled:
            gc.enable()
    with (P / 'filtered_raw.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(date='2026-10-25', queries=len(queries), repetitions=3, warmup_passes=1,
                  rows=len(rows), platform=platform.platform(), python=sys.version,
                  hnswlib_version=importlib.metadata.version('hnswlib'),
                  cpu_affinity=psutil.Process().cpu_affinity(), gc_during_timing=False,
                  hnsw_threads=1, memory_bytes=psutil.virtual_memory().total,
                  timing='One outer timer includes query setup, callback/proof construction, search, exact rerank, and conditional heap repair. Nested prefix timer is diagnostic; graph counters are enabled for every method.',
                  limitations=['Fixed existing graph, vectors, queries, and index seed; repetitions are not independent data samples.',
                               'ef is shared across integrations but is not an equal bound on graph visits or predicate calls.',
                               'Python callback transitions and GIL costs are charged; the experiment does not isolate their causal contribution.',
                               'On an expected contiguous-array RuntimeError no labels are returned; raw prefix is empty, and the practical policy executes exact forward repair.',
                               'Complete in-filter or post-filter output is approximate; acceptance of k legal labels is not a recall certificate.'],
                  source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  csv_hash=hashlib.sha256((P / 'filtered_raw.csv').read_bytes()).hexdigest(),
                  input_hashes={name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                                for name in ['fb15k237_text.npz', 'queries.json', 'truth.json', 'hnsw_index.bin']})
    (P / 'environment.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    run()
