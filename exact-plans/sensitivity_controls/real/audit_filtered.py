"""Independent edge-table and artifact consistency audit for the SENSITIVITY cohort."""
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from provenance_checks import verify_benchmark_source
import numpy as np
import pandas as pd
import benchmark_filtered as implementation

P = Path(__file__).resolve().parent
OLD = P.parents[1] / 'native_comparisons' / 'real'
raw = pd.read_csv(P / 'filtered_raw.csv').fillna({'error_text': ''})
environment = json.loads((P / 'environment.json').read_text())
analysis = json.loads((P / 'analysis.json').read_text())
data = np.load(OLD / 'fb15k237_text.npz')
X, Q, E = data['vectors'], data['query_vectors'], data['triples']
queries = json.loads((OLD / 'queries.json').read_text())
saved = json.loads((OLD / 'truth.json').read_text())
assert len(raw) == 2400
assert raw.groupby(['qid', 'method', 'budget']).size().eq(3).all()
assert raw.groupby('qid').size().eq(30).all()
assert set(raw.rep) == {0, 1, 2}
assert (raw.ms >= raw.prefix_ms).all()
assert (raw.prefix_ms > 0).all()
assert verify_benchmark_source(P / 'benchmark_filtered.py', environment['source_hash'])
assert hashlib.sha256((P / 'filtered_raw.csv').read_bytes()).hexdigest() == environment['csv_hash']
for name, digest in environment['input_hashes'].items():
    assert hashlib.sha256((OLD / name).read_bytes()).hexdigest() == digest

inverse_ids, short_ids, imperfect_policies, comparisons = [], [], [], 0
for qi, query in enumerate(queries):
    layer = {query['anchor']}
    for relation in query['relations']:
        selected = E[E[:, 2] == relation]
        mask = np.isin(selected[:, 0], np.fromiter(layer, dtype=np.int64))
        layer = set(map(int, selected[mask, 1]))
    layer.discard(query['anchor'])
    # Independent ordering construction: score every eligible vertex, then
    # lexicographically sort a list of scalar distance/ID pairs.
    scores = [(float(np.dot(X[v] - Q[qi], X[v] - Q[qi])), v) for v in layer]
    oracle = [v for _, v in sorted(scores)[:query['k']]]
    assert oracle == saved[qi]
    fast, _ = implementation.forward(query)
    assert fast == layer
    proof = implementation.Proof(query['anchor'], query['relations'])
    proof_set = {v for v in range(len(X)) if proof.allow(v)}
    comparisons += len(X)
    assert proof_set == layer
    if len(layer) < query['k']:
        short_ids.append(qi)
    if any(abs(a - b) == 237 for a, b in zip(query['relations'], query['relations'][1:])):
        inverse_ids.append(qi)
    records = raw[raw.qid == qi]
    assert records.eligible.eq(len(layer)).all()
    assert records.truth_k.eq(len(oracle)).all()
    for record in records.itertuples():
        answer, prefix = json.loads(record.answer), json.loads(record.prefix_answer)
        for field, result, recall in [('answer', answer, record.recall), ('prefix', prefix, record.prefix_recall)]:
            assert len(set(result)) == len(result) and set(result) <= layer
            calculated = len(set(result) & set(oracle)) / len(oracle)
            assert abs(calculated - recall) < 1e-12
            ordered = [v for _, v in sorted((float(np.dot(X[v] - Q[qi], X[v] - Q[qi])), v) for v in result)]
            assert result == ordered
        if record.method.startswith('Forward') or record.repaired:
            assert answer == oracle
        else:
            assert len(answer) == query['k']
        if record.method == 'InFilter':
            # An observed property of this cohort, not an API guarantee.
            assert bool(record.exception) == (len(layer) < query['k'])
            assert bool(record.repaired) == bool(record.exception)
            if record.exception:
                assert prefix == [] and 'contiguous 2D array' in record.error_text
            else:
                assert len(prefix) == query['k']
        elif record.method == 'PostFilter':
            assert not record.exception and bool(record.repaired) == (len(prefix) < query['k'])
            assert record.callbacks == record.budget
        else:
            assert not record.exception and not record.repaired
        if record.recall < 1 and record.rep == 0:
            imperfect_policies.append(dict(qid=qi, method=record.method, ef=record.budget, recall=record.recall))
    for (method, budget), group in records.groupby(['method', 'budget']):
        # API outputs, decisions and algorithmic counts are stable over repeats.
        for field in ['answer', 'prefix_answer', 'repaired', 'exception', 'callbacks',
                      'proof_edges', 'proof_entries', 'proof_memo_hits', 'forward_edges',
                      'prefix_distances', 'total_distances']:
            assert group[field].nunique() == 1, (qi, method, budget, field)
assert analysis['metadata']['inverse_query_ids'] == inverse_ids
for label, section in analysis['cohorts'].items():
    ids = section['query_ids']
    data = raw[raw.qid.isin(ids)]
    heap = data[data.method == 'ForwardTopKHeap'].groupby('qid').ms.mean()
    for row in section['sensitivity_policies']:
        selected = data[(data.method == row['method']) & (data.budget == row['ef'])]
        means = selected.groupby('qid').ms.mean()
        assert abs(means.mean() - row['mean_ms']) < 1e-12
        assert abs(means.mean() / heap.mean() - row['mean_ratio_to_heap']) < 1e-12
        assert abs(selected.groupby('qid').recall.mean().mean() - row['mean_recall']) < 1e-12

report = dict(status='PASS', queries=80, raw_rows=2400, repetitions=3,
              independent_edge_scan_queries=80, all_vertex_callback_comparisons=comparisons,
              inverse_pair_count=len(inverse_ids), inverse_pair_query_ids=inverse_ids,
              insufficient_cardinality_query_count=len(short_ids), insufficient_cardinality_query_ids=short_ids,
              infilter_expected_exceptions=int(raw[raw.method == 'InFilter'].exception.sum()),
              imperfect_accepted_policies=imperfect_policies,
              checks=['Independent vectorized edge-table oracle versus saved exact answers, adjacency forward and callback truth on every entity.',
                      'All returned labels are legal, unique and exactly reranked; measured exact controls and repaired answers equal the oracle.',
                      'RuntimeError calls return no prefix; all such calls in this cohort have fewer than k eligible entities and are actually repaired.',
                      'All three repeats have identical outputs, decisions and algorithmic counters; all timings are positive and outer timers include prefix time.',
                      'Environment hashes and query-level analysis means/ratios match retained sources and raw data.'],
              hashes={name: hashlib.sha256((P / name).read_bytes()).hexdigest()
                      for name in ['benchmark_filtered.py', 'filtered_raw.csv', 'analyze_filtered.py',
                                   'analysis.json', 'audit_filtered.py', 'inverse_queries.csv']})
(P / 'audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
