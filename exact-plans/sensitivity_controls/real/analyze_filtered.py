"""Query-level, inverse-pair-stratified analysis of frozen NATIVE and new SENSITIVITY runs."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
OLD = ROOT / 'native_comparisons' / 'real'
queries = json.loads((OLD / 'queries.json').read_text())
relations = json.loads((OLD / 'relations.json').read_text())
assert len(relations) == 474
assert all(relations[r + 237] == 'inverse:' + relations[r] for r in range(237))
df = pd.read_csv(P / 'filtered_raw.csv')
native = pd.read_csv(ROOT / 'native_comparisons' / 'native' / 'strong_real_raw.csv')
original = pd.read_csv(OLD / 'hnsw_raw.csv')
actual = pd.read_csv(OLD / 'actual_policy.csv')
queryrows = []
for q in queries:
    word = q['relations']
    pairs = [(i, a, b) for i, (a, b) in enumerate(zip(word, word[1:]))
             if abs(a - b) == 237]
    vals = df[df.qid == q['qid']]
    queryrows.append(dict(qid=q['qid'], h=q['h'], relations=json.dumps(word),
                          adjacent_inverse=bool(pairs), inverse_pairs=json.dumps(pairs),
                          eligible=int(vals.eligible.iloc[0]),
                          selectivity=int(vals.eligible.iloc[0]) / 14504,
                          anchor_total_degree=q['anchor_degree'],
                          first_label_degree=int(native[native.query_id == q['qid']].anchor_degree.iloc[0])))
querytable = pd.DataFrame(queryrows)
querytable.to_csv(P / 'inverse_queries.csv', index=False)
groups = {'All': list(querytable.qid),
          'NoAdjacentInverse': list(querytable[~querytable.adjacent_inverse].qid),
          'AdjacentInverse': list(querytable[querytable.adjacent_inverse].qid)}


def bootstrap(values, other=None, seed=20261025):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    ix = np.random.default_rng(seed).integers(0, len(values), (5000, len(values)))
    estimates = values[ix].mean(1)
    if other is not None:
        estimates /= np.asarray(other)[ix].mean(1)
    return [float(v) for v in np.quantile(estimates, [.025, .975])]


def common_stats(data, latency, recall, comparison=None):
    grouped = data.groupby('qid')
    times, recalls = grouped[latency].mean(), grouped[recall].mean()
    result = dict(queries=len(times), mean_ms=float(times.mean()),
                  mean_ms_ci95=bootstrap(times), mean_recall=float(recalls.mean()),
                  mean_recall_ci95=bootstrap(recalls))
    if comparison is not None:
        baseline = comparison.reindex(times.index)
        result.update(mean_ratio_to_heap=float(times.mean() / baseline.mean()),
                      ratio_ci95=bootstrap(times, baseline),
                      faster_query_means=int((times < baseline).sum()))
    return result


report, flat = {}, []
for label, ids in groups.items():
    qtable = querytable[querytable.qid.isin(ids)]
    n = native[native.query_id.isin(ids)]
    nwide = n.groupby(['query_id', 'method']).ms.median().unstack()
    f = df[df.qid.isin(ids)]
    heap = f[f.method == 'ForwardTopKHeap'].groupby('qid').ms.mean()
    section = dict(queries=len(ids), query_ids=ids,
                   eligible_min_median_max=[float(qtable.eligible.min()), float(qtable.eligible.median()), float(qtable.eligible.max())],
                   selectivity_min_median_max=[float(qtable.selectivity.min()), float(qtable.selectivity.median()), float(qtable.selectivity.max())],
                   first_label_degree_min_median_max=[float(qtable.first_label_degree.min()), float(qtable.first_label_degree.median()), float(qtable.first_label_degree.max())],
                   h_counts={str(k): int(v) for k, v in qtable.h.value_counts().sort_index().items()},
                   native_winner_counts={k: int(v) for k, v in nwide.idxmin(axis=1).value_counts().items()},
                   native_mean_query_medians_ms={k: float(v) for k, v in nwide.mean().items()},
                   sensitivity_exact={}, sensitivity_policies=[], native_component_prefixes=[], native_actual_policies=[])
    for method in ['ForwardSort', 'ForwardTopKHeap']:
        section['sensitivity_exact'][method] = common_stats(f[f.method == method], 'ms', 'recall')
    for (method, budget), data in f[~f.method.str.startswith('Forward')].groupby(['method', 'budget']):
        result = dict(method=method, ef=int(budget), **common_stats(data, 'ms', 'recall', heap),
                      repair_fraction=float(data.repaired.mean()), exception_fraction=float(data.exception.mean()),
                      prefix_mean_ms=float(data.groupby('qid').prefix_ms.mean().mean()),
                      prefix_mean_ms_ci95=bootstrap(data.groupby('qid').prefix_ms.mean()),
                      mean_callbacks=float(data.groupby('qid').callbacks.mean().mean()),
                      mean_proof_edges=float(data.groupby('qid').proof_edges.mean().mean()),
                      mean_proof_entries=float(data.groupby('qid').proof_entries.mean().mean()))
        # Conditional recall is explicitly labelled. RuntimeError returns no list.
        success = data[data.exception == 0]
        successful = success.groupby('qid').prefix_recall.mean()
        result.update(prefix_successful_queries=len(successful),
                      prefix_successful_call_mean_ms=float(success.groupby('qid').prefix_ms.mean().mean()),
                      prefix_successful_call_mean_ms_ci95=bootstrap(success.groupby('qid').prefix_ms.mean()),
                      prefix_successful_call_recall=float(successful.mean()),
                      prefix_successful_call_recall_ci95=bootstrap(successful),
                      # Diagnostic metric, not a successful-call recall claim.
                      prefix_recall_failure_as_empty=float(data.groupby('qid').prefix_recall.mean().mean()))
        section['sensitivity_policies'].append(result)
        flat.append(dict(cohort=label, **{k: json.dumps(v) if isinstance(v, list) else v for k, v in result.items()}))
    for budget, data in original[original.qid.isin(ids)].groupby('budget'):
        section['native_component_prefixes'].append(dict(budget=int(budget),
                                                       **common_stats(data, 'prefix_ms', 'recall'),
                                                       underfill_fraction=float(data.underfill.mean())))
    a = actual[actual.qid.isin(ids)]
    oldheap = a[a.method == 'ForwardTopKHeap'].groupby('qid').ms.mean()
    section['native_actual_heap'] = common_stats(a[a.method == 'ForwardTopKHeap'], 'ms', 'recall')
    for budget, data in a[a.method == 'HNSWUnderfill'].groupby('budget'):
        section['native_actual_policies'].append(dict(budget=int(budget),
                                                    **common_stats(data, 'ms', 'recall', oldheap),
                                                    repair_fraction=float(data.repaired.mean())))
    report[label] = section

metadata = dict(inverse_definition='An adjacent pair a,b satisfies abs(a-b)==237; either original/inverse order is included.',
                inverse_query_ids=groups['AdjacentInverse'],
                forward_then_inverse_query_count=sum(any(a < 237 and b == a + 237 for a, b in zip(q['relations'], q['relations'][1:])) for q in queries),
                inverse_then_forward_query_count=sum(any(a >= 237 and a == b + 237 for a, b in zip(q['relations'], q['relations'][1:])) for q in queries),
                inference='Paired nonparametric bootstrap of query means (5000 resamples, fixed seed); intervals condition on the fixed graph, embeddings, index and sampled query cohort.',
                exact_control='ForwardTopKHeap is the predeclared comparison; ForwardSort is retained in the same run.',
                timing_scope='Timer ends after answer construction, before execute returns its diagnostic dictionary; final local Proof/cache destruction and caller-side output validation/serialization are outside the timer.',
                prefix_failure='Expected in-filter exceptions produce no returned list; successful-call recall excludes them and always reports its denominator. failure-as-empty recall is retained only as an explicitly defined API-output diagnostic.',
                sources={'hnswlib_v0_8_0_readme': 'https://raw.githubusercontent.com/nmslib/hnswlib/v0.8.0/README.md',
                         'hnswlib_v0_8_0_binding': 'https://raw.githubusercontent.com/nmslib/hnswlib/v0.8.0/python_bindings/bindings.cpp'},
                input_hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [P / 'filtered_raw.csv', OLD / 'queries.json', OLD / 'relations.json',
                                           OLD / 'hnsw_raw.csv', OLD / 'actual_policy.csv',
                                           ROOT / 'native_comparisons' / 'native' / 'strong_real_raw.csv']})
(P / 'analysis.json').write_text(json.dumps(dict(metadata=metadata, cohorts=report), indent=2))
pd.DataFrame(flat).to_csv(P / 'policy_summary.csv', index=False)

# Comparing prefixes on the SAME 56-query successful in-filter cohort avoids
# either fabricating failure recall or juxtaposing different denominators.
success_ids = df[(df.method == 'InFilter') & (df.budget == 64) & (df.exception == 0)].qid.unique()
assert all(set(df[(df.method == 'InFilter') & (df.budget == b) & (df.exception == 0)].qid) == set(success_ids) for b in [64, 256, 1024, 4096])
success_rows = []
for (method, budget), subset in df[(~df.method.str.startswith('Forward')) & df.qid.isin(success_ids)].groupby(['method', 'budget']):
    success_rows.append(dict(method=method, budget=int(budget), **common_stats(subset, 'prefix_ms', 'prefix_recall')))
pd.DataFrame(success_rows).to_csv(P / 'matched_successful_prefixes.csv', index=False)
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.25))
colors = {'PostFilter': '#1764ab', 'InFilter': '#c35419'}
for method in ['PostFilter', 'InFilter']:
    subset = df[(df.method == method) & df.qid.isin(success_ids)]
    prefix = subset.groupby('budget')[['prefix_ms', 'prefix_recall']].mean()
    axes[0].plot(prefix.prefix_recall, prefix.prefix_ms, marker='o', color=colors[method], label=method)
    policies = df[df.method == method].groupby('budget')[['ms', 'recall']].mean()
    axes[1].plot(policies.recall, policies.ms, marker='o', color=colors[method], label=method)
    for budget in prefix.index:
        left_offset = {64: (4, -5), 256: (4, -4), 1024: (4, 0), 4096: (4, 10)}[budget] if method == 'InFilter' else (4, 3)
        right_offset = {64: (-28, 3), 256: (-28, -7), 1024: (8, 0), 4096: (-28, 13)}[budget] if method == 'InFilter' else (4, 3)
        axes[0].annotate(str(budget), (prefix.loc[budget, 'prefix_recall'], prefix.loc[budget, 'prefix_ms']), xytext=left_offset, textcoords='offset points', fontsize=7)
        axes[1].annotate(str(budget), (policies.loc[budget, 'recall'], policies.loc[budget, 'ms']), xytext=right_offset, textcoords='offset points', fontsize=7)
heap_ms = df[df.method == 'ForwardTopKHeap'].ms.mean()
axes[1].scatter([1], [heap_ms], color='#333333', marker='s', label='ForwardTopKHeap')
axes[1].annotate('Exact', (1, heap_ms), xytext=(-32, -4), textcoords='offset points', fontsize=7)
axes[0].set_title(f'Prefixes: same {len(success_ids)} successful-call queries')
axes[1].set_title('Executed repair policies: all 80 queries')
for ax in axes:
    ax.set_xlabel(r'Mean recall against exact top-$\min(10, |C_P|)$')
    ax.set_ylabel('Mean query latency (ms)')
    ax.set_yscale('log')
    ax.set_ylim(.45, 26)
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
axes[1].set_xlim(.9965, 1.0013)
axes[1].ticklabel_format(axis='x', style='plain', useOffset=False)
fig.tight_layout()
fig.savefig(P / 'filtered_quality_cost.pdf', bbox_inches='tight')
fig.savefig(P / 'filtered_quality_cost.png', dpi=180, bbox_inches='tight')
plt.close(fig)

lines = ['# SENSITIVITY real-cohort revision', '',
         'All times below are means of per-query means over three repetitions, with one complete untimed warmup. Ratios use the matched ForwardTopKHeap control. CIs resample queries, not repetitions, and condition on this fixed graph, corpus and index.', '',
         f"## Inverse-pair audit: {len(groups['AdjacentInverse'])}/80 predicates", '',
         'An immediate inverse pair is adjacent labels a,b with |a-b|=237, in either direction. IDs: ' + ', '.join(map(str, groups['AdjacentInverse'])) + '.', '',
         'The original sampler is retained. Inverse-augmented random walks can create attribute-sharing round trips; this is a workload construction effect and not evidence that these are representative graph-retrieval queries.', '',
         '| Cohort | Queries | Eligible min/median/max | Selectivity min/median/max | Native winners |',
         '|---|---:|---|---|---|']
for label, section in report.items():
    counts = '/'.join(f'{v:g}' for v in section['eligible_min_median_max'])
    sels = '/'.join(f'{100*v:.4g}%' for v in section['selectivity_min_median_max'])
    lines.append(f"| {label} | {section['queries']} | {counts} | {sels} | {section['native_winner_counts']} |")
lines += ['', '## New matched timing cohort', '',
          'PostFilter retrieves L candidates at ef=L; InFilter asks for k=10 at the same ef=L with `filter=Proof.allow`. ef is a search parameter, not an equal-work budget. Predicate callbacks, memo construction, Python transitions, graph counters, distance reranking and actual conditional repair are included. The timer ends after constructing the answer but before the diagnostic return dictionary and final local Proof/cache destruction; caller validation/serialization is excluded. This experiment cannot isolate GIL overhead from graph verification or different search work.', '',
          'No free exact cardinality is passed to either plan. The 24 queries with fewer than 10 eligible entities cause the expected hnswlib contiguous-array RuntimeError at every ef. A failed API call returns no list. We report successful-call recall conditionally (56 queries) and the failure fraction (30%); the practical policy catches that error and performs measured forward repair. Underfill acceptance with 10 legal results does not certify top-k recall.', '',
          '| Cohort | Method | ef | Policy ms | Recall | Ratio to heap (95% CI) | Repair fraction |',
          '|---|---|---:|---:|---:|---|---:|']
for label, section in report.items():
    for s in section['sensitivity_policies']:
        lo, hi = s['ratio_ci95']
        lines.append(f"| {label} | {s['method']} | {s['ef']} | {s['mean_ms']:.4f} | {s['mean_recall']:.6f} | {s['mean_ratio_to_heap']:.3f} [{lo:.3f}, {hi:.3f}] | {s['repair_fraction']:.3f} |")
lines += ['', 'The detailed JSON retains confidence intervals for all latency and recall means, all inverse-pair subsets, NATIVE component-prefix measurements and NATIVE executed-policy measurements as separate cohorts. They are not pooled with this run. On the same 56 successful-call queries, raw in-filter recall is 0.998214/1/1/1 versus post-filter 0.482143/0.630357/0.689286/0.841071 across ef=64/256/1024/4096. In-filtering substantially improves prefix quality, at higher measured cost; it is not uniformly inferior on every quality criterion or individual query.', '',
          '## Reproduction and provenance', '',
          'Run `python sensitivity_controls/real/benchmark_filtered.py`, `python sensitivity_controls/real/analyze_filtered.py`, and `python sensitivity_controls/real/audit_filtered.py` from the exact-plans directory. Inputs remain frozen under native_comparisons/real. The benchmark uses Windows logical CPU 4, one hnswlib thread, BLAS thread limits of one, disabled GC during timing, and randomized query and method order. Input hashes and environment are in environment.json; analysis input hashes are in analysis.json.', '',
          'The native CSV column `seed=qid%10` controls method-order randomization only; it does not denote ten independent real graphs. Native comparisons use per-query medians; Python HNSW comparisons use per-query means.', '',
          'hnswlib 0.8.0 documentation: https://raw.githubusercontent.com/nmslib/hnswlib/v0.8.0/README.md (filter API, single-thread guidance, and Python/GIL limitation). The pinned binding checks returned queue size against k and throws before returning arrays when they differ: https://raw.githubusercontent.com/nmslib/hnswlib/v0.8.0/python_bindings/bindings.cpp (lines 556–634).']
(P / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'inverse_queries': len(groups['AdjacentInverse']), 'cohorts': {k: {x: v[x] for x in ['queries', 'eligible_min_median_max', 'native_winner_counts']} for k, v in report.items()}}, indent=2))
