"""Recompute the paper's four-plan results from the released measurements.

Run from any directory with Python 3.8 or newer. An optional positional
argument selects the ordered_access directory of a separately extracted copy.
The comparisons use recorded measurements and do not rerun timed experiments.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, quantiles


METHODS = ("ForwardTopK", "ReverseRecursiveTopK", "LazyShared", "LazyHeap")
SAMPLE_SIZES = {"main": 1500, "degree": 1440, "dimension": 240, "matched": 60, "real": 80}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_cohort(root, name):
    with (root / "analysis" / (name + "_query_medians.csv")).open(newline="") as f:
        queries = list(csv.DictReader(f))
    indexed = {row["query_id"]: row for row in queries}
    check(len(indexed) == len(queries) == SAMPLE_SIZES[name], name + ": query count")
    cells = defaultdict(dict)
    with (root / "results" / (name + "_raw.csv")).open(newline="") as f:
        for row in csv.DictReader(f):
            query = indexed[row["query_id"]]
            elapsed = float(row["ms"])
            check(elapsed > 0 and math.isfinite(elapsed), name + ": invalid time")
            check(row["exact"] == "1" and row["answer_fnv64"] == query["answer_fnv64"],
                  name + ": recorded answer check")
            check(row["method"] in METHODS and row["cohort"] == name, name + ": unexpected method or cohort")
            cell = cells[row["query_id"], row["method"]]
            check(row["rep"] not in cell, name + ": duplicate repetition")
            cell[row["rep"]] = elapsed
    for row in queries:
        for method in METHODS:
            times = cells[row["query_id"], method]
            check(set(times) == {"0", "1", "2"}, name + ": repetitions")
            observed = median(times.values())
            check(math.isclose(observed, float(row[method]), rel_tol=1e-12), name + ": median")
            row[method] = observed
        row["s"] = int(row["eligible"]) / (int(row["n"]) - 1)
    return queries


def strict_winner(row):
    fastest = min(row[m] for m in METHODS)
    winners = [m for m in METHODS if row[m] == fastest]
    return winners[0] if len(winners) == 1 else "tie"


def units(rows):
    return [len({r["graph_fnv64"] for r in rows}),
            len({(r["graph_fnv64"], r["anchor"], r["word"]) for r in rows})]


def materializer_comparison(rows):
    # The comparison chooses the faster materializer after measuring each query.
    best = [min(r["ForwardTopK"], r["ReverseRecursiveTopK"]) for r in rows]
    heap = [r["LazyHeap"] for r in rows]
    ratios = [h / b for h, b in zip(heap, best)]
    q1, _, q3 = quantiles(ratios, n=4, method="inclusive")
    return [round(x, 3) for x in (mean(heap) / mean(best), median(ratios), q1, q3)]


def summarize(root):
    cohorts = {c: read_cohort(root, c) for c in SAMPLE_SIZES}
    main = cohorts["main"]
    unsaturated = [r for r in main if 0 < r["s"] < .9998]
    wins = [r for r in unsaturated if strict_winner(r) == "LazyHeap"]
    degree = [r for r in cohorts["degree"] if 0 < r["s"] < .9998]
    degree_wins = [r for r in degree if strict_winner(r) == "LazyHeap"]
    return {
        "timed_rows_checked": sum(len(rows) * len(METHODS) * 3 for rows in cohorts.values()),
        "graph_fingerprints_and_predicates": {k: units(cohorts[k]) for k in ("main", "degree", "dimension", "matched")},
        "main_unsaturated": len(unsaturated),
        "main_unsaturated_winners": dict(Counter(map(strict_winner, unsaturated))),
        "main_unsaturated_heap_wins_at_or_above_90pct": sum(r["s"] >= .9 for r in wins),
        "degree_unsaturated": len(degree),
        "degree_unsaturated_winners": dict(Counter(map(strict_winner, degree))),
        "lowest_degree_heap_winning_selectivity_pct": round(100 * min(r["s"] for r in degree_wins), 5),
        "real_winners": dict(Counter(map(strict_winner, cohorts["real"]))),
        "heap_over_best_materializer_ratio_of_means_median_q1_q3": {
            "all_main": materializer_comparison(main),
            "nonempty_unsaturated_main": materializer_comparison(unsaturated),
        },
    }


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("ordered_access", nargs="?", type=Path, default=here / "ordered_access")
    findings = summarize(cli.parse_args().ordered_access)
    expected = json.loads((here / "expected_paper_statistics.json").read_text())
    check(findings == expected, "Computed results differ from the paper:\n" + json.dumps(findings, indent=2))
    print(json.dumps(findings, indent=2))
    print("PASS")
