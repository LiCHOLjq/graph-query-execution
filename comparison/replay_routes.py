"""Replay frozen graph-routing decisions under two measured execution menus.

The output is a descriptive execution-time replay. Both costs entering a
comparison come from the same measurement batch and the same request.
Run without arguments from any directory in a complete artifact checkout.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
PLANS = {"Forward": "ForwardTopK", "Reverse": "ReverseRecursiveTopK", "Lazy": "LazyShared"}
STRATEGIES = ("status_quo", "train_best", "most_frequent", "threshold",
              "cost_tree", "penalty_tree", "uncertainty_tree")


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unique(records, key):
    result = {}
    for row in records:
        k = key(row)
        require(k not in result, f"Duplicate key {k}")
        result[k] = row
    return result


def same_cost(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def summarize(base, chosen):
    diffs = [a - b for a, b in zip(base, chosen)]
    return {
        "mean_reference_ms": mean(base), "mean_comparator_ms": mean(chosen),
        "mean_saving_ms": mean(diffs), "median_saving_ms": median(diffs),
        "saving_pct_of_reference_mean": 100 * mean(diffs) / mean(base),
        "faster_equal_slower": [sum(x > 0 for x in diffs), sum(x == 0 for x in diffs),
                                sum(x < 0 for x in diffs)],
    }


def omission_range(base, chosen, groups):
    percentages = []
    for omitted in sorted(set(groups)):
        keep = [i for i, group in enumerate(groups) if group != omitted]
        percentages.append(summarize([base[i] for i in keep],
                                     [chosen[i] for i in keep])["saving_pct_of_reference_mean"])
    return [min(percentages), max(percentages)]


def degree_grid(exact, policy):
    original = unique(read(exact / "sensitivity_controls/native/degree_query_medians.csv"),
                      lambda r: r["query_id"])
    jointly_timed = unique(read(exact / "ordered_access/analysis/degree_query_medians.csv"),
                           lambda r: r["query_id"])
    require(original.keys() == jointly_timed.keys() and len(original) == 1440, "Grid coverage")
    shared_attributes = ("family", "seed", "n", "d", "m", "h", "k", "anchor", "eligible")
    for q, row in original.items():
        require(all(row[k] == jointly_timed[q][k] for k in shared_attributes), (q, "input mismatch"))

    data = policy / "analysis/retained/data"
    historical = [r for r in read(data / "costs.csv") if r["policy"] in PLANS]
    stages = {r["stage"] for r in historical}
    require(len(stages) == 1, "Historical exact stage ambiguous")
    stage, = stages
    for row in historical:
        q = original[row["query_id"]]
        config = f"n{q['n']}_deg{q['degree']}"
        require(row["config"] == config and row["seed"] == q["seed"], "Historical grid identity")
        for k in ("n", "d", "degree", "h", "k"):
            require(float(row["f_" + k]) == float(q[k]), (q["query_id"], k))
        require(same_cost(row["cost_ms"], q[PLANS[row["policy"]]]), "Historical cost mismatch")
    cells = unique(historical, lambda r: (r["query_id"], r["policy"]))
    require(len(cells) == len(original) * len(PLANS), "Incomplete historical matrix")

    predictions = unique((r for r in read(data / "predictions.csv") if r["stage"] == stage),
                         lambda r: (r["query_id"], r["strategy"]))
    require(len(predictions) == len(original) * len(STRATEGIES), "Incomplete strategy matrix")
    folds = {r["held_out_config"]: r for r in read(data / "folds.csv") if r["stage"] == stage}
    require(len(folds) == 16, "Configuration fold count")
    for cfg, row in folds.items():
        require(cfg not in row["train_configs"].split("|") and int(row["test_queries"]) == 90,
                "Fold definition")
    qs = sorted(original, key=int)
    configs = [cells[q, "Forward"]["config"] for q in qs]
    selections = {}
    for strategy in STRATEGIES:
        choices = []
        for q in qs:
            p = predictions[q, strategy]
            plan = p["chosen_policy"]
            require(p["config"] == cells[q, plan]["config"] and p["config"] in folds and
                    same_cost(p["chosen_ms"], original[q][PLANS[plan]]), "Frozen choice mismatch")
            if strategy == "threshold":
                row = original[q]
                lazy = (int(row["n"]) >= 100000 and int(row["degree"]) >= 8
                        and int(row["h"]) >= 6 and int(row["k"]) == 1)
                require(plan == ("Lazy" if lazy else "Forward"), "Retrospective rule mismatch")
            choices.append(PLANS[plan])
        selections[strategy] = choices

    output = {"requests": len(qs), "historical_cost_cells_checked": len(cells),
              "configurations": len(folds), "batches": {}}
    for name, matrix, has_heap in (("historical", original, False), ("joint", jointly_timed, True)):
        def costs(choices):
            return [float(matrix[q][p]) for q, p in zip(qs, choices)]
        baseline = costs(["ForwardTopK"] * len(qs))
        menu = list(PLANS.values())
        oracle = [min(float(matrix[q][p]) for p in menu) for q in qs]
        result = {"oracle_three": summarize(baseline, oracle), "frozen_choices": {}}
        if has_heap:
            four = [min(t, float(matrix[q]["LazyHeap"])) for q, t in zip(qs, oracle)]
            result["oracle_four"] = summarize(baseline, four)
            result["incremental_oracle_saving_ms"] = mean(x - y for x, y in zip(oracle, four))
        for strategy, choices in selections.items():
            replay = {"counts": dict(Counter(choices)), "original_mapping": summarize(baseline, costs(choices))}
            if has_heap:
                substituted = ["LazyHeap" if p == "LazyShared" else p for p in choices]
                heap_costs = costs(substituted)
                replay["heap_mapping"] = summarize(baseline, heap_costs)
                replay["heap_mapping"]["omit_one_configuration_saving_pct_range"] = omission_range(
                    baseline, heap_costs, configs)
            result["frozen_choices"][strategy] = replay
        output["batches"][name] = result
    return output


def prospective(policy):
    data = read(policy / "data/prospective/results/costs.csv")
    matrix = defaultdict(dict)
    for row in data:
        key = (row["config"], row["seed"], row["query_id"])
        require(row["arm"] not in matrix[key] and int(row["repeats"]) == 8, "Unexpected timing cell")
        matrix[key][row["arm"]] = float(row["cost_ms"])
    require(len(matrix) == 3200 and len(data) == 25600, "Prospective coverage")
    require(all(set(row) == set(next(iter(matrix.values()))) for row in matrix.values()), "Missing arm")
    base = [r["Eager"] for r in matrix.values()]
    oracle = [min(r["Eager"], r["RACE"]) for r in matrix.values()]
    return {"requests": len(base), "fixed_pair_oracle": summarize(base, oracle)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, default=ROOT / "exact-plans")
    parser.add_argument("--policy", type=Path, default=ROOT / "policy-selection")
    parser.add_argument("--check", action="store_true", help="Compare all summaries with the stored results")
    args = parser.parse_args()
    output = {"degree": degree_grid(args.exact, args.policy), "prospective": prospective(args.policy)}
    if args.check:
        expected = json.loads(Path(__file__).with_name("expected.json").read_text(encoding="utf-8"))
        require(output == expected, "Published joint summaries differ")
        print("PASS: 1,440 matched degree-sweep requests and 3,200 separate prospective requests")
    else:
        print(json.dumps(output, indent=2))
