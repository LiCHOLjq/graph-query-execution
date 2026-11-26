# Joint replay of routing choices and execution plans

Run `python comparison/replay_routes.py --check` at the repository root to validate the published summaries using the Python standard library. Omit `--check` to print the full result as JSON. The script reads the supplied per-request medians and historical policy decisions. It performs no new timing experiments.

The exact-retrieval routing grid and the degree-sweep benchmark contain the same 1,440 requests across 16 configurations. The script checks their request attributes, all 4,320 historical fixed-plan costs, the held-out configuration assignments, and the cost recorded for every frozen choice. It replays these choices on a separate joint timing matrix for the same requests. Positive percentages are reductions in total execution time relative to always running forward in the corresponding batch. The two alternative lazy backends are the full distance sort and incremental heap. The expanded oracle may use either. Break-even budgets are the resulting mean execution savings before routing costs.

The second output evaluates Eager against the within-request minimum of Eager and RACE on a separate prospective sample of 3,200 queries. The serving portfolios have different contracts and measurements. All reported percentages use a within-cohort fixed baseline.
