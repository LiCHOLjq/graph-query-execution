# Execution Boundaries and Policy Selection in Graph-Constrained Vector Search

This is the code and data companion to the article by Jiaquan Li, Ji Li, Yijie Zhang, Guanning Li, Junxiao Li and Chuanwen Li. It connects two practical decisions. Which complete execution plan should handle a graph-constrained vector query? Given the cost of choosing and dispatching that plan, how much of the retrospective opportunity can a policy actually realize?

## Reproduce the results

From the repository root, run either standard-library check with Python 3.11 or later.

```sh
python exact-plans/verify_paper_statistics.py
python comparison/replay_routes.py --check
```

The first command reconstructs the exact-plan comparisons from 39,840 individual timings. The second validates the shared costs and held-out decisions for 1,440 requests and replays them on the separately timed plan alternatives. It also evaluates prospective policy headroom on 3,200 new requests. Distinct timing sessions retain separate baselines.

For the full offline analysis, use a clean Python environment with the recorded numerical libraries.

```sh
python -m pip install -r policy-selection/requirements.txt
python verify.py
```

The verifier checks distributed file digests, runs the mechanism unit suite, audits measured answers and counters, refits the policy feature controls, and compares 40 recomputed tables with their references. It uses temporary directories for generated results. The statistical checks operate on the distributed measurements. To repeat native execution experiments or build the external corpus, see the module instructions below.

| Directory | Evidence and entry points |
|---|---|
| [Exact plans](exact-plans/README.md) | Reference executor and C++ benchmarks, work counters, timings, exactness audits, filtered approximate search, and instructions for rebuilding the real-graph inputs. |
| [Policy selection](policy-selection/README.md) | Out-of-fold policy choices, measurements of dispatch and inference, timing-session controls, censoring records, and prospective whole-path evaluation. See the [data dictionary](policy-selection/DATA_DICTIONARY.md) for units and query identifiers. |
| [Joint comparison](comparison/README.md) | Request-level reconciliation, cross-batch replay, dispatch break-even calculation, and a separate prospective comparison. |

## Provenance and scope

The execution and policy modules use consistent descriptive stage names. A [normalization ledger](provenance/normalized-files.json) identifies affected files by their distributed paths and records their input and output digests. Operational checksums refer to the distributed representation. Normalization changes identifiers, descriptive comments, paths and dependent audit metadata. It preserves the observed numerical values, record order in the retained inputs, cohort membership, and held-out assignments. The [policy input ledger](policy-selection/provenance/transformations.json) records earlier projections to numerical fields.

For the exact-plan experiments, scripts acquire upstream triples and entity descriptions and check their hashes before rebuilding vectors and indexes. The policy module distributes the numerical evidence needed for offline analysis. Remeasuring its native end-to-end paths requires separately obtained graph data, engines and indexes. Numerical recall and legality audits in this module are retained measurements. Upstream datasets and software remain subject to their respective terms. The authors have not specified a blanket reuse license for this repository.
