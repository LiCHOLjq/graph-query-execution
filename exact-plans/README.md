# Execution Boundaries of State Reuse in Graph-Constrained Vector Search

Research code and measurements accompanying the manuscript by Jiaquan Li, Ji Li, Yijie Zhang, Guanning Li, Junxiao Li, and Chuanwen Li.

The experiments compare when to evaluate a graph reachability predicate during top-*k* vector search. The central comparison times four exact plans on the same configurations. It retains clocks, work counters, answer digests, and per-query attributes for 39,840 executions. Timings are in milliseconds. For each query, compare the medians of three repetitions per plan. The workload grids overlap.

## Check the reported results

From the repository root, run the following with Python 3.11 or later. The statistics check uses only the standard library and the recorded data.

```sh
python verify_paper_statistics.py
python -m unittest discover -s artifact -p test_engine.py -v
```

The first command recomputes the paper's four-plan statistics. The real-graph cohort has 80 queries. Forward materialization has the lowest median on 79, recursive reverse materialization on one. Among 990 nonempty main-grid configurations below 99.98% predicate selectivity, heap-ordered verification has 87 strict wins. Of these, 86 have selectivity at least 90%.

## Files

| Directory | Code and data |
|---|---|
| `artifact/` | Python mechanism experiment, synthetic workload generator, oracle tests, and 8,640 measured executions with individual answers. |
| `native_comparisons/` | C++17 exact plans, raw timings, held-out cost analysis, real-query preparation, and HNSW integration. |
| `sensitivity_controls/` | Degree, vector dimension, query vector, allocation and filtered HNSW controls. |
| `matched_controls/` | Fixed-input small-*k* comparisons, root traces, and callback statistics. |
| `ordered_access/` | Four-plan benchmark and source, original protocol, five query cohorts, raw repetitions and bootstrap summaries. |

In `ordered_access/results/`, each `*_queries.csv` describes a configuration and each `*_raw.csv` contains one measured plan execution per row. Join on cohort and `query_id`. The `ms` field gives elapsed time and `exact` records agreement with the independent oracle. The `answer_fnv64` fingerprint identifies the ordered returned IDs. The query medians, selectivity strata, and matched-*k* results are in `ordered_access/analysis/`. [Four-plan results](ordered_access/RESULTS_EN.md)

## Regenerate statistics and figures

Run these commands in a disposable copy. The analyses regenerate their derived files. The first audit checks the Python mechanism measurements against their stored query answers and timing-source hashes. The final analysis recomputes the four-plan outcomes, compares full-sort and heap root sequences and counters, and checks counter alignment against the earlier studies.

```sh
python -m pip install -r ordered_access/requirements.txt
python artifact/audit_results.py artifact/results/main
python matched_controls/callbacks/analyze_callbacks.py
python ordered_access/analyze.py --history-root .
```

The full analyzer records which original binary and corpus-derived inputs are absent from a public checkout. It checks the files it distributes against the frozen protocol. The module checksum file identifies the distributed CSVs and accompanying files. Historical cohort labels have been normalized in the matched controls; recorded measurements retain their original values. Use `sha256sum -c MEASUREMENT_FILES.sha256` from the repository root to verify the public checkout, including the statistical verifier and its expected results, before regenerating outputs. The root provenance map records changes to descriptive labels and file paths in source and measurements.

## Run the native implementation

Use Linux or WSL with a C++17 compiler. The following commands run correctness tests and one synthetic cohort in a fresh output directory. They use the local machine and produce new timings.

```sh
mkdir fresh_run
g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra ordered_access/source/benchmark.cpp -o fresh_run/benchmark
fresh_run/benchmark test
fresh_run/benchmark matched fresh_run/results
```

The native tests cover 20,736 comparisons with independent exact answers and 1,296 checks of the complete heap ordering. Frozen timings used in the manuscript are under `ordered_access/results/`. The historical four-plan protocol used g++ 13.3.0, a warmup and three repetitions in randomized plan order, and thread affinity to virtual CPU 4 on an Intel Core i9-12900K under WSL2. Absolute latencies and winner counts depend on the host.

## Rebuild the real-query inputs

The original FB15k-237 triples and entity descriptions are provided by their upstream sources. This repository includes the downloader, preparation code, query definitions, and measured search outcomes. The downloader pins the exact data revision and validates three SHA-256 digests. Obtain and use those files under their upstream terms. [Original FB15k-237 dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52312) · [Pinned KG-BERT triples and text](https://github.com/yao8839836/kg-bert/tree/f3f260309c5e2986467a66e0e68fa7f53b1a150e/data/FB15k-237)

In a disposable working copy, run the following to prepare the text-derived vectors, graph, and query inputs. The final, optional command rebuilds the index for the HNSW controls and compares its hash with the measured index. [hnswlib v0.8.0](https://github.com/nmslib/hnswlib/tree/v0.8.0)

```sh
python -m pip install -r native_comparisons/real/requirements.txt
python native_comparisons/real/download_data.py
python native_comparisons/real/prepare_real.py
python native_comparisons/real/build_index.py
```

To run the real graph with the compiled native executable, use `fresh_run/benchmark real fresh_run/results native_comparisons/real`. The prepared graph has 14,505 vertices and 544,230 typed edges including inverses. To regenerate filtered ANN statistics after input preparation, run `python sensitivity_controls/real/analyze_filtered.py`, then `python sensitivity_controls/real/audit_filtered.py`, then `python matched_controls/callbacks/analyze_callbacks.py`. The historical audits check recorded file hashes. Numeric libraries, platform serialization and line endings can affect regenerated file hashes. Raw data downloads and new benchmark runs are optional for analyzing the supplied measurement CSVs.
