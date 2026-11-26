# Measured exact heap-ordering results

All numbers below come from the new jointly randomized four-plan batch. F, R, S and H denote ForwardTopK, ReverseRecursiveTopK, full-sort LazyShared, and LazyHeap. Each plan/configuration has three timings; comparisons use their medians. A strict winner must be faster than every alternative, and ties are kept separately.

| Cohort | Configurations | F | R | S | H | Tie |
|---|---:|---:|---:|---:|---:|---:|
| Main | 1,500 | 1,076 | 36 | 4 | 383 | 1 |
| Degree | 1,440 | 1,095 | 7 | 7 | 331 | 0 |
| Matched k | 60 | 35 | 0 | 0 | 25 | 0 |
| Dimension | 240 | 180 | 0 | 2 | 58 | 0 |
| Real all queries | 80 | 79 | 1 | 0 | 0 | 0 |
| Real no adjacent inverse | 49 | 48 | 1 | 0 | 0 | 0 |

These cohorts overlap or reuse predicates and must not be pooled as independent samples. See the generated `winner_strata.csv` and two-column `heap_winners_table.tex` for all selectivity strata.

## Revised execution boundary

Heap ordering changes the observed boundary materially. Among the main grid's 990 nonempty configurations below 99.98% selectivity, H wins 87 and F wins 903. Removing H from this same new batch gives F/R/S = 953/0/37: 50 heap wins replace forward wins and 37 replace full-sort wins. This same-batch comparison separates the added plan from changes between historical and current execution environments. The main-grid sub-saturation heap wins include k=1/10/100 counts 33/30/24, with 84 fanout and three uniform configurations. Their minimum observed selectivity is 88.7888%; it is not an estimated decision threshold.

The degree sweep has 1,083 nonempty sub-saturation configurations: H wins 80 and F wins 1,003. The corresponding three-plan counts are F/R/S = 1,039/0/44, so 36 heap wins replace forward wins and 44 replace full-sort wins. H's sub-saturation wins comprise 73 cases at k=1 and seven at k=10, with none at k=100. The minimum observed selectivity is 36.00336% at n=100,000, degree six, h=6, k=1, seed zero; H's median is 5.618679 ms versus F's 5.951468 ms. This measured point is a counterexample to a universal near-saturation requirement, not a universal low-selectivity boundary.

Across the entire main grid, H replaces 58 forward, 162 reverse and 163 full-sort winners in the same batch. The sole tied minimum is query 318 (n=1,000, fanout, seed one, h=6, k=100), where R and H both have median 0.030121 ms. Across the degree grid H replaces 78 forward, nine reverse and 244 full-sort winners.

## Matched output-count sweep

| k | H wins / 10 | S beats faster materializer / 10 | Median H/materializer [conditional 95% CI] | Median S/materializer [conditional 95% CI] |
|---:|---:|---:|---|---|
| 1 | 10 | 9 | 0.487 [0.357, 0.633] | 0.713 [0.607, 0.856] |
| 2 | 9 | 6 | 0.729 [0.571, 0.955] | 0.950 [0.813, 1.225] |
| 3 | 4 | 1 | 1.077 [0.759, 1.561] | 1.386 [1.053, 1.736] |
| 5 | 2 | 0 | 1.351 [1.129, 1.731] | 1.662 [1.389, 1.861] |
| 10 | 0 | 0 | 1.681 [1.573, 2.081] | 2.061 [1.892, 2.405] |
| 100 | 0 | 0 | 3.803 [3.614, 4.211] | 4.179 [3.852, 4.375] |

Both lazy plans inspect exactly the same prefixes: 1–2, 2–3, 3–4, 5–7, 10–12 and 107–117 roots at these k values. Median H/S ratios are 0.648, 0.766, 0.777, 0.821, 0.862 and 0.949. Heap ordering therefore extends the observed winning cases to k=5 on two seeds in this cell, although the median ratio at k=5 remains above one and neither lazy plan wins at k=10 or 100. The intervals resample the ten seed clusters with 10,000 paired draws, conditional on this previously selected workload cell; they are not general guarantees about output count.

## Dimension controls and real graph

All six (query condition, dimension) strata have 40 configurations. H beats F on 10/40 in every stratum. S beats F on 10/40 except origin/d=768, where it wins 9/40. The four-plan counts are F180/H58/S2; both S wins occur for Gaussian d=768. All 180 nonempty sub-saturation configurations remain forward wins in this dimension cohort; the dense h=6 cases supply the lazy wins. Paired median ratios and IQRs are reported separately from ratios of means in the generated dimension table, since their distributions are highly heterogeneous. Increasing dimensionality does not produce universal lazy dominance.

On the real graph, neither lazy plan wins any of the 80 queries. F wins 79 and R wins query 49. Its median R time is 0.099112 ms versus F 0.105619 ms, a descriptive 6.16% reduction; the first of three paired repetitions favors F, so this is not a claim of statistically established superiority. The predefined 49-query no-immediate-inverse subset retains this case and has F48/R1. Graph and query files exactly match the historical inputs by SHA-256. All 240 corresponding historical plan cells match graph-work and scoring counters, but their timings are not pooled.

## Validity and limits

All 39,840 timed outputs and 13,280 warmups equal the independent edge-scan oracle. All 9,960 measured heap/full-sort pairs have equal proof work, distance counts, roots and sequence digest; no workload, root or plan was omitted. The heap receives no oracle information. It preserves all-N exact distance scoring and O(N) order storage; its advantage is reduced exact ordering work when only a prefix is consumed.

Measurements use a nonexclusive WSL virtual CPU and have no fixed-frequency guarantee. A possible short host-background affinity event is transparently recorded, without deleting or rerunning observations. There are zero zero/negative durations, but 275 query/plan cells have a repetition max/min ratio above three; the smallest main-grid timing is one approximately 100-ns clock increment. Counts with at least a descriptive 5% advantage over the next-fastest median are main F1071/R24/S2/H349 and degree F1089/R3/S3/H308. These counts and strict winners are descriptive, not significance tests. The new evidence supports an implementation-sensitive execution boundary and conditional benefits of exact lazy verification, not universal superiority of either materialization or laziness.
