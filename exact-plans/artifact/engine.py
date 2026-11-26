"""Independent, standard-library reference implementation of staged graph search.

Graph states are bound to one immutable snapshot/query/predicate identity.
No third-party ANN implementation or borrowed result is used.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import heapq
import math
import pickle
import time


def squared_l2(a, b):
    return math.fsum((x-y)*(x-y) for x, y in zip(a, b))


@dataclass
class Stats:
    graph_entries: int = 0
    graph_edges: int = 0
    distances: int = 0
    score_ns: int = 0
    ann_edges: int = 0
    ann_expansions: int = 0
    cache_hits: int = 0


@dataclass
class Node:
    initialized: bool = False
    value: bool | None = None
    cursor: int = 0
    pending: tuple | None = None


@dataclass
class GraphTask:
    root: tuple
    stack: list = field(default_factory=list)

    def __post_init__(self):
        if not self.stack:
            self.stack = [self.root]


class SuffixEngine:
    """Canonical cursors shared across cooperatively scheduled task stacks."""
    def __init__(self, reverse, anchor, relations, identity, stats):
        self.reverse, self.anchor, self.relations = reverse, anchor, tuple(relations)
        self.identity, self.stats, self.nodes = tuple(identity), stats, {}

    def task(self, v):
        return GraphTask((v, len(self.relations)))

    def _normalize(self, task):
        """Zero primitive bookkeeping, ending immediately before a primitive."""
        while task.stack:
            key = task.stack[-1]
            node = self.nodes.setdefault(key, Node())
            v, depth = key
            if not node.initialized:
                return 'entry'
            if node.value is not None:
                task.stack.pop()
                continue
            if node.pending is not None:
                child = self.nodes.get(node.pending)
                if child is not None and child.value is not None:
                    if child.value:
                        node.value = True
                    node.pending = None
                    continue
                task.stack.append(node.pending)
                continue
            if node.cursor == len(self.reverse[self.relations[depth-1]][v]):
                node.value = False
                continue
            return 'edge'
        return None

    def advance(self, task, budget):
        if budget < 0:
            raise ValueError('negative graph budget')
        used = 0
        while True:
            action = self._normalize(task)
            if action is None or used == budget:
                return used
            key = task.stack[-1]
            v, depth = key
            node = self.nodes[key]
            if action == 'entry':
                node.initialized = True
                self.stats.graph_entries += 1
                if depth == 0:
                    node.value = (v == self.anchor)
            else:
                predecessor = self.reverse[self.relations[depth-1]][v][node.cursor]
                node.cursor += 1
                node.pending = (predecessor, depth-1)
                self.stats.graph_edges += 1
            used += 1

    def finish(self, task):
        self.advance(task, 1 << 60)
        return self.nodes[task.root].value

    def value(self, v):
        node = self.nodes.get((v, len(self.relations)))
        return None if node is None else node.value

    def cardinality_interval(self):
        vals = [self.value(v) for v in range(len(self.reverse[0])) if v != self.anchor]
        return sum(x is True for x in vals), sum(x is not False for x in vals)

    def checkpoint(self):
        return pickle.dumps(self, protocol=5)

    @classmethod
    def restore(cls, payload, identity):
        instance = pickle.loads(payload)
        if instance.identity != tuple(identity):
            raise ValueError('snapshot/query/predicate identity mismatch')
        return instance


class Materializer:
    def __init__(self, engine):
        self.engine, self.next_root, self.current = engine, 0, None
        self.eligible = set()

    @property
    def done(self):
        return self.next_root == len(self.engine.reverse[0]) and self.current is None

    def advance(self, budget):
        used = 0
        while not self.done:
            if self.current is None:
                if self.next_root == self.engine.anchor:
                    self.next_root += 1
                    continue
                self.current = self.engine.task(self.next_root)
                self.next_root += 1
            spent = self.engine.advance(self.current, budget-used)
            used += spent
            if self.current.stack:
                break
            if self.engine.nodes[self.current.root].value:
                self.eligible.add(self.current.root[0])
            self.current = None
            if used == budget:
                break
        return used


class Scorer:
    def __init__(self, vectors, query, stats):
        self.vectors, self.query, self.stats, self.cache = vectors, query, stats, {}

    def score(self, v):
        if v in self.cache:
            self.stats.cache_hits += 1
            return self.cache[v]
        tick = time.perf_counter_ns()
        distance = squared_l2(self.vectors[v], self.query)
        self.stats.score_ns += time.perf_counter_ns() - tick
        self.stats.distances += 1
        self.cache[v] = distance
        return distance


class BestFirst:
    """Unbounded retained queue, insertion-time visited marks, edge cursor.

    This is an experimental best-first proximity-graph traversal, NOT HNSW.
    Each step processes one adjacency edge or one heap pop.
    """
    def __init__(self, index, scorer, identity, entry=0):
        self.index, self.scorer, self.identity = index, scorer, tuple(identity)
        self.visited, self.heap = {entry}, [(scorer.score(entry), entry)]
        self.active, self.cursor, self.expanded = None, 0, 0

    def step(self):
        if self.active is None:
            if not self.heap:
                return False
            _, self.active = heapq.heappop(self.heap)
            self.cursor = 0
            self.expanded += 1
            self.scorer.stats.ann_expansions += 1
            return True
        neighbors = self.index[self.active]
        if self.cursor < len(neighbors):
            v = neighbors[self.cursor]
            self.cursor += 1
            self.scorer.stats.ann_edges += 1
            if v not in self.visited:
                self.visited.add(v)
                heapq.heappush(self.heap, (self.scorer.score(v), v))
            return True
        self.active = None
        self.cursor = 0
        return self.step()

    def advance_expansions(self, target):
        while self.expanded < target or self.active is not None:
            # Finish the active adjacency list before respecting an expansion boundary.
            if self.active is not None and self.cursor == len(self.index[self.active]):
                self.active = None
                self.cursor = 0
                if self.expanded >= target:
                    break
            if not self.step():
                break

    def candidates(self, limit):
        return [v for _, v in sorted((self.scorer.cache[v], v) for v in self.visited)[:limit]]

    def checkpoint(self):
        return pickle.dumps(self, protocol=5)

    @classmethod
    def restore(cls, payload, identity):
        instance = pickle.loads(payload)
        if instance.identity != tuple(identity):
            raise ValueError('snapshot/query/predicate identity mismatch')
        return instance


def forward_membership(forward, anchor, relations, stats=None):
    frontier = {anchor}
    for relation in relations:
        next_frontier = set()
        for u in sorted(frontier):
            for v in forward[relation][u]:
                if stats is not None:
                    stats.graph_edges += 1
                next_frontier.add(v)
        frontier = next_frontier
    return frontier - {anchor}


def exact_oracle(forward, anchor, relations, vectors, query, k):
    eligible = forward_membership(forward, anchor, relations)
    ranked = sorted((squared_l2(vectors[v], query), v) for v in eligible)
    return [v for _, v in ranked[:k]], eligible


def identity_for(graph_digest, vector_digest, query, anchor, relations, index_digest):
    return (graph_digest, vector_digest, index_digest, hashlib.sha256(pickle.dumps(query)).hexdigest(), anchor, tuple(relations))


METHODS = ('Restart', 'GraphShare', 'VectorResume', 'CoExec', 'ColdSuffix', 'ForwardGF')


def execute(case, query, anchor, method, k=10, stages=(8,24,64), graph_quantum=256):
    """Outer timer includes setup, all phase work, exact repair and state cleanup.

    Input generation, index build and independent correctness oracle are excluded.
    State entry/edge counters are logical primitives, NOT cost-normalized time.
    """
    if method not in METHODS:
        raise ValueError('unknown method')
    started = time.perf_counter_ns()
    stats = Stats()
    identity = identity_for(case['graph_digest'], case['vector_digest'], query, anchor, case['relations'], case['index_digest'])
    graph_share = method in ('GraphShare', 'CoExec')
    vector_share = method in ('VectorResume', 'CoExec')
    materializer = verifier = lazy = ann = score = repair_scorer = None
    states_peak = score_entries_peak = 0
    prefix_ids = []
    if method == 'ForwardGF':
        eligible = forward_membership(case['forward'], anchor, case['relations'], stats)
        repair_scorer = Scorer(case['vectors'], query, stats)
    else:
        eager = SuffixEngine(case['reverse'], anchor, case['relations'], identity, stats)
        materializer = Materializer(eager)
        if method != 'ColdSuffix':
            lazy = eager if graph_share else SuffixEngine(case['reverse'], anchor, case['relations'], identity, stats)
            for target in stages:
                if ann is None or not vector_share:
                    score = Scorer(case['vectors'], query, stats)
                    ann = BestFirst(case['index'], score, identity)
                ann.advance_expansions(target)
                prefix_ids = []
                for v in ann.candidates(target):
                    if v != anchor and lazy.finish(lazy.task(v)):
                        prefix_ids.append(v)
                materializer.advance(graph_quantum)
                states_peak = max(states_peak, len(eager.nodes) + (0 if graph_share else len(lazy.nodes)))
                score_entries_peak = max(score_entries_peak, len(score.cache))
        materializer.advance(1 << 60)
        eligible = materializer.eligible
        repair_scorer = score if vector_share else Scorer(case['vectors'], query, stats)
        states_peak = max(states_peak, len(eager.nodes) + (len(lazy.nodes) if lazy is not None and not graph_share else 0))
    result = [v for _, v in heapq.nsmallest(k, ((repair_scorer.score(v), v) for v in sorted(eligible)))]
    live_scores = len(repair_scorer.cache)
    if score is not None and score is not repair_scorer:
        live_scores += len(score.cache)
    score_entries_peak = max(score_entries_peak, live_scores)
    n_eligible = len(eligible)
    # Explicitly destroy query-private state before the endpoint timestamp.
    if materializer is not None:
        eager.nodes.clear()
    if lazy is not None:
        lazy.nodes.clear()
    if ann is not None:
        ann.heap.clear(); ann.visited.clear(); ann.scorer.cache.clear()
    repair_scorer.cache.clear()
    eligible.clear()
    materializer = verifier = lazy = ann = score = repair_scorer = None
    elapsed_ns = time.perf_counter_ns() - started
    return dict(ids=result, prefix_ids=prefix_ids[:k], eligible=n_eligible, latency_ms=elapsed_ns/1e6,
                graph_entries=stats.graph_entries, graph_edges=stats.graph_edges,
                graph_work=stats.graph_entries+stats.graph_edges, distances=stats.distances,
                ann_edges=stats.ann_edges, ann_expansions=stats.ann_expansions,
                cache_hits=stats.cache_hits, score_ms=stats.score_ns/1e6,
                suffix_entries_peak=states_peak, score_entries_peak=score_entries_peak)
