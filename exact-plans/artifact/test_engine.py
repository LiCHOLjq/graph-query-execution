import copy
import pickle
import random
import unittest
from engine import (Stats, SuffixEngine, Materializer, Scorer, BestFirst,
                    forward_membership, exact_oracle, execute, METHODS, squared_l2)
from workloads import make_case, make_queries


def small_graph(seed, n=9):
    rng=random.Random(seed)
    reverse=[[[] for _ in range(n)] for _ in range(2)]
    forward=[[[] for _ in range(n)] for _ in range(2)]
    for r in range(2):
        for u in range(n):
            for v in range(n):
                if rng.random()<0.23:
                    reverse[r][v].append(u); forward[r][u].append(v)
    return forward,reverse


class Correctness(unittest.TestCase):
    def test_every_graph_boundary_and_serialized_resume(self):
        for seed in range(5):
            f,r=small_graph(seed)
            relations=[0,1,0,1]
            truth=forward_membership(f,0,relations)
            for v in range(1,9):
                engine=SuffixEngine(r,0,relations,('g','v','q',0,tuple(relations)),Stats())
                task=engine.task(v)
                while True:
                    restored=SuffixEngine.restore(engine.checkpoint(),engine.identity)
                    restored_task=copy.deepcopy(task)
                    self.assertEqual(restored.finish(restored_task),v in truth)
                    if not task.stack: break
                    self.assertLessEqual(engine.advance(task,1),1)
                self.assertEqual(engine.value(v),v in truth)

    def test_interleaving_canonical_cursors_and_intervals(self):
        for seed in range(12):
            f,r=small_graph(seed)
            for length in range(1,5):
                relations=([0,1]*2)[:length]
                truth=forward_membership(f,0,relations)
                e=SuffixEngine(r,0,relations,('id',),Stats())
                m=Materializer(e); tasks=[e.task(v) for v in range(1,9)]
                rng=random.Random(seed)
                while not m.done:
                    rng.shuffle(tasks)
                    for task in tasks: e.advance(task,1)
                    m.advance(1)
                    lo,hi=e.cardinality_interval()
                    self.assertLessEqual(lo,len(truth)); self.assertGreaterEqual(hi,len(truth))
                self.assertEqual(m.eligible,truth)
                cold=SuffixEngine(r,0,relations,('id',),Stats())
                cm=Materializer(cold); cm.advance(10**9)
                self.assertEqual(e.stats.graph_edges,cold.stats.graph_edges)
                self.assertEqual(e.stats.graph_entries,cold.stats.graph_entries)
                for (v,d),node in e.nodes.items():
                    if d>0:
                        self.assertLessEqual(node.cursor,len(r[relations[d-1]][v]))

    def test_ann_every_primitive_checkpoint_and_boundary(self):
        c=make_case(48,'distributed',True,3,8)
        _,_,q=make_queries(c,1)[0]
        identity=('g','v','q')
        oracle=BestFirst(c['index'],Scorer(c['vectors'],q,Stats()),identity)
        oracle.advance_expansions(15)
        a=BestFirst(c['index'],Scorer(c['vectors'],q,Stats()),identity)
        while a.expanded < 15 or a.active is not None:
            restored=BestFirst.restore(a.checkpoint(),identity)
            restored.advance_expansions(15)
            self.assertEqual(restored.candidates(48),oracle.candidates(48))
            self.assertEqual(restored.scorer.stats.distances,oracle.scorer.stats.distances)
            if a.active is not None and a.cursor==len(a.index[a.active]):
                a.active=None; a.cursor=0
                if a.expanded==15: break
            a.step()
        staged=BestFirst(c['index'],Scorer(c['vectors'],q,Stats()),identity)
        for target in (1,3,7,15): staged.advance_expansions(target)
        self.assertEqual(staged.candidates(48),oracle.candidates(48))
        self.assertEqual(staged.scorer.stats.ann_edges,oracle.scorer.stats.ann_edges)
        self.assertEqual(staged.scorer.stats.distances,len(staged.visited))

    def test_snapshot_identity_guards(self):
        c=make_case(24,'distributed',True,1,4)
        identity=('g','v','index','q',0,(0,1))
        e=SuffixEngine(c['reverse'],0,[0,1],identity,Stats())
        a=BestFirst(c['index'],Scorer(c['vectors'],c['vectors'][0],Stats()),identity)
        for idx in range(len(identity)):
            wrong=list(identity); wrong[idx]='changed'
            with self.assertRaises(ValueError): SuffixEngine.restore(e.checkpoint(),wrong)
            with self.assertRaises(ValueError): BestFirst.restore(a.checkpoint(),wrong)

    def test_all_plans_and_work_relationships(self):
        for family in ('shared_hubs','distributed'):
            for seed in range(4):
                c=make_case(96,family,bool(seed%2),seed,8)
                for _,anchor,q in make_queries(c,3):
                    truth,_=exact_oracle(c['forward'],anchor,c['relations'],c['vectors'],q,10)
                    runs={m:execute(c,q,anchor,m,stages=(2,5,12),graph_quantum=11) for m in METHODS}
                    for result in runs.values(): self.assertEqual(result['ids'],truth)
                    self.assertEqual(runs['CoExec']['graph_work'],runs['ColdSuffix']['graph_work'])
                    self.assertEqual(runs['GraphShare']['graph_work'],runs['ColdSuffix']['graph_work'])
                    self.assertLessEqual(runs['CoExec']['distances'],c['n'])
                    self.assertLessEqual(runs['CoExec']['graph_work'],runs['Restart']['graph_work'])
                    self.assertEqual(runs['CoExec']['graph_work'],runs['GraphShare']['graph_work'])
                    self.assertEqual(runs['CoExec']['distances'],runs['VectorResume']['distances'])

    def test_empty_underfull_ties_and_cycles(self):
        c=dict(n=4,reverse=[[[],[],[],[]]],forward=[[[],[],[],[]]],vectors=[(0.,)]*4,
               index=[[1],[2],[3],[0]],relations=[0],graph_digest='g',vector_digest='v',index_digest='i')
        for edges,expected in [([],[]),([(0,1)],[1]),([(0,3),(0,2),(0,1),(0,0)],[1,2,3])]:
            c['reverse']=[[[],[],[],[]]]; c['forward']=[[[],[],[],[]]]
            for u,v in edges: c['forward'][0][u].append(v); c['reverse'][0][v].append(u)
            for method in METHODS:
                self.assertEqual(execute(c,(0.,),0,method,stages=(1,2,3),graph_quantum=1)['ids'],expected)
        c['relations']=[0,0]
        for method in METHODS:
            self.assertEqual(execute(c,(0.,),0,method,stages=(1,2,3),graph_quantum=1)['ids'],[1,2,3])

    def test_budget_zero_and_negative(self):
        _,r=small_graph(1)
        e=SuffixEngine(r,0,[0,1],('id',),Stats()); task=e.task(3)
        self.assertEqual(e.advance(task,0),0)
        self.assertEqual(e.stats.graph_entries,0)
        with self.assertRaises(ValueError): e.advance(task,-1)


if __name__=='__main__': unittest.main(verbosity=2)
