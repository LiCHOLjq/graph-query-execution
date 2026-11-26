"""FB15k-237 text-derived retrieval workload; no random vectors are generated."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import hashlib,json,time,struct,platform,sys
from pathlib import Path
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

p=Path(__file__).resolve().parent
t=time.perf_counter()
raw=[s.split('\t') for s in (p/'train.tsv').read_text(encoding='utf8').splitlines() if s.strip()]
entities=sorted({x for s,r,o in raw for x in [s,o]}); eid={x:i for i,x in enumerate(entities)}
relations=sorted({r for s,r,o in raw}); rid={x:i for i,x in enumerate(relations)}
triples=np.array(sorted(set((eid[s],eid[o],rid[r]) for s,r,o in raw)),dtype='<u4')
inv=triples[:,[1,0,2]].copy();inv[:,2]+=len(relations)
triples=np.concatenate((triples,inv));n=len(entities)
texts=dict(s.split('\t',1) for s in (p/'entity2textlong.txt').read_text(encoding='utf8').splitlines() if '\t' in s)
names=dict(s.split('\t',1) for s in (p/'entity2text.txt').read_text(encoding='utf8').splitlines() if '\t' in s)
docs=[texts.get(e,names.get(e,'' )).replace('\\n',' ') for e in entities]
graph_seconds=time.perf_counter()-t
t=time.perf_counter()
tfidf=TfidfVectorizer(min_df=2,max_features=50000,sublinear_tf=True,dtype=np.float64)
with threadpool_limits(limits=1): X=tfidf.fit_transform(docs)
tfidf_seconds=time.perf_counter()-t
t=time.perf_counter()
svd=TruncatedSVD(n_components=128,n_iter=7,random_state=20261023)
with threadpool_limits(limits=1): vectors=normalize(svd.fit_transform(X)).astype('<f8')
svd_seconds=time.perf_counter()-t
out=[[] for _ in entities]
for a,b,r in triples:out[int(a)].append((int(b),int(r)))
rng=np.random.default_rng(20261023)
queries=[]; qvec=[]
for h in [1,2,3,4]:
 for j in range(20):
  a=int(rng.integers(n));v=a;word=[]
  for i in range(h):
   v,r=out[v][int(rng.integers(len(out[v])))];word.append(r)
  mode='walk_endpoint_text' if j%2==0 else 'independent_entity_text'
  source=v if j%2==0 else int(rng.integers(n))
  queries.append(dict(qid=len(queries),anchor=a,h=h,relations=word,k=10,query_entity=source,mode=mode,anchor_degree=len(out[a])))
  qvec.append(vectors[source])
qvec=np.array(qvec,dtype='<f8')
np.savez_compressed(p/'fb15k237_text.npz',triples=triples,vectors=vectors,query_vectors=qvec)
with (p/'graph.bin').open('wb') as f:
 f.write(struct.pack('<III',n,128,len(triples)));f.write(triples.tobytes());f.write(vectors.tobytes())
with (p/'queries.txt').open('w',encoding='ascii') as f:
 for q,vec in zip(queries,qvec):f.write(' '.join(map(str,[q['anchor'],q['h'],*q['relations'],q['k'],*vec.tolist()]))+'\n')
(p/'queries.json').write_text(json.dumps(queries,indent=2))
(p/'entities.json').write_text(json.dumps(entities))
(p/'relations.json').write_text(json.dumps(relations+['inverse:'+r for r in relations],indent=2))
files=['train.tsv','entity2text.txt','entity2textlong.txt','graph.bin','queries.txt','fb15k237_text.npz','prepare_real.py']
report=dict(dataset='FB15k-237 training graph; typed inverse edges added explicitly',source_repository='https://github.com/yao8839836/kg-bert',download_base='https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k-237/',vertices=n,original_edges=len(triples)//2,edges_with_inverses=len(triples),original_relations=len(relations),relations_with_inverses=2*len(relations),description_coverage=sum(e in texts for e in entities),name_only_fallback=sum(e not in texts and e in names for e in entities),empty_texts=sum(not x for x in docs),nonzero_vectors=int((np.linalg.norm(vectors,axis=1)>0).sum()),dimension=128,embedding='TF-IDF followed by 128-component truncated SVD and L2 normalization (latent semantic analysis), fitted on actual entity descriptions; not KG-BERT, pretrained neural embeddings, or synthetic vectors',tfidf_features=X.shape[1],tfidf_nnz=X.nnz,svd_explained_variance=float(svd.explained_variance_ratio_.sum()),graph_preprocessing_seconds=graph_seconds,tfidf_fit_seconds=tfidf_seconds,svd_fit_seconds=svd_seconds,query_seed=20261023,queries=len(queries),queries_per_h=20,query_modes=['walk_endpoint_text','independent_entity_text'],split_use='training triples only; dev/test triples are not used; retrieval study does not claim link-prediction generalization',python=sys.version,platform=platform.platform(),numpy=np.__version__,sklearn=sklearn.__version__,hashes={fn:hashlib.sha256((p/fn).read_bytes()).hexdigest() for fn in files})
(p/'provenance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
