"""Fetch a pinned public data revision and verify byte-level provenance."""
from pathlib import Path
import hashlib,urllib.request

p=Path(__file__).resolve().parent
commit='f3f260309c5e2986467a66e0e68fa7f53b1a150e'
expected={
 'train.tsv':'6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae',
 'entity2text.txt':'16beedcfad3d32696e8b0b98cfa0f5eeb144f5142225f227a64fd9ce37447da5',
 'entity2textlong.txt':'0b23b469414ea1f9c84a8d29c48faf9e9d733360a9edc4e4de9febcb3f131842'}
for name,digest in expected.items():
 url=f'https://raw.githubusercontent.com/yao8839836/kg-bert/{commit}/data/FB15k-237/{name}'
 if not (p/name).exists():urllib.request.urlretrieve(url,p/name)
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
 print(name,'verified',digest)
