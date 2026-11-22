"""Data validation and release manifest helpers, no ML dependencies."""
import hashlib, json, math
from pathlib import Path
from .game import flatten, legal, d4

def digest(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def read(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def write(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)

def checked(root):
    lock=json.loads((root/"lock.json").read_text(encoding="utf-8")); data={};keys={};episodes={}
    for name in ("train","dev"):
        path=root/(name+".jsonl")
        if digest(path)!=lock["splits"][name]["sha256"]:raise ValueError("SHA-256 mismatch: "+str(path))
        rs=read(path); k=set(); s=set()
        if len(rs)!=lock["splits"][name]["rows"]:raise ValueError("Fixture count mismatch")
        for r in rs:
            st=flatten(r['board'])
            if any(type(v)!=int or v<0 or (v and (v<2 or v&(v-1))) for v in st):raise ValueError("Bad board value")
            opts=legal(st);q=r['action_values'];z=d4(st)
            if z in k or len(opts)<2 or opts!=r['valid_moves'] or set(q)!=set(opts) or not all(math.isfinite(v) for v in q.values()):raise ValueError("Bad or duplicate fixture")
            best=max(opts,key=q.get)
            if best!=r['action'] or r['best_actions']!=[d for d in opts if q[best]-q[d]<=float(lock['tie_epsilon'])]:raise ValueError("Invalid teacher label")
            r['board']=[list(row) for row in r['board']]
            k.add(z);s.add(r['episode_seed'])
        data[name]=rs;keys[name]=k;episodes[name]=s
    if keys['train']&keys['dev'] or episodes['train']&episodes['dev']:raise ValueError("Dataset leakage")
    return lock,data
