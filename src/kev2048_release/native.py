"""Actual native Kev, lazily imported by the no-dependency CLI."""
import copy, gc, os, random, shutil, statistics, time
import inspect
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
import torch
from torch.nn import functional as F
from kev.checkpoint import Checkpoint, LoadOptions, write_meta
from kev.api import SystemOneRequest, to_record
from kev.model import rows_of
from . import feature_cache
from .assets import FILES
from .data import digest,write
from .game import Game,legal,nested

PROMPT=("Choose the strongest legal slide for 2048. Plan for long-term survival and a high largest tile; "
        "favor open cells and keeping the largest tile in a corner without unnecessarily losing merges. "
        "Zero is empty; equal tiles merge once per slide; after each legal slide a new 2 or 4 appears.")

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def local_model(base,path,device):
    names=("head.pt",*FILES)
    require(base.is_dir() and (base/"config.json").is_file(),"Use --base with a local Qwen snapshot (see TRAINING.md)")
    require(all((path/n).is_file() for n in names),"Use --checkpoint with official local Kev directory or locally built V2")
    if device=="cuda":require(torch.cuda.is_available(),"CUDA unavailable; install a compatible CUDA PyTorch wheel")
    ck=Checkpoint(str(path.resolve())); meta=copy.copy(ck.meta)
    require(meta.weights=="lora" and meta.head_dim==256,"Expected native Kev 0.8B LoRA checkpoint")
    ck.meta.base=str(base.resolve());ck.meta.base_revision=None
    tok,m=ck.load(device,LoadOptions(dtype=torch.float32,merge=True,temperature=1.0))
    return tok,m.requires_grad_(False).eval(),meta

def record(tok,m,r):
    x=dict(state=dict(board=r["board"]),questions=dict(move=dict(type="choice",instructions=PROMPT,
           criteria={d:"Slide "+d for d in r["valid_moves"]})))
    validated,_=to_record(SystemOneRequest.model_validate(x))
    return m.encode(tok,validated)

@torch.no_grad()
def hidden(tok,m,rows):
    layouts=[];at=[]
    for r in rows:
        s,sp,qs=rows_of(record(tok,m,r));q,=qs
        layouts.append((s+q["ids"],sp+q["pos"]))
        at.append((len(s)+q["decide"],[len(s)+n for n in q["opts"]]))
    return [(h[d].clone(),h[k].clone()) for h,(d,k) in zip(m._rows_hidden(layouts),at,strict=True)]

@torch.no_grad()
def parity(tok,m,rows,cache):
    err=[]
    for r,(q,k) in zip(rows,cache,strict=True):
        a=m.head(q,k).softmax(-1).cpu();b,=m.probs(record(tok,m,r));b=b.cpu()
        require(a.shape==b.shape and a.argmax().item()==b.argmax().item(),"Official native argmax mismatch")
        err.append(float((a-b).abs().max()))
    require(max(err)<=1e-5,"Native cache probability parity exceeds 1e-5; use validated CPU FP32")
    return {"rows":len(rows),"max_abs_probability_error":max(err)}

def cache(tok,m,rows,split):
    out=[]
    for i in range(0,len(rows),8):
        vv=hidden(tok,m,rows[i:i+8])
        require(all(a.dtype==b.dtype==torch.float32 and torch.isfinite(a).all() and torch.isfinite(b).all() for a,b in vv),
                "Nonfinite or unexpected-dtype hidden tokens")
        out.extend(vv)
        if len(out)%64==0 or len(out)==len(rows):print(f"{split}: cached {len(out)}/{len(rows)}",flush=True)
    return out

@torch.no_grad()
def evaluate(head,rows,feats):
    out=[]
    for r,(q,k) in zip(rows,feats,strict=True):
        p=head(q,k).softmax(-1).cpu()
        opts=r["valid_moves"]; require(p.numel()==len(opts) and torch.isfinite(p).all(),"Invalid native policy vector")
        pick=opts[int(p.argmax())]
        out.append(dict(seed=r['episode_seed'],ply=r['ply'],stage=r['stage'],crowded=r['crowded'],legal=opts,
                        probabilities=dict(zip(opts,p.tolist())),predicted=pick,best_hit=pick in r['best_actions'],
                        exact_hit=pick==r['action'],raw_teacher_regret=max(r['action_values'].values())-r['action_values'][pick]))
    return out

def metrics(a,b):
    ret={}
    for n,pred in [('all',lambda r:True),('crowded',lambda r:r['crowded']),*((s,lambda r,s=s:r['stage']==s) for s in ('early','middle','tile512','late_ood'))]:
        pairs=[(u,v) for u,v in zip(a,b,strict=True) if pred(u)]
        if not pairs:continue
        ret[n]=dict(n=len(pairs), improved=sum(not u['best_hit'] and v['best_hit'] for u,v in pairs),
              regressed=sum(u['best_hit'] and not v['best_hit'] for u,v in pairs))
        for arm,idx in [('baseline',0),('candidate',1)]:
            rs=[p[idx] for p in pairs]
            ret[n][arm]=dict(best_hits=sum(r['best_hit'] for r in rs),exact_hits=sum(r['exact_hit'] for r in rs),
                            mean_raw_teacher_regret=statistics.mean(r['raw_teacher_regret'] for r in rs))
    return ret

def jsonl(path,rows):
    import json
    with path.open("x",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False,separators=(",",":"))+"\n")

def configure(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(96729100)
    torch.use_deterministic_algorithms(True)
    if args.device=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False

def _encoding_identity(tok, m, meta, device):
    packages = {}
    for name in ("torch", "kev", "transformers", "peft", "tokenizers", "pydantic", "accelerate",
                 "kernels", "causal-conv1d", "flash-linear-attention", "fla-core", "triton", "triton-windows"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    runtime = {
        "device": device,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device == "cuda":
        props = torch.cuda.get_device_properties(device)
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=3,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout.splitlines()
        except (OSError, subprocess.SubprocessError):
            driver = []
        runtime.update(
            gpu=props.name, compute_capability=[props.major, props.minor], gpu_uuid=str(getattr(props, "uuid", None)),
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), driver=sorted(set(driver)),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
            attention_kernels={name: getattr(torch.backends.cuda, name)() for name in
                               ("flash_sdp_enabled", "math_sdp_enabled", "mem_efficient_sdp_enabled", "cudnn_sdp_enabled")},
            cuda_env={name: os.environ.get(name) for name in
                      ("CUBLAS_WORKSPACE_CONFIG", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")},
        )
    return {
        "prompt": PROMPT,
        "encoder_source": [inspect.getsource(fn) for fn in (record, hidden, cache, local_model)],
        "kev_source": [digest(inspect.getfile(obj)) for obj in (Checkpoint, to_record, rows_of)],
        "packages": packages,
        "runtime": runtime,
        # The pointer tensor values are not used by hidden(). Some checkpoint metadata is.
        "encoder_weights": meta.weights,
        "saved_weights_dtype": meta.weights_dtype,
        "loaded_dtype": m.dtype,
        "option_isolation": m.option_isolation,
        "attention": m.lm.config._attn_implementation,
        "tokenizer_type": type(tok).__qualname__,
        "pad_id": m.pad_id,
    }


def _training_features(args, lock, data, tok, model, meta, inputs):
    rows = data["train"]
    if inputs is None:
        features = cache(tok, model, rows, "train")
        return features, parity(tok, model, rows[:8], features[:8]), {"status": "disabled"}

    encoder = _encoding_identity(tok, model, meta, args.device)
    key = feature_cache.cache_key(inputs, lock, data, encoder)
    require(inputs == feature_cache.frozen_inputs(args.base, args.checkpoint),
            "Frozen model inputs changed while loading or hashing them; retry with a stable snapshot")
    path = args.feature_cache.resolve() / f"{key}.features"
    counts = [len(row["valid_moves"]) for row in rows]
    hidden_size = model.lm.config.hidden_size
    features, reason = feature_cache.read_features(path, key, counts, hidden_size, args.device)
    info = {"key": key, "path": str(path), "status": "hit" if reason is None else "miss"}
    if reason is not None:
        info["reason"] = reason
    print(f"train feature cache {info['status']}: {path}" + (f" ({reason})" if reason else ""), flush=True)
    if features is None:
        features = cache(tok, model, rows, "train")
    verification = parity(tok, model, rows[:8], features[:8])
    if reason is not None:
        feature_cache.write_features(path, key, features, counts, hidden_size)
    return features, verification, info

def train(args,lock,data):
    timing={}
    def mark(phase,started):
        # CUDA execution is asynchronous; complete queued work before finishing a phase.
        if args.device=="cuda" and torch.cuda.is_initialized():torch.cuda.synchronize()
        t=time.perf_counter()
        timing[phase+"_seconds"]=t-started
        return t
    configure(args)
    tr=data["train"];dv=data["dev"]
    init_sha=digest(args.checkpoint/"head.pt")
    historical=init_sha==lock["v2_head_sha256"]
    require(not args.require_historical_init or historical,"Historical V2 SHA does not match frozen manifest")
    if args.preset=="smoke":tr=tr[:8];dv=dv[:8]
    else:require(len(tr)==512 and len(dv)==128,"Stock preset requires the 512/128 locked fixtures")
    args.out.mkdir(parents=True,exist_ok=False)
    inputs = feature_cache.frozen_inputs(args.base, args.checkpoint) if args.feature_cache is not None else None
    start=time.perf_counter()
    tok,m,meta=local_model(args.base,args.checkpoint,args.device)
    start=mark("initial_model_load",start)
    trf, p_train, feature_info = _training_features(args, lock, {"train": tr, "dev": dv}, tok, m, meta, inputs)
    start=mark("train_encoding",start)
    head=copy.deepcopy(m.head).requires_grad_(True)
    with torch.no_grad():refs=[m.head(q,k).softmax(-1) for q,k in trf]
    opt=torch.optim.AdamW(head.parameters(),lr=8e-5,weight_decay=.02)
    losses=[]
    for ep in (1,2):
        order=list(range(len(tr))); random.Random(96729100+ep).shuffle(order);head.train();ll=[]
        for i in range(0,len(tr),16):
            opt.zero_grad()
            z=[]
            for j in order[i:i+16]:
                p=head(*trf[j]).log_softmax(-1)
                mask=[d in tr[j]["best_actions"] for d in tr[j]["valid_moves"]]
                z.append(-torch.logsumexp(p[mask],0)+.15*F.kl_div(p,refs[j],reduction="sum"))
            loss=torch.stack(z).mean();require(torch.isfinite(loss),"Nonfinite training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
            ll.append(float(loss.detach()))
        losses.append(statistics.mean(ll))
    start=mark("head_fit",start)
    base_weights=m.head.state_dict()
    changed={k:int((p!=base_weights[k]).sum()) for k,p in head.state_dict().items()}
    require(all(changed.values()),"Pointer parameters did not update")
    heads=args.out/"checkpoint";heads.mkdir()
    for n in FILES:shutil.copyfile(args.checkpoint/n,heads/n)
    meta.head={k:p.detach().cpu() for k,p in head.state_dict().items()};meta.temperature=1.
    meta.extra={"kev2048_release":{"epoch":2,"preset":args.preset,"init_head_sha":init_sha}}
    write_meta(str(heads),meta)
    cksha=digest(heads/"head.pt")
    write(args.out/"head-lock.json",dict(head_sha256=cksha,preset=args.preset,changed=changed))
    start=mark("checkpoint_save",start)
    # No development board passes through the model until the final checkpoint is written and hashed.
    df=cache(tok,m,dv,"dev")
    old=evaluate(m.head,dv,df)
    new=evaluate(head.eval(),dv,df)
    start=mark("development_evaluation",start)
    del tok,m,head,base_weights,opt,trf,refs,meta
    gc.collect()
    if args.device=="cuda":torch.cuda.empty_cache()
    t,reloaded,_=local_model(args.base,heads,args.device)
    pr=parity(t,reloaded,dv[:8],df[:8])
    require(evaluate(reloaded.head,dv,df)==new and digest(heads/"head.pt")==cksha,"Checkpoint reload mismatch")
    mark("official_reload",start)
    data=metrics(old,new)
    def delta(n):return data[n]["improved"]-data[n]["regressed"]
    threshold=(delta('all')>=8 and delta('crowded')>=0 and delta('late_ood')>=-2) if args.preset=="stock" else None
    jsonl(args.out/"baseline-dev.jsonl",old);jsonl(args.out/"candidate-dev.jsonl",new)
    report=dict(preset=args.preset,device=args.device,dtype="fp32",train=len(tr),dev=len(dv),losses=losses,changed=changed,
                candidate_head_sha256=cksha,fixture_manifest=lock["splits"],initialization="historical_v2" if historical else "substitute_local_checkpoint",
                initial_head_sha256=init_sha,timing=timing,training_feature_cache=feature_info,
                train_parity=p_train,reload_parity=pr,metrics=data,meets_stock_numeric_gate=threshold,
                historical_protocol_dev_pass=threshold if historical else None)
    write(args.out/"result.json",report)
    return report

def play(args):
    configure(args)
    args.out.mkdir(parents=True,exist_ok=False)
    tok,m,_=local_model(args.base,args.checkpoint,args.device)
    result=[]
    with (args.out/"moves.jsonl").open("x",encoding="utf-8") as f:
        import json
        for seed in args.seed:
            g=Game.seeded(seed)
            while g.turn<args.cap and (opts:=legal(g.state)):
                before=nested(g.state)
                p,=m.probs(record(tok,m,dict(board=before,valid_moves=opts)));p=p.cpu()
                require(p.numel()==len(opts) and torch.isfinite(p).all(),"Invalid gameplay probability vector")
                a=opts[int(p.argmax())];gain,spawn=g.step(a)
                f.write(json.dumps(dict(seed=seed,turn=g.turn,before=before,after=nested(g.state),action=a,
                    probabilities=dict(zip(opts,p.tolist())),gained=gain,spawn=spawn,score=g.score,tile=max(g.state)))+"\n")
                f.flush()
            result.append(dict(seed=seed,turn=g.turn,score=g.score,tile=max(g.state),stop="cap" if legal(g.state) else "terminal"))
    report=dict(checkpoint_head_sha256=digest(args.checkpoint/"head.pt"),device=args.device,cap=args.cap,games=result)
    write(args.out/"result.json",report);return report
