"""Clone-and-run entry point; no native dependency needed for rules/data smoke."""
import argparse, json, os
from pathlib import Path
from . import data

def cli_parser():
    p=argparse.ArgumentParser(description=__doc__)
    cmds=p.add_subparsers(dest="command",required=True)
    s=cmds.add_parser("verify-data",help="No-model fixture digest/rules/split verification")
    s.add_argument("--data",type=Path,default=Path("data/stock_scale"))
    s=cmds.add_parser("fetch-assets",help="Fetch public pinned model files outside Git")
    s.add_argument("--out",type=Path,required=True)
    for c in ("train","play"):
        s=cmds.add_parser(c)
        s.add_argument("--base",type=Path,required=True)
        s.add_argument("--checkpoint",type=Path,required=True)
        s.add_argument("--out",type=Path,required=True,help="New output directory; will not overwrite")
        s.add_argument("--device",choices=("cpu","cuda"),default="cpu")
        s.add_argument("--threads",type=int,default=2)
        if c=="train":
            s.add_argument("--data",type=Path,default=Path("data/stock_scale"))
            s.add_argument("--preset",choices=("smoke","stock"),default="smoke")
            s.add_argument("--require-historical-init",action="store_true")
        else:
            s.add_argument("--seed",type=int,action="append",required=True)
            s.add_argument("--cap",type=int,default=400)
    return p

def main():
    p=cli_parser();a=p.parse_args()
    if a.command=="verify-data":
        lock,rows=data.checked(a.data)
        out={"no_model":True,"d4_and_episode_disjoint":True,"splits":{k:len(v) for k,v in rows.items()}}
    elif a.command=="fetch-assets":
        from .assets import fetch
        out=fetch(a.out)
    else:
        if a.threads<1 or (a.command=="play" and a.cap<1):p.error("threads/cap must be positive")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        from . import native
        if a.command=="train":out=native.train(a,*data.checked(a.data))
        else:out=native.play(a)
    print(json.dumps(out,allow_nan=False,indent=2))

if __name__=="__main__":main()
