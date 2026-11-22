"""One local, bounded native Kev 2048 session with resumable SSE snapshots."""
import argparse
import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime, timezone
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Use the already published game rules without requiring Torch for monitor-only mode.
sys.path.insert(0, str(ROOT / 'src'))
from kev2048_release.game import Game, legal, nested

DEFAULT_SEED = 96652001
DEFAULT_CAP = 40
MAX_CAP = 120
HISTORICAL = '67df3e0789ccf361c27c21db6f20dbfbfe9330b0c7fbb01bf4b76c1a9c924822'
RELEASED = 'f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6'
FINAL = {'terminal', 'cap', 'stopped', 'error'}
def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def native_loader(args):
    from kev2048_release import native
    native.configure(args)
    tok, model, _ = native.local_model(args.base, args.checkpoint, args.device)
    torch = native.torch
    def decide(board, opts):
        with torch.inference_mode():
            if args.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            (p,) = model.probs(native.record(tok, model, dict(board=board, valid_moves=opts)))
            vals = p.detach().cpu().tolist()
            if args.device == 'cuda':
                torch.cuda.synchronize()
            return dict(zip(opts, vals, strict=True)), (time.perf_counter()-start)*1000
    return decide

class BadRequest(Exception):
    def __init__(self, message, code=400):
        self.message = message
        self.code = code

class Live:
    def __init__(self, load, out, profile, sha):
        self.cv = threading.Condition()
        self.load = load
        self.out = Path(out)
        self.profile = profile
        self.sha = sha
        self.decide = None
        self.active = False
        self.pause_requested = False
        self.stop_requested = False
        self.id = ''
        self.cap = DEFAULT_CAP
        self.seed = DEFAULT_SEED
        self.game = Game.seeded(self.seed)
        self.frames = []
        self.frame_times = []
        self.emit('ready')

    def emit(self, state, **extra):
        with self.cv:
            g = self.game
            row = dict(type=state, run_id=self.id, seq=len(self.frames), seed=self.seed,
                turn=g.turn, score=g.score, tile=max(g.state), board=nested(g.state),
                cap=self.cap, profile=self.profile, checkpoint_sha256=self.sha,
                stop=state if state in FINAL - {'error'} else None)
            row.update(extra)
            self.frames.append(row)
            self.frame_times.append(time.time())
            self.cv.notify_all()
            return row

    def latest(self):
        with self.cv:
            return self.frames[-1]

    def start(self, body):
        if set(body) - {'seed', 'cap'}:
            raise BadRequest('Unknown start parameters')
        seed, cap = body.get('seed', DEFAULT_SEED), body.get('cap', DEFAULT_CAP)
        if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
            raise BadRequest('Invalid seed')
        if type(cap) is not int or not 1 <= cap <= MAX_CAP:
            raise BadRequest(f'Cap must be 1–{MAX_CAP}')
        with self.cv:
            if self.active:
                raise BadRequest('A run is already active', 409)
            self.active = True
            self.pause_requested = self.stop_requested = False
            self.id, self.seed, self.cap = uuid.uuid4().hex, seed, cap
            self.game, self.frames, self.frame_times = Game.seeded(seed), [], []
            first = self.emit('loading' if self.decide is None else 'running')
            threading.Thread(target=self.work, daemon=True).start()
            return first

    def command(self, key, body):
        if body:
            raise BadRequest('Command takes an empty object')
        with self.cv:
            if not self.active:
                raise BadRequest('No active run', 409)
            if key == 'stop':
                self.stop_requested = True
                self.pause_requested = False
            elif self.stop_requested:
                raise BadRequest('Run is stopping', 409)
            elif key == 'pause':
                self.pause_requested = True
            elif key == 'resume':
                self.pause_requested = False
            self.cv.notify_all()
            return self.frames[-1]

    def work(self):
        finish = 'error'
        try:
            if self.decide is None:
                self.decide = self.load()
            self.out.mkdir(parents=True, exist_ok=True)
            with (self.out / f'{self.id}.jsonl').open('x', encoding='utf-8') as fp:
                with self.cv:
                    if not self.stop_requested and self.frames[-1]['type'] == 'loading':
                        self.emit('running')
                while True:
                    with self.cv:
                        while self.pause_requested and not self.stop_requested:
                            if self.frames[-1]['type'] != 'paused':
                                self.emit('paused')
                            self.cv.wait()
                        if self.stop_requested:
                            finish = 'stopped'
                            break
                        if self.frames[-1]['type'] == 'paused':
                            self.emit('running')
                        opts = legal(self.game.state)
                        if not opts:
                            finish = 'terminal'
                            break
                        if self.game.turn >= self.cap:
                            finish = 'cap'
                            break
                        before = nested(self.game.state)
                    p, ms = self.decide(before, opts)
                    if (list(p) != list(opts) or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p.values())
                            or abs(sum(p.values())-1) > 1e-5 or not math.isfinite(ms) or ms < 0):
                        raise ValueError('Invalid native decision')
                    action = max(opts, key=p.get)
                    with self.cv:
                        if self.stop_requested:
                            finish = 'stopped'
                            break
                        gained, spawn = self.game.step(action)
                        stop = ('terminal' if not legal(self.game.state) else
                                'cap' if self.game.turn >= self.cap else None)
                        # Persist before publication, while no next step/command may advance the game.
                        event = dict(type='move', run_id=self.id, seq=len(self.frames), seed=self.seed,
                            turn=self.game.turn, score=self.game.score, tile=max(self.game.state),
                            board=nested(self.game.state), before=before, action=action, gained=gained,
                            spawn=spawn, decision_ms=ms, probabilities=p, stop=stop, cap=self.cap,
                            profile=self.profile, checkpoint_sha256=self.sha)
                        fp.write(json.dumps(event, allow_nan=False, separators=(',', ':')) + '\n')
                        fp.flush()
                        self.frames.append(event)
                        self.frame_times.append(time.time())
                        self.cv.notify_all()
                        if stop:
                            finish = stop
                            break
        except Exception:
            traceback.print_exc()
        finally:
            with self.cv:
                self.active = False
                self.emit(finish, **({'message':'本地 Kev 运行失败，未使用替代策略。'} if finish == 'error' else {}))

    def pending(self, run_id, seq, wait=10):
        def get():
            return [f for f in self.frames if f['run_id'] != run_id or f['seq'] > seq]
        with self.cv:
            if not get():
                self.cv.wait_for(lambda: bool(get()), timeout=wait)
            return get()

MAX_WATCH_BYTES = 4 * 1024 * 1024
MAX_WATCH_ROWS = 48
MAX_WATCHES = 12
ACTIVE_STATES = {'loading','running','paused'}
KINDS = {'training','inference','benchmark'}
SAFE_STATUS = {'waiting','loading','running','paused','completed','terminal','cap','stopped','error','unknown'}
SAFE_LABEL = re.compile(r'^[\w -]{1,80}$')
TIMING_NAME = re.compile(r'^[a-z][a-z0-9_]{0,52}$')
SHA = re.compile(r'^[0-9a-f]{64}$')

def utc(now):
    return datetime.fromtimestamp(now, timezone.utc).isoformat().replace('+00:00','Z')

def timestamp(s, now):
    if not isinstance(s,str) or len(s)>40:
        return None
    try:
        d = datetime.fromisoformat(s.replace('Z','+00:00'))
        q = d.timestamp()
        return q if d.tzinfo is not None and 0<=q<=now+300 else None
    except (ValueError,OverflowError,OSError):
        return None

def measured(x):
    return not isinstance(x,bool) and isinstance(x,(float,int)) and math.isfinite(x) and abs(x)<1e15

def safe_metrics(item):
    out = {}
    for k in ('step','total','loss','elapsed_seconds','decision_ms','score','tile','turn','epoch','epochs'):
        v=item.get(k)
        if measured(v) and (v>=0 or k=='loss'):
            out[k]=v
    b=item.get('board')
    if isinstance(b,list) and len(b)==4 and all(isinstance(r,list) and len(r)==4 for r in b):
        if all(type(v) is int and 0<=v<=2**30 and (v==0 or (v>=2 and not v&(v-1))) for r in b for v in r):
            out['board']=b
    return out

def safe_timing(item):
    out={}
    def add(key,value,prefix=''):
        if isinstance(key,str) and TIMING_NAME.fullmatch(key) and measured(value) and value>=0 and len(out)<48:
            if key.endswith('_ms'):
                key,value=key[:-3]+'_seconds',value/1000
            elif not key.endswith('_seconds'):
                key+='_seconds'
            out[prefix+key]=value
    for field in ('timings','timing'):
        for k,v in (item[field].items() if isinstance(item.get(field),dict) else ()):
            add(k,v)
    for field in ('median_phase_seconds','median_wall_seconds'):
        devices=item.get(field)
        if isinstance(devices,dict):
            for dev,data in devices.items():
                if dev not in ('cpu','cuda','mps'):
                    continue
                if field=='median_wall_seconds':
                    add('median_wall_seconds',data,dev+'_')
                elif isinstance(data,dict):
                    for k,v in data.items():
                        add(k,v,dev+'_median_')
    return out

def event(raw,now):
    if not isinstance(raw,dict):
        return None
    state=raw.get('status')
    status=state if isinstance(state,str) and state in SAFE_STATUS else 'unknown'
    stage=raw.get('phase')
    phase = stage if isinstance(stage,str) and SAFE_LABEL.fullmatch(stage) else None
    t=timestamp(raw.get('timestamp',raw.get('updated_at')),now)
    m=safe_metrics(raw)
    v=safe_timing(raw)
    if status=='unknown' and phase is None and not m and not v:
        return None
    # A missing per-event clock is visibly unknown, not assigned one from mtime.
    return {**m,'status':status,'phase':phase,'timestamp':utc(t) if t is not None else None,'timings':v}

class Watch:
    def __init__(self, i, spec):
        try:
            name,kind,path=spec.split(':',2)
        except ValueError:
            raise ValueError('--watch expects NAME:KIND:PATH') from None
        if not SAFE_LABEL.fullmatch(name) or kind not in KINDS:
            raise ValueError('Invalid watch name or kind')
        p=Path(path)
        if p.suffix.lower() not in ('.json','.jsonl','.ndjson'):
            raise ValueError('Watch paths must end in .json, .jsonl or .ndjson')
        self.path=p.absolute()
        self.base=dict(id=f'watch-{i}',name=name,kind=kind,
                       mode='static_result' if p.suffix.lower()=='.json' else 'progress_log')
        self.cache_key=None
        self.cache_value=None
        self.lock=threading.Lock()

    def snapshot(self, now):
        row={**self.base,'status':'waiting','phase':None,'updated_at':None,'timings':{},'history':[],'latest':None}
        try:
            st=self.path.lstat()
            if not stat.S_ISREG(st.st_mode) or st.st_size>MAX_WATCH_BYTES:
                row['status']='unknown'
                return row
        except FileNotFoundError:
            return row
        except OSError:
            row['status']='unknown'
            return row
        mtime=st.st_mtime
        row['updated_at']=utc(mtime) if 0<=mtime<=now+300 else None
        with self.lock:
            key=(st.st_dev,st.st_ino,st.st_mtime_ns,st.st_size)
            if key != self.cache_key:
                try:
                    fd=os.open(self.path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0)|getattr(os,'O_BINARY',0))
                    with os.fdopen(fd,'rb') as f:
                        seen=os.fstat(f.fileno())
                        if not stat.S_ISREG(seen.st_mode) or (seen.st_dev,seen.st_ino)!=(st.st_dev,st.st_ino):
                            raise ValueError('nonregular file')
                        data=f.read(MAX_WATCH_BYTES+1)
                    if len(data)>MAX_WATCH_BYTES:
                        raise ValueError('file too large')
                    text=data.decode('utf-8-sig')
                    if self.base['mode']=='static_result':
                        normalized=self.static(text,now)
                    else:
                        normalized=self.progress(text,now)
                except (OSError,ValueError,RecursionError,OverflowError):
                    normalized=[]
                self.cache_key=key
                self.cache_value=normalized
            hist=self.cache_value
        if not hist:
            row['status']='waiting' if st.st_size==0 else 'unknown'
            return row
        latest=hist[-1]
        row.update(status=latest['status'],phase=latest['phase'],latest=latest,history=hist,
                   updated_at=latest['timestamp'] or row['updated_at'])
        for unit in hist:
            for k,v in unit['timings'].items():
                if len(row['timings'])<48 or k in row['timings']:
                    row['timings'][k]=v
        if row['status'] in ACTIVE_STATES:
            last=timestamp(latest['timestamp'],now)
            if last is None:
                # Buffered/partially appended data or filesystem mtime alone does not prove activity.
                row['status']='unknown'
            elif now-last>120:
                row['status']='stale'
        return row

    @staticmethod
    def static(text,now):
        raw=json.loads(text)
        if not isinstance(raw,dict):
            return []
        row=event(raw,now)
        if row is None:
            row=dict(status='unknown',phase=None,timestamp=None,timings={})
        explicit=raw.get('status')
        released_result=(isinstance(raw.get('candidate_head_sha256'),str) and SHA.fullmatch(raw['candidate_head_sha256'])
                         and type(raw.get('train')) is int and raw['train']>0 and type(raw.get('dev')) is int and raw['dev']>0
                         and isinstance(raw.get('metrics'),dict) and bool(row['timings']))
        if raw.get('completed') is True or explicit in ('completed','terminal','cap','stopped','error') or released_result:
            row['status']=explicit if explicit in ('terminal','cap','stopped','error') else 'completed'
        else:
            row['status']='unknown'
        return [row]

    @staticmethod
    def progress(text,now):
        out=[]
        lines=text.splitlines()[-MAX_WATCH_ROWS:]
        for i,line in enumerate(lines):
            try:
                r=event(json.loads(line),now)
            except (ValueError,RecursionError,OverflowError):
                # An actively written last row may be incomplete; arbitrary text earlier is not ignored.
                if i==len(lines)-1 and not text.endswith(('\n','\r')):
                    break
                return []
            if r is not None:
                out.append(r)
        return out

def channel_snapshot(server):
    now=time.time()
    channels=[]
    if server.live is not None:
        app=server.live
        with app.cv:
            history=[]
            for f,t in list(zip(app.frames,app.frame_times))[-MAX_WATCH_ROWS:]:
                state={'ready':'waiting','move':'running'}.get(f['type'],f['type'])
                history.append(event({**f,'status':state,'phase':'native_inference','timestamp':utc(t)},now))
            last=history[-1]
            channels.append(dict(id='console',name='Kev local console',kind='inference',mode='in_session',
                status=last['status'],phase=last['phase'],updated_at=last['timestamp'],timings=last['timings'],
                latest=last,history=history))
    channels.extend(w.snapshot(now) for w in server.watches)
    return dict(timestamp=utc(now),channels=channels)

class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 12
    def __init__(self, port, live, index, watches=()):
        super().__init__(('127.0.0.1', port), Handler)
        self.live = live
        self.watches = tuple(watches)
        index = Path(index)
        self.page_root = index.parent.resolve()
        self.pages = {'/': index.name, '/console': 'console.html', '/replay': 'replay.html'}
        self.streams = threading.BoundedSemaphore(6)
        self.hosts = {f'127.0.0.1:{self.server_port}', f'localhost:{self.server_port}'}

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def allowed_host(self):
        return self.headers.get('Host') in self.server.hosts
    def body(self, val, code=200, mime='application/json; charset=utf-8'):
        if isinstance(val, dict):
            val = json.dumps(val, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-type', mime)
        self.send_header('Content-Length', str(len(val)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('X-Frame-Options', 'SAMEORIGIN')
        self.end_headers()
        self.wfile.write(val)
    def do_GET(self):
        origin = self.headers.get('Origin')
        same_origin = origin is None or origin == 'http://' + self.headers.get('Host', '')
        if not self.allowed_host() or not same_origin or self.headers.get('Sec-Fetch-Site') == 'cross-site':
            self.body({'error':'Forbidden'}, 403)
        elif self.path == '/console' and self.server.live is None:
            self.body({'error':'Live console disabled'},404)
        elif self.path in self.server.pages:
            path = self.server.page_root / self.server.pages[self.path]
            if path.is_file() and path.resolve().parent == self.server.page_root:
                self.body(path.read_bytes(), mime='text/html; charset=utf-8')
            else:
                self.body({'error':'Preview HTML not present'}, 404)
        elif self.path == '/channels':
            self.body(channel_snapshot(self.server))
        elif self.path in ('/state','/events') and self.server.live is None:
            self.body({'error':'Live console disabled'},404)
        elif self.path == '/state':
            self.body(self.server.live.latest())
        elif self.path == '/events':
            self.sse()
        else:
            self.body({'error':'Not found'}, 404)

    def sse(self):
        if not self.server.streams.acquire(blocking=False):
            self.body({'error':'Too many live listeners'},503)
            return
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            self.send_header('Connection', 'close')
            self.send_header('X-Accel-Buffering', 'no')
            self.end_headers()
            self.close_connection = True
            self.connection.settimeout(15)
            rid, sep, s = self.headers.get('Last-Event-ID','').rpartition(':')
            seq = int(s) if sep and s.isascii() and s.isdigit() and len(s) <= 6 else -1
            while True:
                rows = self.server.live.pending(rid, seq)
                if not rows:
                    self.wfile.write(b': heartbeat\n\n')
                for row in rows:
                    rid, seq = row['run_id'], row['seq']
                    msg = f'id: {rid}:{seq}\nevent: frame\ndata: {json.dumps(row, allow_nan=False)}\n\n'
                    self.wfile.write(msg.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            self.server.streams.release()

    def reject(self, message, code):
        self.close_connection = True
        self.body({'error':message},code)

    def do_POST(self):
        if not self.allowed_host() or self.headers.get('Origin') != 'http://' + self.headers.get('Host',''):
            self.reject('Same-origin request required',403)
            return
        if self.server.live is None:
            self.reject('Read-only monitoring mode',403)
            return
        if self.path not in ('/start','/pause','/resume','/stop'):
            self.reject('Not found',404)
            return
        if self.headers.get_content_type() != 'application/json':
            self.reject('Expected application/json',415)
            return
        length = self.headers.get('Content-Length','')
        if not length.isascii() or not length.isdigit() or len(length)>4 or not 1 <= int(length) <= 512:
            self.reject('Invalid request length',400)
            return
        try:
            self.connection.settimeout(5)
            raw = json.loads(self.rfile.read(int(length)))
            if not isinstance(raw, dict):
                raise BadRequest('Expected a JSON object')
            target = self.path[1:]
            row = self.server.live.start(raw) if target == 'start' else self.server.live.command(target,raw)
            self.body(row,202)
        except (ValueError, UnicodeError, TimeoutError):
            self.reject('Invalid JSON request',400)
        except BadRequest as e:
            self.reject(e.message,e.code)

    def log_message(self, pattern, *args):
        if self.path != '/events':
            super().log_message(pattern,*args)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--base', type=Path, default=ROOT/'assets'/'qwen-base')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'assets'/'official-kev')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--threads', type=int, choices=(1,2,4), default=2)
    parser.add_argument('--out', type=Path, default=ROOT/'runs'/'live')
    parser.add_argument('--watch',action='append',default=[],metavar='NAME:KIND:PATH')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--monitor-only',action='store_true',help='Serve read-only dashboard (the default)')
    mode.add_argument('--enable-live-console',action='store_true',help='Opt in to the separate local inference console')
    args = parser.parse_args()
    if len(args.watch)>MAX_WATCHES:
        parser.error(f'At most {MAX_WATCHES} watch files')
    try:
        watches=[Watch(i,s) for i,s in enumerate(args.watch,1)]
    except ValueError as e:
        parser.error(str(e))
    if not args.enable_live_console:
        serve(args,None,watches)
        return
    if not (args.checkpoint/'head.pt').is_file():
        parser.error('Local model assets missing; see the launcher instructions')
    sha = digest(args.checkpoint/'head.pt')
    profiles = {HISTORICAL:'frozen_v2', RELEASED:'official_kev_0_8b'}
    if sha not in profiles:
        parser.error('Checkpoint does not match exact frozen V2 or pinned official Kev')
    # File-pinned, independent of any private workspace fallback paths.
    checkpoint_files = {
        'adapter_model.safetensors':'9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a',
        'adapter_config.json':'748acb2cda88454cb1ba69d745ba336f3fcb5486eac349e90960c8b8d8d3e854',
        'tokenizer.json':'06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523',
        'tokenizer_config.json':'8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45',
    }
    base_files = {
        'model.safetensors':'c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c',
        'config.json':'b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204',
        **{k:v for k,v in checkpoint_files.items() if k.startswith('tokenizer')},
    }
    for folder, pins in ((args.base,base_files),(args.checkpoint,checkpoint_files)):
        for artifact, pin in pins.items():
            if not (folder/artifact).is_file() or digest(folder/artifact) != pin:
                parser.error(f'Missing or not pinned: {artifact}')
    app = Live(lambda: native_loader(args), args.out, profiles[sha], sha)
    serve(args,app,watches)

def serve(args,app,watches):
    with Server(args.port, app, HERE/'index.html',watches) as server:
        print(f'http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
