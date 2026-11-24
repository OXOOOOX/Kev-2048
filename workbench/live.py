"""Local Kev 2048 workbench for training progress, replay and interactive play."""
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
from urllib.parse import parse_qs, urlencode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Use the already published game rules without requiring Torch for monitor-only mode.
sys.path.insert(0, str(ROOT / 'src'))
from kev2048_release.game import DIRS, Game, legal, nested
from play_session import PlayRequestError, PlaySession
from replay_library import ReplayLibrary
from comparison_catalog import build_comparison_catalog
from evaluation_catalog import EvaluationCatalog
from gameplay_reuse import GameplayReuse
from gameplay_replay import GameplayReplay, MAX_PAGE, NativeGameplayReplay
from training_boards import ACCUMULATION, TRAINING_SCHEMA, TrainingBoards, recorded_batch
from run_discovery import RunDiscovery
from run_summaries import normalize_result
from training_archive import TrainingArchive

DEFAULT_SEED = 96652001
DEFAULT_CAP = 40
MAX_CAP = 120
PLAY_COMMANDS = {'/play/start':'start', '/play/stop':'stop', '/play/pause':'pause',
                 '/play/resume':'resume', '/play/save':'save', '/play/continue-endless':'continue_endless'}
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
KINDS = {'training','evaluation','inference','benchmark'}
EVALUATION_PHASES = {'gameplay','gameplay_start','episode_completed','screening_completed','baseline_reused'}
SAFE_STATUS = {'waiting','loading','running','paused','completed','terminal','cap','stopped','error','unknown'}
SAFE_LABEL = re.compile(r'^[\w -]{1,80}$')
TIMING_NAME = re.compile(r'^[a-z][a-z0-9_]{0,52}$')
SHA = re.compile(r'^[0-9a-f]{64}$')
GAMEPLAY_ARM = re.compile(r'^[a-z0-9][a-z0-9-]{0,63}$')
MOVE_TAIL_BYTES = 8192
MONITOR_INTERVAL = 0.25
OPTIMIZER_PROGRESS_TIMEOUT = 360

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

def safe_board(board):
    return (isinstance(board,list) and len(board)==4
        and all(isinstance(row,list) and len(row)==4 for row in board)
        and all(type(v) is int and 0<=v<=2**30 and (v==0 or (v>=2 and not v&(v-1)))
                for row in board for v in row))

def safe_metrics(item):
    out = {}
    for k in ('step','total','loss','elapsed_seconds','decision_ms','score','tile','turn','epoch','epochs'):
        v=item.get(k)
        if measured(v) and (v>=0 or k=='loss'):
            out[k]=v
    for k in ('games','total_games','active_games','reference_games_reused','reused_games','complete_games',
              'requested_games','until','turns','seed','encoded_boards','cache_hits','candidate_games',
              'train_rows','dev_rows','repeats','threads'):
        v=item.get(k)
        if type(v) is int and 0<=v<2**53:
            out[k]=v
    for k in ('train_loss','dev_loss','best_dev_teacher_loss','boards_per_second','wall_speedup_cpu_over_cuda',
              'process_speedup','process_seconds_saved','process_percent_saved'):
        if measured(item.get(k)) and item[k]>=0:
            out[k]=item[k]
    optimizer_step, steps = item.get('optimizer_step'), item.get('steps')
    if type(optimizer_step) is int and type(steps) is int and 0<=optimizer_step<=steps<=100_000_000 and steps>0:
        out.update(optimizer_step=optimizer_step,steps=steps)
        if item.get('phase') in ('optimizer','lora_fit'):
            for key,limit in (('batch_size',100_000_000),('samples_seen',2**53-1)):
                value=item.get(key)
                if type(value) is int and 0<value<=limit:
                    out[key]=value
    train_loss = item.get('mean_training_loss')
    if measured(train_loss):
        out['mean_training_loss']=train_loss
    rate=item.get('dev_best_rate')
    if measured(rate) and 0<=rate<=1:
        out['dev_best_rate']=rate
    trial=item.get('trial')
    if isinstance(trial,str) and SAFE_LABEL.fullmatch(trial):
        out['trial']=trial
    stage=item.get('stage')
    if isinstance(stage,str) and SAFE_LABEL.fullmatch(stage):
        out['stage']=stage
    if item.get('phase')=='screening_completed':
        promote=item.get('promote_to_100')
        if type(promote) is bool:
            out['promote_to_100']=promote
        for k in ('candidate_vector','threshold'):
            v=item.get(k)
            if (isinstance(v,list) and len(v)==4 and all(type(n) is int and 0<=n<=300 for n in v[:3])
                    and measured(v[3]) and v[3]>=0):
                out[k]=v
    b=item.get('board')
    if safe_board(b):
        out['board']=b
    return out

def open_watch_file(path, expected):
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0)|getattr(os,'O_BINARY',0))
    stream=os.fdopen(fd,'rb')
    try:
        seen=os.fstat(stream.fileno())
        if not stat.S_ISREG(seen.st_mode) or (seen.st_dev,seen.st_ino)!=(expected.st_dev,expected.st_ino):
            raise ValueError('nonregular file')
    except (OSError,ValueError):
        stream.close()
        raise
    return stream

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
    if state=='failed':
        state='error'
    status=state if isinstance(state,str) and state in SAFE_STATUS else 'unknown'
    stage=raw.get('phase')
    phase = stage if isinstance(stage,str) and SAFE_LABEL.fullmatch(stage) else None
    if phase=='optimizer' and (raw.get('schema')!=TRAINING_SCHEMA or raw.get('kind')!='training'):
        return None
    t=timestamp(raw.get('timestamp',raw.get('updated_at')),now)
    m=safe_metrics(raw)
    v=safe_timing(raw)
    if status=='unknown' and phase is None and not m and not v:
        return None
    # A missing per-event clock is visibly unknown, not assigned one from mtime.
    return {**m,'status':status,'phase':phase,'timestamp':utc(t) if t is not None else None,'timings':v}


def estimate_batch_pace(current,previous,size,previous_size,now):
    """Average actual boards per second between two updates of the same training pass."""
    if (not current.get('trial') or current.get('trial')!=previous.get('trial') or type(previous.get('epoch')) is not int
            or current.get('epoch')!=previous.get('epoch') or current['steps']!=previous.get('steps')):
        return None
    step=current['optimizer_step']
    span=step-previous['optimizer_step']
    start,end=(timestamp(record.get('timestamp'),now) for record in (previous,current))
    if span<=0 or start is None or end is None or end<=start:
        return None
    seconds=end-start
    if 'elapsed_seconds' in previous and 'elapsed_seconds' in current:
        seconds=current['elapsed_seconds']-previous['elapsed_seconds']
        if seconds<=0:
            return None
    if 'samples_seen' in previous and 'samples_seen' in current:
        count=current['samples_seen']-previous['samples_seen']
        if count<size+span-1 or (span==1 and count!=size):
            return None
    elif span==1:
        count=size
    else:
        # These legacy trainers use full batches until the last optimizer update.
        if not previous_size or (size!=previous_size and (step!=current['steps'] or size>previous_size)):
            return None
        count=(span-1)*previous_size+size
    speed=count/seconds
    return speed if measured(speed) else None


def latest_optimizer_record(text, now):
    current,batch=None,None
    for line in reversed(text.splitlines()):
        if TRAINING_SCHEMA not in line:
            continue
        try:
            raw=json.loads(line)
            if (not isinstance(raw,dict) or raw.get('phase')!='optimizer'
                    or raw.get('schema')!=TRAINING_SCHEMA or raw.get('kind')!='training'):
                continue
            step=event(raw,now)
            if step.get('optimizer_step',0)<=0:
                continue
            if current is None:
                if step['timestamp'] is None:
                    continue
                current,batch=step,recorded_batch(raw,step)
                if batch is None:
                    break
                if 'boards_per_second' in batch:
                    batch['pace_source']='measured'
                    break
            elif step.get('trial')==current.get('trial'):
                rate=estimate_batch_pace(current,step,batch['batch_size'],step.get('batch_size'),now)
                if rate is not None:
                    batch.update(boards_per_second=rate,pace_source='estimated')
                break
        except (ValueError,RecursionError,OverflowError):
            continue
    return current,batch

class Watch:
    def __init__(self, i, spec, *, path_guard=None):
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
        self.path_guard=path_guard
        self.base=dict(id=f'watch-{i}',name=name,kind=kind,
                       mode='static_result' if p.suffix.lower()=='.json' else 'progress_log')
        self.cache_key=None
        self.cache_value=([],None,None)
        self.lock=threading.Lock()
        self.native_replay=None
        if kind=='evaluation' and p.name=='result.json':
            arm=self.path.parent.name
            self.native_replay=NativeGameplayReplay(self.path,arm if SAFE_LABEL.fullmatch(arm) else name)

    def snapshot(self, now):
        row={**self.base,'status':'waiting','phase':None,'updated_at':None,'timings':{},'history':[],'latest':None}
        is_static=self.base['mode']=='static_result'
        try:
            st=self.path.lstat()
            if (not stat.S_ISREG(st.st_mode) or (is_static and st.st_size>MAX_WATCH_BYTES)
                    or (self.path_guard is not None and not self.path_guard(self.path))):
                row['status']='unknown'
                return row
        except FileNotFoundError:
            return row
        except (OSError,RuntimeError):
            row['status']='unknown'
            return row
        mtime=st.st_mtime
        row['updated_at']=utc(mtime) if 0<=mtime<=now+300 else None
        if row['kind']=='training':
            row['log_revision']=f'{st.st_ino:x}-{st.st_mtime_ns:x}-{st.st_size:x}'
        with self.lock:
            key=(st.st_dev,st.st_ino,st.st_mtime_ns,st.st_size)
            if key != self.cache_key:
                training_step,training_batch=None,None
                try:
                    with open_watch_file(self.path,st) as f:
                        offset=max(0,st.st_size-MAX_WATCH_BYTES) if not is_static else 0
                        if offset:
                            f.seek(offset)
                        data=f.read(min(st.st_size,MAX_WATCH_BYTES) if not is_static else MAX_WATCH_BYTES+1)
                        if self.path_guard is not None and not self.path_guard(self.path):
                            raise ValueError('discovered watch changed path')
                    if len(data)>MAX_WATCH_BYTES:
                        raise ValueError('file too large')
                    if offset:
                        data=data.partition(b'\n')[2]
                    text=data.decode('utf-8-sig')
                    if is_static:
                        normalized=self.static(text,now,status_file=self.path.name=='status.json')
                    else:
                        normalized=self.progress(text,now)
                        if normalized and row['kind']=='training':
                            training_step,training_batch=latest_optimizer_record(text,now)
                except (OSError,ValueError,RuntimeError,OverflowError):
                    normalized=[]
                self.cache_key=key
                self.cache_value=normalized,training_step,training_batch
            hist,training_step,training_batch=self.cache_value
        if not hist:
            row['status']='waiting' if st.st_size==0 else 'unknown'
            return row
        latest=hist[-1]
        row.update(status=latest['status'],phase=latest['phase'],latest=latest,history=hist,
                   updated_at=latest['timestamp'] or row['updated_at'])
        if row['kind']=='inference' and (latest.get('stage') or any(unit['phase'] in EVALUATION_PHASES for unit in hist)):
            row['kind']='evaluation'
        if training_step is not None:
            row['training_step']=training_step
        if training_batch is not None:
            row['training_batch']=training_batch
        if row['kind']=='inference':
            completed=next((e['complete_games'] for e in reversed(hist) if 'complete_games' in e),None)
            goal=next((e.get('requested_games',e.get('until')) for e in reversed(hist)
                       if 'requested_games' in e or 'until' in e),None)
            if completed is not None:
                row['collection_progress']={'completed':completed,'total':goal}
        if row['kind']=='training' and latest['phase'] in ('lora_fit','optimizer'):
            row['stale_after_seconds']=OPTIMIZER_PROGRESS_TIMEOUT
        for unit in hist:
            for k,v in unit['timings'].items():
                if len(row['timings'])<48 or k in row['timings']:
                    row['timings'][k]=v
        if row['status'] in ACTIVE_STATES:
            last=timestamp(latest['timestamp'],now)
            if last is None:
                # Buffered/partially appended data or filesystem mtime alone does not prove activity.
                row['status']='unknown'
            elif now-last>row.get('stale_after_seconds',120):
                row['status']='stale'
        if self.native_replay is not None and row['status']=='completed':
            row.update(self.native_replay.overview(now) or {})
        return row

    @staticmethod
    def static(text,now,*,status_file=False):
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
        summary=normalize_result(raw)
        # Only a status snapshot can report live activity. A result may carry an
        # old nonterminal status from its inputs without having an active writer.
        if not status_file and summary.get('status') in ACTIVE_STATES | {'waiting'}:
            summary.pop('status')
        row.update(summary)
        return [row]

    @staticmethod
    def progress(text,now):
        def normalize(raw, now):
            if isinstance(raw,dict) and raw.get('phase')=='verification_budget_stop':
                return None
            return event(raw,now)
        return Watch.progress_events(text,now,normalize)

    @staticmethod
    def progress_events(text,now,normalize):
        out=[]
        lines=text.splitlines()[-MAX_WATCH_ROWS:]
        for i,line in enumerate(lines):
            try:
                r=normalize(json.loads(line),now)
            except (ValueError,RecursionError,OverflowError):
                # An actively written last row may be incomplete; arbitrary text earlier is not ignored.
                if i==len(lines)-1 and not text.endswith(('\n','\r')):
                    break
                return []
            if r is not None:
                out.append(r)
        return out

class GameplayWatch(Watch):
    """Observe an explicitly selected evaluation directory without running it."""
    def __init__(self, i, name, directory, *, path_guard=None, runs_root=None):
        self.directory=Path(directory).resolve()
        super().__init__(i,f'{name}:evaluation:{self.directory / "progress.jsonl"}',path_guard=path_guard)
        self.base['mode']='gameplay'
        self.inputs_cache=None
        self.reuse_cache=None
        self.candidate_count_cache=None
        self.move_cache={}
        self.completed_cache=None
        self.finished_seeds_cache=None
        self.reuse=GameplayReuse(runs_root,self.directory)

    @staticmethod
    def progress(text,now):
        def normalize(raw,now):
            item=event(raw,now)
            if item is not None:
                rate=raw.get('decisions_per_second')
                if item['phase']=='gameplay' and measured(rate) and rate>=0:
                    item['decisions_per_second']=rate
            return item
        return Watch.progress_events(text,now,normalize)

    def snapshot(self, now):
        row=super().snapshot(now)
        if self.path_guard is not None and not self.path_guard(self.path):
            return row
        signature=(self.cache_key,row['status'])
        if row['status']=='completed' and self.completed_cache is not None:
            saved_at,previous,cached=self.completed_cache
            if previous==signature and now-saved_at<5:
                return cached
        row.update(episodes=[],games=None,total_games=None,batch=None,arms=[],seed_catalog=[],
                   activity=None,decisions_per_second=None)
        try:
            arms,batches,identity=self.inputs()
        except (OSError,ValueError,RuntimeError):
            return row
        seed_catalog=[seed for group in batches for seed in group]
        total_seeds=len(seed_catalog)
        total=len(arms)*total_seeds
        reused=self.reused_baseline(identity,total_seeds) if 'official' in arms and len(arms)>1 else 0
        if reused and row['history']:
            # The runner's screening count describes that subset, not the full run.
            row['history']=[{key:value for key,value in item.items() if key!='games'}
                            if item['phase']=='screening_completed' else item for item in row['history']]
            row['latest']=row['history'][-1]
        counts=[item['games'] for item in row['history'] if 'games' in item and reused<=item['games']<=total]
        games=max(counts) if counts else None
        if reused:
            row['reused_games']=reused
            latest_count=(row['latest'] or {}).get('games')
            if latest_count is None or not reused<=latest_count<=total:
                # Resume and other control events may hide all real counts in the bounded history.
                games=max(games or reused,reused+self.completed_candidates(arms,batches,row))
        # The runner finishes every arm in a batch before advancing to the next.
        scheduled_arms=len(arms)-int(bool(reused))
        seed_offset=min(((games or 0)-reused)//scheduled_arms,total_seeds-1)
        for number,seeds in enumerate(batches,1):
            if seed_offset<len(seeds):
                break
            seed_offset-=len(seeds)
        rate=next((item['decisions_per_second'] for item in reversed(row['history'])
                   if 'decisions_per_second' in item),None)
        row.update(games=games,total_games=total,arms=arms,seed_catalog=seed_catalog,
                   batch=dict(number=number,total=len(batches),seeds=seeds),decisions_per_second=rate)
        if row['status']=='completed':
            recorded=self.finished_seeds(identity,arms)
            if recorded:
                played=[[n for n in group if n in recorded] for group in batches]
                played=[group for group in played if group]
                if played:
                    seeds=played[-1]
                    row['batch']=dict(number=len(played),total=len(played),seeds=seeds)
                    row['seed_catalog']=[n for group in played for n in group]
        latest=row['latest'] or {}
        if type(latest.get('total_games')) is int and type(latest.get('reused_games')) is int and latest.get('games') is not None:
            # The journal counts newly played games separately from imported paired runs.
            row.update(new_games=latest['games'],reused_games=latest['reused_games'],
                       games=latest['games']+latest['reused_games'],total_games=latest['total_games'])
        elif latest.get('games') is not None and latest.get('total',0)>=latest['games'] and latest.get('total',0)>0:
            row.update(games=latest['games'],total_games=latest['total'])
        elif row['status']=='completed' and latest.get('stage','').startswith('screened_out') and latest.get('games') is not None:
            row.update(games=latest['games'],total_games=latest['games'])
        if latest.get('reference_games_reused') and not reused:
            row['reference_games_reused']=latest['reference_games_reused']
        prior_seeds=self.reuse.seed_catalog(identity,arms,seed_catalog)
        if prior_seeds:
            row['seed_catalog']=list(dict.fromkeys([*prior_seeds,*row['seed_catalog']]))
        for seed in seeds:
            row['episodes'].extend(self.moves_for_seed(arms,seed,now))
        if row['status'] not in ('unknown','waiting','stale'):
            playing=row['status']=='running' and row['phase'] in ('gameplay','episode_completed','screening_completed')
            active=[move for move in row['episodes'] if move['stop'] is None] if playing else []
            row['activity']=dict(seeds=len({move['seed'] for move in active}),
                                 models=len({move['arm'] for move in active}),games=len(active))
        if row['status']=='completed':
            self.completed_cache=now,signature,row
        return row

    def moves_for_seed(self,arms,seed,now):
        return [move for arm in arms if (move:=self.latest_move(arm,seed,now)) is not None]

    def finished_seeds(self,identity,arms):
        if not isinstance(identity,str) or not SHA.fullmatch(identity):
            return None
        path=self.directory/'result.json'
        try:
            info=path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size>4*1024*1024 or path.resolve()!=path:
                return None
            key=(identity,info.st_dev,info.st_ino,info.st_mtime_ns,info.st_size)
            if self.finished_seeds_cache and self.finished_seeds_cache[0]==key:
                return self.finished_seeds_cache[1]
            with open_watch_file(path,info) as stream:
                result=json.loads(stream.read(info.st_size))
            if not isinstance(result,dict) or result.get('identity')!=identity or not isinstance(result.get('results'),dict):
                return None
            seeds={record['seed'] for arm in arms for record in result['results'].get(arm,[])
                   if isinstance(record,dict) and record.get('arm')==arm and record.get('run_identity')==identity
                   and type(record.get('seed')) is int}
            self.finished_seeds_cache=key,seeds
            return seeds
        except (OSError,ValueError,TypeError,RuntimeError):
            return None

    def seed_episodes(self,seed,now):
        if self.path_guard is not None and not self.path_guard(self.path):
            return None
        arms,batches,identity=self.inputs()
        if any(seed in group for group in batches):
            return self.moves_for_seed(arms,seed,now)
        return self.reuse.seed_episodes(identity,arms,[n for group in batches for n in group],seed,now)

    def replay_sources(self,seed,now):
        if self.path_guard is not None and not self.path_guard(self.path):
            return None
        arms,batches,identity=self.inputs()
        seeds=[n for group in batches for n in group]
        if seed not in seeds:
            return self.reuse.replay_sources(identity,arms,seeds,seed,now)
        reused='official' in arms and bool(self.reused_baseline(identity,len(seeds)))
        return [(arm,self.directory/arm/str(seed)/'moves.jsonl',arm=='official' and reused,None) for arm in arms]

    def inputs(self):
        # Only use arm names and development seeds, never checkpoint paths.
        path=self.directory/'resolved-config.json'
        info=path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size>MAX_WATCH_BYTES or path.resolve()!=path:
            raise ValueError('Invalid evaluation manifest file')
        key=(info.st_dev,info.st_ino,info.st_mtime_ns,info.st_size)
        if self.inputs_cache is not None and self.inputs_cache[0]==key:
            return self.inputs_cache[1]
        with open_watch_file(path,info) as stream:
            config=json.loads(stream.read(MAX_WATCH_BYTES+1))
        if not isinstance(config,dict) or config.get('schema_version')!=1:
            raise ValueError('Invalid evaluation manifest')
        arms,seeds=config.get('checkpoints'),config.get('seeds')
        settings=config.get('config')
        size=settings.get('seed_group_size') if isinstance(settings,dict) else None
        screening=config.get('screening_seeds')
        if (not isinstance(arms,dict) or not 1<=len(arms)<=64
                or not all(GAMEPLAY_ARM.fullmatch(arm) for arm in arms)
                or not isinstance(seeds,list) or not 1<=len(seeds)<=10000
                or not all(type(seed) is int and 0<=seed<2**53 for seed in seeds)
                or len(set(seeds))!=len(seeds)
                or type(size) is not int or not 1<=size<=32
                or not isinstance(screening,list) or seeds[:len(screening)]!=screening):
            raise ValueError('Invalid evaluation arms, seeds or batch size')
        # The screening prefix and the remaining seeds are batched separately.
        batches=[block[start:start+size] for block in (seeds[:len(screening)],seeds[len(screening):])
                 for start in range(0,len(block),size)]
        inputs=sorted(arms,key=lambda arm:arm!='official'),batches,config.get('identity')
        self.inputs_cache=key,inputs
        self.reuse_cache=None
        self.candidate_count_cache=None
        self.move_cache.clear()
        return inputs

    def reused_baseline(self,identity,total_seeds):
        if not isinstance(identity,str) or not SHA.fullmatch(identity):
            return 0
        path=self.directory/'baseline-reuse.json'
        try:
            info=path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size>MAX_WATCH_BYTES:
                return 0
            key=(info.st_dev,info.st_ino,info.st_mtime_ns,info.st_size)
            if self.reuse_cache is not None and self.reuse_cache[0]==key:
                return self.reuse_cache[1]
            with open_watch_file(path,info) as stream:
                metadata=json.loads(stream.read(MAX_WATCH_BYTES+1))
            reused=0
            if (isinstance(metadata,dict) and metadata.get('schema_version')==1
                    and metadata.get('identity')==identity and metadata.get('arm')=='official'
                    and type(metadata.get('reused_games')) is int and metadata['reused_games']==total_seeds):
                reused=total_seeds
            self.reuse_cache=key,reused
            self.candidate_count_cache=None
            return reused
        except (OSError,ValueError,RecursionError):
            return 0

    def completed_candidates(self,arms,batches,row):
        # A new control record or progress log refreshes this fallback, not every SSE sample.
        key=self.cache_key if row['latest'] else None
        if self.candidate_count_cache is not None and self.candidate_count_cache[0]==key:
            return self.candidate_count_cache[1]
        count=0
        for arm in arms:
            if arm=='official':
                continue
            for seeds in batches:
                for seed in seeds:
                    path=self.directory/arm/str(seed)/'result.json'
                    try:
                        if stat.S_ISREG(path.lstat().st_mode) and path.resolve()==path:
                            count+=1
                    except (OSError,RuntimeError):
                        continue
        self.candidate_count_cache=key,count
        return count

    def latest_move(self,arm,seed,now):
        path=self.directory/arm/str(seed)/'moves.jsonl'
        try:
            info=path.lstat()
            if not stat.S_ISREG(info.st_mode) or not 0<=info.st_mtime<=now+300:
                return None
            key=(info.st_dev,info.st_ino,info.st_mtime_ns,info.st_size)
            cached=self.move_cache.get((arm,seed))
            if cached is not None and cached[0]==key:
                return cached[1]
            if path.resolve()!=path:
                return None
            with open_watch_file(path,info) as stream:
                offset=max(0,info.st_size-MOVE_TAIL_BYTES)
                stream.seek(offset)
                tail=stream.read(info.st_size-offset)
            # A poll can land during an append. Use only newline-terminated moves.
            lines=tail.split(b'\n')
            if len(lines)<(3 if offset else 2):
                return None
            item=json.loads(lines[-2])
            if (not isinstance(item,dict) or item.get('seed')!=seed
                    or item.get('stop',False) not in (None,'target','terminal','capped')
                    or not all(safe_board(item.get(key)) for key in ('before','after'))
                    or not all(type(item.get(key)) is int and 0<=item[key]<2**53 for key in ('turn','score','max_tile'))):
                return None
            probabilities,action=item.get('probabilities'),item.get('action')
            options=item.get('legal_moves')
            if (not isinstance(options,list) or not options or not set(options)<=set(DIRS)
                    or not isinstance(probabilities,dict) or set(probabilities)!=set(options)
                    or not all(measured(p) and 0<=p<=1 for p in probabilities.values())
                    or abs(sum(probabilities.values())-1)>1e-5
                    or action!=max(options,key=probabilities.__getitem__)):
                return None
            move=dict(arm=arm,seed=seed,updated_at=utc(info.st_mtime),
                        **{key:item[key] for key in ('turn','before','after','probabilities','action','score','max_tile','stop')})
            self.move_cache[arm,seed]=key,move
            return move
        except (OSError,ValueError,RecursionError,RuntimeError,TypeError,OverflowError):
            return None

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
    for watch in server.current_watches():
        row=watch.snapshot(now)
        if (server.training_boards is not None and row['kind']=='training'
                and 'training_step' not in row and watch.path.name=='progress.jsonl'):
            fits=(item for item in reversed(row['history']) if item['phase']=='lora_fit' and item.get('optimizer_step',0)>0)
            last_fit=next(fits,None)
            if last_fit is not None:
                batch=server.training_boards.batch(watch.path.parent,last_fit)
                if batch is not None:
                    batch['timestamp']=last_fit['timestamp']
                    previous=next((item for item in fits if item.get('trial')==last_fit.get('trial')),None)
                    if previous is not None:
                        # The legacy board reader verified this trainer's fixed accumulation schedule.
                        rate=estimate_batch_pace(last_fit,previous,len(batch['samples']),ACCUMULATION,now)
                        if rate is not None:
                            batch.update(boards_per_second=rate,pace_source='estimated')
                    row['training_batch']=batch
        channels.append(row)
    return dict(timestamp=utc(now),channels=channels)

class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 12
    def __init__(self, port, live, index, watches=(), play=None, replays=None, training_root=None):
        super().__init__(('127.0.0.1', port), Handler)
        self.live = live
        self.play = play
        self.replays = replays
        self.gameplay_replay = GameplayReplay()
        self.runs_root=training_root
        self.watches = tuple(self.with_gameplay(w) for w in watches)
        self.training_discovery = (RunDiscovery(training_root, excluded=(w.path for w in self.watches))
                                   if training_root is not None else None)
        self.training_boards = TrainingBoards(training_root) if training_root is not None else None
        self.evaluation_catalog = EvaluationCatalog(training_root) if training_root is not None else None
        self.auto_watches = {}
        index = Path(index)
        self.page_root = index.parent.resolve()
        self.pages = {'/': index.name, '/console': 'console.html', '/replay': 'replay.html'}
        self.archive = TrainingArchive(boards=self.training_boards)
        self.studio = index.name=='replay.html'
        self.studio_assets = {f'/launcher/studio-{part}.{ext}': ('text/javascript' if ext=='js' else 'text/css')
                              for part in ('training','monitor') for ext in ('js','css')}
        self.studio_assets['/launcher/studio-import-library.js'] = 'text/javascript'
        self.studio_assets['/launcher/studio-compare.js'] = 'text/javascript'
        self.studio_assets['/launcher/play-model-info.js'] = 'text/javascript'
        self.streams = threading.BoundedSemaphore(6)
        self.hosts = {f'127.0.0.1:{self.server_port}', f'localhost:{self.server_port}'}
        self.channel_lock=threading.Lock()
        self.channel_sampled=0
        self.channel_state=None
        self.channel_message=b''
        self.closing=threading.Event()

    def current_watches(self):
        if self.training_discovery is None:
            return self.watches
        auto_watches={}
        for source_entry in self.training_discovery.sources():
            if len(source_entry)==3:
                key,name,path=source_entry
                kind='training'
            else:
                key,name,kind,path=source_entry
            watch=self.auto_watches.get(key)
            source=path/'progress.jsonl' if kind=='gameplay' else path
            if (watch is None or watch.path!=source or watch.base['name']!=name
                    or (kind=='gameplay')!=isinstance(watch,GameplayWatch)
                    or (kind!='gameplay' and watch.base['kind']!=kind)):
                guard=self.training_discovery.safe_progress
                watch=(GameplayWatch(key,name,path,path_guard=guard,runs_root=self.runs_root) if kind=='gameplay' else
                       Watch(key,f'{name}:{kind}:{path}',path_guard=guard))
            auto_watches[key]=watch
        self.auto_watches=auto_watches
        return (*self.watches,*auto_watches.values())

    def with_gameplay(self,watch):
        if isinstance(watch,GameplayWatch):
            watch.reuse=GameplayReuse(self.runs_root,watch.directory)
            return watch
        if watch.base['kind']!='evaluation' or watch.path.name!='progress.jsonl':
            return watch
        richer=GameplayWatch(watch.base['id'].removeprefix('watch-'),watch.base['name'],watch.path.parent,
                             path_guard=watch.path_guard,runs_root=self.runs_root)
        try:
            richer.inputs()
        except (ValueError,OSError,RuntimeError):
            return watch
        return richer

    def find_watch(self,channel):
        with self.channel_lock:
            return next((watch for watch in self.current_watches() if watch.base['id']==channel),None)

    def channel_update(self):
        # All browser connections share one bounded local sampling pass.
        with self.channel_lock:
            now=time.monotonic()
            if self.channel_state is None or now-self.channel_sampled>=MONITOR_INTERVAL:
                self.channel_sampled=now
                updated=channel_snapshot(self)
                if updated['channels']!=self.channel_state:
                    self.channel_state=updated['channels']
                    self.channel_message=f"event: channels\ndata: {json.dumps(updated,allow_nan=False)}\n\n".encode()
            return self.channel_message

    def server_close(self):
        self.closing.set()
        super().server_close()

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def allowed_host(self):
        return self.headers.get('Host') in self.server.hosts
    def body(self, val, code=200, mime='application/json; charset=utf-8', *, etag=None):
        if isinstance(val, dict):
            val = json.dumps(val, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-type', mime)
        self.send_header('Content-Length', str(len(val)))
        self.send_header('Cache-Control', 'no-store')
        if etag is not None:
            self.send_header('ETag', etag)
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('X-Frame-Options', 'SAMEORIGIN')
        self.end_headers()
        self.wfile.write(val)
    def do_GET(self):
        route,_,query=self.path.partition('?')
        origin = self.headers.get('Origin')
        same_origin = origin is None or origin == 'http://' + self.headers.get('Host', '')
        if not self.allowed_host() or not same_origin or self.headers.get('Sec-Fetch-Site') == 'cross-site':
            self.body({'error':'Forbidden'}, 403)
        elif self.path == '/console' and self.server.live is None:
            self.body({'error':'Live console disabled'},404)
        elif self.path in self.server.pages or (route in ('/','/replay') and self.server.studio):
            path = self.server.page_root / self.server.pages[route]
            if path.is_file() and path.resolve().parent == self.server.page_root:
                self.body(path.read_bytes(), mime='text/html; charset=utf-8')
            else:
                self.body({'error':'Preview HTML not present'}, 404)
        elif self.path in self.server.studio_assets:
            path=self.server.page_root/self.path[1:]
            if path.is_file() and path.resolve()==path:
                self.body(path.read_bytes(),mime=self.server.studio_assets[self.path]+'; charset=utf-8')
            else:
                self.body({'error':'工作台资源尚未安装。'},404)
        elif route in ('/training/batches','/training/batch'):
            self.training_archive(query,detail=route=='/training/batch')
        elif self.path == '/channels':
            with self.server.channel_lock:
                snapshot=channel_snapshot(self.server)
            self.body(snapshot)
        elif self.path == '/channels/events':
            self.sse(self.channel_events)
        elif route in ('/gameplay/seed','/gameplay/replay'):
            self.gameplay_seed(query,replay=route=='/gameplay/replay')
        elif self.path == '/replay/catalog':
            self.replay_catalog()
        elif self.path == '/replay/comparisons':
            self.replay_comparisons()
        elif self.path.partition('?')[0] == '/replay/record':
            self.replay_record(self.path.partition('?')[2])
        elif self.path in ('/play/state','/play/events') and self.server.play is None:
            self.body({'error':'当前服务未启用即时游玩。'},404)
        elif self.path == '/play/state':
            self.body(self.server.play.latest())
        elif self.path == '/play/events':
            self.sse(self.play_events)
        elif self.path.partition('?')[0] == '/play/frame':
            self.play_frame(self.path.partition('?')[2])
        elif self.path in ('/state','/events') and self.server.live is None:
            self.body({'error':'Live console disabled'},404)
        elif self.path == '/state':
            self.body(self.server.live.latest())
        elif self.path == '/events':
            self.sse(self.native_events)
        else:
            self.body({'error':'Not found'}, 404)

    def gameplay_seed(self, query, *, replay=False):
        try:
            fields=parse_qs(query,keep_blank_values=True,strict_parsing=True,max_num_fields=4 if replay else 2)
            if replay and set(fields)-{'channel','seed','start','limit'}:
                raise ValueError('Unknown query field')
            channel,=fields['channel']
            number,=fields['seed']
            if not re.fullmatch('[A-Za-z0-9_-]{1,80}',channel) or not re.fullmatch('[0-9]{1,16}',number):
                raise ValueError('Invalid channel or seed')
            seed=int(number)
            if seed>=2**53:
                raise ValueError('Seed is outside the supported range')
            if replay:
                offset,=fields.get('start',['1'])
                size,=fields.get('limit',[str(MAX_PAGE)])
                if not re.fullmatch('[1-9][0-9]{0,8}',offset) or not re.fullmatch('[1-9][0-9]{0,2}',size):
                    raise ValueError('Invalid replay page')
                start,limit=int(offset),int(size)
                if limit>MAX_PAGE:
                    raise ValueError('Replay page too large')
        except (KeyError,ValueError):
            self.body({'error':'测评来源、种子编号或步数范围无效。' if replay else '测评来源或种子编号无效。'},400)
            return
        watch=self.server.find_watch(channel)
        native=watch.native_replay if isinstance(watch,Watch) else None
        if not isinstance(watch,GameplayWatch) and native is None:
            self.body({'error':'没有这个整局测评来源。'},404)
            return
        try:
            now=time.time()
            if native is not None:
                if watch.path_guard is not None and not watch.path_guard(watch.path):
                    data=None
                else:
                    data=native.page(seed,start,limit,now) if replay else native.seed_episodes(seed,now)
            else:
                with self.server.channel_lock:
                    data=watch.replay_sources(seed,now) if replay else watch.seed_episodes(seed,now)
                if replay and data is not None:
                    data=self.server.gameplay_replay.page(data,seed,start,limit,now)
        except (OSError,ValueError,RuntimeError):
            self.body({'error':'测评配置暂时无法读取。'},503)
            return
        if data is None:
            self.body({'error':'这个种子不在本次测评中。'},404)
        elif replay:
            self.body(dict(channel=channel,**data))
        else:
            self.body(dict(channel=channel,seed=seed,episodes=data))

    def training_archive(self,query,*,detail):
        try:
            fields=parse_qs(query,keep_blank_values=True,strict_parsing=True,max_num_fields=2)
            if set(fields)!=({'channel','key'} if detail else {'channel'}):
                raise ValueError('invalid fields')
            channel,=fields['channel']
            if not re.fullmatch('[A-Za-z0-9_-]{1,80}',channel):
                raise ValueError('invalid source')
            if detail:
                key,=fields['key']
                if not re.fullmatch('[a-f0-9]{24}',key):
                    raise ValueError('invalid batch')
        except (KeyError,ValueError):
            self.body({'error':'训练来源或批次编号无效。'},400)
            return
        watch=self.server.find_watch(channel)
        if watch is None or watch.base['kind']!='training':
            self.body({'error':'未找到这项训练。'},404)
            return
        try:
            data=self.server.archive.batch(watch,key) if detail else self.server.archive.index(watch)
        except (OSError,RuntimeError,ValueError):
            self.body({'error':'训练记录暂时不可读。'},503)
            return
        self.body(data if data is not None else {'error':'该批次已不在当前日志中。'},
                  200 if data is not None else 404)

    def replay_comparisons(self):
        with self.server.channel_lock:
            watches = self.server.current_watches()
            now = time.time()
            channels = [watch.snapshot(now) if isinstance(watch, GameplayWatch) or watch.native_replay else watch.base
                        for watch in watches]
        index = self.server.evaluation_catalog
        evaluations = index.catalog(watches) if index is not None else ()
        self.body(build_comparison_catalog(watches, channels, evaluations, self.server.runs_root))

    def replay_catalog(self):
        library = self.server.replays
        evaluations = self.server.evaluation_catalog
        if library is None and evaluations is None:
            self.body({'error':'当前服务未启用对局存档。'},404)
            return
        catalog = library.catalog() if library is not None else dict(games=[], truncated=False)
        if evaluations is not None:
            with self.server.channel_lock:
                watches = self.server.current_watches()
            catalog['evaluations'] = evaluations.catalog(watches)
        raw = json.dumps(catalog, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        etag = '"' + hashlib.sha256(raw).hexdigest() + '"'
        if self.headers.get('If-None-Match') == etag:
            self.send_response(304)
            self.send_header('ETag', etag)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
        else:
            self.body(raw, etag=etag)

    def replay_record(self, query):
        if self.server.replays is None:
            self.body({'error':'当前服务未启用对局存档。'},404)
            return
        try:
            fields = parse_qs(query, keep_blank_values=True, strict_parsing=True, max_num_fields=1)
            identifier, = fields['id']
            record = self.server.replays.record(identifier)
        except (KeyError, ValueError):
            self.body({'error':'对局编号无效。'},400)
            return
        if record is None:
            self.body({'error':'这份对局存档不存在或无法读取。'},404)
        else:
            self.body(record)

    def play_frame(self, query):
        if self.server.play is None:
            self.body({'error':'当前服务未启用即时游玩。'},404)
            return
        try:
            fields=parse_qs(query,keep_blank_values=True,strict_parsing=True,max_num_fields=2)
            if set(fields)!={'id','turn'}:
                raise ValueError('Expected playback id and turn')
            identifier,=fields['id']
            turn,=fields['turn']
            if not re.fullmatch('[0-9a-f]{32}',identifier) or not re.fullmatch('[0-9]{1,6}',turn):
                raise ValueError('Invalid playback id or turn')
            self.body(self.server.play.frame(identifier,int(turn)))
        except ValueError:
            self.body({'error':'回看步数或对局编号无效。'},400)
        except PlayRequestError as e:
            self.body({'error':e.message},e.code)

    def sse(self, send_events):
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
            send_events()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            pass
        finally:
            self.server.streams.release()

    def channel_events(self):
        self.wfile.write(b'retry: 1000\n\n')
        previous=None
        heartbeat=time.monotonic()
        while not self.server.closing.is_set():
            message=self.server.channel_update()
            now=time.monotonic()
            if message is not previous:
                self.wfile.write(message)
                self.wfile.flush()
                previous=message
                heartbeat=now
            elif now-heartbeat>=10:
                self.wfile.write(b': heartbeat\n\n')
                self.wfile.flush()
                heartbeat=now
            self.server.closing.wait(MONITOR_INTERVAL)

    def native_events(self):
        rid, sep, s = self.headers.get('Last-Event-ID','').rpartition(':')
        seq = int(s) if sep and s.isascii() and s.isdigit() and len(s) <= 6 else -1
        while not self.server.closing.is_set():
            rows = self.server.live.pending(rid, seq)
            if not rows:
                self.wfile.write(b': heartbeat\n\n')
            for row in rows:
                rid, seq = row['run_id'], row['seq']
                msg = f'id: {rid}:{seq}\nevent: frame\ndata: {json.dumps(row, allow_nan=False)}\n\n'
                self.wfile.write(msg.encode())
            self.wfile.flush()

    def play_events(self):
        revision=-1
        self.wfile.write(b'retry: 1000\n\n')
        while not self.server.closing.is_set():
            state=self.server.play.pending(revision,wait=5)
            if state is None:
                self.wfile.write(b': heartbeat\n\n')
            else:
                revision=state['revision']
                self.wfile.write(f'event: play\ndata: {json.dumps(state,allow_nan=False)}\n\n'.encode())
            self.wfile.flush()

    def reject(self, message, code):
        self.close_connection = True
        self.body({'error':message},code)

    def do_POST(self):
        if (not self.allowed_host() or self.headers.get('Origin') != 'http://' + self.headers.get('Host','')
                or self.headers.get('Sec-Fetch-Site') == 'cross-site'):
            self.reject('Same-origin request required',403)
            return
        play_method=PLAY_COMMANDS.get(self.path)
        if play_method and self.server.play is None:
            self.reject('当前服务未启用即时游玩。',403)
            return
        if not play_method and self.server.live is None:
            self.reject('Read-only monitoring mode',403)
            return
        if not play_method and self.path not in ('/start','/pause','/resume','/stop'):
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
            if play_method:
                row = getattr(self.server.play,play_method)(raw)
            else:
                target = self.path[1:]
                row = self.server.live.start(raw) if target == 'start' else self.server.live.command(target,raw)
            self.body(row,202)
        except (ValueError, UnicodeError, TimeoutError):
            self.reject('Invalid JSON request',400)
        except (BadRequest,PlayRequestError) as e:
            self.reject(e.message,e.code)

    def log_message(self, pattern, *args):
        if self.path not in ('/events','/channels/events','/play/events'):
            super().log_message(pattern,*args)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--base', type=Path, default=ROOT/'assets'/'qwen-base')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'assets'/'official-kev')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--threads', type=int, choices=(1,2,4), default=2)
    parser.add_argument('--out', type=Path, default=ROOT/'runs'/'live')
    parser.add_argument('--play-out', type=Path, default=ROOT/'runs'/'instant-play',
                        help='Local directory for new interactive game replays')
    parser.add_argument('--teacher-preparation', type=Path,
                        help='Use the search teacher from this calibrated preparation directory')
    parser.add_argument('--watch',action='append',default=[],metavar='NAME:KIND:PATH',
                        help='Watch a log/result file or use KIND=gameplay to watch an evaluation directory')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--monitor-only',action='store_true',help='Disable interactive play')
    mode.add_argument('--enable-live-console',action='store_true',help='Opt in to the separate local inference console')
    args = parser.parse_args()
    if len(args.watch)>MAX_WATCHES:
        parser.error(f'At most {MAX_WATCHES} watch files')
    try:
        watches=[]
        for i,s in enumerate(args.watch,1):
            parts=s.split(':',2)
            if len(parts)==3 and parts[1]=='gameplay' and parts[2]:
                watches.append(GameplayWatch(i,parts[0],parts[2]))
            else:
                watches.append(Watch(i,s))
    except ValueError as e:
        parser.error(str(e))
    if not args.enable_live_console:
        play=None
        if not args.monitor_only:
            from play_models import ModelLibrary
            library=ModelLibrary(args,ROOT/'runs')
            play=PlaySession(args.play_out,library.models,library.load,move_interval=.30,refresh_models=library.refresh)
        serve(args,None,watches,play)
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

def serve(args,app,watches,play=None):
    with Server(args.port, app, HERE/'replay.html', watches, play=play, replays=ReplayLibrary(args.play_out),
                training_root=ROOT/'runs') as server:
        print(f'http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
