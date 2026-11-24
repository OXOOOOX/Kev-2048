"""Backend black-box tests; optional actual native smoke with --native URL."""
import http.client
import json
import tempfile
import time
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode, urlsplit

from kev2048_release.training_progress import TrainingProgress
from live import DIRS, Game, GameplayWatch, Live, PlaySession, Server, Watch, legal, nested, utc, MAX_WATCH_BYTES
from replay_library import ReplayLibrary
from run_discovery import RunDiscovery
from training_boards import TRAINING_SCHEMA
from training_discovery import TrainingDiscovery

def getconn(addr):
    u=urlsplit(addr)
    return http.client.HTTPConnection(u.hostname,u.port,timeout=25)

def req(addr,path,body=None,origin=None):
    c=getconn(addr)
    hdrs={'Origin':origin or addr,'Content-Type':'application/json'} if body is not None else {}
    c.request('POST' if body is not None else 'GET',path,body=None if body is None else json.dumps(body),headers=hdrs)
    r=c.getresponse();data=json.loads(r.read()); c.close();return r.status,data

def events(addr,last=''):
    c=getconn(addr)
    c.request('GET','/events',headers={'Last-Event-ID':last})
    res=c.getresponse()
    assert res.status==200 and res.getheader('Content-Type').startswith('text/event-stream')
    while True:
        line=res.readline()
        if not line:break
        if line.startswith(b'data: '):yield json.loads(line[6:])
    c.close()

def validate(addr, start, seed,cap):
    g=Game.seeded(seed)
    ev=events(addr); check=[]; seq=-1
    for r in ev:
        assert r['run_id']==start['run_id'] and r['seq']==seq+1
        seq=r['seq']
        if r['type']=='move':
            assert r['before']==nested(g.state) and list(r['probabilities'])==legal(g.state)
            assert r['action']==max(r['probabilities'],key=r['probabilities'].get)
            gain,spawn=g.step(r['action'])
            assert (r['gained'],r['spawn'],r['turn'],r['score'],r['tile'],r['board'])==(gain,spawn,g.turn,g.score,max(g.state),nested(g.state))
            assert r['decision_ms']>=0
            check.append(r)
        if r['type'] in ('cap','terminal','error'):
            assert r['type']=='cap' and g.turn==cap and r==req(addr,'/state')[1]
            ev.close(); return check
    raise AssertionError('No final status')

class Tests(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory()
        self.run=threading.Event()
        self.entered=threading.Event()
        self.calls=[]
        def decide(board,opts):
            self.calls.append(1)
            self.entered.set()
            if not self.run.wait(5):raise RuntimeError('test timeout')
            return dict(zip(opts,[1]+[0]*(len(opts)-1))),0.1
        self.live=Live(lambda:decide,self.dir.name,'test','fixture')
        self.s=Server(0,self.live,Path(self.dir.name)/'index.html')
        self.work=threading.Thread(target=self.s.serve_forever,daemon=True); self.work.start()
        self.url=f'http://127.0.0.1:{self.s.server_port}'
    def tearDown(self):
        self.run.set()
        self.s.shutdown()
        self.s.server_close()
        self.dir.cleanup()
    def test_static_route_allowlist(self):
        d=Path(self.dir.name)
        for filename,text in [('index.html','launcher'),('console.html','console'),('replay.html','replay'),
                              ('secret.jsonl','secret')]:
            (d/filename).write_text(text,encoding='utf-8')
        def raw(path,headers=None):
            c=getconn(self.url)
            c.request('GET',path,headers=headers or {})
            r=c.getresponse()
            ans=(r.status,r.headers,r.read())
            c.close()
            return ans
        for path,text in [('/','launcher'),('/console','console'),('/replay','replay')]:
            code,h,b=raw(path)
            self.assertEqual((code,b),(200,text.encode()))
            self.assertEqual(h.get('Cross-Origin-Resource-Policy'),'same-origin')
            self.assertEqual(h.get('X-Frame-Options'),'SAMEORIGIN')
            self.assertTrue(h.get('Content-Type').startswith('text/html'))
        for path in ['/console.html','/index.html','/secret.jsonl','/../index.html','/console?file=secret.jsonl','/runs/foo.jsonl','/trace']:
            self.assertEqual(raw(path)[0],404,path)
        for bad in [{'Host':'evil.invalid'},{'Origin':'http://evil.invalid'},{'Sec-Fetch-Site':'cross-site'}]:
            self.assertEqual(raw('/console',bad)[0],403)
        self.assertEqual(raw('/state',{'Host':'evil.invalid'})[0],403)
        self.assertEqual(raw('/events',{'Origin':'http://evil.invalid'})[0],403)
        (d/'replay.html').unlink()
        self.assertEqual(raw('/replay')[0],404)

    def test_cap_sse_replay_and_validation(self):
        self.run.set()
        self.assertEqual(req(self.url,'/state')[1]['type'],'ready')
        self.assertEqual(req(self.url,'/start',{'cap':5},origin='http://evil.invalid')[0],403)
        self.assertEqual(req(self.url,'/start',{'cap':False})[0],400)
        self.assertEqual(req(self.url,'/start',{'cap':121})[0],400)
        code,start=req(self.url,'/start',{'cap':4,'seed':17})
        self.assertEqual(code,202)
        x=validate(self.url,start,17,4)
        self.assertEqual(x,[json.loads(s) for s in (Path(self.dir.name)/(start['run_id']+'.jsonl')).read_text().splitlines()])
        ev=events(self.url,f'{start["run_id"]}:{x[-1]["seq"]}')
        self.assertEqual(next(ev)['type'],'cap')
        ev.close()

    def test_instant_play_routes_are_separate_and_push_updates(self):
        self.assertEqual(req(self.url,'/play/state')[0],404)
        self.assertEqual(req(self.url,'/play/frame')[0],404)
        self.assertEqual(req(self.url,'/replay/catalog')[0],404)
        self.assertEqual(req(self.url,'/play/start',{'models':['official']})[0],403)
        self.assertEqual(req(self.url,'/play/save',{'id':'unused'})[0],403)

        def load(_):
            def decide(requests):
                if requests[0][1].turn == 1:
                    self.entered.set()
                    if not self.run.wait(5):
                        raise RuntimeError('test timeout')
                result={}
                for name,game in requests:
                    options=legal(game.state)
                    result[name]=dict(probabilities=dict(zip(options,[1]+[0]*(len(options)-1))),decision_ms=1.)
                return result
            return decide

        playback = Path(self.dir.name)/'instant-play'
        self.s.play=PlaySession(playback,[{'id':'official','name':'Official','kind':'model'}],load)
        self.s.replays=ReplayLibrary(playback)
        self.assertEqual(req(self.url,'/play/start',{'models':['official']},origin='http://evil.invalid')[0],403)
        self.assertEqual(req(self.url,'/play/state')[1]['models'][0]['id'],'official')
        connection=getconn(self.url)
        self.addCleanup(connection.close)
        connection.request('GET','/play/events')
        stream=connection.getresponse()
        self.addCleanup(stream.close)
        self.assertTrue(stream.getheader('Content-Type').startswith('text/event-stream'))

        def snapshot():
            for line in stream:
                if line.startswith(b'data: '):
                    return json.loads(line[6:])
            self.fail('Missing live play snapshot')

        self.assertIsNone(snapshot()['session'])
        code,response=req(self.url,'/play/start',{'models':['official'],'seed':23,'cap':2})
        self.assertEqual(code,202)
        session=response['session']['id']
        self.assertTrue(self.entered.wait(5))
        for path in ('/play/pause','/play/resume','/play/save'):
            self.assertEqual(req(self.url,path,{'id':session},origin='http://evil.invalid')[0],403)
        code,paused=req(self.url,'/play/pause',{'id':session})
        self.assertEqual((code,paused['session']['status']),(202,'paused'))
        self.assertEqual(req(self.url,'/play/save',{'id':session})[0],409)
        path=f'/play/frame?id={session}&turn='
        for invalid in ('', '-1', '0&turn=2', '0&path=relative'):
            self.assertEqual(req(self.url,path+invalid)[0],400)
        code,frame=req(self.url,path+'0')
        self.assertEqual((code,frame['id'],frame['turn']),(200,session,0))
        self.assertEqual(frame['games'][0]['board'],nested(Game.seeded(23).state))
        code,frame=req(self.url,path+'1')
        self.assertEqual((code,frame['games'][0]['turn']),(200,1))
        self.assertIn('probabilities',frame['games'][0])
        code,resumed=req(self.url,'/play/resume',{'id':session})
        self.assertEqual((code,resumed['session']['status']),(202,'running'))
        self.run.set()
        for _ in range(10):
            current=snapshot()['session']
            if current and not current['active']:
                break
        self.assertEqual(current['id'],session)
        self.assertEqual((current['status'],current['games'][0]['turn']),('completed',2))
        self.assertFalse(current['saved'])
        archive=playback/session/'official'/'moves.jsonl'
        self.assertFalse(archive.exists())
        self.assertEqual(req(self.url,'/replay/catalog')[1]['games'],[])
        record=f'/replay/record?id={session}%2Fofficial'
        self.assertEqual(req(self.url,record)[0],404)
        self.assertEqual(req(self.url,'/replay/record?id=..%2F..%2Fsecret')[0],400)
        self.assertEqual(req(self.url,record+'&path=anything')[0],400)
        code,saved=req(self.url,'/play/save',{'id':session})
        self.assertEqual((code,saved['session']['saved']),(202,True))
        raw=archive.read_text(encoding='utf-8')
        self.assertEqual(len(raw.splitlines()),2)
        code,catalog=req(self.url,'/replay/catalog')
        self.assertEqual(code,200)
        self.assertEqual(len(catalog['games']),1)
        card=catalog['games'][0]
        self.assertEqual((card['id'],card['model'],card['seed'],card['turns'],card['starred']),
                         (f'{session}/official','Official',23,2,True))
        code,bundle=req(self.url,record)
        self.assertEqual((code,bundle['id'],bundle['manifest']['variant'],bundle['moves']),
                         (200,card['id'],'Official',raw))
        cross_site=getconn(self.url)
        try:
            cross_site.request('GET',record,headers={'Origin':'http://evil.invalid'})
            blocked=cross_site.getresponse()
            self.assertEqual(blocked.status,403)
            blocked.read()
        finally:
            cross_site.close()
        self.assertEqual(snapshot()['session']['saved'],True)
        self.assertEqual(req(self.url,'/play/stop',{'id':session})[0],409)
        self.assertEqual(req(self.url,'/state')[1]['type'],'ready')
        self.assertNotIn(session,json.dumps(req(self.url,'/channels')[1]))
    def test_private_watchers_and_monitor_only(self):
        d=Path(self.dir.name)
        journal=d/'safe-progress.jsonl'
        static=d/'benchmark-result.json'
        unknown=d/'old-running.json'
        waiting=d/'not-created.ndjson'
        wrong=d/'oversized.json'
        symlink=d/'link.json'
        now=time.time()
        journal.write_text(json.dumps({'timestamp':utc(now-400),'status':'running','phase':'encode_private',
            'step':8,'total':128,'loss':0.42,'trial':'lr8e-5-kl015','dev_best_rate':0.5375,
            'private_path':str(d),'traceback':'secret','bad':123})+'\n{"status":',encoding='utf-8')
        static.write_text(json.dumps({'completed':True,'median_phase_seconds':{'cpu':{'head_fit_seconds':1.2}},
            'median_wall_seconds':{'cpu':21.3},'error':'secret path','phase':'verified_result'}),encoding='utf-8')
        unknown.write_text(json.dumps({'status':'running','timestamp':utc(now),'phase':'native','timing':{'head_fit_seconds':8.1}}),encoding='utf-8')
        with wrong.open('wb') as o:o.truncate(MAX_WATCH_BYTES+1)
        symlink.symlink_to(static)
        records=[('Real training','training',journal),('Real benchmark','benchmark',static),('Unproven static','training',unknown),('Pending','training',waiting),('Oversized','benchmark',wrong),('Link','benchmark',symlink)]
        w=[Watch(i,f'{name}:{kind}:{p}') for i,(name,kind,p) in enumerate(records,1)]
        m=Server(0,None,d/'index.html',w)
        work=threading.Thread(target=m.serve_forever,daemon=True);work.start()
        addr=f'http://127.0.0.1:{m.server_port}'
        try:
            n,r=req(addr,'/channels'); self.assertEqual(n,200)
            a=r['channels']; self.assertEqual(len(a),6)
            self.assertEqual([u['status'] for u in a],['stale','completed','unknown','waiting','unknown','unknown'])
            self.assertEqual(a[0]['latest']['step'],8); self.assertEqual(len(a[0]['history']),1)
            self.assertEqual(a[0]['latest']['trial'],'lr8e-5-kl015')
            self.assertEqual(a[0]['latest']['dev_best_rate'],0.5375)
            self.assertEqual(a[1]['timings'],{'cpu_median_head_fit_seconds':1.2,'cpu_median_wall_seconds':21.3})
            self.assertNotIn(str(d),json.dumps(r));self.assertNotIn('secret',json.dumps(r))
            self.assertNotIn('private_path',json.dumps(r));self.assertNotIn('traceback',json.dumps(r))
            self.assertEqual(req(addr,'/start',{'cap':2})[0],403)
            self.assertEqual(req(addr,'/events')[0],404)
            self.assertEqual(req(addr,'/console')[0],404)
            self.assertEqual(req(addr,'/state')[0],404)
            self.assertEqual(req(addr,'/channels?path=secret.json')[0],404)
            journal.write_text(''.join(json.dumps({'timestamp':utc(now),'status':'running','step':i,'total':60})+'\n' for i in range(60)),encoding='utf-8')
            b=req(addr,'/channels')[1]['channels'][0]
            self.assertEqual(b['status'],'running');self.assertEqual(b['latest']['step'],59)
            self.assertEqual(len(b['history']),48)
            journal.write_text('{"status":"running","phase":"read-no-clock"}\n',encoding='utf-8')
            self.assertEqual(req(addr,'/channels')[1]['channels'][0]['status'],'unknown')
            journal.write_text(''.join(json.dumps(row)+'\n' for row in [
                dict(phase='screening_completed',games=30,promote_to_100=False,candidate_vector=[0,0,0,1042.],threshold=[0,0,2,1548.]),
                dict(status='completed',phase='completed',stage='screened_out_30',games=30,reference_games_reused=90,dev_best_rate=2),
                dict(status='stopped',phase='verification_budget_stop',games=0),
            ]),encoding='utf-8')
            m.watches=(Watch(0,f'Candidate screening:inference:{journal}'),)
            channel=req(addr,'/channels')[1]['channels'][0]
            self.assertEqual((channel['kind'], channel['status']),('evaluation','completed'))
            gate,done=channel['history']
            self.assertEqual((gate['candidate_vector'],gate['threshold'],gate['promote_to_100']),([0,0,0,1042.],[0,0,2,1548.],False))
            self.assertEqual((done['stage'],done['reference_games_reused']),('screened_out_30',90))
            self.assertNotIn('dev_best_rate',done)
        finally:
            m.shutdown();m.server_close()

    def test_new_training_joins_existing_sse_and_reports_optimizer_updates(self):
        root=Path(self.dir.name)/'runs'
        root.mkdir()
        self.s.training_discovery=TrainingDiscovery(root,interval=0)
        batch_calls=[]

        def batch(directory,event):
            batch_calls.append(directory)
            return dict(trial=event['trial'], epoch=event['epoch'], optimizer_step=event['optimizer_step'],
                        steps=event['steps'], samples=[dict(row=i+1,board=nested(Game.seeded(1).state),
                             stage='early',source_policy='teacher',best_actions=['left']) for i in range(16)])

        self.s.training_boards=SimpleNamespace(batch=batch)
        connection=getconn(self.url)
        self.addCleanup(connection.close)
        connection.request('GET','/channels/events')
        stream=connection.getresponse()
        self.addCleanup(stream.close)

        def snapshot():
            for line in stream:
                if line.startswith(b'data: '):
                    return json.loads(line[6:])['channels']
            self.fail('Missing monitoring snapshot')

        self.assertFalse(any(row['kind']=='training' for row in snapshot()))
        run=root/'new-training'
        run.mkdir()
        (run/'resolved-config.json').write_text(json.dumps({'training_scope':'pointer_head_only',
            'config':{'train_file':'train.jsonl','trials':[{}]}}),encoding='utf-8')
        journal=run/'progress.jsonl'
        now=time.time()
        first=dict(timestamp=utc(now-150),status='running',phase='lora_fit',trial='lora',epoch=1,
                   optimizer_step=70,steps=125,mean_training_loss=1.3897014)
        journal.write_text(json.dumps(first)+'\n',encoding='utf-8')
        enrolled=next(row for row in snapshot() if row['kind']=='training')
        self.assertEqual((enrolled['status'],enrolled['stale_after_seconds']),('running',360))
        self.assertEqual(tuple(enrolled['latest'][key] for key in ('optimizer_step','steps','mean_training_loss')),
                         (70,125,1.3897014))
        self.assertEqual(enrolled['training_batch']['optimizer_step'],70)
        self.assertNotIn('boards_per_second',enrolled['training_batch'])
        self.assertEqual(batch_calls[-1],run)
        with journal.open('a',encoding='utf-8') as stream_file:
            stream_file.write(json.dumps(dict(first,timestamp=utc(now),optimizer_step=80,
                                              mean_training_loss=1.3595210))+'\n')
        update=next(row for row in snapshot() if row['id']==enrolled['id'])
        self.assertEqual((update['latest']['optimizer_step'],update['latest']['steps']),(80,125))
        self.assertEqual(update['latest']['mean_training_loss'],1.3595210)
        self.assertEqual(update['training_batch']['optimizer_step'],80)
        self.assertEqual(update['training_batch']['pace_source'],'estimated')
        self.assertAlmostEqual(update['training_batch']['boards_per_second'],160/150)
        with journal.open('a',encoding='utf-8') as stream_file:
            stream_file.write(json.dumps(dict(timestamp=utc(time.time()),status='running',phase='dev_encoding'))+'\n')
        during_validation=next(row for row in snapshot() if row['id']==enrolled['id'])
        self.assertEqual(during_validation['phase'],'dev_encoding')
        self.assertEqual(during_validation['training_batch']['optimizer_step'],80)

    def test_shared_publisher_discovers_custom_training_and_serves_its_batch(self):
        root = Path(self.dir.name) / 'runs'
        root.mkdir()
        self.s.training_discovery = RunDiscovery(root, interval=0)
        run = root / 'custom-training'
        board = Game.seeded(1).state
        moves = legal(board)
        row = dict(board=nested(board), valid_moves=moves, best_actions=[moves[-1]])

        def snapshot(name='custom-training'):
            status, payload = req(self.url, '/channels')
            self.assertEqual(status, 200)
            return next(channel for channel in payload['channels'] if channel['name'] == name)

        with TrainingProgress(run) as progress:
            self.assertEqual(snapshot()['status'], 'loading')
            progress.emit_cache(split='train', done=1, total=1)
            self.assertEqual(snapshot()['phase'], 'train_cache')
            for step in (1, 2):
                progress.emit_optimizer(trial='custom', epoch=1, optimizer_step=step, steps=2,
                                        mean_training_loss=1 / step, samples_seen=step,
                                        batch_rows=[row], batch_indices=[3],
                                        pre_update_logits=[[0.] * (len(moves) - 1) + [float(step)]])
                channel = snapshot()
                self.assertEqual(channel['training_step']['optimizer_step'], step)
            self.assertGreater(channel['training_batch']['boards_per_second'], 0)

        channel = snapshot()
        self.assertEqual(channel['status'], 'completed')
        query = {'channel': channel['id']}
        status, archive = req(self.url, '/training/batches?' + urlencode(query))
        self.assertEqual(status, 200)
        self.assertEqual([batch['step'] for batch in archive['batches']], [1, 2])
        query['key'] = archive['batches'][0]['key']
        status, batch = req(self.url, '/training/batch?' + urlencode(query))
        self.assertEqual(status, 200)
        sample, = batch['samples']
        self.assertEqual((sample['row'], sample['board'], sample['best_actions']), (4, row['board'], row['best_actions']))
        self.assertEqual(sample['candidate']['choice'], moves[-1])
        probabilities = sample['candidate']['probabilities']
        self.assertAlmostEqual(sum(probabilities[d] for d in moves), 1.)
        self.assertTrue(all(probabilities[d] is None for d in DIRS if d not in moves))
        latest = channel['training_batch']['samples'][0]['candidate']['probabilities']
        self.assertGreater(latest[moves[-1]], probabilities[moves[-1]])

        for exception, state in ((RuntimeError, 'error'), (KeyboardInterrupt, 'stopped')):
            with self.assertRaises(exception), TrainingProgress(root / f'custom-{state}'):
                raise exception('test lifecycle')
            self.assertEqual(snapshot(f'custom-{state}')['status'], state)

    def test_standard_journal_retains_last_step_and_real_samples_after_fitting(self):
        root=Path(self.dir.name)/'runs'
        journal=root/'native-stock'/'progress.jsonl'
        journal.parent.mkdir(parents=True)
        self.s.training_discovery=TrainingDiscovery(root,interval=0)
        board=Game.seeded(1).state
        moves=legal(board)
        candidate=dict(choice=moves[0],probabilities={d:1/len(moves) if d in moves else None for d in DIRS})
        sample=dict(row=4,board=nested(board),best_actions=moves[:2],candidate=candidate,private='do not expose')
        first=dict(schema=TRAINING_SCHEMA,kind='training',timestamp=utc(time.time()),status='running',
                   phase='optimizer',trial='stock',epoch=1,optimizer_step=1,steps=2,mean_training_loss=1.45,
                   batch_size=1,samples=[sample])

        def assert_monitor():
            channel=next(row for row in req(self.url,'/channels')[1]['channels'] if row['name']=='native-stock')
            self.assertEqual((channel['training_step']['optimizer_step'],channel['training_step']['mean_training_loss']), (2,1.25))
            self.assertEqual(channel['training_batch']['samples'],[{k:v for k,v in sample.items() if k!='private'}])
            self.assertEqual(channel['training_batch']['pace_source'],'measured')
            self.assertEqual(channel['training_batch']['boards_per_second'],12.5)
            self.assertNotIn('samples',channel['training_step'])
            self.assertTrue(all('samples' not in event for event in channel['history']))
            return channel

        latest=dict(first,optimizer_step=2,mean_training_loss=1.25,boards_per_second=12.5,samples_seen=2)
        journal.write_text(json.dumps(first)+'\n'+json.dumps(latest)+'\n',encoding='utf-8')
        self.assertEqual(assert_monitor()['phase'],'optimizer')
        finished=dict(schema=TRAINING_SCHEMA,kind='training',timestamp=utc(time.time()),phase='completed',status='completed')
        with journal.open('a',encoding='utf-8') as stream:
            stream.write((json.dumps(dict(finished,phase='dev_encoding',status='running'))+'\n')*50)
            stream.write(json.dumps(finished)+'\n')
        self.assertEqual(assert_monitor()['status'],'completed')
        with journal.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(first,epoch=2,samples=[dict(sample,best_actions=['not a move'])]))+'\n')
        channel=next(row for row in req(self.url,'/channels')[1]['channels'] if row['name']=='native-stock')
        self.assertNotIn('training_batch',channel)

    def test_historical_batch_speed_uses_only_consistent_optimizer_intervals(self):
        now=time.time()
        board=Game.seeded(2).state
        samples=[dict(row=i,board=nested(board),best_actions=legal(board)[:1]) for i in range(1,5)]
        first=dict(schema=TRAINING_SCHEMA,kind='training',phase='optimizer',status='running',trial='arm',
                   epoch=1,optimizer_step=1,steps=3,timestamp=utc(now-10),batch_size=4,samples=samples)
        last=dict(first,optimizer_step=3,timestamp=utc(now),batch_size=2,samples=samples[:2])
        path=Path(self.dir.name)/'history.jsonl'
        def pace(*records):
            path.write_text(''.join(json.dumps(record)+'\n' for record in records),encoding='utf-8')
            batch=Watch(1,f'Historic:training:{path}').snapshot(now)['training_batch']
            return batch.get('pace_source'),batch.get('boards_per_second')

        self.assertEqual(pace(first,dict(first,trial='other'),last),('estimated',.6))
        self.assertEqual(pace(first,dict(last,optimizer_step=2)),('estimated',.2))
        self.assertEqual(pace(dict(first,samples_seen=4),dict(last,samples_seen=12)),('estimated',.8))
        for previous in (dict(first,trial='elsewhere'),dict(first,epoch=2),dict(first,timestamp=utc(now+1)),
                         dict(first,optimizer_step=3),dict(first,steps=4)):
            self.assertEqual(pace(previous,last),(None,None))
        self.assertEqual(pace(dict(first,elapsed_seconds=8),dict(last,elapsed_seconds=1)),(None,None))
        self.assertEqual(pace(dict(first,samples_seen=12),dict(last,samples_seen=4)),(None,None))
        self.assertEqual(pace(dict(first,samples_seen=4),dict(last,samples_seen=6)),(None,None))

    def test_long_training_feed_retains_latest_safe_row(self):
        path=Path(self.dir.name)/'large.progress.jsonl'
        first=dict(timestamp=utc(time.time()-400),status='running',phase='lora_fit',trial='lora',
                   optimizer_step=10,steps=125,mean_training_loss=1.4)
        row=(json.dumps(first)+'\n').encode()
        with path.open('wb') as stream:
            stream.write(row*(MAX_WATCH_BYTES//len(row)+1))
        watch=Watch(1,f'Long training:training:{path}')
        snapshot=watch.snapshot(time.time())
        self.assertEqual((snapshot['status'],snapshot['latest']['optimizer_step']),('stale',10))
        with path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(first,timestamp=utc(time.time()),optimizer_step=True,
                                         mean_training_loss='unknown'))+'\n')
        latest=watch.snapshot(time.time())['latest']
        self.assertNotIn('optimizer_step',latest)
        self.assertNotIn('steps',latest)
        self.assertNotIn('mean_training_loss',latest)

    def test_pause_stop_between_moves(self):
        code,start=req(self.url,'/start',{'cap':5})
        self.assertTrue(self.entered.wait(3))
        self.assertEqual(req(self.url,'/start',{'cap':3})[0],409)
        self.assertEqual(req(self.url,'/pause',{})[0],202)
        self.run.set()
        with self.live.cv:
            self.assertTrue(self.live.cv.wait_for(lambda:self.live.latest()['type']=='paused',3))
        self.assertEqual(req(self.url,'/state')[1]['turn'],1)
        self.assertEqual(len(self.calls),1)
        self.run.clear(); self.entered.clear()
        self.assertEqual(req(self.url,'/resume',{})[0],202)
        self.assertTrue(self.entered.wait(3))
        self.assertEqual(req(self.url,'/stop',{})[0],202)
        self.run.set()
        with self.live.cv:
            self.assertTrue(self.live.cv.wait_for(lambda:self.live.latest()['type']=='stopped',3))
        self.assertEqual(req(self.url,'/state')[1]['turn'],1)
        self.assertEqual(len((Path(self.dir.name)/(start['run_id']+'.jsonl')).read_text().splitlines()),1)

    def test_gameplay_watch_refreshes_complete_decisions(self):
        root=Path(self.dir.name)/'evaluation'
        root.mkdir()
        manifest=dict(schema_version=1,checkpoints={'official':'private-path','candidate':'private-path'},
                      seeds=[17,18,19,20],screening_seeds=[17,18,19],config={'seed_group_size':2})
        (root/'resolved-config.json').write_text(json.dumps(manifest),encoding='utf-8')
        now=time.time()
        journal=root/'progress.jsonl'
        def progress(**fields):
            with journal.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(timestamp=utc(now),**fields))+'\n')
        def move(game,seed):
            options=legal(game.state)
            before=nested(game.state)
            action=options[0]
            game.step(action)
            return dict(seed=seed,turn=game.turn,action=action,legal_moves=options,
                        before=before,after=nested(game.state),score=game.score,max_tile=max(game.state),
                        probabilities=dict(zip(options,[1]+[0]*(len(options)-1))),stop=None)
        game=Game.seeded(17)
        first=move(game,17)
        paths=[root/'official'/'17'/'moves.jsonl',root/'official'/'19'/'moves.jsonl']
        for path in paths:
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(first)+'\n{"turn":',encoding='utf-8')
        progress(phase='gameplay',status='running',games=1,decisions_per_second=5.125)
        self.s.watches=(GameplayWatch(1,'Evaluation',root),)
        def snapshot():
            status,body=req(self.url,'/channels')
            self.assertEqual(status,200)
            self.assertNotIn('private-path',json.dumps(body))
            return body['channels'][-1]
        channel=snapshot()
        self.assertEqual((channel['games'],channel['total_games']),(1,8))
        self.assertEqual(channel['arms'],['official','candidate'])
        self.assertEqual(channel['batch'],{'number':1,'total':3,'seeds':[17,18]})
        self.assertEqual(channel['activity'],{'seeds':1,'models':1,'games':1})
        self.assertEqual(channel['decisions_per_second'],5.125)
        self.assertEqual([(x['seed'],x['turn']) for x in channel['episodes']],[(17,1)])
        self.assertEqual(channel['episodes'][0]['probabilities'],first['probabilities'])
        second=move(game,17)
        paths[0].write_text('\n'.join(map(json.dumps,(first,second)))+'\n',encoding='utf-8')
        progress(phase='episode_completed',status='running',games=2)
        channel=snapshot()
        self.assertEqual(channel['games'],2)
        self.assertEqual(channel['decisions_per_second'],5.125)
        self.assertEqual(channel['episodes'][0]['before'],first['after'])
        self.assertEqual(channel['episodes'][0]['after'],second['after'])
        self.assertEqual(channel['episodes'][0]['turn'],2)
        terminal={**second,'stop':'terminal'}
        paths[0].write_text('\n'.join(map(json.dumps,(first,terminal)))+'\n',encoding='utf-8')
        paths[0].with_name('result.json').write_text('{}',encoding='utf-8')
        terminal_snapshot=snapshot()
        self.assertEqual(terminal_snapshot['episodes'][0]['stop'],'terminal')
        self.assertEqual(terminal_snapshot['activity'],{'seeds':0,'models':0,'games':0})
        progress(phase='gameplay',status='running',games=4)
        self.assertEqual(snapshot()['batch'],{'number':2,'total':3,'seeds':[19]})
        self.assertEqual(snapshot()['episodes'],[])
        progress(phase='screening_completed',status='running',games=6)
        self.assertEqual(snapshot()['batch'],{'number':3,'total':3,'seeds':[20]})
        progress(phase='completed',status='completed',games=8)
        self.assertEqual((snapshot()['status'],snapshot()['games']),('completed',8))

    def test_gameplay_catalog_reads_one_selected_seed_across_arms(self):
        root=Path(self.dir.name)/'evaluation'
        root.mkdir()
        seeds=[90,12,78,6]
        arms=['official','candidate-b','candidate-a']
        manifest=dict(schema_version=1,checkpoints=dict.fromkeys([arms[1],arms[0],arms[2]],'private-path'),
                      seeds=seeds,screening_seeds=seeds[:2],config={'seed_group_size':2})
        config=root/'resolved-config.json'
        config.write_text(json.dumps(manifest),encoding='utf-8')
        (root/'progress.jsonl').write_text(json.dumps(dict(timestamp=utc(time.time()),phase='gameplay',
                status='running',games=9))+'\n',encoding='utf-8')
        watch=GameplayWatch(1,'Evaluation',root)
        self.s.watches=(watch,Watch(2,f'Other:training:{root}/progress.jsonl'))
        moves=[]
        for i,arm in enumerate(arms):
            game=Game.seeded(12)
            options=legal(game.state)
            action=options[i%len(options)]
            before=nested(game.state)
            game.step(action)
            move=dict(seed=12,turn=game.turn,action=action,legal_moves=options,before=before,
                      after=nested(game.state),score=game.score,max_tile=max(game.state),
                      probabilities={option:int(option==action) for option in options},stop='capped')
            moves.append(move)
            folder=root/arm/'12'
            folder.mkdir(parents=True)
            (folder/'moves.jsonl').write_text(json.dumps(move)+'\n{"turn":',encoding='utf-8')
        channel=next(c for c in req(self.url,'/channels')[1]['channels'] if c['id']=='watch-1')
        self.assertEqual(channel['seed_catalog'],seeds)
        self.assertEqual(channel['batch']['seeds'],[78,6])
        self.assertEqual(channel['episodes'],[])
        self.assertEqual(watch.move_cache,{})
        route='/gameplay/seed?channel=watch-1&seed='
        code,detail=req(self.url,route+'12')
        self.assertEqual((code,detail['channel'],detail['seed']),(200,'watch-1',12))
        self.assertEqual([e['arm'] for e in detail['episodes']],arms)
        for episode,raw in zip(detail['episodes'],moves,strict=True):
            for key in ('seed','turn','action','before','after','score','max_tile','probabilities','stop'):
                self.assertEqual(episode[key],raw[key])
        self.assertNotIn('private-path',json.dumps(detail))
        self.assertEqual(set(watch.move_cache),{(arm,12) for arm in arms})
        self.assertEqual(req(self.url,route+'78'),(200,dict(channel='watch-1',seed=78,episodes=[])))
        self.assertEqual(req(self.url,route+'11')[0],404)
        self.assertEqual(req(self.url,'/gameplay/seed?channel=watch-2&seed=12')[0],404)
        self.assertEqual(req(self.url,'/gameplay/seed?channel=unknown&seed=12')[0],404)
        for query in ('','?channel=watch-1','?seed=12','?channel=watch-1&seed=12&seed=90',
                      '?channel=..%2Fprivate&seed=12','?channel=watch-1&seed=..%2F12',
                      '?channel=watch-1&seed=-4','?channel=watch-1&seed=9007199254740992'):
            with self.subTest(query=query):
                self.assertEqual(req(self.url,'/gameplay/seed'+query)[0],400)
        config.write_text(json.dumps(dict(manifest,seeds=[6],screening_seeds=[])),encoding='utf-8')
        self.assertEqual(req(self.url,route+'12')[0],404)
        config.unlink()
        self.assertEqual(req(self.url,route+'6')[0],503)

    def test_gameplay_watch_counts_reused_baseline_without_advancing_batches(self):
        root=Path(self.dir.name)/'reused-evaluation'
        root.mkdir()
        identity='a'*64
        arms=('official','candidate-one','candidate-two')
        manifest=dict(schema_version=1,identity=identity,checkpoints=dict.fromkeys(arms,'private-path'),
                      seeds=list(range(100)),screening_seeds=list(range(30)),config={'seed_group_size':8})
        (root/'resolved-config.json').write_text(json.dumps(manifest),encoding='utf-8')
        now=time.time()
        journal=root/'progress.jsonl'
        sidecar=root/'baseline-reuse.json'
        provenance=dict(schema_version=1,identity=identity,arm='official',reused_games=100,source='private-path')
        self.s.watches=(GameplayWatch(1,'Evaluation',root),)

        def progress(phase='gameplay',status='running',**fields):
            with journal.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(timestamp=utc(now),phase=phase,status=status,**fields))+'\n')

        def snapshot():
            status,body=req(self.url,'/channels')
            self.assertEqual(status,200)
            self.assertNotIn('private-path',json.dumps(body))
            return body['channels'][-1]

        sidecar.write_text(json.dumps({**provenance,'identity':'b'*64}),encoding='utf-8')
        progress(games=104)
        old=snapshot()
        self.assertNotIn('reused_games',old)
        self.assertEqual(old['batch']['number'],5)

        journal.unlink()
        sidecar.write_text(json.dumps(provenance),encoding='utf-8')
        initial=snapshot()
        self.assertEqual((initial['games'],initial['total_games'],initial['reused_games']),(100,300,100))
        self.assertEqual(initial['batch']['number'],1)
        progress('baseline_reused',games=100)

        game=Game.seeded(0)
        options=legal(game.state)
        before=nested(game.state)
        action=options[0]
        game.step(action)
        move=dict(seed=0,turn=game.turn,action=action,legal_moves=options,before=before,
                  after=nested(game.state),score=game.score,max_tile=max(game.state),
                  probabilities=dict(zip(options,[1]+[0]*(len(options)-1))))
        for arm,stop in zip(arms,('terminal',None,None)):
            path=root/arm/'0'/'moves.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(move,stop=stop))+'\n',encoding='utf-8')
        progress(games=115)
        first=snapshot()
        self.assertEqual(first['batch']['number'],1)
        self.assertEqual([(item['arm'],item['stop']) for item in first['episodes']],
                         [('official','terminal'),('candidate-one',None),('candidate-two',None)])
        self.assertEqual(first['activity'],{'seeds':1,'models':2,'games':2})
        progress(games=116)
        self.assertEqual(snapshot()['batch']['seeds'],list(range(8,16)))

        for arm in arms[1:]:
            for seed in range(30):
                result=root/arm/str(seed)/'result.json'
                result.parent.mkdir(parents=True,exist_ok=True)
                result.write_text('{}',encoding='utf-8')
        progress('screening_completed',games=90)
        screened=snapshot()
        self.assertEqual((screened['games'],screened['batch']['number']),(160,5))
        self.assertNotIn('games',screened['latest'])
        for _ in range(50):
            progress('loading_model')
        resumed=snapshot()
        self.assertEqual((resumed['games'],resumed['batch']['seeds']),(160,list(range(30,38))))
        progress('completed',status='completed',games=300)
        completed=snapshot()
        self.assertEqual((completed['games'],completed['batch']['number']),(300,13))

    def test_channel_stream_pushes_updates_and_reconnects(self):
        journal=Path(self.dir.name)/'progress.jsonl'
        self.s.watches=(Watch(1,f'Training:training:{journal}'),)
        def write(step):
            journal.write_text(json.dumps(dict(timestamp=utc(time.time()),status='running',
                                               phase='encoding',step=step,total=8))+'\n',encoding='utf-8')
        def connect():
            c=http.client.HTTPConnection('127.0.0.1',self.s.server_port,timeout=3)
            self.addCleanup(c.close)
            c.request('GET','/channels/events')
            stream=c.getresponse()
            self.addCleanup(stream.close)
            self.assertEqual(stream.status,200)
            self.assertTrue(stream.getheader('Content-Type').startswith('text/event-stream'))
            return stream
        def frame(stream):
            for line in stream:
                if line.startswith(b'data: '):
                    return json.loads(line[6:])['channels'][-1]['latest']['step']
            self.fail('Missing channel snapshot')
        write(1)
        stream=connect()
        self.assertEqual(frame(stream),1)
        write(2)
        self.assertEqual(frame(stream),2)
        self.assertEqual(frame(connect()),2)
        cross=getconn(self.url)
        self.addCleanup(cross.close)
        cross.request('GET','/channels/events',headers={'Origin':'http://example.invalid'})
        self.assertEqual(cross.getresponse().status,403)

if __name__=='__main__':
    import sys
    if len(sys.argv)==3 and sys.argv[1]=='--native':
        addr=sys.argv[2]; seed=96652001; cap=4
        code,start=req(addr,'/start',{'cap':cap,'seed':seed})
        assert code==202 and start['profile']=='frozen_v2'
        print(json.dumps(validate(addr,start,seed,cap),ensure_ascii=False,indent=2))
    else:
        unittest.main()
