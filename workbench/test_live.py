"""Backend black-box tests; optional actual native smoke with --native URL."""
import http.client
import json
import tempfile
import time
import threading
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from live import Game, Live, Server, Watch, legal, nested, utc, MAX_WATCH_BYTES

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
        for filename,text in [('index.html','launcher'),('console.html','console'),('replay.html','replay'),('secret.jsonl','secret')]:
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
            'step':8,'total':128,'loss':0.42,'private_path':str(d),'traceback':'secret','bad':123})+'\n{"status":',encoding='utf-8')
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
        finally:
            m.shutdown();m.server_close()

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

if __name__=='__main__':
    import sys
    if len(sys.argv)==3 and sys.argv[1]=='--native':
        addr=sys.argv[2]; seed=96652001; cap=4
        code,start=req(addr,'/start',{'cap':cap,'seed':seed})
        assert code==202 and start['profile']=='frozen_v2'
        print(json.dumps(validate(addr,start,seed,cap),ensure_ascii=False,indent=2))
    else:
        unittest.main()
