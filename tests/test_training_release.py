import json, os, subprocess, sys, unittest
from pathlib import Path
from kev2048_release.game import Game, flatten, legal, slide,d4
from kev2048_release.data import checked,read
ROOT=Path(__file__).resolve().parents[1]

class ReleaseTest(unittest.TestCase):
    def test_merge_once(self):
        a=(2,2,2,2)+(0,)*12
        self.assertEqual(slide(a,'left'),((4,4,0,0)+(0,)*12,8))
        self.assertEqual(slide((2,2,4,0)+(0,)*12,'left'),((4,4,0,0)+(0,)*12,4))

    def test_seed_and_first_divergence(self):
        g=Game.seeded(96626001)
        self.assertEqual(g.state,flatten([[0,0,0,0],[2,0,0,0],[2,0,0,0],[0,0,0,0]]))
        self.assertEqual(g.step('right'),(0,[0,0,2]))
        self.assertEqual(g.step('down'),(4,[1,1,2]))
        self.assertEqual(g.state,flatten([[0,0,0,0],[0,2,0,0],[0,0,0,0],[2,0,0,4]]))

    def test_transforms_and_data(self):
        a=(0,2,4,0,0,8,0,0,16,0,0,0,2,0,2,0)
        b=tuple(a[r*4+c] for r in range(4) for c in (3,2,1,0))
        self.assertEqual(d4(a),d4(b))
        lock,rows=checked(ROOT/'data'/'stock_scale')
        self.assertEqual((len(rows['train']),len(rows['dev'])),(512,128))
        self.assertEqual(sum(r['max_tile']==1024 for r in rows['dev']),32)

    def test_archived_seeded_transitions(self):
        for name, expected in [('d4',1998),('wide',3695)]:
            path=ROOT/'experiments'/'traces'/(name+'.jsonl')
            rows=read(path)
            self.assertEqual(len(rows),expected)
            games={}
            for r in rows:
                k=(r['seed'],r['policy'])
                if k not in games:games[k]=Game.seeded(r['seed'])
                g=games[k]
                self.assertEqual(g.state,tuple(r['before']))
                _,(y,x,_)=g.step(r['action'])
                self.assertEqual((g.state,g.score,g.turn,max(g.state),4*y+x),
                    (tuple(r['after']),r['score'],r['turn'],r['max_tile'],r['spawn_index']))

    def test_cli_no_native_import(self):
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
        cmd=[sys.executable,'-m','kev2048_release.cli','verify-data']
        out=subprocess.check_output(cmd,cwd=ROOT,env=env,text=True)
        self.assertTrue(json.loads(out)['no_model'])

if __name__=='__main__':unittest.main()
