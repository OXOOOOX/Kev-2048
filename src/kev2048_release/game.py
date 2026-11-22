"""Independent 2048 rules implementation; no sample Jev code dependency."""
from dataclasses import dataclass
import random

DIRS = ("up", "down", "left", "right")
COLS = tuple(tuple(range(c,16,4)) for c in range(4))
ROWS = tuple(tuple(range(r,r+4)) for r in range(0,16,4))
INDICES = dict(zip(DIRS,(COLS,tuple(x[::-1] for x in COLS),ROWS,tuple(x[::-1] for x in ROWS))))

def flatten(rows):
    if len(rows)!=4 or any(len(r)!=4 for r in rows):raise ValueError("Expected 4x4 board")
    return tuple(v for row in rows for v in row)

def nested(state):
    return [list(state[i:i+4]) for i in range(0,16,4)]

def slide(state, action):
    result = [0]*16
    gain = 0
    for line in INDICES[action]:
        xs=[state[i] for i in line if state[i]]
        j=k=0
        while j<len(xs):
            v=xs[j]
            j+=1
            if j<len(xs) and xs[j]==v:
                v*=2;gain+=v;j+=1
            result[line[k]]=v
            k+=1
    return tuple(result),gain

def legal(state):
    return [d for d in DIRS if slide(state,d)[0]!=state]

def d4(state):
    return min(tuple(state[(c if transpose else r)*4+(r if transpose else c)]
                     for r in (range(3,-1,-1) if a else range(4))
                     for c in (range(3,-1,-1) if b else range(4)))
               for transpose in (False,True) for a in (False,True) for b in (False,True))

@dataclass
class Game:
    state: tuple
    rng: random.Random
    score: int = 0
    turn: int = 0

    @classmethod
    def seeded(cls, seed):
        g=cls((0,)*16,random.Random(seed));g.spawn();g.spawn();return g

    def spawn(self):
        j=self.rng.choice([i for i,v in enumerate(self.state) if not v])
        v=4 if self.rng.random()<.1 else 2
        self.state=self.state[:j]+(v,)+self.state[j+1:]
        return [j//4,j%4,v]

    def step(self, direction):
        nxt, gain=slide(self.state,direction)
        if nxt==self.state:raise ValueError("Illegal move: "+direction)
        self.state=nxt;self.score+=gain;self.turn+=1
        return gain,self.spawn()
