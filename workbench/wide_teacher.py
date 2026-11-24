"""Fixed-depth search for instant-play boards beyond the native teacher's tile range.

Heuristic and search policy adapted from nneonneo/2048-ai, revision
41e298f4571a9505e421e3a19af7a1cb372a368c.

MIT License

Copyright (c) 2014-2019 Robert Xiao <nneonneo@gmail.com> and contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from functools import lru_cache
from itertools import groupby

from kev2048_release.game import DIRS, slide
from kev2048_release.teacher import Teacher


@lru_cache(maxsize=8192)
def _line_value(line):
    ranks = tuple(value.bit_length() - 1 if value else 0 for value in line)
    merges = 0
    for _, group in groupby(rank for rank in ranks if rank):
        run = len(tuple(group))
        if run > 1:
            merges += run
    ascending = descending = 0
    for left, right in zip(ranks, ranks[1:]):
        change = right**4 - left**4
        ascending += max(0, change)
        descending += max(0, -change)
    return (200000 + 270 * ranks.count(0) + 700 * merges
            - 47 * min(ascending, descending) - 11 * sum(rank**3.5 for rank in ranks))


@lru_cache(maxsize=8192)
def _board_value(board):
    rows = sum(_line_value(board[i:i+4]) for i in range(0, 16, 4))
    columns = sum(_line_value(board[i::4]) for i in range(4))
    return rows + columns


@lru_cache(maxsize=4096)
def _legal_slides(board):
    moves = []
    for action in DIRS:
        after, _ = slide(board, action)
        if after != board:
            moves.append((action, after))
    return tuple(moves)


def _search(after, depth):
    # As in the native search, each root direction has its own transposition table.
    table = {}

    def chance(board, level, probability):
        if level >= depth or probability < 0.0001:
            return _board_value(board)
        cached = table.get(board)
        if cached is not None and cached[0] <= level:
            return cached[1]
        empty = board.count(0)
        position_probability = probability / empty
        total = 0.0
        for index, value in enumerate(board):
            if value:
                continue
            for tile, weight in ((2, 0.9), (4, 0.1)):
                spawned = board[:index] + (tile,) + board[index+1:]
                best = 0.0
                for _, moved in _legal_slides(spawned):
                    best = max(best, chance(moved, level+1, position_probability * weight))
                total += weight * best
        score = total / empty
        if cached is not None or len(table) < 32768:
            table[board] = (level, score)
        return score

    return chance(after, 0, 1.0) + 0.000001


class PlayTeacher(Teacher):
    def values(self, state):
        # The DLL accepts inputs through 16384 and simulates merges through 32768.
        # The search can add at most one 4 per ply to the board's total mass.
        if max(state) <= 16384 and sum(state) + 4 * self.depth < 65536:
            return super().values(state)
        if len(state) != 16 or any(type(v) is not int or v < 0 or v == 1 or v & (v-1) for v in state):
            raise ValueError("Expected 16 zero/power-of-two cells")
        return {action: _search(after, self.depth) for action, after in _legal_slides(tuple(state))}
