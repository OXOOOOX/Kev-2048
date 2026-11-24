"""Instant play must still have a search policy after its first large merge."""

import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from kev2048_release.game import legal
from kev2048_release.teacher import Teacher

if __package__:
    from .wide_teacher import PlayTeacher, _legal_slides, _search
else:
    from wide_teacher import PlayTeacher, _legal_slides, _search


NATIVE = Path(__file__).resolve().parents[1] / ".cache" / "teacher-build-v2"


class WideTeacherTests(unittest.TestCase):
    def setUp(self):
        self.teacher = PlayTeacher.__new__(PlayTeacher)
        self.teacher.depth = 3

    def test_native_route_for_safe_4096_8192_and_16384_boards(self):
        for tile in (4096, 8192, 16384):
            board = (tile, 2, 4, 8, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2, 4, 0)
            with self.subTest(tile=tile), patch.object(Teacher, "values", return_value={"down": 12.0}) as native:
                self.assertEqual(self.teacher.values(board), {"down": 12.0})
                native.assert_called_once_with(board)

    def test_all_high_directions_are_scored_and_double_32768_merges_to_65536(self):
        tail = (4, 8, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2, 4, 8)
        board = (32768, 32768, *tail)
        self.assertEqual(dict(_legal_slides(board))["left"][0], 65536)
        self.assertEqual(dict(_legal_slides(board))["right"][:4], (0, 65536, 4, 8))
        for high in (32768, 65536, 131072):
            sample = (high, *board[1:-1], 0)
            with self.subTest(tile=high):
                scores = self.teacher.values(sample)
                self.assertEqual(list(scores), legal(sample))
                self.assertTrue(scores)
                self.assertTrue(all(math.isfinite(value) for value in scores.values()))
        # Even when the root fits the DLL, a subsequent search move can exceed it.
        dangerous = (16384,) * 4 + board[4:]
        self.assertEqual(set(self.teacher.values(dangerous)), set(legal(dangerous)))
        dead = (131072, 2, 4, 8, 2, 4, 8, 16, 4, 8, 16, 32, 8, 16, 32, 64)
        self.assertEqual(self.teacher.values(dead), {})

    @unittest.skipUnless(os.name == "nt" and (NATIVE / "teacher.dll").is_file(), "pinned native teacher unavailable")
    def test_wide_search_agrees_with_calibrated_depth_three_teacher(self):
        teacher = Teacher(NATIVE, depth=3)
        boards = (
            (2, 4, 8, 16, 32, 64, 128, 256, 512, 2, 4, 8, 16, 32, 0, 0),
            (4096, 16, 4, 2, 128, 16, 4, 0, 128, 32, 2, 0, 32, 2, 0, 0),
            (16384, 8192, 512, 256, 128, 64, 32, 16, 4, 8, 16, 32, 2, 4, 0, 2),
        )
        for board in boards:
            with self.subTest(board=board):
                native = teacher.values(board)
                wide = {move: _search(after, teacher.depth) for move, after in _legal_slides(board)}
                self.assertEqual(set(wide), set(native))
                self.assertEqual(max(wide, key=wide.get), max(native, key=native.get))
                for move in native:
                    self.assertAlmostEqual(wide[move], native[move], delta=2.0)


if __name__ == "__main__":
    unittest.main()
