"""Local fixed-depth search teacher. No model, network, or game RNG in the DLL."""
import ctypes
import json
import math
from pathlib import Path

from .data import digest
from .game import DIRS, legal

REVISION = "41e298f4571a9505e421e3a19af7a1cb372a368c"


def pack(state):
    if len(state) != 16 or any(type(v) != int or v < 0 or v > 16384 or
                             (v and (v < 2 or v & (v - 1))) for v in state):
        raise ValueError("Expected 16 zero/power-of-two cells, at most 16384")
    return sum((v.bit_length() - 1 if v else 0) << (4 * i) for i, v in enumerate(state))


def unpack(board):
    return tuple((1 << exponent) if (exponent := (board >> (4 * i)) & 15) else 0 for i in range(16))


class Teacher:
    def __init__(self, folder, depth=3):
        if depth not in (3, 4, 5):
            raise ValueError("Teacher depth must be 3, 4, or 5")
        folder = Path(folder).resolve()
        self.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["revision"] != REVISION or self.manifest["chance_cutoff"] != 1e-4:
            raise ValueError("Unexpected teacher source or search threshold")
        if digest(folder / "teacher.dll") != self.manifest["dll_sha256"]:
            raise ValueError("Teacher DLL digest mismatch")
        self.depth = depth
        self.dll = ctypes.CDLL(str(folder / "teacher.dll"))
        self.dll.kev_teacher_abi.restype = ctypes.c_int
        if self.dll.kev_teacher_abi() != 1:
            raise ValueError("Unexpected teacher ABI")
        self.dll.init_tables.argtypes = []
        self.dll.init_tables.restype = None
        self.dll.execute_move.argtypes = [ctypes.c_int, ctypes.c_uint64]
        self.dll.execute_move.restype = ctypes.c_uint64
        self.dll.kev_score_move.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
        self.dll.kev_score_move.restype = ctypes.c_float
        self.dll.init_tables()

    def slide(self, state, action):
        return unpack(self.dll.execute_move(DIRS.index(action), pack(state)))

    def values(self, state):
        board = pack(state)
        options = legal(state)
        scores = {a: float(self.dll.kev_score_move(board, DIRS.index(a), self.depth)) for a in options}
        if not all(math.isfinite(v) for v in scores.values()):
            raise RuntimeError("Nonfinite teacher value")
        return scores


def labels(values):
    """New data only: near ties within 1% of the legal-action value range."""
    if len(values) < 2 or not all(math.isfinite(v) for v in values.values()):
        raise ValueError("At least two finite legal action values required")
    best = max(values, key=values.get)
    tolerance = (values[best] - min(values.values())) * .01
    return best, [a for a in DIRS if a in values and values[best] - values[a] <= tolerance], tolerance
