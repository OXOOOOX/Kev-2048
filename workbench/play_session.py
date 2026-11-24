"""Record one local, same-seed comparison and save its replays on request."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import math
from pathlib import Path
import re
import secrets
from tempfile import TemporaryDirectory
import threading
import time
import uuid

from kev2048_release.game import Game, legal, nested
from kev2048_release.artifacts import stop_reason


MAX_MODELS = 8
MAX_SEED = 2**31 - 1
MAX_TURNS = 5000
PLAY_MODES = ("classic", "endless")
MODEL_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
PLAYING = {"loading", "running"}
REPLAY_STOPS = {"terminal", "target", "cap"}
LOGGER = logging.getLogger(__name__)


class PlayRequestError(Exception):
    def __init__(self, message, code=400):
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass
class _Track:
    game: Game
    view: dict
    frames: list[dict]


def _position(game):
    return dict(turn=game.turn, score=game.score, max_tile=max(game.state), board=nested(game.state))


def _validate_decision(decision, options, kind):
    field = "action_values" if kind == "teacher" else "probabilities"
    if not isinstance(decision, dict) or set(decision) != {field, "decision_ms"}:
        raise ValueError("Unexpected decision fields")
    values = decision[field]
    if not isinstance(values, dict) or set(values) != set(options):
        raise ValueError("Decision must score each legal direction")
    decision_ms = decision["decision_ms"]
    if (type(decision_ms) not in (int, float) or not math.isfinite(decision_ms) or decision_ms < 0
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values())):
        raise ValueError("Decision contains an invalid score or duration")
    if field == "probabilities" and (any(not 0 <= value <= 1 for value in values.values())
                                      or abs(sum(values.values()) - 1) > 1e-5):
        raise ValueError("Decision probabilities must sum to one")
    return max(options, key=values.get), {field: {move: values[move] for move in options},
                                          "decision_ms": decision_ms}


class PlaySession:
    """Own the active game group without loading or running a policy until start."""

    def __init__(self, out, models, load, *, move_interval=0.0, refresh_models=None):
        self._out = Path(out)
        self._models = self._model_catalog(models)
        self._catalog_source = models
        self._refresh_models = refresh_models
        self._catalog_lock = threading.Lock()
        self._load = load
        self._move_interval = move_interval
        self._condition = threading.Condition()
        self._session = None
        self._tracks = {}
        self._revision = 0

    @staticmethod
    def _model_catalog(models):
        catalog = {}
        for model in models:
            key = model["id"]
            if (not isinstance(key, str) or not MODEL_ID.fullmatch(key)
                    or key in catalog or model["kind"] not in {"model", "teacher"}):
                raise ValueError("Invalid local play model catalog")
            item = dict(id=key, name=model["name"], kind=model["kind"])
            for field in ("training_summary", "evaluation"):
                if field in model:
                    item[field] = deepcopy(model[field])
            catalog[key] = item
        return catalog

    def _refresh_catalog(self):
        if self._refresh_models is None:
            return
        with self._catalog_lock:
            models = self._refresh_models()
            if models is self._catalog_source:
                return
            catalog = self._model_catalog(models)
            with self._condition:
                self._catalog_source = models
                if list(catalog.values()) != list(self._models.values()):
                    self._models = catalog
                    self._publish()

    def _state(self):
        return deepcopy(dict(revision=self._revision, models=list(self._models.values()),
                             modes=list(PLAY_MODES), can_continue_endless=bool(self._continuation_models()),
                             session=self._session))

    def _continuation_models(self):
        session = self._session
        if (not session or session["active"] or session["status"] != "completed"
                or session.get("mode", "classic") != "classic"
                or any(track.view["status"] not in REPLAY_STOPS for track in self._tracks.values())):
            return ()
        playable = tuple(key for key, track in self._tracks.items()
                         if track.view["status"] in ("target", "cap") and legal(track.game.state))
        return playable if any(self._tracks[key].view["status"] == "target" for key in playable) else ()

    def _publish(self):
        self._revision += 1
        self._condition.notify_all()

    def latest(self):
        self._refresh_catalog()
        with self._condition:
            return self._state()

    def pending(self, revision, wait=10):
        self._refresh_catalog()
        with self._condition:
            if self._condition.wait_for(lambda: revision != self._revision, timeout=wait):
                return self._state()
            return None

    def _current(self, identifier):
        if not isinstance(identifier, str) or not identifier:
            raise PlayRequestError("缺少有效的对局编号。")
        if self._session is None or self._session["id"] != identifier:
            raise PlayRequestError("指定对局已切换。", 409)
        return self._session

    def _command_session(self, body):
        if not isinstance(body, dict) or set(body) != {"id"}:
            raise PlayRequestError("缺少有效的对局编号。")
        return self._current(body["id"])

    def frame(self, identifier, turn):
        with self._condition:
            session = self._current(identifier)
            max_turn = max(game["turn"] for game in session["games"])
            if type(turn) is not int or not 0 <= turn <= max_turn:
                raise PlayRequestError("回看步数超出当前对局记录。")
            games = [track.frames[min(turn, len(track.frames) - 1)] for track in self._tracks.values()]
            return dict(id=identifier, turn=turn, games=deepcopy(games))

    def start(self, body):
        if not isinstance(body, dict) or set(body) - {"models", "seed", "mode", "cap"}:
            raise PlayRequestError("新开对局参数无效。")
        models = body.get("models")
        if (not isinstance(models, list) or not 1 <= len(models) <= MAX_MODELS
                or any(not isinstance(key, str) for key in models)
                or len(set(models)) != len(models)):
            raise PlayRequestError(f"请选择 1 至 {MAX_MODELS} 个不同的可用模型。")
        mode = body.get("mode", "classic")
        if mode not in PLAY_MODES:
            raise PlayRequestError("请选择经典模式或无尽模式。")
        cap = body.get("cap", None if mode == "endless" else MAX_TURNS)
        if mode == "endless" and cap is not None:
            raise PlayRequestError("无尽模式应省略步数上限。")
        if mode == "classic" and (type(cap) is not int or not 1 <= cap <= MAX_TURNS):
            raise PlayRequestError(f"步数上限应为 1 至 {MAX_TURNS} 的整数。")
        seed = body["seed"] if "seed" in body else secrets.randbelow(MAX_SEED + 1)
        if type(seed) is not int or not 0 <= seed <= MAX_SEED:
            raise PlayRequestError(f"种子应为 0 至 {MAX_SEED} 的整数。")

        self._refresh_catalog()
        with self._condition:
            if any(key not in self._models for key in models):
                raise PlayRequestError(f"请选择 1 至 {MAX_MODELS} 个不同的可用模型。")
            if self._session is not None and self._session["active"]:
                raise PlayRequestError("当前对局还在进行。", 409)
            tracks = {}
            for key in models:
                model = self._models[key]
                game = Game.seeded(seed)
                view = dict(model=key, name=model["name"], kind=model["kind"], status="loading", **_position(game))
                tracks[key] = _Track(game, view, [dict(view, status="ready")])
            return self._launch(seed, mode, cap, tracks)

    def continue_endless(self, body):
        with self._condition:
            previous = self._command_session(body)
            playable = self._continuation_models()
            if not playable:
                raise PlayRequestError("只有已达到 2048 且还有合法走法的经典对局可以继续无尽。", 409)
            tracks = {}
            for key, old in self._tracks.items():
                continuing = key in playable
                view = dict(old.view, status="loading" if continuing else "terminal")
                frames = old.frames.copy()
                # The former target or cap becomes an ordinary move in the extended recording.
                frames[-1] = dict(frames[-1], status="running" if continuing else "terminal")
                tracks[key] = _Track(old.game, view, frames)
            return self._launch(previous["seed"], "endless", None, tracks)

    def _launch(self, seed, mode, cap, tracks):
        session = dict(id=uuid.uuid4().hex, seed=seed, mode=mode, cap=cap, status="loading", active=True,
                       saved=False, games=[track.view for track in tracks.values()])
        # Both callers hold the condition. A failed thread launch leaves the previous game available.
        worker = threading.Thread(target=self._work, args=(session, tracks), daemon=True, name="instant-play")
        worker.start()
        self._session = session
        self._tracks = tracks
        self._publish()
        return self._state()

    def stop(self, body):
        with self._condition:
            session = self._command_session(body)
            if not session["active"]:
                raise PlayRequestError("指定对局已结束。", 409)
            if session["status"] != "stopping" and any(game["status"] in PLAYING for game in session["games"]):
                session["status"] = "stopping"
                self._publish()
            return self._state()

    def pause(self, body):
        with self._condition:
            session = self._command_session(body)
            if session["status"] == "paused":
                return self._state()
            if session["status"] != "running" or not any(game["status"] == "running" for game in session["games"]):
                raise PlayRequestError("当前对局无法暂停。", 409)
            session["status"] = "paused"
            self._publish()
            return self._state()

    def resume(self, body):
        with self._condition:
            session = self._command_session(body)
            if session["status"] == "running":
                return self._state()
            if session["status"] != "paused":
                raise PlayRequestError("当前对局无法继续。", 409)
            session["status"] = "running"
            self._publish()
            return self._state()

    def save(self, body):
        with self._condition:
            session = self._command_session(body)
            if session["saved"]:
                return self._state()
            if session["active"] or not any(game["turn"] for game in session["games"]):
                raise PlayRequestError("对局结束并至少落子一步后才能保存。", 409)
            try:
                self._out.mkdir(parents=True, exist_ok=True)
                archive = self._out.resolve()
                destination = archive / session["id"]
                if destination.exists():
                    raise FileExistsError(destination)
                replay = dict(session, mode=session.get("mode", "classic"),
                              saved_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"))
                with TemporaryDirectory(prefix=".instant-play-save-", dir=archive.parent,
                                        ignore_cleanup_errors=True) as tmp:
                    recording = Path(tmp) / "recording"
                    recording.mkdir()
                    for key, track in self._tracks.items():
                        if track.view["turn"]:
                            self._write_replay(recording / key, replay, track)
                    recording.rename(destination)
            except OSError as error:
                LOGGER.exception("Could not save the instant-play replay")
                raise PlayRequestError("保存对局失败，请重试。", 500) from error
            session["saved"] = True
            self._publish()
            return self._state()

    @classmethod
    def _write_replay(cls, directory, session, track):
        directory.mkdir()
        manifest = dict(schema_version=1, kind="play", variant=track.view["name"], run_id=session["id"],
                        seeds=[session["seed"]], mode=session["mode"], max_moves_per_game=session["cap"],
                        files={"moves": "moves.jsonl"},
                        starred=True, saved_at=session["saved_at"])
        with (directory / "run.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False)
            stream.write("\n")
        fields = ("before", "action", "gained", "spawn", "decision_ms",
                  "action_values" if track.view["kind"] == "teacher" else "probabilities")
        with (directory / "moves.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for turn in range(1, len(track.frames)):
                frame = track.frames[turn]
                stop = frame["status"] if frame["status"] in REPLAY_STOPS else None
                row = cls._replay_row(session, frame, "move", stop)
                row.update({key: frame[key] for key in fields})
                cls._write_row(stream, row)
            status = track.view["status"]
            if status in ("stopped", "error"):
                stop = "stopped" if status == "stopped" else None
                cls._write_row(stream, cls._replay_row(session, track.view, status, stop))

    @staticmethod
    def _write_row(stream, row):
        stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    @staticmethod
    def _replay_row(session, position, event_type, stop=None):
        sequence = position["turn"] if event_type == "move" else position["turn"] + 1
        return dict(type=event_type, run_id=session["id"], seq=sequence, mode=session["mode"],
                    seed=session["seed"], cap=session["cap"], turn=position["turn"], score=position["score"],
                    tile=position["max_tile"], board=position["board"], stop=stop)

    def _move(self, session, track, action, decision):
        game = track.game
        before = track.frames[-1]["board"]
        gained, spawn = game.step(action)
        if session.get("mode") == "endless":
            final = None if legal(game.state) else "terminal"
        else:
            reason = stop_reason(game, session["cap"])
            final = "cap" if reason == "capped" else reason
        track.view.update(_position(game), status=final or "running", before=before, action=action,
                          gained=gained, spawn=spawn, **decision)
        track.frames.append(track.view.copy())

    def _play(self, session, tracks):
        decide = self._load(tuple(key for key, track in tracks.items() if track.view["status"] == "loading"))
        with self._condition:
            if session["status"] != "stopping":
                session["status"] = "running"
                for track in tracks.values():
                    if track.view["status"] == "loading":
                        track.view["status"] = "running"
                self._publish()

        while True:
            with self._condition:
                self._condition.wait_for(lambda: session["status"] != "paused")
                if session["status"] == "stopping":
                    return "stopped"
                active = [(key, track.game) for key, track in tracks.items() if track.view["status"] == "running"]
                if not active:
                    return "completed"
            started = time.monotonic()
            decisions = decide(active)
            with self._condition:
                if session["status"] == "stopping":
                    return "stopped"
                if session["status"] == "paused":
                    continue
                if not isinstance(decisions, dict) or decisions.keys() != {key for key, _ in active}:
                    raise ValueError("Unexpected models in the decision batch")
                actions = [_validate_decision(decisions[key], legal(game.state), tracks[key].view["kind"])
                           for key, game in active]
                for (key, _), (action, decision) in zip(active, actions, strict=True):
                    self._move(session, tracks[key], action, decision)
                self._publish()
                remaining = self._move_interval - (time.monotonic() - started)
                if remaining > 0 and any(track.view["status"] == "running" for track in tracks.values()):
                    # Give the browser time to show fast moves, with an immediate wake-up on pause or stop.
                    self._condition.wait_for(lambda: session["status"] != "running", timeout=remaining)

    def _finish(self, session, status):
        session.update(active=False, status=status)
        if status == "error":
            session["message"] = "即时游玩失败，请查看本机服务日志。"
        for game in session["games"]:
            if game["status"] in PLAYING:
                game["status"] = status
                track = self._tracks[game["model"]]
                if game["turn"]:
                    track.frames[-1] = game.copy()
        self._publish()

    def _work(self, session, tracks):
        status = "error"
        try:
            status = self._play(session, tracks)
        except Exception:
            LOGGER.exception("Instant play failed")
        finally:
            with self._condition:
                self._finish(session, status)
