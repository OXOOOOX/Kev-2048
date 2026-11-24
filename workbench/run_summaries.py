"""Public, bounded metrics for the workbench's aggregate result formats."""

import math
import re


MAX_METRICS = 12
MAX_GAMES = 20_000
COUNT_UNITS = {"局", "张", "步", "轮", "次", "个"}
STATES = {"waiting", "loading", "running", "paused", "completed", "terminal", "cap", "stopped", "error"}


def _number(value):
    return type(value) in (int, float) and abs(value) < 2**53 and math.isfinite(value)


def _dict(value):
    return value if isinstance(value, dict) else {}


def _add(metrics, label, value, unit="", *, signed=False):
    if (len(metrics) >= MAX_METRICS or not _number(value) or (not signed and value < 0)
            or (unit in COUNT_UNITS and type(value) is not int)):
        return
    metric = {"label": label, "value": value}
    if unit:
        metric["unit"] = unit
    metrics.append(metric)


def _native_games(raw, metrics):
    games = raw.get("games")
    head = raw.get("checkpoint_head_sha256")
    if (not isinstance(games, list) or not 1 <= len(games) <= MAX_GAMES
            or not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{64}", head)
            or type(raw.get("cap")) is not int or not 0 < raw["cap"] < 2**53):
        return False
    max_score = max_tile = reached = 0
    for game in games:
        if (not isinstance(game, dict) or not all(type(game.get(k)) is int and 0 <= game[k] < 2**53
                for k in ("seed", "turn", "score", "tile")) or game.get("stop") not in ("cap", "terminal")
                or game["turn"] > raw["cap"] or game["tile"] < 2 or game["tile"] & (game["tile"] - 1)):
            return False
        max_score = max(max_score, game["score"])
        max_tile = max(max_tile, game["tile"])
        reached += game["tile"] >= 2048
    _add(metrics, "记录对局", len(games), "局")
    _add(metrics, "单局最高分", max_score, "分")
    _add(metrics, "最大方块", max_tile)
    _add(metrics, "达到 2048", reached, "局")
    # The native play command writes this aggregate only after the seed loop finishes.
    return True


def _evaluation(raw, metrics):
    _add(metrics, "测评对局", raw.get("total_games", raw.get("total_completed_games")), "局")
    _add(metrics, "本次新跑", raw.get("fresh_games", raw.get("newly_evaluated_games")), "局")
    _add(metrics, "复用对局", raw.get("reused_games", raw.get("reference_games_reused")), "局")
    _add(metrics, "候选对局", raw.get("candidate_games"), "局")
    arms = _dict(raw.get("metrics"))
    if not arms:
        # In paired reports, use only the aggregate that covers the reported total.
        for name, value in raw.items():
            if not isinstance(name, str) or not re.fullmatch(r"all_\d{1,6}", name):
                continue
            cohort = _dict(value)
            seeds = cohort.get("seeds")
            if isinstance(seeds, list) and 2 * len(seeds) == raw.get("total_games"):
                arms = _dict(cohort.get("metrics"))
                break
    for key, label in (("candidate_id", "候选"), ("reference_id", "对照")):
        arm = raw.get(key)
        if isinstance(arm, str):
            result = _dict(arms.get(arm))
            _add(metrics, label + "得分中位数", result.get("score_median"), "分")
            _add(metrics, label + "达到 512", result.get("n512"), "局")
            _add(metrics, label + "达到 2048", result.get("n2048"), "局")


def _training(raw, metrics):
    dev = _dict(raw.get("best_dev")) or _dict(raw.get("candidate_dev"))
    train = _dict(raw.get("best_train"))
    _add(metrics, "训练盘面", raw.get("train_rows", train.get("n")), "张")
    _add(metrics, "验证盘面", raw.get("dev_rows", dev.get("n")), "张")
    _add(metrics, "已训练轮数", raw.get("completed_epochs"), "轮")
    _add(metrics, "选用轮次", raw.get("best_epoch"), "轮")
    _add(metrics, "参数更新", raw.get("total_update_steps", raw.get("optimizer_steps")), "步")
    _add(metrics, "最佳验证老师损失", raw.get("best_dev_teacher_loss", dev.get("teacher_loss")))
    rate = dev.get("best_rate")
    if _number(rate) and 0 <= rate <= 1:
        _add(metrics, "验证集老师走法命中率", 100 * rate, "%")
    _add(metrics, "训练用时", raw.get("elapsed_seconds"), "秒")


def normalize_result(raw):
    """Return only verified status and metric values, without paths, commands or raw metadata."""
    if not isinstance(raw, dict):
        return {}
    reported = raw.get("status")
    state = reported if isinstance(reported, str) and reported in STATES else None
    if reported == "failed":
        state = "error"
    elif not state and raw.get("completed") is True:
        state = "completed"
    metrics = []
    if "total_collected_episodes" in raw and "teacher_labels_rechecked" in raw:
        _add(metrics, "采集对局", raw.get("total_collected_episodes"), "局")
        _add(metrics, "训练盘面", _dict(raw.get("train")).get("actual"), "张")
        _add(metrics, "验证盘面", _dict(raw.get("dev")).get("actual"), "张")
        _add(metrics, "训练子集", _dict(raw.get("train_subset")).get("actual"), "张")
        _add(metrics, "老师校准成功", raw.get("teacher_calibration_successes"), "局")
        _add(metrics, "老师校准总数", raw.get("teacher_calibration_games"), "局")
        if reported == "ready" and raw.get("verification_passed") is True:
            state = "completed"
    elif "collection_identity" in raw or "selected_count" in raw and "collection_seed_count" in raw:
        _add(metrics, "采集对局", raw.get("games"), "局")
        _add(metrics, "记录走法", raw.get("moves"), "步")
        _add(metrics, "采集种子", raw.get("collection_seed_count"), "个")
        _add(metrics, "选入训练盘面", raw.get("selected_count"), "张")
        _add(metrics, "选中种子", raw.get("selected_seed_count"), "个")
    elif isinstance(raw.get("runs"), list) and "preset" in raw and "repeats" in raw:
        runs = raw["runs"]
        wall = _dict(raw.get("median_wall_seconds"))
        _add(metrics, "CPU 耗时中位数", wall.get("cpu"), "秒")
        _add(metrics, "CUDA 耗时中位数", wall.get("cuda"), "秒")
        _add(metrics, "CUDA 加速比", raw.get("wall_speedup_cpu_over_cuda"), "倍")
        _add(metrics, "每设备计划重复", raw.get("repeats"), "次")
        _add(metrics, "已记录运行", len(runs), "次")
        if raw.get("completed") is False and isinstance(raw.get("error"), str) and raw["error"]:
            state = "error"
    elif "process_speedup" in raw and isinstance(raw.get("runs"), dict):
        runs = raw["runs"]
        _add(metrics, "冷缓存用时", _dict(runs.get("cold")).get("process_wall_seconds"), "秒")
        _add(metrics, "热缓存用时", _dict(runs.get("hot")).get("process_wall_seconds"), "秒")
        _add(metrics, "热缓存加速比", raw.get("process_speedup"), "倍")
        _add(metrics, "节省用时", raw.get("process_seconds_saved"), "秒", signed=True)
        _add(metrics, "节省编码用时", raw.get("encoding_seconds_saved"), "秒", signed=True)
    elif "control_timing" in raw and isinstance(raw.get("optimizer"), dict):
        optimizer = raw["optimizer"]
        _add(metrics, "计时参数更新", optimizer.get("steps"), "步")
        _add(metrics, "计时训练盘面", optimizer.get("train_rows_seen"), "张")
        _add(metrics, "优化器用时", optimizer.get("seconds"), "秒")
        _add(metrics, "对照优化器用时", _dict(raw.get("control_timing")).get("seconds"), "秒")
        _add(metrics, "总用时", raw.get("wall_seconds"), "秒")
        if state is None and _dict(raw.get("native_reload")).get("argmax_equal") is True and metrics:
            state = "completed"
    elif isinstance(raw.get("games"), list):
        if _native_games(raw, metrics) and state is None:
            state = "completed"
    elif any(key in raw for key in ("total_games", "total_completed_games", "fresh_games", "candidate_games")):
        _evaluation(raw, metrics)
    elif any(key in raw for key in ("completed_epochs", "total_update_steps", "train_rows", "best_epoch")):
        _training(raw, metrics)
    result = {}
    if state:
        result["status"] = state
    if metrics:
        result["summary_metrics"] = metrics
    return result
