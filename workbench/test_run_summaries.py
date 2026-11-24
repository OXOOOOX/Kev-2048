import unittest

from run_summaries import normalize_result


def metrics(result):
    return {item["label"]: item["value"] for item in result.get("summary_metrics", ())}


class ResultSummaryTests(unittest.TestCase):
    def test_verified_preparation_and_collection_keep_separate_meanings(self):
        data = dict(status="ready", total_collected_episodes=670, teacher_labels_rechecked=87,
                    verification_passed=True, train={"actual": 8000}, dev={"actual": 2000})
        summary = normalize_result(data)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(metrics(summary), {"采集对局": 670, "训练盘面": 8000, "验证盘面": 2000})
        self.assertNotIn("status", normalize_result({**data, "verification_passed": False}))
        selected = normalize_result(dict(status="completed", collection_identity="x", selected_count=200,
                                         collection_seed_count=28, games=28, moves=4535))
        self.assertEqual(metrics(selected)["选入训练盘面"], 200)
        self.assertEqual(metrics(selected)["记录走法"], 4535)

    def test_native_results_only_count_terminal_games_in_the_known_aggregate(self):
        report = dict(checkpoint_head_sha256="0" * 64, cap=50, games=[
            dict(seed=10, turn=15, score=1240, tile=256, stop="terminal"),
            dict(seed=11, turn=50, score=35000, tile=2048, stop="cap"),
        ])
        result = normalize_result(report)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(metrics(result), {"记录对局": 2, "单局最高分": 35000, "最大方块": 2048, "达到 2048": 1})
        report["games"][-1]["stop"] = "in_progress"
        self.assertEqual(normalize_result(report), {})
        self.assertEqual(normalize_result({"games": [{"score": 7}]}), {})

    def test_public_metrics_use_fixed_labels_and_no_private_command_arguments(self):
        inputs = {"runs": {"cold": {"argv": ["sensitive"], "process_wall_seconds": 59},
                           "hot": {"argv": ["private/path"], "process_wall_seconds": 29.5}},
                  "completed": True, "checkpoints_identical": True, "process_speedup": 2,
                  "process_seconds_saved": 29.5, "encoding_seconds_saved": float("nan")}
        output = normalize_result(inputs)
        self.assertEqual(metrics(output), {"冷缓存用时": 59, "热缓存用时": 29.5, "热缓存加速比": 2, "节省用时": 29.5})
        self.assertNotIn("argv", str(output))
        failure = normalize_result(dict(preset="smoke", repeats=2, completed=False, runs=[], error="private/path"))
        self.assertEqual(failure["status"], "error")
        self.assertNotIn("private/path", str(failure))
        self.assertNotIn("status", normalize_result(dict(preset="stock", repeats=2, completed=False, runs=[])))

    def test_selected_epoch_and_final_paired_cohort(self):
        fit = normalize_result(dict(status="completed", train_rows=4000, best_epoch=0, completed_epochs=1,
                                    total_update_steps=125, candidate_dev={"n": 2000, "best_rate": .5}))
        self.assertEqual(metrics(fit)["选用轮次"], 0)
        self.assertEqual(metrics(fit)["验证集老师走法命中率"], 50)
        report = dict(status="completed", total_games=4, fresh_games=2, reused_games=2,
                      candidate_id="candidate", reference_id="baseline",
                      all_1={"seeds": [10], "metrics": {"candidate": {"score_median": 999}}},
                      all_2={"seeds": [10, 11], "metrics": {"candidate": {"score_median": 150, "n512": 0},
                                                          "baseline": {"score_median": 100, "n2048": 0}}})
        output = metrics(normalize_result(report))
        self.assertEqual(output["候选得分中位数"], 150)
        self.assertEqual(output["对照得分中位数"], 100)
        self.assertEqual(output["复用对局"], 2)
        self.assertNotIn("对照达到 512", output)

    def test_unknown_or_nonfinite_data_falls_back_without_guessing(self):
        self.assertEqual(normalize_result("bad"), {})
        self.assertEqual(normalize_result({"status": "ready", "paths": ["private"]}), {})
        self.assertEqual(normalize_result({"total_games": True, "fresh_games": -3, "reused_games": []}), {})
        self.assertEqual(normalize_result({"train_rows": 10**200, "dev_rows": float("inf")}), {})
