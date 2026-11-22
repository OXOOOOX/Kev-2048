# Kev 2048 测试结果怎么存

**每次运行一个目录。** 名字用 UTC 时间、类型和短标签，目录由本机指定，不写进模型、网页或 GitHub。推荐：

```text
runs/
  YYYY-MM-DD/
    YYYYMMDDTHHMMSSfffZ-play-v2-cpu/
      moves.jsonl               # 必要：逐步回放记录
      result.json               # 运行结束后由原生 CLI 写入
      run.json                  # 可选：说明这是什么运行
    YYYYMMDDTHHMMSSfffZ-train-smoke-cpu/
      progress.jsonl            # 可选：监控训练进度，已启用观察脚本时生成
      artifacts/                # 独立、全新的原生 CLI 输出目录
        result.json
        checkpoint/             # 本地；不上传
    YYYYMMDDTHHMMSSfffZ-bench-stock/
      benchmark.json            # 测速结果；完成后看 completed 和实测耗时
```

## 用哪个文件

| 我想做什么 | 选哪个文件 |
|---|---|
| 回放具体动作，或选择目录批量导入 | 选择对局目录，或直接打开其中的 `moves.jsonl`；同一次运行的不同种子/策略可各是一条轨道 |
| 看正在进行的训练 | 将 `progress.jsonl` 添加为监控通道；只显示进程实际发出的阶段和进度 |
| 看已结束的训练或 CPU/GPU 测速 | 添加 `artifacts/result.json` 或 `benchmark.json`；不要把结果摘要当作棋盘回放 |
| 以后复现、比较两次运行 | 可复制 [run.template.json](run.template.json) 为本次目录的 `run.json`，填实际种子、步数上限、设备、代码提交、模型和数据哈希、基线运行 ID |

进度和回放分开写，避免目录里到处放副本。工作台不会把 `run.json` 当回放，也不要求旧结果补这个文件。

## 保存时的约定

- **逐步回放**：每次有效滑动追加一条 JSON 并刷新到文件。记录 `seed`、可选但建议固定的 `policy`（同文件有多次运行时也记录 `run_id`）、从 1 连续递增的 `turn`、英文方向 `action`、移动前 `before`、实际生成新块后的 `after`、累计 `score`。棋盘可用 16 元素行优先数组或 4×4 嵌套数组。原生实时日志的移动后棋盘也可叫 `board`。可另记 `gained`、`max_tile`/`tile`，以及零基 `spawn: [行, 列, 数值]` 或行优先 `spawn_index`。最后一行如已知可写 `stop: "terminal" | "cap" | "stopped"`，上限可另记 `cap`。临时文件不要伪造缺少的走子、计分或结束标记。
- **终局与截断不同**：`terminal` 表示已无合法方向；`cap` 表示仍可能继续，但达到了本次测试上限；`stopped` 是人为停止。原生 `play` 当前把最终类型写在 `result.json`，不回写 `moves.jsonl`。旧日志照常可看；若回放文件单独无法证明上限/停止，而最后仍有合法动作，播放器会标记记录未完，不代表 CLI 一定未结束。400 步测试不是 2048 达成测试。
- **训练进度**：另用追加式 NDJSON `progress.jsonl`，写实际 UTC `timestamp`、`status`、`phase`，有实测时才填 `step`、`total`、`elapsed_seconds` 等。静态的 `result.json` 或 `benchmark.json` 不是实时信息。以工具实际写出的完成状态、耗时为准，不从文件名或文件存在推算训练完成。
- **本地留档**：不覆盖已存在的 `--out`。不需要自动迁移、重命名或改写历史。整个 `runs/` 应被 Git 忽略；原始日志、绝对路径、密钥、权重、缓存和本机检查点不发布。要把结果提升为可公开复现实例，先审查、核验轨迹和汇总，再单独整理进 `experiments/`；不要整目录提交。

## 现有命令示例（PowerShell）

从仓库根目录执行，先按项目现有说明准备并安装本地 Kev 依赖及模型。以下是可选操作，不属于只读工作台。每次新建标记；**不要提前创建传给原生 `--out` 的目录**。

```powershell
$utc = [DateTime]::UtcNow
$day = $utc.ToString("yyyy-MM-dd")
$tag = $utc.ToString("yyyyMMdd'T'HHmmssfff'Z'")
$parent = "runs/$day"
New-Item -ItemType Directory -Force $parent | Out-Null
$base = "assets/qwen-base"
$checkpoint = "assets/official-kev"

$play = "$parent/$tag-play-official-cpu"
python -m kev2048_release.cli play --base "$base" --checkpoint "$checkpoint" --out "$play" --device cpu --threads 2 --seed 96652001 --cap 40

$train = "$parent/$tag-train-smoke-cpu"
python workbench/watch_train.py --events "$train/progress.jsonl" -- --base "$base" --checkpoint "$checkpoint" --data "data/stock_scale" --device cpu --threads 2 --preset smoke --out "$train/artifacts"

$bench = "$parent/$tag-bench-smoke"
python tools/bench_training.py --dry-run --base "$base" --checkpoint "$checkpoint" --out "$bench" --preset smoke

python workbench/live.py --watch "本轮训练:training:$train/progress.jsonl" --watch "训练结果:training:$train/artifacts/result.json" --watch "测速结果:benchmark:$bench/benchmark.json"
```

工作台仍只读；运行最后一行不会启动训练或模型。测速示例仅预览命令，不生成结果；准备好 CPU 与 CUDA 后，明确想运行时才去掉 `--dry-run`。原生 CLI 从检查点读取实际模型身份，例子使用原版公开 Kev，不把它标成历史 V2。
