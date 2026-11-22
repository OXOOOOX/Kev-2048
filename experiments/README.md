# 实验记录与验证

这里是云端测试的**已完成记录**，不是训练权重或排行榜。发布数据由此项目记录逐步导出，未复制第三方示例的代码、素材或模型文件。保留了验证棋盘变化所需的字段，不包含任何机器路径、账号、凭据、云端环境凭证或私有配置。

| 记录 | 实验与结论 |
|---|---|
| [D4 对局](results/d4.json) | 4 个相同种子比较冻结 V2 原版与 8 方向预测均值，另有同种子贪心参考，统一 240 步上限。D4 整局按各自终点得分 **0 胜 4 负**，未过预设门槛。纯模型实际决策不调用搜索老师或贪心兜底。 |
| [V2 与 V3-hard 扩展对照](results/wide.json) | 10 个相同新种子，统一 400 步上限，20 局全部自然结束。V2 与 V3-hard 各赢 5 局，未证实 V3-hard 有稳定优势。对贪心的同预算得分表使用各模型对局的步数，或贪心更早终局的步数。 |
| [盘面研究](results/board-studies.json) | 保留拥挤盘复盘假设未通过、V4 小训练未通过、D4 盘面测试通过但不等价于对局胜利，以及原版 Kev 扩充训练的独立盘面测试通过。后者未通过整局门槛，详情见整局测试。 |
| [512 盘面训练后的整局测试](results/stock.json) | 4 个新开局，最多 400 步。候选虽然在独立盘面测试把老师动作命中数从 43 提到 54（共 128），实战比 V2 **0 胜 4 负**。8 局原生模型均自然结束，候选 0/4 到达 512、V2 1/4。贪心单列，4 局中 3 局到上限。 |

测试小样本用于项目筛选，不能说明 2048 达成率稳定。归档的 D4 和 V2/V3 模型对局均未达到 512 方块；这只是这些测试的观察结果。不同方案对步数和种子的选择须分开解释。特别地，相同随机种子会产生相同起局，并不意味着走法分歧后各步落点继续相同。

## 校验（Python 3.9+，无依赖）

在仓库根目录运行：

```text
python experiments/verify_traces.py verify
```

脚本核对 [SHA-256 清单](SHA256.json)、同种子起始棋盘、连续动作、合并得分、生成格、合法动作概率、自然终局和步数上限，并把最终结果与 JSON 摘要对照。它重演棋盘迁移，不运行模型，也不验证随机生成器的内部状态。实验原始日志若可用，也能独立导出到相同格式，须使用不同的输出路径以避免覆盖此发行版中的冻结合集：

```text
python experiments/verify_traces.py normalize d4 path/to/raw-d4.jsonl exported-d4.jsonl
python experiments/verify_traces.py normalize wide path/to/raw-wide.jsonl exported-wide.jsonl
python experiments/verify_traces.py normalize stock path/to/raw-stock.jsonl exported-stock.jsonl --extra-source path/to/raw-greedy.jsonl
```

[对局逐步记录](traces)中的每行是一个 JSON 记录，所有棋盘是从左到右、从上到下展开的一维 16 元素数组。0 表示空格。

| 字段 | 含义 |
|---|---|
| `seed`, `policy`, `turn` | 开局种子、策略标识、从 1 开始的步序。对局以种子与策略一起识别。 |
| `before`, `after` | 滑动前、滑动并生成新方块后的棋盘。 |
| `action` | 本步的实际滑动方向，四个英文方向之一。 |
| `spawn_index` | 新方块在 `after` 中的坐标，将滑动合并后的棋盘与记录盘面比较获得。0 是左上角。 |
| `score`, `max_tile` | 本步后的累计棋分与最大方块。 |
| `stop` | `terminal` 自然结束、`capped` 还有合法走法但到达实验上限，或 `null` 尚未结束。 |
| `choice_probabilities` | 原生模型在合法方向中的选择概率，D4 为 8 个变换映射回原方向后的平均。不是获胜概率；贪心记录无此字段。 |

训练/部署方法详见仓库的训练说明。它们与这里的已录制实验结果不是同一资源。
