# Kev-2048 Windows 本地实验日志

2026-11-22 至 2026-11-23，Asia/Shanghai。以仓库提交 `8f733011689e5f290dcca7e755d44c631c07c9c7` 为基础。先跑原有的 smoke、stock 和对局，再实现并验证训练特征缓存。以下文件路径相对 Kev-2048 仓库根目录。模型、运行日志和缓存留在本机，驱动保持 `596.47`，未推送。

## 环境和模型

原目录已指向 `https://github.com/OXOOOOX/Kev-2048.git`，但当时只有 `.git`，`main` 尚未检出且没有项目文件。确认状态后执行 `git fetch origin main` 和 `git switch -c main --track origin/main`，补齐检出。

| 项目 | 本机实测环境 |
|---|---|
| 操作系统 | Windows 10 Pro，64 位，build 19045 |
| CPU | Intel Core i9-11950H，8 核 16 线程 |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU，16 GiB，计算能力 8.6 |
| Python / PyTorch | 项目 `.venv` 中的 Python 3.12.11、PyTorch `2.15.0+cu132` |
| CUDA | PyTorch 构建版本 13.2，`torch.cuda.is_available()` 为 `True`，GPU 上的 FP32 矩阵乘法通过 |
| 测试电源 | 接交流电，Windows「平衡」方案 |

1. 用现有的 64 位 Python 3.12.11 建立 `.venv`，升级 pip，并执行 `pip install -e .`。按照 `requirements-native.txt` 安装八项固定版本依赖，用 `--no-deps` 安装 Kev 源码提交 `65b1e763943ace53fd3a4aae00d8d08dbd849772`。
2. 官方 PyTorch wheel 的直接下载速度偏慢，改用官方 CDN 的 HTTPS 分段下载。Windows CPython 3.12 wheel 的 SHA-256 核对为 `14424b1514fc10a8a2352710234869c9af86a9223b582c3767b7512ec038f979`。pip 未从索引顺利解析 `numpy==2.3.5`，随后用 uv 从官方 PyPI 安装原定版本，八项版本均与清单一致。
3. 基座使用 `Qwen/Qwen3.5-0.8B-Base` 的 `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`，下载到 `assets/qwen-base/`。官方 Kev 使用 `jaredpalmer/kev-0.8b` 的 `9a45d25eb2ab761841196625383fa1dff0e56c1e`，下载到 `assets/official-kev/`。
4. Hugging Face Xet 遇到 TLS 握手中断，改用普通 HTTPS。基础权重仍多次断流，改用同一固定修订 URL 的分段下载。原始文件名为 `model.safetensors-00001-of-00001.safetensors`，完整文件 SHA-256 为 `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`。校验后在同目录新增 `model.safetensors` 硬链接，兼容原资产校验脚本。原分片、索引和权重字节保持原样。
5. 离线执行 `kev2048 fetch-assets --out assets`，通过基础权重、官方 head、adapter 权重和 tokenizer 的规定哈希检查。数据校验确认 512 条训练、128 条开发盘面的摘要及隔离条件。测前 5 项训练测试和 3 项基准启动器测试通过。

## Smoke 和 stock 训练基准

两个预设均使用官方初始 head、FP32、2 线程，每种设备运行两次，进程顺序为 CPU → CUDA → CUDA → CPU。Smoke 使用 8 条训练和 8 条开发盘面。Stock 是完整标准配方，使用 512 条训练和 128 条开发盘面，冻结底座和 LoRA，只训练决策头两个 epoch。

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py tools/bench_training.py --preset smoke --threads 2 --repeats 2 --out runs/bench-smoke
& $py tools/bench_training.py --preset stock --threads 2 --repeats 2 --out runs/bench-stock
```

下表是每次进程的总墙钟时间，包含启动、载入、训练、评估、保存和官方重载。运行间没有清除操作系统文件缓存。复跑需更换 `--out`。

| 预设 | 设备 | 第一次（秒） | 第二次（秒） | 中位数（秒） |
|---|---|---:|---:|---:|
| Smoke | CPU | 67.14 | 59.22 | 63.18 |
| Smoke | CUDA | 20.27 | 21.14 | 20.70 |
| Stock | CPU | 944.94 | 926.73 | 935.83 |
| Stock | CUDA | 60.94 | 55.40 | 58.17 |

CUDA 全程加速比为 **3.05 倍（smoke）**和 **16.09 倍（stock）**。两个报告的 `completed` 均为 `true`。所有运行完成两个 epoch，更新了决策头，并通过训练特征和官方重载一致性检查。CPU 与 CUDA 产出的 head SHA-256 不同，各设备的两次训练分别可复现。

| 阶段中位数（秒） | Smoke CPU | Smoke CUDA | Stock CPU | Stock CUDA |
|---|---:|---:|---:|---:|
| 启动后首次模型加载 | 6.47 | 5.09 | 5.13 | 5.83 |
| 训练特征编码与一致性校验 | 21.63 | 1.84 | 718.32 | 27.83 |
| 决策头优化 | 0.11 | 0.23 | 0.74 | 2.21 |
| 保存检查点 | 0.05 | 0.05 | 0.06 | 0.05 |
| 开发集编码和评估 | 10.53 | 0.39 | 186.27 | 6.39 |
| 官方重载与比对 | 13.82 | 4.88 | 15.53 | 5.09 |
| 六阶段以外的进程耗时 | 10.58 | 8.22 | 9.78 | 10.77 |

Smoke 的 CUDA 总耗时主要由加载、重载和六阶段以外的进程开销构成，合计约 88%。Stock 的主要瓶颈是冻结模型的前向计算。训练编码与开发集评估合计占 stock 全程 CPU 的 96.7%、CUDA 的 58.8%。日志记录 `causal_conv1d` 和 `flash-linear-attention` 的可选内核未安装，内核使用 PyTorch 参考实现，CUDA 运行仍在 GPU 上。未测量安装这些内核能节省多少时间。

原始结果位于 `runs/bench-smoke/benchmark.json` 和 `runs/bench-stock/benchmark.json`，各进程保留 `.log`、`result.json` 和 `checkpoint/`。

## CUDA 对局与回放

比较 `assets/official-kev` 和 `runs/bench-stock/01-cuda/checkpoint`，共用相同基座、LoRA 和 tokenizer。统一用 2 线程、种子 `202618` 至 `202622`，每局上限 1,000 步。原版先跑、训练版后跑。模型直接选择合法动作。相同种子保证起局相同，走法分歧后各局按自己的盘面继续生成方块。

```powershell
$play = @('play', '--device', 'cuda', '--threads', '2', '--base', 'assets/qwen-base', '--cap', '1000',
          '--seed', '202618', '--seed', '202619', '--seed', '202620', '--seed', '202621', '--seed', '202622')
& $py -m kev2048_release.cli @play --checkpoint assets/official-kev --out runs/stock-gameplay-cuda/official-kev
& $py -m kev2048_release.cli @play --checkpoint runs/bench-stock/01-cuda/checkpoint --out runs/stock-gameplay-cuda/stock-01-cuda
```

| 种子 | 官方分数 | 最大块 | 步数 | Stock 分数 | 最大块 | 步数 |
|---|---:|---:|---:|---:|---:|---:|
| 202618 | 640 | 64 | 90 | 364 | 32 | 65 |
| 202619 | 680 | 64 | 93 | 644 | 64 | 90 |
| 202620 | 432 | 32 | 70 | 1,216 | 128 | 126 |
| 202621 | 756 | 64 | 101 | 564 | 64 | 76 |
| 202622 | 640 | 64 | 86 | 1,456 | 128 | 155 |

10 局均自然结束，逐步复算了全部 952 步的盘面、落子和分数。这组开局中，Stock 得分 2 次高于官方版、3 次低于官方版，平均分分别为 848.8 和 629.6。5 个开局的结果不足以判断稳定强弱。

- 并排离线回放 `runs/stock-gameplay-cuda/2048-replay.html`，支持选种子、逐步播放和跳到首次走法分岔。
- 官方版逐步记录 `runs/stock-gameplay-cuda/official-kev/moves.jsonl`，汇总在同目录 `result.json`。
- Stock 版逐步记录 `runs/stock-gameplay-cuda/stock-01-cuda/moves.jsonl`，汇总在同目录 `result.json`。

## Stock 训练特征缓存

「Stock 训练特征」是冻结模型对 512 个训练盘面的输出。每个盘面保存一个决策向量，以及每个合法动作的向量。决策头拿这些向量给动作打分。原实现已经在单次训练的两个 epoch 之间复用特征，这次加入跨进程持久化。

缓存由 `--feature-cache` 启用。换决策头时可以读回同一批 FP32 向量。底座、LoRA、tokenizer、校验过的数据、编码源码或配置、设备和运行环境共同决定缓存键，任一变化都会触发重算。缓存校验数据完整性，损坏时重建，写入先生成临时文件再原子替换。每次训练仍用当前初始头重算参考概率，并抽查 8 条样本与官方 Kev 的一致性。开发集在新 head 保存并完成哈希后另行编码。

本次冷、热跑都从官方初始 head 开始，CUDA FP32、2 线程，各启动一个新进程。复跑冷测时指定新的空缓存目录，两次共用该目录，并使用新的输出目录。下例保留本次用过的路径。

```powershell
$fit = @('train', '--preset', 'stock', '--device', 'cuda', '--threads', '2',
         '--base', 'assets/qwen-base', '--checkpoint', 'assets/official-kev',
         '--feature-cache', '.cache/stock-training-features')
& $py -m kev2048_release.cli @fit --out runs/stock-feature-cache-cuda/cold
& $py -m kev2048_release.cli @fit --out runs/stock-feature-cache-cuda/hot
```

| 指标 | 冷跑（未命中） | 热跑（命中） | 节省 |
|---|---:|---:|---:|
| 端到端耗时 | 58.07 秒 | 29.47 秒 | **28.60 秒（49.3%）** |
| 训练特征阶段，含查询与一致性校验 | 28.81 秒 | 2.85 秒 | **25.96 秒（90.1%）** |

两次检查点的 5 个文件 SHA-256 逐项一致，训练 loss 和开发集逐行输出也一致。`head.pt` 的 SHA-256 为 `bb69e3ce559ac0335fb767a4fbacf2eaa0b9148a361c0c6b2c413f5907c95f98`，也与此前未启用缓存的 `01-cuda` stock 基准一致。另用该训练头作为不同的初始头做了一次 CUDA 缓存读取验证，仍命中，并通过 8 条样本一致性校验，最大概率误差约 `1.13e-6`。

本机缓存文件为 `.cache/stock-training-features/e6a726ed15d028ba47d0e727051d669f861a3b60cba4780babd66260bbb450e6.features`，共 9,687,330 字节（约 9.24 MiB）。完整计时和检查点哈希见 `runs/stock-feature-cache-cuda/comparison.json`，`cold.log`、`hot.log` 和两次训练结果在同目录。这是一组冷、热对照，冷指磁盘上尚无训练特征缓存。端到端差值也包含加载等阶段的波动。

缓存相关变更在 `src/kev2048_release/feature_cache.py`、`src/kev2048_release/native.py`、`src/kev2048_release/cli.py` 和 `TRAINING.md`。冷、热实测后补齐了链接目录和 Windows 短时文件占用的处理。最终 12 项单元测试通过，覆盖换头复用、数据或编码配置变化失效、链接目录内的模型文件变化、损坏缓存，以及模拟 Windows 文件占用后的重试。数据校验通过，缓存和本地运行产物由 Git 忽略。
