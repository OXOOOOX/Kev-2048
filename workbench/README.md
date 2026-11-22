# Kev 2048 本地工作台

一个入口，两种查看方式：**监控总览**只读观察已经联接的训练、推理和测速来源；**即时回看**打开你选择的本地逐步对局记录。工作台不会替你开始、暂停、停止训练或改参数。

## 不安装任何东西：即时回看

1. 从克隆的仓库中双击 `index.html`，在 Edge 或 Chrome 打开。
2. 点「即时回看」，选择你本机 `runs/…/moves.jsonl`。也可先用 `samples/sample-events.jsonl` 试一下：这是上一场云端 Kev 推理的真实四步记录，不是实时。
3. 导入一个文件可看单局，同时选多份可按种子比较。界面会优先选择开局一致的两局；如果盘面从开局就不同，会明确提示。不足以还原棋盘的 `benchmark.json` 只能做监控摘要，不能回放。

文件内容只在当前浏览器读取，离线回看不连服务、不上传、不加载模型。

## 有程序在跑：只读监控

默认模式无需模型或 PyTorch。用 Python 3.12 或 3.13，从仓库根目录运行：

```powershell
python workbench/live.py
```

打开 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)。没有指定来源时，页面确实为空；不会自动产生示例数据。要观察单独运行的程序，把允许读取的进度/结果文件显式传给服务；可重复使用 `--watch`：

```powershell
python workbench/live.py --watch "本轮训练:training:runs/local-a.progress.jsonl" --watch "测速结果:benchmark:runs/bench-smoke/benchmark.json" --watch "本轮推理:inference:runs/live/当前对局.jsonl"
```

第一个是训练程序产生的进度流，第二个是完成后才出现的静态结果，第三个是明确联接的推理步序。不存在的文件显示「等数据」；历史结果即使有数据也不会被当成仍在跑。活动日志停止更新会标记过期，服务断线时保留最后一次收到的数据并显示断线。卡片只显示来源已经报告的阶段、计数、耗时或棋盘。绝对路径也可用，含空格或 Windows 盘符时保留引号。

### 如何接入真实训练进度

训练必须由独立进程启动。发行版的训练 CLI 只在部分阶段写出实际特征编码计数；没有记录的阶段不会被补成估计的百分比或损失曲线。可选训练侧采集程序及输入字段见 [TRAINING-WATCH.md](TRAINING-WATCH.md)。看板启动与关闭不影响独立训练进程。

### 想连同一服务中的真实推理

默认只读模式没有启用会下棋的模型，也不接收控制请求。若运行模型的操作者明示选择，可单独启动 `python workbench/live.py --enable-live-console`；工作台仍只读，已有本机会话作为一张推理卡呈现。旧的独立操作者控制台只在此显式模式下存在于 `http://127.0.0.1:8765/console`，不从工作台显示控制。要运行实际模型，使用原先的 [Kev-2048 训练依赖及权重说明](../TRAINING.md)；仓库没有权重、密钥或云端检查点。此处的同机服务只监听回环地址，不支持公网访问，不要转发端口。

## 检查及重建

```sh
python workbench/build_launcher.py
python workbench/test_live.py
python workbench/test_watch_train.py
node workbench/test_dashboard.cjs
```

Python 测试使用模拟工作进程和真实本机 HTTP，不加载模型。Node 测试检查工作台的只读请求、显示区隔以及复用的本地回放控件。未在 Windows 或用户机器上实测。
