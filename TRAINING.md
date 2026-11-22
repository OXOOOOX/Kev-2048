# Running the native Kev training from a clone

This is real offline PyTorch training through the official upstream **Kev** API, not a hosted TypeSafe Jev call. It leaves the Qwen base and Kev LoRA frozen and trains a copy of the existing pointer head only. It changes no original weights, uses no search at gameplay inference, and writes a separate, officially reloadable checkpoint directory. The no-model validation command does not train.

## Setup

Use Python 3.12 or 3.13 (cloud verified with 3.12), Git, and a clean virtual environment. From the repository root:

```sh
python -m venv .venv
# POSIX: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
kev2048 verify-data
python -m unittest discover -s tests -p test_training_release.py
```

The rule and archived seeded-trace tests and 640-fixture digest/split verifier need no ML dependencies or network. For native commands, install PyTorch using its [official platform selector](https://pytorch.org/get-started/locally/) (CPU recommended for the historical comparison; Windows/NVIDIA users choose a CUDA wheel matching their driver). Then install the tested dependency snapshot and pinned Kev source:

```sh
python -m pip install -r requirements-native.txt
python -m pip install --no-deps "git+https://github.com/jaredpalmer/kev.git@65b1e763943ace53fd3a4aae00d8d08dbd849772"
python -c "import torch, kev; print(torch.__version__, torch.cuda.is_available())"
```

The cloud test used PyTorch 2.15.0+cpu, Transformers 5.24.0 and the other versions listed. The upstream checkout's dependency constraints differ from the cloud runtime, so the source install deliberately does not replace those versions. CPU smoke is verified. Windows CUDA is an opt-in FP32 path, and its results can differ numerically from CPU. If deterministic CUDA kernels or native parity reject it, use CPU; no precision switch happens automatically. Budget 8 GB free host RAM and approximately 2 GB disk for model assets. GPU memory needs depend on its runtime, not only on the size of the weight files.

## Pinned model downloads

From the repository root, the following downloads public files into a local gitignored directory. No training data, prompt, or new checkpoint is sent to the hub.

```sh
kev2048 fetch-assets --out assets
```

It resolves [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` and [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) revision `9a45d25eb2ab761841196625383fa1dff0e56c1e`. The command verifies model, adapter, tokenizer and released pointer hashes against pinned manifests. If one of those assets already exists locally, use its directory directly instead of downloading. The native commands are offline and expect the base snapshot's config, tokenizer, and safetensors; checkpoint directory with `head.pt`, `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.json`, and `tokenizer_config.json`. Only use a trusted official or locally generated checkpoint.

## Genuine smoke, full preset, and a new game trace

```sh
kev2048 train --preset smoke --base assets/qwen-base --checkpoint assets/official-kev --out runs/smoke
kev2048 play --base assets/qwen-base --checkpoint runs/smoke/checkpoint --seed 202618 --cap 3 --out runs/smoke-game
kev2048 train --preset stock --base assets/qwen-base --checkpoint assets/official-kev --out runs/stock-from-released
```

The smoke uses 8 train and 8 dev boards, runs both optimization epochs, checks that pointer parameters moved, reloads the final checkpoint using official Kev, and writes legal probabilities/results. The play command invokes that actual checkpoint once per board, writes each move and its spawned tile, then writes final score, max tile, and terminal/cap. The Linux CPU release smoke completed both epochs and matched all eight official reload argmaxes (max probability error about 2.3e-6). Smoke has no gameplay or quality gate. A complete stock run fits 512 train and screens 128 disjoint dev boards; runtime depends on available compute. Change the new output directory name for each run; existing output paths are refused rather than overwritten.

For CUDA, append `--device cuda` to `train` and `play` commands. Development scoring, cache, training, and inference retain FP32 and ordered native legal argmax. Published numbers are CPU only.

The historical 512-board experiment started from an earlier privately generated V2 pointer, not directly from the official released pointer. This repo does not contain the historical checkpoint. If you have a separately authorized local copy of its checkpoint directory, run the same exact recipe with its path and a strict SHA check:

```sh
kev2048 train --preset stock --base assets/qwen-base --checkpoint assets/your-v2 --require-historical-init --out runs/stock-from-v2
```

A fresh clone using official weights genuinely executes the same fit but is **not** expected to reproduce the cloud V2→candidate outcomes. Its mathematical gate statistic is separated from eligibility for the historical protocol, and the artifact records which initialization it actually used. Torch/platform versions may change numerical results even with the exact V2 starting SHA.

## Reuse stock training features across runs

Pass the same `--feature-cache` directory to stock training commands. Use a fresh `--out` for each run:

```sh
kev2048 train --preset stock --device cuda --threads 2 --base assets/qwen-base --checkpoint assets/official-kev --feature-cache .cache/stock-training-features --out runs/stock-cached
```

The cache stores the frozen encoder's FP32 vector for the decision position and one vector per legal action for each of the 512 training boards. A second run can load those vectors from disk even when initialized with a different pointer head. Each run computes its own reference probabilities and checks eight cached boards against official Kev. Development features are computed after saving and hashing the new checkpoint.

The cache key includes the local base and adapter files, tokenizer, checked data, encoding implementation and settings, and device/runtime. Changes to those inputs create a new entry. Invalid or corrupted cache entries are rebuilt. `training_feature_cache` in `result.json` records the hit or miss, and the existing training encoding timer includes the lookup, parity check, and any encoding or write. The cache is optional, and `.cache/` is gitignored.

## Benchmark the Windows laptop (PowerShell)

After the setup and the one-time model download above, run from the cloned repository root. The full benchmark is offline. First confirm that PyTorch can see the NVIDIA GPU; if `cuda_available` is False, install a CUDA-enabled PyTorch build for your driver before attempting a comparison.

```powershell
python -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda, 'cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python tools/bench_training.py --preset smoke --threads 2 --repeats 2 --out runs/bench-smoke --dry-run
python tools/bench_training.py --preset smoke --threads 2 --repeats 2 --out runs/bench-smoke
```

Use AC power and the same laptop power profile. After smoke passes, use `--preset stock --out runs/bench-stock` for the 512-train/128-dev comparison. The output path must be new. You can pass `--base`, `--checkpoint` and `--data` if your files are elsewhere. The launcher uses the active Python environment, exactly the same initial head and data on CPU/CUDA, fixed thread settings, separate new output directories, and alternates the run order between repeats. Every run starts in a fresh process; OS file caches are not cleared.

Read `runs/bench-smoke/benchmark.json` for the median, per-run elapsed time, and speed ratio. The total includes imports, model loading, training, dev evaluation, checkpoint save and official reload. Each training result also times initial model load, feature encoding with the parity check, head optimization, checkpoint save, dev evaluation, and reload separately. CUDA is synchronized at phase boundaries so queued work is included in the phase that launched it. Use the stock result when judging training throughput; the tiny smoke run is more sensitive to startup. If CUDA fails or the inputs do not match, the script records the failing log and reports no speed ratio. It never silently changes precision or retries on CPU.

The [Windows CUDA run log](experiments/windows-cuda-2026-11-22.md) records measured smoke and stock timings on an RTX 3080 Laptop GPU, plus the paired gameplay and feature cache experiments.

## Protocol and limits

The 512/128 fixtures are synthetic reachable 2048 positions. Rewards are standard merge points; teacher labels are local depth-3-after-root expectimax heuristic values, not game scores. Fixtures are already labeled and verified by SHA; the clone does not need to download the teacher or relabel. The full historical source corpus, raw caches, model weights, and older experimentation remain cloud-side. Only this bounded, parameterized final experiment uses the bundled fixtures.

Stock fit: FP32, AdamW LR 8e-5 / decay .02, two epochs, batch 16, gradient cap 1, ties handled as summed mass over best set, plus 0.15 KL against frozen initial policy. No dev row goes through the model until the final head bytes are hashed. Official reload checks eight dev predictions and rechecks cached-head outputs for all dev boards. Run results contain paired hits, exact hits and raw teacher regret, plus separate crowded and difficulty slices. Gameplay uses the repository's independent Python 2048 rules. The capped score is not a natural loss, and best-set hit rates are not a measured chance of reaching 2048.
