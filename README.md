# 俄罗斯方块强化学习

我们的强化学习课程项目：用 DQN 教程序玩俄罗斯方块，并和随机乱放、人工规则做对比。

游戏用的是 [Tetris Gymnasium](https://github.com/Max-We/Tetris-Gymnasium) 0.3.1。程序每次只做一个决定：当前这块放到哪一列、转几次（10 列 × 4 种旋转 = 40 个选择）。

## 结果

每种 DQN 方法独立训练 5 次（种子 0–4），每次都在同样的 100 局测试上算平均消行数，下表是 5 次的平均。每局最多放 3000 块。

| 方法 | 平均消行数 | 说明 |
|---|---:|---|
| 随机放置 | 0.01 | 下限 |
| 人工规则 | 50.51 | 按高度、空洞、凹凸度打分 |
| DQN（3 个特征） | 11.89 | |
| Double DQN | 12.28 | 和普通 DQN 差不多 |
| DQN + 奖励塑形 | 78.05 | 5 次之间差别很大，见报告 |
| DQN（13 个特征） | **1191.69** | 98.8% 的局放满 3000 块都没死 |

结论：**换特征比换算法重要得多**。只加上"每列高度"，成绩从十几行涨到一千多行；把 DQN 换成 Double DQN 几乎没变化。

- 图表和演示视频：[results/final](results/final/README.md)
- 完整实验报告：[report/results.md](report/results.md)
- 全部 2200 局的逐局成绩：[results/final/episodes.csv](results/final/episodes.csv)

![学习曲线](results/final/figures/learning_curves.png)

补充图：
- [5 次训练各自的成绩](results/final/figures/training_seed_scores.png)：看同一方法重复训练稳不稳
- [放满 3000 块的比例](results/final/figures/cap_hit_rates.png)
- [6 种方法的棋盘对比](results/final/figures/board_comparison.png)：同一串方块，各方法堆成什么样
- [每局消行数分布](results/final/figures/episode_lines_distribution.png)
- [存活长度曲线](results/final/figures/survival_curves.png)
- [训练耗时与成绩](results/final/figures/training_time_vs_score.png)

![13 维 DQN 演示](results/final/videos/features13_seed0.gif)

## 怎么做的

| 概念 | 在本项目里 |
|---|---|
| 状态 | 棋盘。网络看到的是压缩后的特征：最高高度、空洞数、凹凸度（3 个），13 维版再加上 10 列各自的高度 |
| 动作 | 当前方块的落点：10 列 × 4 种旋转，撞墙的不能选 |
| 奖励 | 每放一块 +1，一次消 k 行再加 k² × 10，死了 −10 |
| 折扣 γ | 0.95 |

核心做法：先把 40 个落点"放下去之后的棋盘"都算出来，网络给每个棋盘打分，选分最高的那个。

```
落点得分 = 这一步能拿的分（直接算出来） + 0.95 × 网络对落子后棋盘的估值
```

这一步拿多少分是确定的，不需要网络去学；网络只负责判断"这个局面以后还能拿多少分"。

## 6 种方法

**两个基线**（不用训练，用来对比）：

| 方法 | 怎么下 |
|---|---|
| 随机放置 | 在能放的位置里随便挑一个 |
| 人工规则 | 给每个落点打分：−0.5×最高高度 −0.36×空洞数 −0.18×凹凸度 + 消行数²，选最高分。权重是人定的，不会学习 |

**四个 DQN 方法**（都要训练）。它们共用同一套网络和训练流程，只差下面三个开关：

| 方法 | 配置文件 | 算法 | 棋盘特征 | 奖励塑形 | 想验证什么 |
|---|---|---|---|---|---|
| 基础 DQN | `configs/dqn.yaml` | DQN | 3 个 | 关 | 基准 |
| Double DQN | `configs/double_dqn.yaml` | Double DQN | 3 个 | 关 | 换个更准的算法有没有用 |
| 奖励塑形 | `configs/dqn_shaped.yaml` | DQN | 3 个 | 开 | 额外给奖励引导有没有用 |
| 13 维特征 | `configs/dqn_features13.yaml` | DQN | 13 个 | 关 | 给网络更多棋盘信息有没有用 |

- **3 个特征**：最高高度、空洞数、凹凸度（相邻列高度差之和）。
- **13 个特征**：在这 3 个基础上，再加上 10 列各自的高度。
- **Double DQN**：普通 DQN 估计"以后能拿多少分"时容易估高。Double DQN 让两个网络分工：一个选动作，另一个打分，以减少高估。
- **奖励塑形**：除了游戏本身的分数，空洞变多、堆得变高也会被扣一点分，好让程序学得快些。具体权重在 `configs/common.yaml` 的 `shaping` 部分。

## 怎么切换方法

**训练时换方法**：把 `--config` 换成对应的配置文件就行。

```powershell
python -m tetris_rl.train --config configs/dqn.yaml            --seed 0 --output runs/try_dqn
python -m tetris_rl.train --config configs/double_dqn.yaml     --seed 0 --output runs/try_double
python -m tetris_rl.train --config configs/dqn_shaped.yaml     --seed 0 --output runs/try_shaped
python -m tetris_rl.train --config configs/dqn_features13.yaml --seed 0 --output runs/try_f13
```

也可以写成 `--experiment 方法名`，方法名有 `dqn`、`double_dqn`、`dqn_shaped`、`dqn_features13` 四种：

```powershell
python -m tetris_rl.train --experiment double_dqn --seed 0 --output runs/try_double
```

两个参数都不写，默认训练基础 DQN。`--seed` 换成 0–4，就是同一方法的另一次独立训练。

**测试时换方法**：用 `--policy` 选 `random`、`heuristic` 或 `dqn`。选 `dqn` 时要用 `--checkpoint` 指定模型文件，程序会自动识别它是哪种方法、用几个特征。

**批量跑时只跑一种方法**：给 `run_experiments.py` 加上 `--config`，会自动跑这个方法的 5 个种子，再加上两个基线。

```powershell
python scripts/run_experiments.py --config configs/dqn_features13.yaml --batch runs/f13_batch --workers 4
```

**想加一个新方法**：在 `configs/` 里复制一个配置文件（必须放在这个文件夹，它会自动读同目录的 `common.yaml`），改这几行，然后用 `--config` 指定它来训练。

```yaml
id: dqn_features13_double     # 新名字，不能和已有方法重名
algorithm: double_dqn         # dqn 或 double_dqn
board_feature_dim: 13         # 3 或 13
shaping_enabled: false        # true 或 false
```

学习率、训练步数、种子这些所有方法共用的参数，都在 `configs/common.yaml` 里改。

## 安装（Windows，Python 3.13）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest -q    # 测试全过就说明装好了
```

下面的命令都在项目根目录执行，前面统一是 `.\.venv\Scripts\python.exe`，这里简写成 `python`。

## 常用命令

```powershell
# 1. 先跑个小训练，确认流程能走通
python -m tetris_rl.train --steps 2000 --validation-episodes 3 --test-episodes 3 --evaluation-cap 200 --output runs/smoke_test

# 2. 训练一个方法（怎么换方法见上面"怎么切换方法"）
python -m tetris_rl.train --config configs/dqn_features13.yaml --seed 0 --output runs/my_run

# 3. 测试基线
python -m tetris_rl.evaluate --policy random    --output results/random_test
python -m tetris_rl.evaluate --policy heuristic --output results/heuristic_test

# 4. 测试一个训练好的模型
python -m tetris_rl.evaluate --policy dqn --checkpoint runs/formal_v1/dqn_features13_seed0_attempt1/best.pt --output results/my_eval

# 5. 一次跑完全部实验（4 种方法 × 5 个种子 + 2 个基线）
python scripts/run_experiments.py --batch runs/my_batch --workers 4

# 6. 把一批实验汇总成表格和图
python -m tetris_rl.plot --batch runs/my_batch --output results/my_batch

# 7. 生成全部六张补充图及随图数据
python scripts/plot_additional_results.py --batch runs/formal_v1 --output results/my_figures

# 8. 生成演示 GIF / MP4
python scripts/render_demo.py --batch runs/formal_v1 --output results/my_videos
```

注意：
- `--output` 每次都写一个新目录，别覆盖已有结果。
- **不要改** `runs/formal_v1` 和 `results/final`，这是正式结果，报告里的数字都来自这里。
- 批量运行中断了，再执行一遍同样的命令就行，已经完成的任务会自动跳过。

## 代码在哪

| 文件 | 干什么 |
|---|---|
| `src/tetris_rl/env.py` | 算出 40 个落点放下去的结果（棋盘、得分、会不会死），再交给游戏引擎真正执行 |
| `src/tetris_rl/features.py` | 把棋盘变成 3 或 13 个数字 |
| `src/tetris_rl/network.py` | 打分网络（两层 64 个神经元） |
| `src/tetris_rl/agents/` | 三种玩家：随机、人工规则、DQN |
| `src/tetris_rl/replay.py` | 存过去的落子记录，训练时反复抽来学 |
| `src/tetris_rl/train.py` | 训练主程序：边玩边学，定期验证，保存最好的模型 |
| `src/tetris_rl/evaluate.py` | 测试：用固定的一批对局给模型打分 |
| `src/tetris_rl/plot.py` | 汇总成绩、画图 |
| `configs/` | 实验参数。`common.yaml` 是共用参数，另外 4 个文件各对应一种方法 |
| `scripts/` | 批量运行、生成补充图和演示、生成报告 |
| `tests/` | 自动检查，改完代码跑一下 `pytest` |

想看"网络怎么学"，按这个顺序读：`network.py` → `agents/dqn_agent.py` → `train.py`。

## 一次训练会产生什么

每次训练在 `runs/` 下生成一个目录，例如 `runs/formal_v1/dqn_features13_seed0_attempt1/`（第一批正式实验 / 13 维 DQN / 种子 0 / 第 1 次运行）：

| 文件 | 内容 |
|---|---|
| `config.yaml` | 这次用的参数 |
| `train.csv` | 训练时每局的成绩 |
| `validation.csv` | 每 1 万步验证一次的成绩 |
| `best.pt` | 验证成绩最好的模型（测试用这个） |
| `last.pt` | 训练结束时的模型 |
| `episodes.csv` / `summary.json` | 最终测试的逐局成绩 / 汇总 |

`runs/formal_v1/source_snapshot` 存了跑正式实验时的代码，之后代码改了也能对照。

## 实验设置

- 每种方法训练 10 万步（1 步 = 放 1 块），训练时每局最多 500 块
- 每 1 万步用 20 局验证一次，挑成绩最好的存成 `best.pt`
- 最终测试用另外 100 局（种子 40000–40099），每局最多 3000 块，测试时不随机探索
- 验证局和测试局完全分开，保证不是"在考卷上挑模型"

## GPU

支持 `--device cuda`，但网络很小，实测 **CPU 反而更快**（同样任务 CPU 2.2 秒、GPU 3.5 秒），所以正式实验都是 CPU 跑的。GPU 环境安装方法见 `requirements-gpu.txt`。
