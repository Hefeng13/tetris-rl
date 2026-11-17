# Tetris 强化学习：基于 Afterstate 的 DQN

本项目将俄罗斯方块建模为 MDP，采用基于 afterstate 价值的 DQN 学习落点策略，并与随机策略、人工启发式进行对比。环境为 [Tetris Gymnasium](https://github.com/Max-We/Tetris-Gymnasium) 0.3.1，棋盘 20×10。动作空间是对当前方块"旋转 × 列位置"的离散化，共 40 个动作。

## 结果

每种 DQN 配置使用 5 个训练种子（0–4）。每个模型在同一组 100 个测试种子上评估，单局上限 3000 块。下表为 5 个模型测试均值的平均。

| 方法 | 平均消行数 | 备注 |
|---|---:|---|
| 随机策略 | 0.01 | |
| 人工启发式 | 50.51 | |
| DQN（3 维特征） | 11.89 | |
| Double DQN（3 维特征） | 12.28 | 与 DQN 无显著差异 |
| DQN + 奖励塑形 | 78.05 | 跨种子方差大，见报告 |
| DQN（13 维特征） | **1191.69** | 98.8% 的测试局达到上限 |

主要结论：**状态表示比算法改进影响更大**。特征从 3 维扩展到 13 维，平均消行数从约 12 提升到约 1192；从 DQN 换成 Double DQN 则无明显收益。

- 结果页（图表、演示）：[results/final](results/final/README.md)
- 实验报告：[report/results.md](report/results.md)
- 逐局测试数据（2200 局）：[results/final/episodes.csv](results/final/episodes.csv)

![学习曲线](results/final/figures/learning_curves.png)

补充图：
- [各训练种子的测试成绩](results/final/figures/training_seed_scores.png)
- [达到 3000 块上限的比例](results/final/figures/cap_hit_rates.png)
- [相同方块序列下各方法的棋盘对比](results/final/figures/board_comparison.png)
- [单局消行数分布](results/final/figures/episode_lines_distribution.png)
- [存活曲线（含截断）](results/final/figures/survival_curves.png)
- [训练耗时与成绩](results/final/figures/training_time_vs_score.png)

![13 维 DQN 演示](results/final/videos/features13_seed0.gif)

## MDP 建模

| 元素 | 定义 |
|---|---|
| 状态 S | 棋盘、当前方块及方块队列。网络输入为棋盘的压缩特征（3 维或 13 维） |
| 动作 A | 当前方块的旋转（4 种）× 列位置（10 列），越界动作由 action mask 屏蔽 |
| 奖励 R | 每次落子 +1，一次消除 k 行另加 10k²，游戏结束 −10。训练时奖励统一除以 10 |
| 转移 P | 落子与消行由引擎确定性计算，下一方块按 7-bag 规则随机生成 |
| 折扣 γ | 0.95 |
| 终止 / 截断 | 方块无法生成为终止；达到放块上限为截断，截断时仍进行 bootstrap |

## 价值函数

环境对 40 个候选动作分别模拟落子，得到即时奖励 r(s,a) 和落子后的棋盘（afterstate）。网络只学习 afterstate 的价值 V，动作价值由下式给出：

```
Q(s, a) = r(s, a) + γ · V(afterstate(s, a))      终止动作的 V 记为 0
TD 目标 = r + γ · max_{a'} Q_target(s', a')
```

即时奖励由模拟器精确算出，网络只需拟合后续回报，因此学习难度更低。网络结构为 MLP（两个 64 单元隐藏层，ReLU），损失函数为 Huber loss，并使用经验回放、目标网络和 ε-greedy 探索。

## 方法

**基线（无需训练）**

| 方法 | 策略 |
|---|---|
| 随机策略 | 在合法动作中均匀采样 |
| 人工启发式 | 对每个 afterstate 打分：−0.5·最高高度 − 0.36·空洞数 − 0.18·凹凸度 + 消行数²，选取最高分动作；终止动作强制最低分 |

**DQN 配置**：四种配置共用网络结构、训练流程和超参数，仅以下三项不同。

| 配置 | 配置文件 | 算法 | 状态特征 | 奖励塑形 | 实验目的 |
|---|---|---|---|---|---|
| DQN | `configs/dqn.yaml` | DQN | 3 维 | 否 | 基准配置 |
| Double DQN | `configs/double_dqn.yaml` | Double DQN | 3 维 | 否 | 消融：缓解 Q 值高估 |
| 奖励塑形 | `configs/dqn_shaped.yaml` | DQN | 3 维 | 是 | 消融：势函数奖励塑形 |
| 13 维特征 | `configs/dqn_features13.yaml` | DQN | 13 维 | 否 | 消融：状态表示 |

- **3 维特征**：最高高度、空洞数、凹凸度（相邻列高度差的绝对值之和），均除以 10。
- **13 维特征**：在 3 维特征之外加入 10 列各自的高度。
- **Double DQN**：由在线网络选择下一步动作，再由目标网络评估该动作，以降低 max 操作带来的高估偏差。
- **奖励塑形**：基于势函数，附加奖励为 γΦ(s') − Φ(s)，其中 Φ = −(0.05·最高高度 + 0.5·空洞数)，终止状态 Φ = 0。权重见 `configs/common.yaml` 的 `shaping` 字段。

## 切换方法

**训练**：用 `--config` 指定配置文件，或用 `--experiment` 指定配置 ID（`dqn`、`double_dqn`、`dqn_shaped`、`dqn_features13`）。两者都省略时默认为 `dqn`。

```powershell
python -m tetris_rl.train --config configs/dqn.yaml            --seed 0 --output runs/try_dqn
python -m tetris_rl.train --config configs/double_dqn.yaml     --seed 0 --output runs/try_double
python -m tetris_rl.train --config configs/dqn_shaped.yaml     --seed 0 --output runs/try_shaped
python -m tetris_rl.train --config configs/dqn_features13.yaml --seed 0 --output runs/try_f13

python -m tetris_rl.train --experiment double_dqn --seed 0 --output runs/try_double
```

`--seed` 指定训练种子，正式实验使用 0–4。

**评估**：`--policy` 可选 `random`、`heuristic` 或 `dqn`。选 `dqn` 时需用 `--checkpoint` 指定模型，配置 ID 和特征维度会从模型文件的元数据中自动读取。

**批量运行单一配置**：给 `run_experiments.py` 传入 `--config`，会运行该配置的 5 个训练种子和两个基线。

```powershell
python scripts/run_experiments.py --config configs/dqn_features13.yaml --batch runs/f13_batch --workers 4
```

**新增配置**：在 `configs/` 下新建 YAML 文件（加载时会自动合并同目录的 `common.yaml`），然后用 `--config` 指定。

```yaml
id: dqn_features13_double     # 配置 ID，需唯一
algorithm: double_dqn         # dqn | double_dqn
board_feature_dim: 13         # 3 | 13
shaping_enabled: false        # true | false
```

共享超参数（学习率、训练步数、种子等）统一在 `configs/common.yaml` 中修改。

## 安装（Windows，Python 3.13）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest -q
```

以下命令均在项目根目录执行，`python` 指 `.\.venv\Scripts\python.exe`。

## 常用命令

```powershell
# 冒烟测试：短时训练，检查流程能否跑通
python -m tetris_rl.train --steps 2000 --validation-episodes 3 --test-episodes 3 --evaluation-cap 200 --output runs/smoke_test

# 训练单个配置
python -m tetris_rl.train --config configs/dqn_features13.yaml --seed 0 --output runs/my_run

# 评估基线
python -m tetris_rl.evaluate --policy random    --output results/random_test
python -m tetris_rl.evaluate --policy heuristic --output results/heuristic_test

# 评估已训练模型
python -m tetris_rl.evaluate --policy dqn --checkpoint runs/formal_v1/dqn_features13_seed0_attempt1/best.pt --output results/my_eval

# 完整实验：4 种配置 × 5 个种子 + 2 个基线
python scripts/run_experiments.py --batch runs/my_batch --workers 4

# 汇总结果表与学习曲线
python -m tetris_rl.plot --batch runs/my_batch --output results/my_batch

# 生成补充图及对应数据
python scripts/plot_additional_results.py --batch runs/formal_v1 --output results/my_figures

# 生成演示 GIF / MP4
python scripts/render_demo.py --batch runs/formal_v1 --output results/my_videos
```

- `--output` 请指向新目录。
- `runs/formal_v1` 和 `results/final` 是正式结果，报告中的数据均来自这两处，请勿修改。
- 批量运行支持断点续跑：重复执行同一命令时，已完成的任务会自动跳过。

## 代码结构

| 路径 | 说明 |
|---|---|
| `src/tetris_rl/env.py` | 环境封装：批量模拟 40 个候选落子（afterstate、奖励、是否终止），再调用引擎执行真实转移 |
| `src/tetris_rl/features.py` | 棋盘特征提取（3 维 / 13 维）与势函数 |
| `src/tetris_rl/network.py` | 价值网络与 Q 值计算 |
| `src/tetris_rl/agents/` | 随机策略、人工启发式、DQN / Double DQN 智能体 |
| `src/tetris_rl/replay.py` | 经验回放缓冲区，保存下一状态的全部候选及其 action mask |
| `src/tetris_rl/train.py` | 训练主循环：ε-greedy 采样、TD 更新、目标网络同步、验证与检查点选择 |
| `src/tetris_rl/evaluate.py` | 在固定测试种子上评估策略 |
| `src/tetris_rl/plot.py` | 结果汇总与绘图 |
| `configs/` | 实验配置：`common.yaml` 为共享参数，其余 4 个文件各对应一种配置 |
| `scripts/` | 批量运行、补充图、演示和报告生成 |
| `tests/` | 回归测试：候选模拟与引擎一致性、TD 目标、模型保存与加载等 |

建议阅读顺序：`env.py` → `network.py` → `agents/dqn_agent.py` → `train.py`。

## 运行产物

每次训练会在 `runs/` 下生成一个目录，命名格式为 `批次/配置_种子_尝试次数`，例如 `runs/formal_v1/dqn_features13_seed0_attempt1/`。

| 文件 | 内容 |
|---|---|
| `config.yaml` | 本次运行的完整配置 |
| `metadata.json` | 设备、依赖版本、随机种子 |
| `train.csv` | 每个训练回合的消行数、回报、ε、TD loss |
| `validation.csv` | 每 10,000 步的验证成绩 |
| `best.pt` / `last.pt` | 验证成绩最优的检查点 / 训练结束时的检查点 |
| `episodes.csv` / `summary.json` | 最终测试的逐局成绩 / 汇总统计 |

`runs/formal_v1/source_snapshot` 保存了正式实验所用的源码版本。

## 实验设置

| 项目 | 设置 |
|---|---|
| 训练步数 | 每个种子 100,000 步（1 步 = 1 次落子），训练回合上限 500 块 |
| 超参数 | Adam，学习率 1e-3，batch 64，回放容量 20,000，每 4 步更新一次，目标网络每 1,000 步同步 |
| 探索 | ε 在前 60% 的训练步数内从 1.0 线性衰减到 0.05 |
| 检查点选择 | 每 10,000 步在验证种子 10000–10019 上评估，保留平均消行数最高的检查点（包括第 0 步） |
| 测试 | 测试种子 40000–40099，与验证集互不重叠；单局上限 3000 块，ε = 0 |
| 统计方式 | 先计算每个模型 100 局的均值，再计算 5 个训练种子间的均值和标准差 |

## GPU

支持 `--device cuda | cpu | auto`。由于网络规模小、决策逐步进行，实测 CPU 比 GPU 更快（同一评估任务 CPU 2.2 s，CUDA 3.5 s），因此正式实验均在 CPU 上完成。GPU 依赖见 `requirements-gpu.txt`。
