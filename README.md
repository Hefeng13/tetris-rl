# 俄罗斯方块强化学习课程项目

项目使用 Tetris Gymnasium 0.3.1 的游戏引擎，训练落点决策智能体，并比较随机策略、固定启发式、DQN、Double DQN、奖励塑形和棋盘特征数量。

## 直接查看本次结果

队友登录 GitHub 并接受仓库协作邀请后，可以使用浏览器查看成绩、图表和演示。

[简明结果入口](results/final/README.md) · [完整实验报告](report/results.md) · [逐局测试数据](results/final/episodes.csv)

正式批次 22/22 项已完成，包括 20 次训练、200 万次训练交互和 2200 局独立测试。每份学习模型使用相同的 100 个测试种子，每局上限为 3000 次落子决策。

| 方法 | 平均消行数 |
|---|---:|
| 随机放置 | 0.01 |
| 人工固定规则 | 50.51 |
| 3 维 DQN | 11.89 |
| Double DQN | 12.28 |
| DQN＋奖励塑形 | 78.05 |
| 13 维 DQN | **1191.69** |

学习方法的成绩为 5 次独立训练各自测试成绩的平均值。13 维 DQN 的跨训练标准差为 7.01，98.8% 的测试局达到 3000 次落子上限。奖励塑形组波动较大，其中两份验证选择的检查点位于训练前，详细情况见报告。

### 学习曲线

![五次独立训练的验证学习曲线](results/final/figures/learning_curves.png)

### 13 维 DQN 演示

演示固定使用一个测试种子，上限为 500 次落子。完整评估采用每局 3000 次落子的上限。

![13 维 DQN 游戏演示](results/final/videos/features13_seed0.gif)

[观看或下载 MP4](results/final/videos/features13_seed0.mp4) · [查看全部图表和演示](results/final/README.md)

CPU 核心检查 13 项通过，GPU 环境 14 项检查通过。正式逐局记录、模型、配置与源码快照已经随仓库共享。

预实验记录保存在 runs/pilot_* 和 runs/smoke_*，正式统计使用 formal_v1 批次。

## 团队协作

私有仓库为 [Hefeng13/tetris-rl](https://github.com/Hefeng13/tetris-rl)。队友接受协作邀请后，可以克隆代码、查看正式结果和模型，并通过 Issues 讨论实验，通过 Pull Request 查看和合并改动。

第一次加入的安装步骤、分支操作和实验记录方式见 [队友协作指南](CONTRIBUTING.md)。正式实验依据保存在 runs/formal_v1 和 results/final，新的实验使用独立目录。

## 安装

| 目的 | 使用的工具 |
|---|---|
| 浏览报告、成绩、图片和演示 | 浏览器与已获得仓库访问权限的 GitHub 账号 |
| 在自己的电脑复现本次 CPU 实验 | Python 3.13 与下方命令安装的项目依赖 |
| 同步代码、提交修改和参与评审 | GitHub Desktop 或 Git |
| 使用 NVIDIA GPU 训练或推理 | README“GPU 运行”中的 CUDA 环境与依赖 |

Python 官方下载见 [Python 3.13.9](https://www.python.org/downloads/release/python-3139/)，这是本次 CPU 实验使用的版本。GitHub Desktop 的下载和安装入口见 [官方网站](https://github.com/apps/desktop)。

已验证的本机环境为 Windows、Python 3.13.9 和 CPU PyTorch。完整版本记录见 requirements-lock.txt。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest -q
```

## 运行

从项目根目录执行。output 指向新目录，便于保留每次运行的记录。

```powershell
# 单独评估两个基线
.\.venv\Scripts\python.exe -m tetris_rl.evaluate --policy random --output results/random_new
.\.venv\Scripts\python.exe -m tetris_rl.evaluate --policy heuristic --output results/heuristic_new

# 短训练闭环
.\.venv\Scripts\python.exe -m tetris_rl.train --steps 2000 --validation-episodes 3 --test-episodes 3 --evaluation-cap 200 --output runs/smoke_new

# 单个正式配置
.\.venv\Scripts\python.exe -m tetris_rl.train --experiment dqn --seed 0 --output runs/dqn_new

# 完整批次，4 个配置、每个 5 个训练种子，加 2 个基线
.\.venv\Scripts\python.exe scripts/run_experiments.py --batch runs/formal_new --workers 4

# 汇总已经完成的批次
.\.venv\Scripts\python.exe -m tetris_rl.plot --batch runs/formal_v1 --output results/final

# 生成同一固定种子的 GIF 和 MP4
.\.venv\Scripts\python.exe scripts/render_demo.py --batch runs/formal_v1 --output results/final/videos
```

重新执行已有批次时，运行器校验冻结配置、跳过成功任务，并将失败任务保留在独立的 attempt 目录。训练中断后从该次配置的初始状态重新运行。best.pt 与 last.pt 用于推理评估。

## 目录

```text
configs/experiments.yaml     冻结的正式实验参数
configs/pilot*.yaml          预实验参数及诊断记录
src/tetris_rl/env.py         候选落点和游戏引擎适配
src/tetris_rl/features.py    3/13 维棋盘特征和 float32 缩放
src/tetris_rl/network.py     后续价值网络与候选动作评分
src/tetris_rl/replay.py      经验回放，保留下一候选集合与 mask
src/tetris_rl/agents/        随机、启发式与 DQN/DDQN
src/tetris_rl/train.py       单配置训练和验证
src/tetris_rl/evaluate.py    共用的独立评估入口
src/tetris_rl/plot.py        逐局结果汇总和绘图
scripts/run_experiments.py   有界并行批量运行
scripts/render_demo.py       GIF 与 MP4 演示生成
scripts/write_report.py     结果报告与进度表
tests/                     环境与 TD 目标回归检查
runs/                       配置快照、日志和模型
results/final/              正式表格、图表和演示
report/                     进度表与实验报告
```

## MDP 与网络表示

游戏完整状态涉及棋盘、当前方块、队列和出块机制。3/13 维向量是压缩棋盘观测，存在状态混叠。

每次决策对应一个方块的旋转和落点。10 列乘 4 种旋转共 40 个候选槽位，环境的边框碰撞规则决定合法动作 mask。终止候选保留在合法集合中。所有方法获得相同的候选信息。

棋盘统计为最高高度、空洞数、相邻列高度差之和。13 维设置额外包含每列高度。棋盘特征统一除以 10，另附已知的本次学习奖励和候选终止标记。3/13 维仅指棋盘部分，实际候选表示分别为 5/15 维。网络的可学习输入使用棋盘部分。

候选评分采用以下定义。

```text
Q(s,a) = known_learning_reward(s,a) + gamma * V(afterstate(s,a))
terminal candidate continuation = 0
TD target = observed_learning_reward + gamma * max Q_target(next candidates)
```

Double DQN 使用在线网络选取下一动作，再由目标网络评估。损失为 Huber loss。网络为两层 64 单元 ReLU MLP。已知的一步奖励由模拟器计算，网络学习后续价值。这属于使用一步模型的 afterstate 价值学习设定。

实际落子调用安装包的游戏引擎，候选模拟采用 NumPy 批量计算。回归检查将多个棋盘高度下的所有合法候选与真实引擎执行逐一比较，检查棋盘、奖励、消行和终止。全零特征的正常空棋盘与终止哨兵使用显式标记区分。

## 奖励与评估

基础环境奖励为存活落子 1 加消行数平方乘 10，游戏结束奖励为 -10。学习时基础奖励统一除以 10。安装版本的引擎在游戏结束时覆盖当步奖励，实际统计沿用该行为。

奖励塑形使用 gamma * phi(next) - phi(current)，其中 phi = -(0.05 * maximum_height + 0.5 * holes)，真实终止状态的 phi 为 0。截断继续 bootstrap。

启发式评分固定为 -0.5 * maximum_height - 0.36 * holes - 0.18 * bumpiness + lines_cleared^2，并对已知终止候选施加较低评分。全部候选均终止时仍正常执行合法动作结束游戏。

正式训练每配置使用种子 0 到 4，每次 100000 次落子，训练回合上限 500 块。每 10000 次落子使用验证种子 10000 到 10019，选择平均消行数最高的检查点。验证与最终测试的上限均为 3000 块。步数 0 的初始化模型也参与验证选择，验证并列时选择较晚的检查点。

独立测试种子为 40000 到 40099。开发与预实验使用了另外的测试种子，配置选择发生在正式测试前。测试关闭随机探索。随机基线的动作种子为 42 加测试种子。

先对每个模型的 100 局结果取平均，再统计五个训练种子之间的均值和样本标准差。基线单独报告逐局波动。达到放块上限的比例另列。pieces 记录落子决策次数，包含导致终止的最后一次尝试，作为存活长度指标。

## 产物

每次训练保存完整配置、依赖和源码摘要、训练和验证 CSV、best.pt、last.pt、逐局测试 CSV、summary.json 与 progress.json。源码摘要覆盖 src 下的 Python 实现。

results/final/episodes.csv 汇总所有正式测试回合，summary.csv 汇总配置级结果。report/results.md 在批次完成后记录实际结果、分析和限制。

## 来源

游戏引擎及其依赖沿用各自许可证。本项目的训练、评估、候选批量模拟及实验脚本独立实现。

- [Tetris Gymnasium](https://github.com/Max-We/Tetris-Gymnasium)
- [Gymnasium 终止与截断说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

## GPU 运行

本机已配置 .venv-gpu，复用了 rl-basics 环境中的 PyTorch 2.14.0+cu130，并安装了项目所需依赖。GPU 为 RTX 3060 Laptop 6GB，CUDA 训练、模型保存加载与推理已经验证，GPU 环境中的 14 项检查通过。

```powershell
# 使用已配好的 GPU 环境训练
.\.venv-gpu\Scripts\python.exe -m tetris_rl.train --experiment dqn_features13 --seed 0 --device cuda --output runs/gpu_new

# 用 GPU 加载已有模型评估
.\.venv-gpu\Scripts\python.exe -m tetris_rl.evaluate --policy dqn --checkpoint runs/formal_v1/dqn_features13_seed0_attempt1/best.pt --device cuda --episodes 2 --piece-cap 500 --output results/gpu_evaluation_new

# 重新测量当前小网络的 CPU/CUDA 开销
.\.venv-gpu\Scripts\python.exe scripts/benchmark_device.py --output report/device_benchmark_new.json
```

新建独立 CUDA 环境时，可使用 Python 3.10 及 requirements-gpu.txt 安装经过验证的核心依赖，再执行 pip install -e . --no-deps。CUDA wheel 的来源为 PyTorch 官方 cu130 索引。[官方安装说明](https://pytorch.org/get-started/locally/)

设备选择支持 cpu、cuda 和 auto。当前正式批次的 20 次训练使用 CPU，源码快照保存在 runs/formal_v1/source_snapshot。当前主代码增加了设备选择，同种子 2000 步 CPU 回归训练的最终权重与快照版本逐位一致。新一轮批量实验使用新的 batch 目录。

同一已训练模型、同样两个种子、每局上限 500 次落子的实测为 CPU 2.209 秒、CUDA 3.547 秒，两者平均消行数均为 198。本次小网络和逐步决策方式下，CPU 路径耗时较低。详细记录位于 report/gpu_verification.json 与 report/device_benchmark.json。

完整 GPU 批次也支持设备参数。

```powershell
.\.venv-gpu\Scripts\python.exe scripts/run_experiments.py --batch runs/gpu_formal_new --device cuda --workers 1
```
