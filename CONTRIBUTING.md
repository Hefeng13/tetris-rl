# 队友协作指南

仓库地址为 https://github.com/Hefeng13/tetris-rl。接受协作邀请后，可以克隆、提交分支并参与讨论。

## 第一次加入

1. 在 GitHub 接受仓库协作邀请。
2. 用 GitHub Desktop 登录自己的 GitHub 账号，选择 File → Clone repository，输入仓库地址。
3. 打开克隆后的文件夹，按照 README 安装依赖。
4. 运行测试，并加载现有模型看一段游戏结果。

也可以在终端克隆。

下面的 CPU 复现命令对应 Python 3.13。GPU 环境的核心依赖和运行命令见 README 的“GPU 运行”。

```powershell
git clone https://github.com/Hefeng13/tetris-rl.git
cd tetris-rl
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest -q
```

主项目支持 Python 3.10 及以上版本。本次 CPU 结果使用 Python 3.13.9 和 requirements-lock.txt 中的依赖，GPU 核心依赖见 requirements-gpu.txt。GPU 环境运行方法见 README。

## 怎样讨论一个想法

在 Issues 点击 New issue，选择“实验与改进讨论”。说明想研究的问题、准备修改的内容和对照方法。每个问题对应一个 Issue，后续的成绩、图表和讨论可以继续写在该 Issue 中。

例如可以建立一个题为“比较 gamma=0.95 与 0.99”的 Issue。写清楚其他配置沿用哪个版本、使用哪些种子，再运行对应的实验。

## 怎样一起修改代码

main 保存已核验的共享版本，个人修改使用独立分支，例如 feat/reward-demo、feat/features 或 docs/report。

GitHub Desktop 的操作顺序如下。

1. 点击 Fetch origin，获取队友的更新。
2. 选择 main 并更新到最新版本。
3. 点击 New branch，为当前改动创建分支。
4. 修改代码或配置，运行相关测试和短实验。
5. 在 Changes 写明修改内容，点击 Commit，再点击 Publish branch。
6. 点击 Create Pull Request，填写改了什么、怎样验证和关联的 Issue。
7. 请队友查看改动和结果，讨论后将改动合并到 main。

对应的终端流程如下，分支名可以按自己的任务修改。

```powershell
git switch main
git pull --ff-only
git switch -c feat/reward-demo

# 完成修改和验证后执行
git add .
git commit -m "Compare a new reward setting"
git push -u origin feat/reward-demo
```

随后在 GitHub 仓库的 Pull requests 页面创建合并请求。

## 怎样安排实验

- 配置文件放在 configs，复制一份并使用能说明实验内容的文件名。
- 一个对照实验集中改变一个因素，记录训练预算、种子、放块上限和评价指标。
- 每次运行使用新的输出目录，例如 runs/alice_gamma99_seed0。
- 在 Issue 或 Pull Request 中记录所用配置、代码提交编号、运行命令和结果。
- 训练和测试入口分别为 src/tetris_rl/train.py 与 src/tetris_rl/evaluate.py。

四组方法分别位于 configs/dqn.yaml、configs/double_dqn.yaml、configs/dqn_shaped.yaml 和 configs/dqn_features13.yaml。每个文件记录该方法的算法、棋盘信息维度和奖励引导开关，训练和评估共用同一份代码。共用参数位于 configs/common.yaml，完整实验清单位于 configs/experiments.yaml。

复制某个方法文件开始新实验时，将副本保存在 configs 目录，配置读取会使用同目录的 common.yaml。运行时指定 --config，例如 --config configs/dqn_features13.yaml，程序会自动选择对应的方法。

建议先执行短训练验证流程。

```powershell
.\.venv\Scripts\python.exe -m tetris_rl.train --steps 2000 --validation-episodes 3 --test-episodes 3 --evaluation-cap 200 --output runs/my_smoke
```

## 文件放在哪里

| 内容 | 位置 |
|---|---|
| 可继续修改的实现 | src/tetris_rl |
| 实验设置 | configs |
| 回归检查 | tests |
| 正式批次的模型、原始记录和源码快照 | runs/formal_v1 |
| 正式逐局成绩、汇总、图表和演示 | results/final |
| 项目报告、核验记录和进度表 | report |

正式批次 formal_v1 作为历史实验依据。新的实验使用新的目录。需要让队友复现时，可以将所需的配置、摘要和图表放进一个新的 results/<实验名> 目录，并在 Pull Request 中明确列出需要加入版本记录的文件。

例如保存了自己整理的实验摘要后，可以明确加入这些产物。

```powershell
git add configs/my_experiment.yaml
git add -f results/my_experiment/summary.csv results/my_experiment/learning_curve.png
```

模型和日志的记录方式可以在对应 Issue 中讨论，再决定共享哪些文件。

## 正式结果的复现

队友克隆后可以加载已有模型进行短评估。

```powershell
.\.venv\Scripts\python.exe -m tetris_rl.evaluate --policy dqn --checkpoint runs/formal_v1/dqn_features13_seed0_attempt1/best.pt --episodes 2 --piece-cap 500 --output results/my_model_check
```

重新汇总已经完成的正式批次。

```powershell
.\.venv\Scripts\python.exe -m tetris_rl.plot --batch runs/formal_v1 --output results/my_formal_summary
```

历史批次保存了 Windows 格式的路径，当前汇总和演示入口会根据克隆后的批次目录定位模型和记录。原始运行代码位于 runs/formal_v1/source_snapshot。新的训练通过当前源代码生成跨平台格式的源码摘要。

## 怎么交流更清楚

Issue 记录问题、想法和实验讨论，Pull Request 展示具体修改和验证结果。修改同一模块前，可以先在 Issue 写明负责范围。分工可以围绕游戏与特征、训练与奖励、结果分析与报告安排。
