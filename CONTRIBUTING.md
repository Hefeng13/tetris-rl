# 协作指南

仓库地址：https://github.com/Hefeng13/tetris-rl

仓库公开，任何人都可以浏览、克隆，并通过 Issue 或 Pull Request 参与讨论。小组成员接受协作邀请后，可以直接向仓库推送分支。

## 1. 首次配置

**方式一：GitHub Desktop**

1. 在 GitHub 上接受仓库的协作邀请。
2. 在 GitHub Desktop 中登录账号，选择 File → Clone repository，输入仓库地址。
3. 打开克隆下来的文件夹，按下面方式二的第 3 步起安装依赖。

**方式二：命令行**

```powershell
git clone https://github.com/Hefeng13/tetris-rl.git
cd tetris-rl
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest -q
```

测试全部通过，即表示环境配置完成。

- 正式结果是在 Python 3.13.9 下、按 `requirements-lock.txt` 的依赖版本跑出来的。项目最低支持 Python 3.10。
- GPU 环境见 README 的「GPU」一节。
- 激活虚拟环境、运行各方法的方式见 README 的「启动」一节。

## 2. 分支与提交流程

`main` 只放已经验证过的共享版本。任何改动都先在自己的分支上做，再通过 Pull Request 合并。

分支命名：

| 前缀 | 用途 | 示例 |
|---|---|---|
| `feat/` | 新实验或新功能 | `feat/gamma99`、`feat/cnn-features` |
| `fix/` | 修复问题 | `fix/heuristic-height` |
| `docs/` | 文档与报告 | `docs/midterm-report` |

**命令行流程**

```powershell
git switch main
git pull --ff-only                  # 先同步最新的 main
git switch -c feat/gamma99          # 新建分支

# 修改代码或配置，运行测试和冒烟实验后：
git add <改动的文件>
git commit -m "Compare gamma 0.95 vs 0.99"
git push -u origin feat/gamma99
```

推送后，在 GitHub 的 Pull requests 页面创建 PR。

**GitHub Desktop 流程**：Fetch origin → 切到 `main` 并 Pull → New branch → 修改、测试 → Commit → Publish branch → Create Pull Request。

**PR 内容**：写清楚改了什么、怎么验证的（测试结果、运行命令、关键数字），以及关联的 Issue。合并前至少找一位队友看过。

## 3. 用 Issue 讨论实验

新实验先开一个 Issue：New issue → 选择「实验与改进讨论」模板。一个问题对应一个 Issue，之后的结果、图表和讨论都接着写在这个 Issue 下面。

Issue 里要写清楚：

- **研究问题**：例如「γ = 0.99 是否优于 0.95」
- **改动**：改了哪个配置项，其余配置沿用哪个版本
- **对照**：和哪个已有配置比较
- **设置**：训练种子、训练步数、评估上限

## 4. 实验规范

**新增配置**：在 `configs/` 下复制一个方法文件，文件名要能看出实验内容（例如 `dqn_gamma99.yaml`），并修改 `id`。加载时会自动合并同目录下的 `common.yaml`。

```powershell
python -m tetris_rl.train --config configs/dqn_gamma99.yaml --seed 0 --output runs/gamma99_seed0
```

**控制变量**：一次只改一个因素。训练步数、种子、评估上限和评价指标，应与对照配置保持一致。

**输出目录**：每次运行都写到新目录，例如 `runs/<名字>_<实验>_seed<N>`，不要覆盖已有结果。

**先跑冒烟测试**：正式训练前，先确认流程能跑通。

```powershell
python -m tetris_rl.train --steps 2000 --validation-episodes 3 --test-episodes 3 `
    --evaluation-cap 200 --output runs/my_smoke
```

**记录**：在 Issue 或 PR 里记下配置文件、代码提交号（`git rev-parse --short HEAD`）、运行命令和结果。

## 5. 文件放在哪里

| 内容 | 位置 | 能否修改 |
|---|---|---|
| 源代码 | `src/tetris_rl/` | 可以，经 PR 合并 |
| 实验配置 | `configs/` | 可以新增；`common.yaml` 改动会影响所有方法，需在 Issue 里先讨论 |
| 回归测试 | `tests/` | 可以，改代码时同步补测试 |
| 正式实验记录 | `runs/formal_v1/` | **不要修改** |
| 正式结果 | `results/final/` | **不要修改** |
| 新实验结果 | `results/<实验名>/` | 新建 |
| 报告 | `report/` | 可以 |

`runs/` 默认不纳入版本管理。要分享新实验的结果，把汇总表和图放进 `results/<实验名>/`，再用 `-f` 强制加入：

```powershell
git add configs/dqn_gamma99.yaml
git add -f results/gamma99/summary.csv results/gamma99/learning_curves.png
```

模型文件（`.pt`）和训练日志要不要上传，先在对应 Issue 里讨论好再决定。

## 6. 复现正式结果

加载已有模型做一次短评估：

```powershell
python -m tetris_rl.evaluate --policy dqn `
    --checkpoint runs/formal_v1/dqn_features13_seed0_attempt1/best.pt `
    --episodes 2 --piece-cap 500 --output results/my_model_check
```

重新汇总正式批次：

```powershell
python -m tetris_rl.plot --batch runs/formal_v1 --output results/my_formal_summary
```

- 正式批次里记录的是 Windows 路径。汇总和演示脚本会按克隆后的目录重新定位文件，在其他系统上也能直接运行。
- 正式实验所用的源码版本，保存在 `runs/formal_v1/source_snapshot/`。

## 7. 分工

改同一个模块之前，先在 Issue 里说明自己负责的范围，避免冲突。可以按下面三块分工：

| 方向 | 涉及文件 |
|---|---|
| 环境与特征 | `env.py`、`features.py` |
| 训练与奖励 | `train.py`、`agents/`、`configs/` |
| 结果分析与报告 | `plot.py`、`scripts/`、`report/` |
