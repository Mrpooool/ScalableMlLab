# Repository Guidelines

本仓库用于三人小组完成 Scalable Machine Learning 课程的 Lab 1（CIFAR-10 CNN）与 Lab 2（RAG QA），默认运行与最终复现环境为 **Google Colab（免费 T4 GPU）**。

## 1. 目录结构

```text
ScalableMlLab/
├── AGENTS.md                   # 本文件：团队与 AI 助手统一协作规范
├── task_plan.md                # 当前 Lab 阶段计划、分工与验收清单（planning-with-files）
├── findings.md                 # 作业要求拆解、待定口径、技术调研与踩坑记录
├── progress.md                 # 里程碑进度与已验证实验结果总览
├── docs/
│   └── planning_archive/       # 切换 Lab 时归档旧版规划三文件，保持根目录精简
├── lab1/
│   ├── configs/                # 统一超参配置、固定 train/val 划分索引、损坏集配置
│   ├── src/                    # 可复用 Python 模块（数据、模型、训练、鲁棒性、对抗、指标）
│   ├── notebooks/              # 角色实验 Notebook 与最终提交版 Executed Notebook
│   ├── results/                # 轻量实验指标 JSON、summary.csv、manifest、报告图表
│   └── docs/                   # experiment_contract.md、角色笔记、最终报告草稿
└── lab2/                       # Lab 1 完成后按相同模式展开
```

## 2. 核心协作原则

1. **代码在 `.py`，展示在 `.ipynb`**：
   - 所有模型定义、数据加载、训练循环、评测与绘图逻辑写在 `src/` 的纯 Python 模块中。
   - Notebook 只做 Colab 环境初始化、调用 `src/`、展示关键输出与图表，避免多人修改同一 `.ipynb` 引发 Git 冲突。
2. **契约先行，统一口径**：
   - 开始并行实验前必须先冻结 `lab1/docs/experiment_contract.md`（固定 split、随机种子、计算预算、结果 JSON 字段、计时口径）。
   - 修改共享口径或公共模块接口时，必须同步更新契约文档。
3. **版本锁定交接（大文件不上 Git）**：
   - 严禁提交数据集、`.pt`/`.pth` 权重、向量库或密钥。
   - 共享模型权重存放在团队约定的 Google Drive 目录，仓库内通过 `results/checkpoints_manifest.json` 记录精确文件名、对应 commit、超参和验证集指标。
4. **硬件一致性（Task 6 公平对比）**：
   - 平时调试可使用本地 GPU 或 Colab；每次实验结果 JSON 必须自动记录 `device_name`（如 `Tesla T4`）。
   - 进入最终报告横向对比的训练耗时与推理延迟，必须在统一硬件（默认 Colab T4）上测量并注明环境。
5. **诚实汇报验证边界**：
   - 明确区分 smoke test（少量 epoch / 小样本跑通）与正式全量实验结果；未实测的配置不得写为已验证结论。

## 3. 规划文件与防冲突规范

- 本仓库使用 `planning-with-files` 工作流：`task_plan.md`、`findings.md`、`progress.md`。
- 并行开发期间：
  - 每人在自己的功能分支（如 `feat/lab1-role-a`）工作。
  - 实验产出写入独立文件 `lab1/results/<exp_id>.json`，个人记录写入 `lab1/docs/role_<a|b|c>_notes.md`，减少直接并发修改根目录 `progress.md` 的冲突。
  - 合并 PR 到 `main` 时，同步更新 `task_plan.md` 勾选状态与 `progress.md` 里程碑摘要。
- 提交前运行 `git diff --check`，确保无多余空白或误提交大文件。

## 4. 代码与沟通风格

- Python 使用 4 空格缩进、`snake_case` 函数/变量、`PascalCase` 类名、`UPPER_CASE` 常量，保持简洁直观、避免过度抽象。
- 面向用户的沟通与协作文档默认使用简洁中文（报告正文语言按课程要求为英文）。
