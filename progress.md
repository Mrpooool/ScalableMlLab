# 项目进度

更新：2026-11-23。**Lab 1 Phase 1 公共代码底座与 Colab Smoke Test Notebook 已完成并推送至 GitHub `main`；下一步在 Colab T4 运行 1-epoch smoke test 并固化 `split_indices.json`。**

## 2026-11-23：实现 Phase 1 公共模块与 Colab 启动模板

- 录入公开 GitHub 仓库（`https://github.com/Mrpooool/ScalableMlLab.git`）与团队共享 Google Drive 链接至 [lab1/docs/experiment_contract.md](lab1/docs/experiment_contract.md) 及 [lab1/results/checkpoints_manifest.json](lab1/results/checkpoints_manifest.json)。
- 实现 `lab1/src/` 六个公共核心模块：
  - `seed.py`：固定 Python/NumPy/PyTorch/CUDA 与 DataLoader worker 随机种子。
  - `data.py`：CIFAR-10 Brainchip 镜像加载、`SPLIT_SEED=2223` 分层 45k/5k 划分与 train-only 归一化、`RawImages` + 可微 `prepare_inputs`、以及 Challenge 2 的 `STL10Shared9Dataset`（9 共享类映射）。
  - `models.py`：从零手写 Part A9 规范的 `AlexNetCIFAR`、`BasicBlock`、`ResNetCIFAR`（支持 Task 3 宽度/深度/Dropout 消融）与 `trace_shapes`。
  - `metrics.py`：确定性评测、CIFAR-10 9 共享类与 STL-10 跨域双口径评测（10-way + `frog` 泄露率 + 9-way 掩码）、CUDA 同步推理 benchmark（`bs=1` 与 `bs=128`）。
  - `io_utils.py`：原子 JSON/CSV 落盘、环境信息采集、`checkpoints_manifest.json` 登记与 `summary.csv` 汇总。
  - `train.py`：统一 `ExperimentConfig` 与 `run_experiment` 训练循环（含 `train_eval`、峰值显存、已训练样本数、checkpoint 保存与标准 JSON 输出）。
- 新增 [lab1/configs/baseline.json](lab1/configs/baseline.json) 与开箱即用的 [lab1/notebooks/colab_smoke_test.ipynb](lab1/notebooks/colab_smoke_test.ipynb)。
- **验证边界**：本地 Python 3.14 环境未安装 PyTorch，已完成全部 `.py` 编译检查（`py_compile`）、跨模块导入符号检查、JSON 配置解析与 Notebook AST 语法检查；**实际 PyTorch 前向/反向训练、CUDA 峰值显存与 Drive 挂载读写尚未实测**，待在 Colab T4 运行 `colab_smoke_test.ipynb` 验证。

## 2026-11-21：初始化协作底座、核对 Part A9、冻结分工与 Challenge 2 方案

- 阅读 [Lab Assignment 1.docx.md](Lab%20Assignment%201.docx.md) 与 [Lab Assignment 2.md](Lab%20Assignment%202.md)，确定默认环境为 **Google Colab（T4 GPU）**。
- 将本机 `planning-with-files` skill 通过 Junction 链接到 Antigravity 全局 skills 目录。
- 解析课程官方模板 [ID2223_Lab_Session_2.ipynb](lab1/notebooks/ID2223_Lab_Session_2.ipynb)，对齐 Part A1–A9（CIFAR-10 适配版 AlexNet / ResNet-18、Brainchip 镜像、`SPLIT_SEED=2223` 45k/5k 划分、train-only 归一化、`RawImages` + 可微 `prepare_inputs`、`train_eval`、CUDA 同步测速）及 Part B 补充指标。
- 确认最终三人并行分工结构：
  - **角色 A**：Task 1 + Task 2 + 公共 `lab1/src/` 底座与双基线权重
  - **角色 B**：Task 3 + 完整 Task 5（错例诊断 + 对抗样本）
  - **角色 C**：Task 4 + **Challenge 2（跨图像域迁移：CIFAR-10 → STL-10 9 共享类别）** + Task 6 & 最终统稿
- 冻结 [lab1/docs/experiment_contract.md](lab1/docs/experiment_contract.md) 为 **v1.0**，明确 STL-10 标签映射表、双口径跨域评估（10-way 含 `frog` 泄露率 + 9-way 掩码对照 CIFAR-10 同 9 类）及所有实验落盘 JSON 字段。
- **下一步待办**：
  1. 进入 Phase 1：实现 `lab1/src/` 公共核心模块与 `lab1/configs/`，生成固定 `split_indices.json` 与 Colab 启动脚本/Notebook。
  2. 在 Colab T4 上完成 1 epoch smoke test，确认 AlexNet / ResNet 学习率配置后进入 Phase 2 正式基线训练。
