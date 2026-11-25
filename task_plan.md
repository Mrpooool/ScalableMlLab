# 项目执行方案（Lab 1）

更新：2026-11-25。当前任务依据：用户提供的[原始作业要求](https://docs.google.com/document/d/1_WypZO1CwatDjMs1Mm4kfbG2GtAOk8y6/edit)、仓库副本 [Lab Assignment 1.docx.md](Lab%20Assignment%201.docx.md) 与官方模板 [ID2223_Lab_Session_2.ipynb](lab1/notebooks/ID2223_Lab_Session_2.ipynb)（Part A9 & Part B）。
目标运行与复现环境：**Google Colab（免费 T4 GPU）**。Lab 2（[Lab Assignment 2.md](Lab%20Assignment%202.md)）待 Lab 1 完成后归档规划文件再启动。

## 目标与交付范围

从零用 PyTorch 基础层实现 CIFAR-10 适配版 **AlexNet** 与 **ResNet-18**（严禁使用预训练权重或 `torchvision.models` 现成构造器），完成 **Task 1–6 + Challenge 2（跨图像域迁移：CIFAR-10 → STL-10）**，最终交付：
1. **已运行完的 Executed Notebook**（基于 Part B 结构，含完整代码、配置、复现步骤、图表、数值对比和参考文献；若 Colab 额度受限可按任务拆分为多本）。
2. **一份完整的最终 Report**（涵盖 Task 1–6 与 Challenge 2 的方法、证据、权衡讨论与选型建议）。

---

## 执行阶段

### Phase 0：需求对齐与起点核对
**Status:** `complete`（剩余三人姓名与共享 Drive 目录名在 Phase 1 同步补填即可，不阻塞开工）

- [x] 阅读 [Lab Assignment 1.docx.md](Lab%20Assignment%201.docx.md) 并拆解 Task 1–6 及 Challenge 要求。
- [x] 建立仓库协作底座（`AGENTS.md`、`.gitignore`、`planning-with-files` 三文件、`lab1/` 目录与契约）。
- [x] 下载并核对 [ID2223_Lab_Session_2.ipynb](lab1/notebooks/ID2223_Lab_Session_2.ipynb) 中的 **Part A1–A9** 与 **Part B** 全部要求，锁定 AlexNet / ResNet-18 适配架构与 `RawImages` + `prepare_inputs` 范式。
- [x] 确定优化后的三人分工（A = Task 1+2+底座；B = Task 3+完整 Task 5；C = Task 4+Challenge 2+Task 6+汇总）。
- [x] 确定目标 Challenge：**Challenge 2 — 跨图像域迁移（CIFAR-10 → STL-10 9 共享类别）**，并冻结 [lab1/docs/experiment_contract.md](lab1/docs/experiment_contract.md) v1.0。

### Phase 1：公共代码底座与本地/Colab 端到端验证（角色 A 牵头）
**Status:** `complete`

- [x] 补填公开仓库地址（`https://github.com/Mrpooool/ScalableMlLab.git`）与团队共享 Google Drive 链接（`experiment_contract.md`、`checkpoints_manifest.json`）。
- [x] 提交默认基线配置 `lab1/configs/baseline.json`、`lab1/requirements.txt` 与初始 `lab1/results/checkpoints_manifest.json`。
- [x] 实现公共模块 `lab1/src/`（含 `RawImages` 无损张量快路径与双镜像下载）与单元测试 `lab1/tests/test_core.py`（6/6 通过）。
- [x] 创建本地 GPU / Colab 双环境自适应启动本：`lab1/notebooks/colab_smoke_test.ipynb`。
- [x] 生成并固化 `lab1/configs/split_indices.json`（45,000 train / 5,000 val，`SPLIT_SEED=2223`，官方 MD5 `c58f30108f718f92721af3b95e74349a` 校验通过）。
- [x] 在本地 **NVIDIA GeForce RTX 5070 Ti Laptop GPU**（`.venv` Python 3.11.9 + `torch 2.11.0+cu128`）跑通 AlexNet & ResNet-18 1-epoch smoke test。

### Phase 2：数据探索与双基线模型（角色 A：Task 1 & Task 2）
**Status:** `complete`

- [x] **Task 1**：完成 CIFAR-10 样本形态、类别分布（`t1_class_distribution.png`、`t1_sample_grid.png`、`task1_data_summary.json`）、train-only 归一化、固定 `image_id` 追踪与随机性控制验证。
- [x] **Task 2**：验证 AlexNet（6.98M）与 ResNet-18（11.17M）各层特征图尺寸与参数量；在 30-epoch / 1.35M 样本预算下完成双基线训练与对比（AlexNet test `82.14%` / 248.2s vs ResNet-18 test `87.01%` / 685.0s），产出学习曲线图、`summary.csv`、[role_a_task1_task2.ipynb](lab1/notebooks/role_a_task1_task2.ipynb) 与 [role_a_notes.md](lab1/docs/role_a_notes.md)。
- [x] 在 `lab1/results/checkpoints_manifest.json` 登记两基线最优 checkpoint（本地权重已生成，待同步上传至共享 Google Drive 供 B/C 直接拉取）。

### Phase 3：并行专题实验（角色 B & 角色 C 并行）
**Status:** `complete`（B/C 结果已整合到提交分支）

- [x] **角色 B — Task 3（容量与正则化）**：在 T4 完成 6 条 30-epoch 受控配置，覆盖半宽、L2 与 Dropout 0/0.25/0.5，并保留完整 train_eval/val 曲线和成本记录。
- [x] **角色 B — Task 5（预测分析与对抗样本）**：在 T4 对 A 冻结权重评测全部 10,000 张测试图，完成类别指标、混淆矩阵、正/误例、FGSM 与同预算随机控制，记录约束检查和 ASR。
- [x] **角色 C — Task 4（鲁棒性与数据增强）**：已在 Colab T4 完成 seed-42 ResNet 无增强/crop+flip 对照及全部冻结配置训练；固定逐图损坏配置 `lab1/configs/corruptions_t4.json`，在相同 10,000 个 test ID 上完成干净/旋转/平移/高斯噪声精度、预测翻转、逐类错误、配对样例与额外开销分析。

### Phase 4：跨图像域泛化与统一硬件基准（角色 C 牵头，A/B 配合）
**Status:** `complete`（仅角色 C 专题交付；不包含 Phase 5 团队统稿）

- [x] **角色 C — Challenge 2（跨图像域迁移：STL-10）**：已对 5 个冻结 checkpoint 在严格 held-out 的 STL-10 test 9 共享类（7,200 张）上完成纯前向评估；保留 10-way/frog 泄露率与 9-way 掩码双口径，对照 CIFAR-10 test 9,000 张同共享类计算总体/逐类 domain drop，并输出混淆矩阵与相同 STL ID 的配对失败案例。STL 未参与模型或增强选择。
- [x] **角色 C — Task 6（统一硬件基准与权衡分析）**：5 个配置均保留 Colab T4、FP32、30-epoch 训练成本；因当日 Colab GPU 配额耗尽，经用户确认，统一推理评测改在 SSH Jade 的 **NVIDIA RTX 2000 Ada Generation**、FP32 上完成（`bs=1/128`、warmup 10、measured 50）。训练耗时只在 T4 组内比较，推理只在 Jade 组内比较；已完成训练时长、seed 与跨硬件敏感性分析，未将不同硬件直接排名。

### Phase 5：最终汇总、复现校验与交付（角色 C 统稿，全员审校）
**Status:** `complete`（提交文件已准备；待团队合并分支和课程平台上传）

- [x] 生成 `lab1/results/submission_summary.csv` 汇总 15 份训练/冻结权重评测记录，分别标明训练与评测硬件；保留 A 原有的两行 `summary.csv`。
- [x] 在 Colab CPU 顺序执行统一 `lab1_submission.ipynb` 的 9 个代码单元，保存 24 幅图和完整源码附录；核对 13 个训练历史、41 个预测条件及 CPU 架构前向。沿用已完成的正式 GPU 实验，不把展示重执行称为重新训练。
- [x] A 的原始结果展示本已在 Colab CPU 执行并保存 5 个单元输出；保留 RTX 训练/测速来源。
- [x] 完成统一英文报告，覆盖 Task 1–6、Challenge 2、引用和复现说明；论文排版修订版为 `lab1_report_paper.pdf`（27 页）及可编辑 Word，原 18 页稿保留。34 项现有单元测试通过；实验与论文格式检查分别见 `submission_validation.json`、`paper_submission_validation.json`。

---

## 已确认三人分工结构（待补填姓名）

| 角色 | 成员 | 负责核心任务 | 关键交付与边界 |
| :--- | :--- | :--- | :--- |
| **角色 A：底座与基线** | TBD | Task 1、Task 2、公共 `lab1/src/` 底座与 Colab 挂载模板 | 固化 `split_indices.json`，实现 AlexNet / ResNet-18 与公共训练/测速模块，交付基线权重与 `checkpoints_manifest.json` |
| **角色 B：容量/正则与预测/对抗分析** | TBD | Task 3、完整 Task 5（错例诊断 + 对抗样本） | 控制变量消融曲线与表格、混淆矩阵与典型正/误例、FGSM/PGD 对抗评估；产出 `results/*.json` 与对应报告章节 |
| **角色 C：鲁棒性、跨域泛化与全局汇总** | TBD | Task 4、**Challenge 2（STL-10 跨域迁移）**、Task 6、最终统稿 | 固定损坏集与数据增强对照、STL-10 跨域泛化评估、T4 统一 benchmark、汇总 Notebook 与 Report 统稿（A/B 提供各自章节与图表） |

> 协作边界：角色 B 和角色 C 均直接复用角色 A 冻结的 `lab1/src/` 与基线权重，Phase 3 起两条线完全并行、互不阻塞；Task 5 中对比“随机损坏 vs 对抗扰动”时，角色 B 直接读取角色 C 冻结的 `corruptions_t4.json` 即可同口径对比。

---

## Errors Encountered

| 日期 | 错误 / 问题 | 尝试次数 | 解决方案 |
| :--- | :--- | :--- | :--- |
| — | 暂无 | — | — |
