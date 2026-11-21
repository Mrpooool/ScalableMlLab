# 项目执行方案（Lab 1）

更新：2026-11-21。当前任务依据：[Lab Assignment 1.docx.md](Lab%20Assignment%201.docx.md) 与官方模板 [ID2223_Lab_Session_2.ipynb](lab1/notebooks/ID2223_Lab_Session_2.ipynb)（Part A9 & Part B）。
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

### Phase 1：公共代码底座与 Colab 验证（角色 A 牵头）
**Status:** `pending`

- [ ] 补填三人姓名映射与共享 Google Drive 路径（`experiment_contract.md`）。
- [ ] 生成并提交 `lab1/configs/split_indices.json`（`SPLIT_SEED=2223`，45k/5k 索引 + train-only RGB 均值/标准差）与默认配置 `lab1/configs/baseline.json`。
- [ ] 实现并 smoke test 公共模块 `lab1/src/`：`seed.py`、`data.py`（含 `RawImages`、`prepare_inputs`、快速镜像下载）、`models.py`（手写 `AlexNetCIFAR`、`BasicBlock`、`ResNet18CIFAR` 及 `trace_shapes`）、`train.py`（含 `train_eval` 确定性评测、峰值显存、已训练样本数统计）、`metrics.py`、`io_utils.py`。
- [ ] 在 Colab T4 跑通 1 epoch smoke test（挂 Drive → 调 `src/` → 落盘 `results/<exp_id>.json` & checkpoint）。

### Phase 2：数据探索与双基线模型（角色 A：Task 1 & Task 2）
**Status:** `pending`

- [ ] **Task 1**：完成 CIFAR-10 样本形态、类别分布、train-only 归一化、固定 `image_id` 追踪与随机性控制验证。
- [ ] **Task 2**：验证 AlexNet 与 ResNet-18 各层特征图尺寸与参数量；在统一计算预算下训练并对比两模型（训练/验证曲线、测试精度、参数量、训练耗时、已训练样本数、峰值 GPU 显存），写清相对原版改动与残差连接作用。
- [ ] 将两基线最优 checkpoint 存至共享 Drive 并登记到 `lab1/results/checkpoints_manifest.json` 供角色 B / C 复用。

### Phase 3：并行专题实验（角色 B & 角色 C 并行）
**Status:** `pending`

- [ ] **角色 B — Task 3（容量与正则化）**：控制变量对比宽度/深度、L2 正则、以及 **无 Dropout + 2 组 Dropout 设置**；用学习曲线 + `train_eval` 确定性评估区分欠拟合、过拟合与优化不充分。
- [ ] **角色 B — Task 5（预测分析与对抗样本）**：输出混淆矩阵、类别级指标、正/误预测代表性样本；基于可微 `prepare_inputs` 在 `[0, 1]` 像素空间实现对抗攻击（FGSM + 可选 PGD），明确攻击成功定义与扰动约束 $\epsilon$，区分随机损坏与定向对抗扰动。
- [ ] **角色 C — Task 4（鲁棒性与数据增强）**：在验证集上确定合适的数据增强策略，并冻结可复现损坏测试集配置 `lab1/configs/corruptions_t4.json`（如旋转、几何形变、高斯噪声）；在同一组评估图像上对比原训练 vs 增强训练在干净/损坏图上的精度、错误模式变化、额外开销及标签歧义。

### Phase 4：跨图像域泛化与统一硬件基准（角色 C 牵头，A/B 配合）
**Status:** `pending`

- [ ] **角色 C — Challenge 2（跨图像域迁移：STL-10）**：在严格 held-out（不参与任何调参/增强选择）的 **STL-10 test 集（9 个与 CIFAR-10 兼容类别，共 7,200 张）** 上评估选定模型（如基线 AlexNet/ResNet 及增强版 ResNet）；对齐标签映射与 `96×96 → 32×32` 预处理，同时报告 10-way 精度（含 `frog` 误判泄露率）与 9-way 掩码精度，并对照 CIFAR-10 test 9 类子集量化域偏移掉点与典型失败案例。
- [ ] **角色 C — Task 6（统一硬件基准与权衡分析）**：在统一 Colab T4 环境（FP32，`bs=1` 与 `bs=128`）汇总训练成本与推理延迟/吞吐；分析训练时长、随机种子、硬件对结论的敏感性，结合精度与鲁棒性给出选型建议。

### Phase 5：最终汇总、复现校验与交付（角色 C 统稿，全员审校）
**Status:** `pending`

- [ ] 汇总全部 `lab1/results/*.json` 生成 `lab1/results/summary.csv`，逐项核对报告数字与图表来源。
- [ ] 在 Colab T4 从头运行最终提交版 Executed Notebook（保留完整输出与引用列表）。
- [ ] 完成并审校最终英文 Report（Task 1–6 + Challenge 2）。

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
