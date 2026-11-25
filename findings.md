# 项目发现与决策依据（Lab 1）

更新：2026-11-21。依据：[Lab Assignment 1.docx.md](Lab%20Assignment%201.docx.md) 与官方模板 [ID2223_Lab_Session_2.ipynb](lab1/notebooks/ID2223_Lab_Session_2.ipynb)。

## 2026-11-25 原文核对与提交口径

用户提供的 [Google 文档原文](https://docs.google.com/document/d/1_WypZO1CwatDjMs1Mm4kfbG2GtAOk8y6/edit) 与仓库要求副本的 Task 1-6/Challenge/Deliverables 内容一致。原文要求一份已执行 Notebook 和一份覆盖全部任务的报告；Colab 受限时允许分开 Notebook，报告仍须统一。原文没有要求将所有已有实验重新在 Colab 训练，也没有规定所有测量必须同一张卡。团队 AGENTS.md 要求成本横向比较使用一致硬件；本次保留 C 已约定的 T4 训练/Jade 推理分组和 B 自己的 T4 消融组，A 的 RTX 数字仅作独立历史记录。

最终统一 Notebook 在 Colab CPU 审计并展示真实保存结果，包含完整实现的可读附录。该执行不改变历史 GPU 训练和测速来源。整合使用 `submission_summary.csv`，避免覆盖 A Notebook 使用的原始两行 `summary.csv`。

## 1. 作业与 Part B 硬性要求摘录

| 项目 | 作业 + Part B 补充要求 |
| :--- | :--- |
| 数据集与下载 | CIFAR-10（50k train / 10k test，RGB 32×32，10 类）；Part A1 提供高速镜像 `https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz`（md5 `c58f30108f718f92721af3b95e74349a`） |
| 严禁事项 | **禁止使用预训练权重，禁止调用 `torchvision.models` 现成模型构造器**；必须用 `nn.Conv2d`、`nn.BatchNorm2d`、`nn.ReLU`、`nn.MaxPool2d`、`nn.Linear` 等手写 |
| Task 1 | 样本与类别分布分析、固定 stratified train/val 划分（test 严格保留至最后）、**仅在训练集上计算归一化均值/标准差**、保存 `image_id` 便于逐图追踪、控制随机性 |
| Task 2 | 从 Part A9 适配版起步手写 AlexNet & ResNet-18；追踪特征图形状、统计参数量、训练耗时、**已训练样本总数（number of trained samples）**、**峰值 GPU 显存**；说明相对原版改动与残差连接作用 |
| Task 3 | 对比宽度/深度、L2 正则、以及 **无 Dropout + 2 组 Dropout 概率**；要求结合学习曲线与**确定性 `train_eval` / `val` 评估**区分欠拟合、过拟合与优化不充分 |
| Task 4 | 区分随机训练增强与可复现评估损坏；**在验证集上开发方法，冻结后再在测试集做最终评估**；同一组评估图对比干净/损坏性能、错误模式变化、额外开销及标签歧义 |
| Task 5 | 定量评估 + 混淆矩阵 + 类别级误差 + 正/误例可视化；生成对抗样本（明写攻击方法、扰动约束与**攻击成功定义**），区分随机损坏与定向优化扰动 |
| Task 6 | 明确说明推理测速协议（**设备、精度、batch size、warmup + CUDA sync**），结合精度、鲁棒性与计算预算讨论训练时长/随机种子/硬件敏感性并给出选型建议 |
| **Challenge 2（已锁定：跨图像域迁移）** | 在标签兼容的新图像域（**STL-10**）上做严格 held-out 评估；不得参与模型或增强选择；解释分布偏移、标签兼容性与预处理，分析迁移效果、失效案例与泛化边界 |

---

## 2. 从 `ID2223_Lab_Session_2.ipynb` 提取的关键技术起点

### 2.1 Part A 沿用设计
1. **划分与统计口径（A2）**：`SPLIT_SEED = 2223`，`train_test_split(..., test_size=0.1, stratify=labels)` → `45,000 train / 5,000 val`（索引升序保存）；仅用 `train_ids` 算 RGB 通道均值/标准差。
2. **`RawImages` + 可微 `prepare_inputs`（A2 & A8）**：Dataset 返回 `(raw_tensor [0,1], target, image_id)`；归一化与可选 resize 放进 `prepare_inputs(raw, stats, size=32)`，支持逐样本配对与 `[0, 1]` 像素空间对抗梯度反传。
3. **确定性训练评估 `train_eval`（A5 & Part B Task 3）**：无 shuffle、`eval()` 模式，消除训练过程 Dropout/权重滑动干扰。
4. **层输出形状追踪 `trace_shapes`（A4）**与 **GPU 同步测速规范（A8）**。

### 2.2 Part A9 推荐的 CIFAR-10 架构规范
- **AlexNet（`3×32×32`，约 6.97M 参数）**：5 个 `3×3` 卷积（64 → 192 → 384 → 256 → 256）+ 3 个 `2×2` MaxPool → `Flatten(4096)` → `FC(4096→1024)+ReLU+Drop(0.5)` → `FC(1024→512)+ReLU+Drop(0.5)` → `FC(512→10)`。
- **ResNet-18（`3×32×32`，约 11.17M 参数）**：`Conv3×3(3→64, s=1, p=1)+BN+ReLU`（去掉 7×7 与初始 MaxPool）→ 4 Stage（64 / 128 / 256 / 512，各 2 个 `BasicBlock`，后 3 Stage 首 block `stride=2` + `1×1 Conv+BN` shortcut，先加残差再 ReLU）→ `AdaptiveAvgPool2d(1)` → `Linear(512→10)`。

---

## 3. 已锁定分工与 Challenge 2（跨图像域迁移）设计

### 3.1 优化后的三人分工
- **角色 A**：Task 1 + Task 2 + `lab1/src/` 公共底座与双基线 checkpoint。
- **角色 B**：Task 3（容量/正则）+ 完整 Task 5（错例分析 + 对抗样本）。
- **角色 C**：Task 4（数据增强与损坏）+ **Challenge 2（跨图像域迁移）** + Task 6 基准汇总与最终统稿。
- **优势**：Task 3+5 聚焦模型容量/决策边界/错误诊断；Task 4+Challenge 2 聚焦鲁棒性与分布偏移；角色 B、C 拿到 A 的底座后完全并行，互不等待。

### 3.2 Challenge 2：CIFAR-10 → STL-10 跨图像域迁移协议
选用 `torchvision.datasets.STL10(split="test", download=True)` 作为严格 held-out 域外测试集（Colab 原生 `torchvision` 一行下载，无需手动传数据；原始共 8,000 张 `96×96` RGB 图像，10 类每类 800 张）。

**1. 分布偏移来源（报告重点讨论）**：
- **图像来源与构图偏移**：CIFAR-10 来自 Tiny Images（原生 32×32、主体常居满画面且较模糊）；STL-10 来自 ImageNet 高清摄影图（原生 96×96、背景更复杂、物体尺度与视角变化更大）。
- **下采样预处理偏移**：评测时用与 `prepare_inputs` 一致的 `bilinear + antialias=True` 将 STL-10 从 `96×96` 下采样至 `32×32`，并严格复用 **CIFAR-10 训练集统计量** 归一化（不偷看目标域统计量，防止数据泄漏）。

**2. 类别兼容性与精确映射（极其关键，严防标签错位！）**：

| STL-10 索引 | STL-10 类名 | 映射至 CIFAR-10 索引 | CIFAR-10 类名 | 说明 |
| :---: | :--- | :---: | :--- | :--- |
| 0 | `airplane` | 0 | `airplane` | 共享 |
| 1 | `bird` | 2 | `bird` | **索引不同！** |
| 2 | `car` | 1 | `automobile` | **类名与索引均不同！** |
| 3 | `cat` | 3 | `cat` | 共享 |
| 4 | `deer` | 4 | `deer` | 共享 |
| 5 | `dog` | 5 | `dog` | 共享 |
| 6 | `horse` | 7 | `horse` | **索引不同！** |
| 7 | `monkey` | — | *(剔除)* | CIFAR-10 无此类，从评测集中移除 |
| 8 | `ship` | 8 | `ship` | 共享 |
| 9 | `truck` | 9 | `truck` | 共享 |
| — | *(无对应)* | 6 | `frog` | STL-10 无此类，仅可能作为模型误判目标 |

- 过滤 `monkey` 后保留 **9 个共享类别 × 800 = 7,200 张** STL-10 测试图。
- **双口径指标（强烈建议同时汇报）**：
  1. **10-way 标准推理精度** + **`frog` 泄露率（预测为 `frog` 的比例）**：反映未修改模型直接部署到新域时的真实表现。
  2. **9-way 掩码精度（屏蔽 `frog` logit 后的 argmax）**：并与模型在 **CIFAR-10 官方 test 集同 9 个共享类别（9,000 张）** 上的精度直接对比，精准量化纯域偏移造成的准确率下降（$\Delta\text{Acc}$）与逐类别退化（如 `cat`/`dog`/`bird` vs `ship`/`airplane`）。

*(注：若后续想看无分辨率变化的 10 类全覆盖跨域，CINIC-10 的 ImageNet 子集可作为备选参考，但 STL-10 的 `torchvision` 原生复现性和标签兼容性讨论维度更契合 Challenge 2 要求。)*

---

## 4. 实施前微调项（Phase 1 探针验证）

- **优化器与学习率**：建议 `batch_size = 128`、`epochs = 30`（或 `40`）；AlexNet 无 BN 对大 lr 敏感，Phase 1 在 Colab T4 上做 3–5 epoch 短探针确认 AlexNet 与 ResNet 的最优默认 lr 并记录结果。

---

## 5. 角色 C 实施记录（2026-11-24）

- **Colab CLI 实测可用**：通过官方 `google-colab-cli` 完成 T4 smoke、5 次正式训练及 Drive 持久化。当日 GPU 配额耗尽后，经用户明确确认，不再等待或重训；保留 T4 训练记录，将统一评测迁移至 SSH Jade 的 RTX 2000 Ada，训练与推理硬件字段分别保存。
- **同图损坏的复现关键**：每个样本使用独立 CPU generator，种子为 `2223 + image_id`；旋转、平移和高斯噪声均在 `[0,1]` raw tensor 上执行，loader `shuffle=False`。单元测试验证 workers 0/2 逐像素一致，5 份 NPZ 的 CIFAR/STL ID 顺序一致。
- **train/evaluate 隔离**：训练阶段不迭代 test、不加载 STL；只有 5 份完整 T4 JSON/history/checkpoint 全部通过配置与权重校验后才开放 evaluate。评测只前向冻结模型，不按 test/STL 结果挑选 seed 或增强。
- **结果解释边界**：crop+flip 明显改善旋转和平移，却使 seed-42 ResNet 的高斯噪声准确率下降 3.19 个百分点；后续受控实验应单独研究噪声增强，不能泛化为“增强提高所有鲁棒性”。损坏评测时间包含 DataLoader/CPU 变换，不是纯变换 FLOPs。
- **产物口径**：大数组只保存到压缩 NPZ，JSON 保存 prediction 相对路径和 corruption config SHA-256；配对图按 ID 从原始数据集和固定配置重建。权重不进 Git，轻量 JSON/history/NPZ/CSV/图和已执行 Notebook 回收入库。
