# Role A 实验记录与交接文档（Task 1 & Task 2）

更新：2026-11-23 | 代码版本：`b1f9a7e`（及本轮提交）| 运行硬件：`NVIDIA GeForce RTX 5070 Ti Laptop GPU`（PyTorch `2.11.0+cu128`，FP32）

---

## 1. Task 1：数据探索与可复现准备

- **样本形态与分布**：
  - CIFAR-10 共 60,000 张 `32 × 32 × 3` RGB 彩色图像（`uint8`，像素范围 `[0, 255]`），涵盖 10 个完全均衡类别（官方训练集每类 5,000 张，官方测试集每类 1,000 张）。
  - 详见 [task1_data_summary.json](../results/task1_data_summary.json)、类别分布图 [t1_class_distribution.png](../results/figures/t1_class_distribution.png) 与样例网格图 [t1_sample_grid.png](../results/figures/t1_sample_grid.png)。
- **分层划分与测试集隔离**：
  - 按 `SPLIT_SEED = 2223` 做 90/10 分层划分，固化至 [split_indices.json](../configs/split_indices.json)：**45,000 train（每类 4,500）/ 5,000 val（每类 500）/ 10,000 test（每类 1,000）**。
  - 官方 `test` 集严格保留至模型冻结后的最终评估，不参与统计量计算、超参选择或早停。
- **预处理与随机性控制**：
  - RGB 通道均值/标准差**仅在 45,000 张 train split 上计算**：`mean = [0.4913, 0.4822, 0.4464]`，`std = [0.2470, 0.2434, 0.2615]`。
  - 保持原始 `[0, 1]` 像素在 `RawImages` / `TensorBatchLoader` 中输出并携带原始 `image_id`；缩放与归一化通过可微函数 `prepare_inputs(raw, stats, size=32)` 放在前向之前，同时满足 AlexNet/ResNet 统一 `32×32` 输入、Task 5 `[0, 1]` 像素空间对抗梯度反传，以及 Challenge 2 STL-10 `96×96 → 32×32` 统一下采样。
  - 随机性通过 `lab1/src/seed.py` 统一固定（Python/NumPy/PyTorch/CUDA + `torch.Generator` 固定 DataLoader 顺序）。

---

## 2. Task 2：AlexNet 与 ResNet-18 实现及基线对比

两模型均在 [models.py](../src/models.py) 中从零使用 PyTorch 基础层手写（无预训练权重、无 `torchvision.models` 构造器），严格对齐课程 Part A9 CIFAR-10 适配方案。

### 2.1 相对原版 ImageNet 架构的改动与设计依据
- **AlexNet (`alexnet_cifar`)**：
  - 原版（224×224）首层 `11×11 stride=4` 与大池化会让 `32×32` 特征图在浅层迅速塌陷为 `1×1`。适配版全部改用 `3×3 stride=1 padding=1` 卷积与 3 次 `2×2 stride=2` MaxPool，保持 5 卷积 + 3 全连接结构，输出特征图依次为 `64×16×16 → 192×8×8 → 256×4×4 → Flatten(4096) → FC(1024) → FC(512) → 10`（Dropout=0.5）。
  - 总参数 **6,976,842**，其中分类头 FC 层占约 4.72M（67.7%）。
- **ResNet-18 (`resnet18_cifar`)**：
  - 原版（224×224）使用 `7×7 stride=2` + `3×3 MaxPool stride=2`，会直接把 `32×32` 压缩 4 倍至 `8×8`。适配版改用 `3×3 stride=1 padding=1 + BN + ReLU` 并**移除初始 MaxPool**，在 Stage 1 保留完整的 `32×32` 空间分辨率；后续 4 个 Stage（64/128/256/512，每 stage 2 个 `BasicBlock`）依次输出 `64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → GAP(512) → Linear(512→10)`。
  - 总参数 **11,173,962**，分类头仅 5,130 个参数（0.05%），主要容量集中在深层卷积特征提取。
  - **残差连接作用**：每个 `BasicBlock` 学习残差映射 $F(x) = H(x) - x$，前向保留恒等捷径（维度变化时用 `1×1 Conv+BN` 投影），反向传播提供无衰减梯度直通路径；结合每层后的 `BatchNorm2d`，使 18 层网络能稳定使用更大初始学习率（`lr=0.1`），而无 BN 的 AlexNet 需用较小学习率（`lr=0.01`）平稳收敛。

### 2.2 统一计算预算下的 30-Epoch 实测对比（无数据增强，`seed=42`）

计算预算：`30 epochs × 45,000 = 1,350,000` 已训练样本，`batch_size = 128`，`SGD(momentum=0.9, weight_decay=5e-4)` + `CosineAnnealingLR`，按最高 `val_acc`（并列取更低 `val_ce`）保存 `best.pt`。

| 指标 | `alexnet_cifar` | `resnet18_cifar` | 对比观察 |
| :--- | :---: | :---: | :--- |
| 总参数量 | 6,976,842 | 11,173,962 | ResNet-18 为 AlexNet 的 1.60× |
| Checkpoint 体积 | 26.62 MB | 42.70 MB | — |
| 初始学习率 | `0.01` | `0.1` | AlexNet 无 BN 对大 lr 更敏感 |
| Best Val Epoch | 24 | 28 | Cosine 退火后期仍在微幅提升 |
| **Best Val Acc** | **83.00%** | **87.30%** | ResNet-18 **+4.30%** |
| **Best Val CE** | 0.9030 | 0.4223 | ResNet-18 损失不到 AlexNet 一半 |
| **Clean Test Acc** | **82.14%** | **87.01%** | ResNet-18 **+4.87%** |
| **Clean Test CE** | 0.9310 | 0.4357 | AlexNet 错例过度自信（校准差） |
| **Test Macro F1** | 0.8205 | 0.8698 | 与准确率几乎一致（类别均衡） |
| Final `train_eval` Acc / CE | 100.00% / 0.0005 | 100.00% / 0.0012 | 无增强下两者均明显过拟合 |
| 30-Epoch 训练总耗时（5070 Ti） | **248.2 s**（8.28 s/ep） | 685.0 s（22.83 s/ep） | ResNet-18 训练成本为 2.76× |
| 峰值 GPU 显存 | **316.1 MB** | 834.6 MB | ResNet-18 激活显存为 2.64× |
| 推理延迟 (`bs=1`, FP32) | **3.06 ms** | 11.63 ms | AlexNet 单图延迟快 3.80× |
| 推理吞吐 (`bs=128`, FP32) | **41,854 imgs/s** | 11,405 imgs/s | AlexNet 批吞吐高 3.67× |

相关产出：
- 详细指标 JSON：[t2_alexnet_cifar_baseline_s42.json](../results/t2_alexnet_cifar_baseline_s42.json)、[t2_resnet18_cifar_baseline_s42.json](../results/t2_resnet18_cifar_baseline_s42.json)
- 汇总表：[summary.csv](../results/summary.csv)
- 学习曲线与对比图：[t2_learning_curves.png](../results/figures/t2_learning_curves.png)、[t2_baseline_comparison.png](../results/figures/t2_baseline_comparison.png)
- 交互展示本：[role_a_task1_task2.ipynb](../notebooks/role_a_task1_task2.ipynb)

---

## 3. 给 Role B 与 Role C 的交接说明

1. **共享权重已登记在 [checkpoints_manifest.json](../results/checkpoints_manifest.json)**：
   - 本地路径：`lab1/checkpoints/t2_alexnet_cifar_baseline_s42/best.pt`（26.6 MB）与 `lab1/checkpoints/t2_resnet18_cifar_baseline_s42/best.pt`（42.7 MB）。
   - 请将这两个 `.pt` 上传至[团队共享 Google Drive](https://drive.google.com/drive/folders/1eRebAfEqnmBqUf3MgwX3NlcAFC54Izzp?usp=sharing) 对应路径 `lab1/checkpoints/<exp_id>/best.pt`，供队友直接复用。
2. **Role B（Task 3 + Task 5）衔接建议**：
   - 复用 `ResNetCIFAR(layers=..., base_width=..., dropout=...)` 直接做容量（宽度/深度）、L2（`weight_decay`）与 Dropout（`0.0` vs 两组非零值）控制变量实验。
   - Task 5 错例诊断与 FGSM/PGD 对抗攻击可直接加载上述两个冻结的 `best.pt`，通过 `prepare_inputs` 在 `[0, 1]` 像素空间求梯度。
3. **Role C（Task 4 + Challenge 2 + Task 6）衔接建议**：
   - Task 4 在 `make_loaders(..., train_pil_transform=...)` 传入数据增强即可与本轮无增强基线严格同口径对比；损坏测试集请固化参数到 `lab1/configs/corruptions_t4.json` 并同时提供给 Role B。
   - Challenge 2 直接调用 `data.make_stl10_test_loader` 与 `metrics.evaluate_stl10_cross_domain` / `evaluate_cifar10_shared9` 评估两个基线与 Task 4 增强模型。
