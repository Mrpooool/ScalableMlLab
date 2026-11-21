# Lab 1 实验协议契约（v1.0 — Frozen）

状态：**v1.0 已冻结**（已对齐 [ID2223_Lab_Session_2.ipynb](../../lab1/notebooks/ID2223_Lab_Session_2.ipynb) Part A9 & Part B，锁定分工与 Challenge 2 跨图像域迁移协议；仅共享 Drive 目录名与成员姓名可在 Phase 1 补填）  
更新：2026-11-21

本契约确保角色 A / B / C 的所有实验在**相同数据划分、相同计算预算、相同评测口径、相同落盘格式**下进行，保证最终报告横向对比公平、可复现。修改本契约前须经三人同步并更新版本号。

---

## 1. 运行环境与共享存储约定（Colab T4）

- **标准基准硬件**：Google Colab 免费 **NVIDIA Tesla T4**（默认 `fp32`；每次运行自动写入 `environment.device_name`）。
- **代码来源**：在 Colab 中 `git clone` 本仓库并将 `lab1/src/` 加入 `sys.path`；Notebook 仅做环境初始化、调用 `src/` 与展示图表。
- **数据集下载**：
  - CIFAR-10：优先使用 Part A1 镜像 `https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz`（md5 `c58f30108f718f92721af3b95e74349a`），失败回退官方源。
  - STL-10（仅 Challenge 2 评测阶段使用）：`torchvision.datasets.STL10(root=..., split="test", download=True)`。
- **共享权重路径（Google Drive）**：
  - 挂载 `/content/drive/MyDrive/<TEAM_SHARED_FOLDER>/lab1/`（Phase 1 补填实际共享目录名）
  - 子目录约定：`data/`、`checkpoints/<exp_id>/best.pt`、`cache/`
- **版本锁定交接**：任何被下游复用的模型权重，必须在仓库内 `lab1/results/checkpoints_manifest.json` 登记精确路径、代码 commit 和验证指标后才算正式交接。

---

## 2. 数据划分、预处理与随机性契约（Task 1）

沿用 Part A2 规范：
- **数据集**：CIFAR-10（50,000 train / 10,000 test，RGB `32 × 32`，10 类）。
- **分层划分**：`SPLIT_SEED = 2223`，`train_test_split(..., test_size=0.1, stratify=labels, random_state=2223)` → **45,000 train / 5,000 val**（索引升序排列）。索引与训练统计量固化至 `lab1/configs/split_indices.json`。
- **测试集隔离**：官方 `test`（10,000 张）严禁用于任何超参调节、早停、增强或损坏方法开发，仅用于已冻结模型的最终评估。
- **Train-only 归一化与可微前处理**：
  - RGB 均值/标准差仅基于 45,000 张 `train_ids` 计算。
  - Dataset（`RawImages`）返回 `(raw_tensor in [0, 1], target, image_id)`；缩放与归一化通过可微函数 `prepare_inputs(raw, stats, size=32)` 在模型前向时完成，以支持逐图 ID 追踪、STL-10 统一下采样，以及 `[0, 1]` 像素空间对抗梯度计算。
- **随机种子**：默认主种子 `seed = 42`（Task 6 种子敏感性测试追加 `seed = 0, 123`）；统一通过 `lab1/src/seed.py` 固定 `random`、`numpy`、`torch`、`torch.cuda` 与 `DataLoader` generator。

---

## 3. 模型架构与计算预算契约（Task 2，已对齐 Part A9）

**硬性禁止**：不得使用任何预训练权重或 `torchvision.models` 现成模型构造器，全部用 PyTorch 基础层手写。

### 3.1 AlexNet（`alexnet_cifar`，输入 `3 × 32 × 32`）
- `Conv1(3→64, k=3, s=1, p=1) + ReLU → MaxPool(2,2)` (`64×16×16`)
- `Conv2(64→192, k=3, s=1, p=1) + ReLU → MaxPool(2,2)` (`192×8×8`)
- `Conv3(192→384, k=3, s=1, p=1) + ReLU` (`384×8×8`)
- `Conv4(384→256, k=3, s=1, p=1) + ReLU` (`256×8×8`)
- `Conv5(256→256, k=3, s=1, p=1) + ReLU → MaxPool(2,2)` (`256×4×4`)
- `Flatten(4096) → FC1(4096→1024) + ReLU + Dropout(p) → FC2(1024→512) + ReLU + Dropout(p) → FC3(512→10)`（基线默认 `p=0.5`）

### 3.2 ResNet-18（`resnet18_cifar`，输入 `3 × 32 × 32`）
- `Initial Conv(3→64, k=3, s=1, p=1, bias=False) + BN + ReLU`（无初始 MaxPool，`64×32×32`）
- `BasicBlock`：`Conv3×3(bias=False) → BN → ReLU → Conv3×3(bias=False) → BN → (+ shortcut) → ReLU`；下采样或通道变化时 shortcut 为 `Conv1×1(stride=s, bias=False) + BN`。
- Stage 1：2 blocks, 64 ch, stride 1 (`64×32×32`)
- Stage 2：2 blocks, 128 ch, first stride 2 (`128×16×16`)
- Stage 3：2 blocks, 256 ch, first stride 2 (`256×8×8`)
- Stage 4：2 blocks, 512 ch, first stride 2 (`512×4×4`)
- `AdaptiveAvgPool2d(1) → Flatten(512) → Dropout(p, 默认 0.0) → Linear(512→10)`

### 3.3 计算预算与训练协议
- **Batch size**：`128`
- **Epochs**：`30`（默认计算预算 = `45,000 × 30 = 1,350,000` trained samples；若 Phase 1 短探针发现需 40 epochs 收敛，则在 `baseline.json` 中统一调整为 `40` 并同步本节）
- **优化器与调度**：`SGD(momentum=0.9)` + `CosineAnnealingLR(T_max=epochs)`；学习率在 Phase 1 通过验证集小探针确认（默认参考：ResNet `lr=0.1`、AlexNet 无 BN 若 `0.1` 梯度爆炸则用 `lr=0.01` 并在报告中记录实测原因）。
- **Checkpoint 选择规则**：每个 epoch 后在 `val` 上评测，保存 `val_accuracy` 最高（并列时 `val_ce` 更低）的权重为 `best.pt`。
- **确定性训练评估（Task 3 要求）**：每个 epoch 同步在无 shuffle、`model.eval()` 模式的 `train_eval` loader 上计算 `train_eval_ce` 与 `train_eval_acc`，避免训练态 Dropout / BN 统计波动干扰过拟合与欠拟合诊断。

---

## 4. 专题与 Challenge 2 隔离契约（Task 4 / Task 5 / Challenge 2）

### 4.1 Task 4（角色 C）：数据增强与固定损坏测试集
- **训练增强**：在验证集上开发与定型（推荐起步：`RandomCrop(32, padding=4)` + `RandomHorizontalFlip(p=0.5)`），与无增强基线做控制变量对比。
- **固定损坏测试集**（`lab1/configs/corruptions_t4.json`）：固定随机种子与参数（如旋转、仿射/平移形变、加性高斯噪声），保证所有待对比模型（含角色 B 的 Task 5 随机损坏 vs 对抗对比）使用**完全相同的损坏图像与 `image_id`**。

### 4.2 Task 5（角色 B）：错例分析与对抗攻击
- 攻击在原始 `[0, 1]` 像素空间通过可微 `prepare_inputs` 计算梯度，扰动后 `clamp(0, 1)`。
- 明确定义 **攻击成功率（ASR）**：在**干净图像预测正确**的样本子集上，施加满足 $\|\delta\|_\infty \le \epsilon$ 约束的扰动后，模型预测翻转的比例；并汇报对抗集整体准确率。
- 与角色 C 的 `corruptions_t4.json` 损坏结果做同口径对比，区分随机损坏与定向优化扰动。

### 4.3 Challenge 2（角色 C）：跨图像域迁移（CIFAR-10 → STL-10）
- **严格 held-out 铁律**：STL-10 严禁用于任何超参、架构或数据增强选择，仅对 Task 2/4 已冻结的最终模型执行纯前向推理评估。
- **数据集与预处理**：`STL10(split="test")` 原生 `96×96` RGB，共 8,000 张；通过 `prepare_inputs(..., size=32)` 以 `bilinear + antialias=True` 下采样至 `32×32`，并严格使用 **CIFAR-10 训练集均值/标准差** 归一化。
- **标签兼容与映射**：剔除 STL-10 独有类 `7: monkey`，保留 9 个共享类（共 **7,200 张**），按如下字典映射到 CIFAR-10 标签索引：
  ```python
  STL10_TO_CIFAR10 = {
      0: 0,  # airplane -> airplane
      1: 2,  # bird -> bird
      2: 1,  # car -> automobile
      3: 3,  # cat -> cat
      4: 4,  # deer -> deer
      5: 5,  # dog -> dog
      6: 7,  # horse -> horse
      8: 8,  # ship -> ship
      9: 9,  # truck -> truck
  }
  CIFAR10_SHARED_9_IDS = [0, 1, 2, 3, 4, 5, 7, 8, 9]  # excludes 6 (frog)
  ```
- **统一汇报指标**：
  1. `stl10_10way_acc`（原 10 类输出直接 argmax 准确率）与 `stl10_frog_leakage_rate`（被误判为 CIFAR-10 第 6 类 `frog` 的样本占比）；
  2. `stl10_9way_masked_acc`（将第 6 类 logit 置为 $-\infty$ 后的 9 类准确率）与逐类别准确率 / 混淆矩阵；
  3. `cifar10_test_9way_acc`（同一模型在 CIFAR-10 test 对应 9 个共享类共 9,000 张上的准确率）及域偏移掉点 `domain_drop_9way = cifar10_test_9way_acc - stl10_9way_masked_acc`。

---

## 5. 统一实验结果 JSON 格式（`lab1/results/<exp_id>.json`）

```json
{
  "exp_id": "t2_resnet18_baseline_s42",
  "task": "task2",
  "owner": "role_a",
  "git_commit": "short_sha",
  "timestamp_utc": "2026-11-22T10:00:00Z",
  "environment": {
    "platform": "colab",
    "device_name": "Tesla T4",
    "precision": "fp32",
    "torch_version": "2.x.x",
    "cuda_version": "12.x"
  },
  "config": {
    "model": "resnet18_cifar",
    "model_options": {},
    "seed": 42,
    "epochs": 30,
    "batch_size": 128,
    "optimizer": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "dropout": 0.0,
    "augmentation": "none"
  },
  "model_stats": {
    "total_params": 11173962,
    "trainable_params": 11173962,
    "checkpoint_size_mb": 42.6
  },
  "training": {
    "num_trained_samples": 1350000,
    "total_train_time_sec": 0.0,
    "avg_epoch_time_sec": 0.0,
    "peak_gpu_mem_mb": 0.0,
    "best_epoch": 0,
    "best_val_acc": 0.0,
    "best_val_ce": 0.0,
    "final_train_eval_acc": 0.0,
    "final_train_eval_ce": 0.0,
    "history_file": "lab1/results/histories/t2_resnet18_baseline_s42.csv"
  },
  "evaluation": {
    "clean_test_acc": 0.0,
    "clean_test_ce": 0.0,
    "macro_f1": 0.0,
    "corrupted_metrics": {},
    "adversarial_metrics": {},
    "challenge2_stl10": {}
  },
  "inference_benchmark": {
    "device_name": "Tesla T4",
    "precision": "fp32",
    "bs1_latency_ms": 0.0,
    "bs128_latency_ms_per_batch": 0.0,
    "bs128_throughput_imgs_per_sec": 0.0,
    "warmup_runs": 10,
    "measured_runs": 50
  },
  "checkpoint_ref": {
    "drive_rel_path": "checkpoints/t2_resnet18_baseline_s42/best.pt",
    "is_registered_in_manifest": true
  }
}
```

---

## 6. 推理性能计时与显存测量规范（Task 2 & Task 6）

统一调用 `lab1/src/metrics.py`：
1. **峰值显存**：训练前调用 `torch.cuda.reset_peak_memory_stats()`，训练后记录 `torch.cuda.max_memory_allocated() / (1024 ** 2)`。
2. **推理计时**（对齐 Part A8）：`model.eval()` + `torch.inference_mode()`，对预准备好的输入张量先 warmup 10 次，再对后续 50 次前向用 `torch.cuda.synchronize()` 包裹 `time.perf_counter()`，同时报告 `batch_size = 1`（单图延迟）与 `batch_size = 128`（批量延迟与吞吐）。
