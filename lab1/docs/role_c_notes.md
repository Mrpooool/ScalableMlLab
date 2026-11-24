# 角色 C 专题实验记录：Task 4 / Challenge 2 / Task 6

本文档是角色 C 的专题实验记录，不是团队 Report。所有数值均来自 `lab1/results/` 下已回收的真实实验产物（5 份结果 JSON、`role_c_summary.csv`、3 份 Task 6 CSV）。**训练与推理在不同硬件上完成，全文分开标注，不做跨硬件排名：**

- 训练：Google Colab **Tesla T4**，FP32，PyTorch 2.14.1+cu130 / CUDA 13.0。
- 推理 benchmark 与全部 test/STL 评测：SSH 主机 Jade 上 **NVIDIA RTX 2000 Ada Generation**，FP32（warmup=10，measured=50）。
- 角色 A 的 RTX 5070 Ti Laptop 数据仅作跨硬件敏感性参考，不与 T4/Jade 数字混排。

## 冻结协议

- 实验矩阵固定为 5 个配置，全部 30 epochs、SGD（momentum=0.9，weight_decay=5e-4；AlexNet lr=0.01，ResNet lr=0.1）、cosine 调度、batch 128、FP32、1,350,000 trained samples，固定 CIFAR-10 train/val split（train 45,000 / val 5,000）：

| exp_id | 模型 | seed | 增强 |
|---|---|---|---:|
| `t6_alexnet_cifar_no_aug_s42` | alexnet_cifar（6,976,842 参数） | 42 | none |
| `t4_resnet18_cifar_no_aug_s42` | resnet18_cifar（11,173,962 参数） | 42 | none |
| `t4_resnet18_cifar_crop_flip_s42` | resnet18_cifar | 42 | crop32_pad4_hflip05 |
| `t6_resnet18_cifar_crop_flip_s0` | resnet18_cifar | 0 | crop32_pad4_hflip05 |
| `t6_resnet18_cifar_crop_flip_s123` | resnet18_cifar | 123 | crop32_pad4_hflip05 |

- 损坏协议冻结于 `lab1/configs/corruptions_t4.json`（version=1，seed=2223，SHA256 `b21f36…edd3`，5 份 JSON 一致）：对同一组 CIFAR-10 test ID 0–9999 施加三种逐图确定性损坏（每图独立 generator，seed=2223+image_id）：`rotation_15deg`（±15° 均匀采样，bilinear，fill=0）、`translation_3px`（x/y 各取整数 [-3,+3]，fill=0）、`gaussian_noise_std_0_05`（std=0.05，clamp [0,1]）。clean 与三种损坏的 10,000 个 ID 顺序完全一致，可逐图配对。
- 模型与增强选择全部在 test/STL 评测前冻结；STL-10 仅用于冻结后的纯前向泛化评估，不参与任何选择。

## Task 4：数据增强与固定损坏鲁棒性

### 增强选择与标签合理性

选用 `RandomCrop(32, padding=4)` + `RandomHorizontalFlip(p=0.5)`（配置名 `crop32_pad4_hflip05`）。两者作为 CIFAR-10 的标签保持增强使用：4 像素反射/零填充裁剪引入小位移，水平翻转不改变物体类别语义。标签合理性并不意味着每次随机裁剪都完整保留主体；过强几何变化或噪声仍可能让标签对人也不明确，因此保留同 ID 配对样例作定性检查。

### seed-42 ResNet 对照（同一 10,000 个 test ID）

| 指标 | no-aug | crop+flip | 差值 |
|---|---:|---:|---:|
| clean acc | 0.8658 | 0.9308 | +0.0650 |
| rotation_15deg acc | 0.7680 | 0.8871 | +0.1191 |
| translation_3px acc | 0.8014 | 0.9281 | +0.1267 |
| gaussian_noise_std_0_05 acc | 0.7133 | 0.6814 | **−0.0319** |
| 三种损坏平均 acc | 0.7609 | 0.8322 | +0.0713 |

关键逐图配对统计（相对各自 clean 预测）：

| 条件 | 模型 | accuracy drop | prediction flip rate | clean 对→错数 | clean 对→错率 |
|---|---|---:|---:|---:|---:|
| rotation | no-aug | 0.0978 | 0.1892 | 1270 | 0.1467 |
| rotation | crop+flip | 0.0437 | 0.0878 | 593 | 0.0637 |
| translation | no-aug | 0.0644 | 0.1715 | 1032 | 0.1192 |
| translation | crop+flip | 0.0027 | 0.0373 | 173 | 0.0186 |
| noise | no-aug | 0.1525 | 0.2460 | 1767 | 0.2041 |
| noise | crop+flip | 0.2494 | 0.3106 | 2658 | 0.2856 |

解读：crop+flip 大幅提升 clean 与几何类损坏（旋转、平移）鲁棒性——训练时见过 ±4px 裁剪平移，与 ±3px 平移损坏同族；旋转也有明显迁移。**但噪声是最大弱点，且 crop+flip 在噪声上反而更差**（0.6814 vs 0.7133，drop 0.2494 vs 0.1525）：增强模型的 final train_eval acc 为 0.9972，噪声扰动下翻转更多（flip rate 0.3106）；训练拟合程度本身不能解释噪声退化，这里仅报告观察，不声称 crop+flip 提升噪声鲁棒性。噪声下逐类错误集中在 dog（0.422）、airplane（0.474）、cat（0.543），frog 几乎不受影响（0.966）；一种可能解释是噪声破坏了细粒度纹理，而 frog 预测更多依赖颜色或大块形状，但该归因仍需额外受控实验验证。配对样例见 `figures/t4_corruption_examples.png`（同 ID 原图/损坏图对照）。

### 增强的训练开销（T4 组内比较）

- 训练总时间：crop+flip 1859.826 s vs no-aug 1855.352 s（+4.474 s，约 +0.24%；平均每 epoch 61.994 s vs 61.845 s）。
- 峰值显存：两者完全相同，802.99 MB。
- 结论：该增强在 T4 上几乎零成本。

### 损坏评测时间口径

同一模型 clean 与三种损坏的单次评测时间均在 1.418–1.435 s（Jade，10,000 图），损坏与 clean 之差 ≤0.01 s。该时间**包含 DataLoader 与 CPU 端损坏处理**，只说明端到端评测开销可忽略，不代表纯变换 FLOPs。

## Challenge 2：CIFAR-10 → STL-10 跨域泛化

### 标签兼容与预处理

STL-10 test 7,200 张图过滤为与 CIFAR-10 共享的 9 类（airplane/bird/car/cat/deer/dog/horse/ship/truck，monkey 无对应类被排除），标签按 `STL10_TO_CIFAR10` 映射到 CIFAR 索引。原始 96×96 图像经双线性插值缩放到 32×32 并用 CIFAR train 统计量归一化后前向。两个口径：10-way（不屏蔽，统计 frog 泄漏率）与 9-way masked（在 9 个共享 logit 上取 argmax）。CIFAR 侧对照为 test 中 9,000 个共享类样本的 9-way masked 准确率。

### 三模型域下降（seed-42）

| 模型 | CIFAR 9-way | STL 10-way | frog 泄漏率 | STL 9-way masked | 9-way macro F1 | domain drop |
|---|---:|---:|---:|---:|---:|---:|
| alexnet no-aug | 0.8310 | 0.6675 | 0.0314 | 0.6825 | 0.6790 | 0.1485 |
| resnet no-aug | 0.8683 | 0.7107 | 0.0351 | 0.7267 | 0.7221 | 0.1417 |
| resnet crop+flip | 0.9339 | 0.8014 | 0.0185 | 0.8108 | 0.8088 | 0.1231 |

增强 ResNet 不仅 clean 更高，绝对域下降也最小（0.1231 vs 0.1417/0.1485），frog 泄漏率也最低（1.85%）；其 STL 9-way 绝对增益（+0.0841 vs no-aug ResNet）大于 CIFAR 9-way 增益（+0.0656）。该对照与 crop+flip 改善跨域几何不变性的解释一致，但单次架构/增强对照不能单独证明因果。dog/cat 等细粒度动物类的下降绝对值仍然最大，泛化边界依旧存在。

### 逐类域下降与失败模式

增强 ResNet 逐类 domain drop 最大三项：**dog 0.3013（CIFAR 0.890 → STL 0.589）、cat 0.2118（0.873 → 0.661）、horse 0.1507（0.942 → 0.791）**；最小为 ship 0.0395、deer 0.0420。no-aug ResNet 同样是 dog（0.3488）、truck（0.2015）、cat（0.2005）最大；AlexNet 为 dog（0.3108）、truck（0.2788）、horse（0.1733）。三模型共同突出 dog 的跨域风险，truck 在两种无增强模型上也明显下降。一种可能解释是 96→32 降采样削弱了 STL 高分辨率图的细粒度纹理，而模型只学过 32×32 的 CIFAR 分布；该解释与逐类指标和失败图一致，但不作为已证明的因果结论。配对失败样例见 `figures/challenge2_stl10_failures.png`（按三模型错误 ID 的确定性并集取样，在相同 STL ID 上对照三模型预测；并非每个样本都被三模型同时误判），混淆结构见 `figures/challenge2_stl10_confusion.png`，逐类对比见 `figures/challenge2_domain_drop.png`。

**验证边界**：STL-10 结果仅用于冻结后的泛化讨论，未参与模型/增强/seed 选择，也不进入下方推荐。

## Task 6：统一成本与权衡

### 五模型成本/效果（训练=T4，推理=Jade RTX 2000 Ada，分列标注）

| exp_id | 参数量 | T4 训练时间(s) | T4 峰值显存(MB) | Jade bs=1 延迟(ms) | Jade bs=128 吞吐(img/s) | clean acc | 平均损坏 acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| alexnet no-aug s42 | 6.98M | 353.7 | 316.9 | 0.309 | 28,467 | 0.8266 | 0.7207 |
| resnet no-aug s42 | 11.17M | 1855.4 | 803.0 | 0.906 | 7,621 | 0.8658 | 0.7609 |
| resnet crop+flip s42 | 11.17M | 1859.8 | 803.0 | 0.914 | 7,593 | 0.9308 | 0.8322 |
| resnet crop+flip s0 | 11.17M | 1858.5 | 803.0 | 0.852 | 7,575 | 0.9279 | 0.8253 |
| resnet crop+flip s123 | 11.17M | 1856.0 | 803.0 | 0.849 | 7,577 | 0.9292 | 0.8221 |

### 推荐（仅依据 CIFAR test、固定损坏与计算成本）

**推荐 resnet18_cifar + crop32_pad4_hflip05 策略**：三个 seed 的 clean 均值为 0.9293，平均损坏均值为 0.8265（范围 0.8221–0.8322），且每个 seed 的这两项都高于 no-aug ResNet 和 AlexNet。seed 42 的 0.9308/0.8322 是单次最高值，但不能据此认定该 seed 优于 0/123；训练开销相对 no-aug ResNet 仅 +0.24%、显存不变，bs=1 延迟 0.914 ms。若部署场景延迟/吞吐优先，选 alexnet_cifar：bs=1 0.309 ms（约为 ResNet 的 1/3）、吞吐 28,467 img/s（约 3.7 倍）、训练仅 354 s，代价是 clean 低约 10.4 个百分点、平均损坏低 11.2 个百分点。三个增强 ResNet 的共同短板是高斯噪声（seed 范围为 0.6615–0.6814）；五配置中噪声准确率最高的是 AlexNet 的 0.7756，但其平均损坏准确率仍为五配置最低的 0.7207。后续实验建议针对这一噪声弱点。

### seed 敏感性（resnet crop+flip，seeds 0/42/123）

| 指标 | mean | sample std | min | max |
|---|---:|---:|---:|---:|
| best val acc | 0.9314 | 0.0019 | 0.9302 | 0.9336 |
| clean test acc | 0.9293 | 0.0015 | 0.9279 | 0.9308 |
| rotation acc | 0.8823 | 0.0042 | 0.8795 | 0.8871 |
| translation acc | 0.9261 | 0.0017 | 0.9250 | 0.9281 |
| noise acc | 0.6712 | 0.0100 | 0.6615 | 0.6814 |
| 平均损坏 acc | 0.8265 | 0.0052 | 0.8221 | 0.8322 |
| STL 9-way masked | 0.8110 | 0.0017 | 0.8093 | 0.8128 |
| domain drop | 0.1202 | 0.0040 | 0.1156 | 0.1231 |
| 训练时间(s) | 1858.1 | 1.93 | 1856.0 | 1859.8 |

clean 与几何损坏的 seed 波动很小（sample std ≤0.0042），噪声 acc 波动最大（sample std 0.0100，极差 0.0199）。seed 42 的 clean、平均损坏、旋转和平移最高，seed 0 的噪声最高，因此没有单一 seed 在所有指标上占优；这些单策略观测支持策略级推荐，不能用于选出最优 seed。见 `results/task6_seed_sensitivity.csv`。

### 训练时长敏感性（epoch 10/20/30，T4 累计时间）

- AlexNet 收敛快且便宜：epoch 10/20/30 val acc 0.8046/0.8236/0.8294，累计 112/233/354 s；其训练成本约为 ResNet 的 1/5，但最终 val 仍低于两种 ResNet 配置。
- ResNet no-aug：0.8048/0.8712/0.8734（620/1238/1855 s），epoch 20 后基本饱和。
- ResNet crop+flip（s42）：0.7854/0.8832/0.9304（622/1241/1860 s）——epoch 10 时**低于** no-aug（0.7854 vs 0.8048），epoch 20 反超，epoch 30 大幅领先（0.9304 vs 0.8734）。增强的收益依赖完整训练日程，短训练预算下增强反而不划算；s0/s123 曲线形状一致（epoch 30 分别 0.9302/0.9318）。全部 5 行数据见 `results/task6_training_duration.csv`。

### 跨硬件敏感性（角色 A RTX 5070 Ti Laptop，仅参考，非性能排名）

| 模型 | RTX 5070 Ti 训练(s) | T4 训练(s) | RTX bs=1(ms) | Jade bs=1(ms) | RTX 吞吐 | Jade 吞吐 | RTX clean | T4 clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alexnet s42 | 248.2 | 353.7 | 3.059 | 0.309 | 41,854 | 28,467 | 0.8214 | 0.8266 |
| resnet no-aug s42 | 685.0 | 1855.4 | 11.628 | 0.906 | 11,405 | 7,621 | 0.8701 | 0.8658 |

T4/RTX 5070 Ti 训练时间比分别约为 1.42（AlexNet）和 2.71（ResNet）；clean 精度在跨硬件/跨会话下接近（差约 0.005）。两侧 benchmark 的硬件与运行会话不同，以上数字只说明观测到的硬件敏感性，**不构成同硬件性能排名**。见 `results/task6_hardware_comparison.csv`。

## 后续受控实验建议（针对 ResNet 最大实测弱点：高斯噪声）

高斯噪声是四个 ResNet 配置的最弱损坏（无增强 0.7133，增强 0.6615–0.6814，seed-42 增强 drop 达 0.2494）；AlexNet 的最弱损坏则是旋转（0.6829），其噪声准确率为 0.7756。建议的单一受控实验：**保持冻结矩阵中 `t4_resnet18_cifar_crop_flip_s42` 的全部设置不变（同模型、seed=42、30 epochs、优化器/调度/split/批次），仅在训练增强中加入与评测分布匹配的高斯噪声（std=0.05，PIL/张量级随机注入），在同一固定损坏套件与 STL 协议下复评。** 对照组已存在（crop+flip s42），唯一变量是训练期噪声暴露；该实验可检验噪声 acc 是否从 0.6814 提升且 clean/几何损坏不回退，但不能仅凭结果确认或排除架构上限。

## 产物与复现入口

- 结果 JSON（含损坏指标、逐类准确率、Challenge 2 汇总、推理 benchmark）：`lab1/results/t4_*.json`、`t6_*.json`（5 份）。
- 汇总表：`lab1/results/role_c_summary.csv`、`task6_seed_sensitivity.csv`、`task6_training_duration.csv`、`task6_hardware_comparison.csv`；训练曲线：`lab1/results/histories/*.csv`。
- 逐图预测：`lab1/results/predictions/<exp_id>.npz`（clean/三损坏各 10,000 ID、STL 7,200 ID，五份 ID 数组逐一相等）。
- 图：`lab1/results/figures/`（`t4_learning_curves.png`、`t4_corruption_accuracy.png`、`t4_per_class_drops.png`、`t4_corruption_examples.png`、`challenge2_domain_drop.png`、`challenge2_stl10_confusion.png`、`challenge2_stl10_failures.png`、`t6_tradeoffs.png`、`t6_training_duration.png`）。
- 损坏协议：`lab1/configs/corruptions_t4.json`；实现：`lab1/src/corruptions.py`、`run_role_c_experiments.py`（`--stage {smoke,train,evaluate,all}`，断点续跑带完整性校验）。
- 已执行 Notebook（含全部输出）：`lab1/notebooks/role_c_task4_challenge2_task6.ipynb`，在 Jade（`/u/10/lix33/unix/ScalableMlLab`）上以 `--stage evaluate` 复现；训练记录为 Colab T4 实测，不在 Jade 重训。
- **checkpoint 不在 Git 中**：5 个 `best.pt` 仅存于 Google Drive（`checkpoints/<exp_id>/best.pt`，见 `lab1/results/checkpoints_manifest.json`）与 Jade 工作目录。
