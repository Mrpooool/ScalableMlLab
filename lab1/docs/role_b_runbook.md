# B 部分运行与交接

## 包含内容

- Task 3：六个 AlexNet 配置，分别比较容量、L2 与 Dropout；新建模型、固定数据划分、同一预算、确定性 train_eval/val。
- Task 5：队友两个冻结基线的类别指标、混淆矩阵、正/误例、FGSM 与相同 L∞ 预算的随机均匀噪声；另复现 C 的固定旋转、平移和高斯噪声，全部按 image_id 配对。
- `role_b_experiments.json` 是预先确定的实验矩阵。测试集仅在协议冻结后的评测阶段使用。

## 本地运行

从仓库根目录执行，先安装支持当前 GPU 的 PyTorch/torchvision 和 `lab1/requirements.txt`。

```bash
python -m unittest discover -s lab1/tests -p "test_role_b.py" -v
python lab1/src/run_role_b_experiments.py --stage smoke
python lab1/src/run_role_b_experiments.py --stage train
python lab1/src/run_role_b_experiments.py --stage evaluate
```

`train` 完成六组正式训练；不读取测试集指标。`evaluate` 先验证六组配置、30 epoch 历史、1,350,000 trained samples 与权重 SHA256，再执行最终测试和 Task 5。`predictions` 可以单独执行已冻结基线的 Task 5；不得根据其测试结果调整训练矩阵。

中断后重新运行 `train` 会校验并跳过完整实验。公共训练器目前只在每个完整实验结束时保存 best.pt，因此中断的那个实验需要重跑，不声称支持 epoch 内恢复。

Task 5 需要以下队友权重，按目录放置：

```text
lab1/checkpoints/t2_alexnet_cifar_baseline_s42/best.pt
lab1/checkpoints/t2_resnet18_cifar_baseline_s42/best.pt
```

来源是用户提供的团队共享 Drive；加载时核对 exp_id、模型配置、训练集 mean/std 和文件 SHA256，不混用 C 的另一份权重的损坏指标。

## Colab T4

1. `role_b_task3_task5.ipynb` 保留本次真实执行输出；重新运行使用 `role_b_reproduce.ipynb`。
2. 运行时 → 更改运行时类型 → 选择 T4 GPU（资源可用时）。
3. 运行初始化单元格，上传随交付包提供的 `role_b_source.zip`。它含完整模块和冻结配置，不含数据集或权重。
4. 挂载自己的 Google Drive，将团队共享文件夹添加到「我的云端硬盘」快捷方式；按实际名称调整 Notebook 的 `SHARED_ROOT`。
5. 先运行 smoke，再运行六组训练和最终评估。每个实验结束后立即将权重、历史和 JSON 复制到个人 Drive 的 `role_b` 目录。
6. 保存保留输出的 Notebook，并回收 JSON、CSV、图表与权重 manifest。上传共享权重后才将 `shared_drive_verified` 设为 true，并合并到公共 manifest。

GPU 类型与额度由 Google 动态分配。如果没有 T4，本地结果按实际设备标注；成本不能与 T4/Jade 数据直接排名。

## 方法口径

- 宽度倍率仅改变五层卷积的通道；全连接隐藏宽度保持 1024/512，第一层 FC 输入维度随末层通道适配。
- Dropout 放在 AlexNet 的两个 FC 隐藏层 ReLU 后。每个配置从头训练；不能只在已训练权重上修改 p。
- SGD weight_decay 对 `model.parameters()` 全部参数施加衰减（含 bias），对应 λ/2 的 L2 惩罚梯度；不额外再加一次 L2 loss。
- 曲线使用 eval 模式的纯交叉熵，避免训练态 Dropout 和 L2 项造成误导。泛化差距用同一 epoch 的 train_eval 与 val 计算。
- 训练总时间沿用公共代码，包含每 epoch 的验证和 train_eval，排除最终测试和推理 benchmark。
- FGSM 非定向，ε = 0/2/4/8 ÷ 255，通过可微前处理求原始像素梯度；参数权重与参数梯度不改变。
- ASR = 干净预测正确且攻击后错误的数量 / 干净预测正确的数量。整体对抗准确率使用全部评估图像。FGSM 未成功也照实报告。
- 同预算随机对照是按 image_id 固定的均匀 L∞ 噪声；C 的高斯噪声 std=0.05 不是同一扰动预算，单独解释。
- 原始/攻击图保持真实像素；扰动图显示 `clip(0.5 + 8*delta, 0, 1)`，仅为可视化，不是实际攻击幅度。
- 正/误例按 image_id 取首五张；攻击图取首两张成功及首一张失败（若存在），避免手选图片造成结论偏差。

## 完成交付检查

- [x] 六组正式实验及学习曲线、成本表。
- [x] 两模型预测分析与完整测试集对抗指标。
- [x] 扰动范围、ε=0 一致性、ASR 分母及 image_id 配对验证。
- [x] 英文 Task 3/5 报告与已执行 Notebook，数字来自结果文件。
- [x] 权重在 Drive，仓库只含代码、轻量结果与图表。
- [x] `git diff --check`；角色进度已记录在 `role_b_notes.md`。
- [ ] 合并到 main 时由整合者同步根目录 `task_plan.md` 和 `progress.md`（遵循团队防冲突规范）。

交接目录：[团队 results_role_b](https://drive.google.com/drive/folders/1AS8A1paMyDSuYmoCEv_bSH5f-T_FnGHQ)。六个权重的精确 Drive 文件链接、配置和 SHA256 位于公共 `checkpoints_manifest.json` 以及 B 的独立 manifest。
