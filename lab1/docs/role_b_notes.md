# 角色 B：Task 3 / Task 5 执行记录

状态：六组正式 T4 训练及 Task 5 全量评估完成；结果、Notebook 和权重已回收核验，六个权重已上传团队 Drive 并登记 manifest。工作分支 `feat/lab1-role-b`，起点 `d2eec8535fe105becf4ba035ce5fb762c57a687c`。

## 范围与验收

1. 复用 `data.py`、`models.py`、`train.py`、`metrics.py`；不重写公共训练循环。
2. Task 3：AlexNet 六配置，单因素比较宽度、L2、Dropout（0 / 0.25 / 0.5）；固定 split、seed、30 epochs、batch 128、SGD/cosine、无增强；每 epoch 做确定性 train_eval/val。
3. Task 5：冻结的 AlexNet/ResNet checkpoint；完整测试集指标、混淆矩阵、逐类指标、固定规则正/误例、原始像素空间 FGSM、攻击约束检查、整体对抗准确率及 clean-correct 子集 ASR；同 ID 固定损坏及同预算随机噪声对照。
4. 已执行 Notebook、英文报告章节、JSON/CSV 和图表均从实测产物生成。

## 实验隔离

- 先跑少量训练/验证样本 smoke test，产出写入独立目录。
- 正式训练只使用 train/val；全部配置与攻击协议预先固定，测试结果不参与调参或 checkpoint 选择。
- Task 3 每个配置从头训练，含同一 T4 会话的参照组，以免跨硬件、跨会话混排耗时。
- 本地 GPU 可用于验证；最终报告的团队成本排名仍须统一硬件。记录真实设备，不将 RTX 3060 结果标成 T4。
- 未改共享契约；B 的新增协议位于 `role_b_experiments.json`，沿用已有接口与结果字段。
- 权重存放忽略的 checkpoint 目录。只有实际同步至共享 Drive 后，才标记为正式共享交接。

## 当前发现

- 已有公共模型支持 `width_mult` 和分类头两处 Dropout，无需修改模型。
- 现有 manifest 缺少 A 基线记录；复用时从权重内部元数据核对并记录 SHA256，不覆盖 C 的记录。
- `train.py` 的训练时间包括每 epoch 的验证与 train_eval；B 沿用这一口径。
- `SGD(model.parameters(), weight_decay=...)` 对所有可训练参数（包括 bias）施加衰减；报告按实际实现描述，不另加一次 L2 loss。
- 本机 GPU 为 RTX 3060 Laptop 6GB；项目独立虚拟环境已准备，本机验证使用 CPU，正式实验使用 Colab T4。

## 2026-11-25 验证与执行

- 已按 `karpathy-guidelines` 复用共享训练器；新增三个薄模块，覆盖任务编排、原始像素攻击和分析绘图，不改公共数据/模型/训练接口。
- 本地 `python -m unittest discover -s lab1/tests -v`：34 项通过，含 5 项新增 B 测试（81.237 s，CPU）；语法编译与 `git diff --check` 通过。
- 队友两个基线权重已下载并核对内部 exp_id、配置、归一化统计量和版本 `b1f9a7e`；恢复公共 manifest 中缺失的 A 记录并保留 C 记录。
- CIFAR-10 下载包 MD5 为 `c58f30108f718f92721af3b95e74349a`，已验证。
- Colab 会话成功连接 Tesla T4，PyTorch 2.14.1+cu130 / CUDA 13.0，FP32。用户明确授权上传 B 源码和两个共享基线权重。
- T4 上 5 项 B 测试及 128 train / 64 val、1 epoch smoke test 完成。正式阶段使用 45,000 / 5,000 split，6 个配置各 30 epochs；所有训练结束后才执行 test。
- 六组全部完成，共 8,100,000 trained samples。reference 的 best val accuracy 为 0.8346（epoch 30），test accuracy 0.8266，训练 358.596 s；半宽模型 test accuracy 0.8052，训练 175.935 s。
- Task 5 使用 A 的两个冻结权重，在完整 10,000 张测试图上评估。clean accuracy：AlexNet 0.8214，ResNet 0.8702；FGSM 8/255：0.1901 / 0.0478；相同预算随机均匀噪声：0.8167 / 0.8607。
- 共生成 11 张图、六组完整训练历史、两组类别指标和 18 份预测数组。曲线、混淆矩阵、预测及攻击示例已目视核对。
- 已执行 Notebook：[Colab](https://colab.research.google.com/drive/1AToRXGtOKvgQrBCv3dtDmdo4K1wEVsn4)，本地 `lab1/notebooks/role_b_task3_task5.ipynb`；重新运行入口为 `role_b_reproduce.ipynb`。
- Colab 导出验证通过。原始结果包和六权重包已保存到个人 Drive `Role_B_Results`，回收时逐包验证 SHA256。Google 挂载需要完整七项 OAuth 权限，用户逐项知情并明确授权后才继续。
- 独立数值核验通过：六组训练配置、每组 30 epoch / 1,350,000 trained samples、最佳 checkpoint 选择、全部源码哈希，以及 18 份预测数组的 image_id、目标、准确率、ASR 与 epsilon=0 一致性。
- 六个 `best.pt` 已存入团队 `checkpoints/<exp_id>/best.pt`。从共享 Drive 逐个重新下载后，SHA256 与原始训练文件全部相同。公共 manifest 保留 A/C 原记录并新增 B 六条；`shared_drive_verified=true`。
- 团队交接目录：[results_role_b](https://drive.google.com/drive/folders/1AS8A1paMyDSuYmoCEv_bSH5f-T_FnGHQ)。原始 Colab Notebook 保留 T4 实验与保存日志；本地交付 Notebook 补充了从 T4 JSON 整理出的损坏对照表，明确标注为本地整理，不重跑模型。
- 新模块在训练时尚未提交；结果中的 base commit 配合逐文件 SHA256 标识实际执行源码，不能把 base commit 单独当作完整 B 实现版本。
