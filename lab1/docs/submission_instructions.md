# Lab 1 提交说明

正式提交的两个主文件：

- `lab1_submission.ipynb`：顺序执行的统一 Notebook，覆盖 Task 1-6 和 Challenge 2。
- `lab1_report.pdf`：统一英文报告，讨论所有必做任务及已尝试的 Challenge 2。

`lab1_submission.zip` 同时附带报告 Markdown、复现说明、完整源码与结果包。
姓名和学号请在课程提交页面按小组成员填写；文件中未猜测成员身份。

## 结果来源

统一 Notebook 会执行保存结果的一致性检查、CPU 架构前向检查并展示真实图表，
不重新训练 13 个模型。它打印的运行环境属于本次审计/展示；JSON 中的训练和推理
设备仍属于原始测量。A 原始训练在 RTX 5070 Ti Laptop，B 的训练和评测在 T4；
C 的训练在 T4，推理/损坏/STL-10 评测在 Jade RTX 2000 Ada。
所有耗时排名只在相同硬件和测量流程内进行。

A 的 Notebook 已在 Colab CPU 实际执行其 5 个展示单元并保存结果。
B/C 的原始已执行 Notebook 随源码包保留。原文允许受 Colab 限制时提交分开的
已执行 Notebook，但本包提供一个完整的统一入口和单份报告。

## 查看与重现最终展示

直接打开已执行的 `lab1_submission.ipynb` 即可查看输出，不需要权重或下载数据集。
点击 Colab “全部运行”时，初始化单元从 GitHub 拉取固定提交，安装依赖，
随后检查 13 个完整训练历史、41 个预测条件并重新展示结果。CPU 即可。
完整实现及冻结配置同时显示在 Notebook 末尾的 source appendix 中；
50,000 个 split ID 在 `submission_source.zip` 的 JSON 中保留，避免占满输出。

离线使用时，将 `submission_source.zip` 解压到工作目录，把统一 Notebook 放在
`lab1/notebooks/`，在安装 PyTorch 的 Python 环境中执行：

```bash
python -m pip install -r lab1/requirements.txt
jupyter nbconvert --to notebook --execute lab1/notebooks/lab1_submission.ipynb --output lab1_submission_verified.ipynb
```

需要 Jupyter 时另装 `nbconvert`。PDF 可由 Markdown 重新生成：

```bash
python -m pip install reportlab
python lab1/scripts/build_report_pdf.py
```

## 重新运行模型实验

这是完整训练路径，不是查看已有输出的必要步骤。数据与 `.pt` 权重不上 Git。
共享权重目录：
[团队 checkpoints](https://drive.google.com/drive/folders/1NxrOQhUD63Y5_rhIs2WjYNVn4WetlEWL)。
`lab1/results/checkpoints_manifest.json` 记录精确的 `exp_id/best.pt` 路径；
已验证的 B 权重还记录 Drive file ID 和 SHA256。C 历史记录中未知的 Git commit
保留为 unknown，提交包的 SHA256 清单标识当前提供的源码快照。

按实验复现时使用全新结果目录，避免覆盖已提交的测量。

### A：数据探索与两条基线

在仓库根目录运行：

```bash
python lab1/src/run_phase2_baselines.py
```

该命令会准备 CIFAR-10，按 `baseline.json` 和固定 split 训练两条 30-epoch 基线，
写入 `lab1/results/`、`lab1/checkpoints/`。复现正式历史应在干净副本运行。
GPU 变化时应保留新环境信息，不能将新耗时当作原始 RTX 数值。

### B：容量/正则化与预测/对抗评测

在 Colab T4 使用附带的 `role_b_reproduce.ipynb`，或运行：

```bash
python lab1/src/run_role_b_experiments.py --stage smoke --device cuda --quiet
python lab1/src/run_role_b_experiments.py --stage train --device cuda --quiet
python lab1/src/run_role_b_experiments.py --stage evaluate --device cuda --quiet
```

`evaluate` 需要 A 的两个冻结权重分别位于
`lab1/checkpoints/t2_alexnet_cifar_baseline_s42/best.pt` 和
`lab1/checkpoints/t2_resnet18_cifar_baseline_s42/best.pt`。
六条 Task 3 训练与 Task 5 完整 10,000 张测试评测分开记录。
已有记录只有通过 runner 的完整性/配置检查才复用。

### C：增强、成本与 STL-10

训练使用 Colab T4；原始评测路径按 C 已冻结的实现要求使用 Jade GPU：

```bash
python lab1/src/run_role_c_experiments.py --repo-root . --work-lab1 lab1 --stage smoke
python lab1/src/run_role_c_experiments.py --repo-root . --work-lab1 lab1 --stage train
```

把五个冻结 checkpoint 与训练记录复制到 Jade 对应工作目录后运行：

```bash
python lab1/src/run_role_c_experiments.py --repo-root . --work-lab1 lab1 --stage evaluate
```

C runner 使用 `lab1/results_role_c/` 保存独立结果；回收交付时映射至
`lab1/results/`。原始 C Notebook 保留实际 Jade 路径、设备、输出和开关。
不要删除设备校验后把另一张卡的测量冒充 Jade/T4。

## 提交前核对

- Notebook 的所有代码单元有执行编号且无 error 输出，覆盖 Task 1-6、Challenge 2。
- 报告中的模型、权重来源、split、seed、预算和硬件与 JSON 一致。
- 权重、数据集、账号信息均不在 Git 或源码包中。
- `submission_inventory.json` 保存源码与结果包内文件的 SHA256；
  `submission_validation.json` 保存本次结构与数值检查结论。
- Task 5 ResNet 在 T4 复评为 87.02%，A 原始记录为 87.01%；报告明确分开。
- Challenge 1 未尝试；原文将其列为 optional。

文件已准备不等于已向课程平台提交。请在课程平台上传主文件，并按要求填入小组身份。
