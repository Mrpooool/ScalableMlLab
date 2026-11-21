# 项目进度

更新：2026-11-21。**Lab 1 Phase 0 完成，实验协议契约 v1.0 已冻结；准备进入 Phase 1（实现 `lab1/src/` 公共底座与 Colab smoke test）。**

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
