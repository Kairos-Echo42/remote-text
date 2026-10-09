# ADR 0005：一等 TrainingJob 与独立 Trainer

- 状态：已接受
- 日期：2026-09-27

## 背景

训练可能持续数分钟，并且必须支持 Worker 重启、预算准入、取消和 GPU 排队。如果由一个长耗时 Capability 直接拥有训练过程，会把 DAG 执行与计算生命周期混在一起。

## 决策

DatasetVersion、Experiment、TrainingJob、TrainingAttempt、Checkpoint 和 ModelVersion 都是一等资源。独立 Trainer 服务负责抢占 Job、启动隔离的 PyTorch/sklearn 容器并监控执行。Agent Capability 只负责提交、等待和查询，不在 Agent 进程中运行训练。

## 影响

- 训练状态独立于 DAG Worker Lease 持久化。
- Agent 被中断不会直接终止训练容器。
- GPU 并发和预算预留由 Platform 统一控制。
- 需要可选的 Compose profile 和 CUDA Trainer 镜像。