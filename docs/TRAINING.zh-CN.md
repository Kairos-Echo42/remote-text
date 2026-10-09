# AgentForge 训练实验系统

## 核心原则

- Agent 只负责实验决策。
- Platform 负责 Recipe、预算、安全和审计。
- Trainer 负责容器生命周期。
- PyTorch/sklearn 负责实际计算。

## 资源关系

`Dataset` → `DatasetVersion` → `Experiment` → `TrainingJob` → `TrainingAttempt` → `Checkpoint` → `ModelVersion`。

每个 Experiment 必须绑定可信 BaselineStrategy。内置 DummyClassifier、DummyRegressor 和图像先验 Baseline；专用 BaselineStrategy 只能由平台注册。

选择流程是确定性的：先执行 Baseline，再执行已注册 Recipe 的第一轮候选，随后按 Top-K 进行 refinement，最后应用 `SelectionPolicy`。如果所有候选都未达到 `minimum_improvement`，平台保留 Baseline，并记录 `baseline_retained`。

## 数据泄漏防护

- 训练前固定 split。
- 缺失值填充、编码和缩放只在 train split fit。
- validation/test 只执行 transform。
- 图像增强只用于 train。
- 模型选择只使用 validation 指标。
- Final test evaluation 在 selection 之后执行，只作为人工审核元数据。

## 预算账本

`ExperimentBudget` 限制 Job 数、轮数、单 Job 时间、总运行时间和 GPU 时间。提交任务时原子预留硬上限，Job 完成后释放未使用部分并记录实际消耗。支持 `required_gpu`、`preferred_gpu` 和 `cpu_only`。

预算账本包含 `reserved_total_seconds`、`consumed_total_seconds`、`reserved_gpu_seconds` 和 `consumed_gpu_seconds`。`preferred_gpu` 回退 CPU 时立即释放 GPU 预留；final test evaluation 始终由 Platform 执行，不参与 Agent 模型选择。

## Recipe

RecipeRegistry 是平台受信任且不可变的组件，定义允许的参数、范围、资源和运行限制。Agent 不能创建、修改或注册 Recipe。

## 可复现性和 Lineage

TrainingAttempt 记录 DatasetVersion/split checksum、Recipe/config checksum、随机种子、运行环境版本、Docker digest、Git commit（可用时）和硬件信息。

Artifact 节点保存 SHA-256；Lineage 边连接输入/输出节点，并记录操作、时间和执行者。Model Bundle 内部生成 `checksums.json`，ModelVersion 额外保存最终 zip SHA-256。

Checkpoint 记录 epoch/step、validation metric、Artifact ID 和 checksum。v0.2 不从 Checkpoint 自动恢复训练，重试会创建新的 Attempt。Model Bundle 包含 `model.pt`、`preprocessor.joblib`、`schema.json`、`metrics.json`、`training-config.json`、`reproducibility.json`、`lineage.json` 和 `checksums.json`。

## API 与审核边界

- `GET /api/v1/workspaces/{workspace_id}/training/baselines`：受信任 Baseline 列表。
- `GET /api/v1/experiments/{experiment_id}/leaderboard`：仅 validation 指标排行榜。
- `GET /api/v1/experiments/{experiment_id}/report`：预算、selection reason 与人工 final test 摘要。
- 人工会话可查看 test metrics；Agent API Key 会被隐藏或拒绝。

## 取消

Docker 先发送 SIGTERM，等待宽限期后发送 SIGKILL。训练入口收到 SIGTERM 后刷新指标与不可续训 Checkpoint。v0.2 不支持从 Checkpoint 自动恢复。

## 边界

不包含预训练模型、迁移学习、Hugging Face、LLM Fine-tuning、多 GPU、分布式训练、Kubernetes、在线推理、自动 Production、任意训练代码生成和复杂 AutoML。

Trainer 默认使用 `python:3.12-slim`，由 PyTorch CUDA wheel 自带所需的 CUDA 运行库，因此不依赖 Ubuntu `apt` 软件源。如果 Docker Hub 不可直连，可在 `.env` 中把 `AGENTFORGE_TRAINER_BASE_IMAGE` 设置为受信任的 Python 3.12 基础镜像代理。
