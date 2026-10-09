# AgentForge 中文演示指南

> 英文版本：[DEMO.md](DEMO.md)


## 1. 演示目标

一次完整运行需要证明：

1. Gateway、Web 工作台和 API 可以正常访问；
2. Workflow 会生成不可变版本；
3. Scheduler 按依赖和并发限制派发节点；
4. DeepSeek 可以执行 Planner、Researcher、Analyst 和 Writer；
5. DashScope 可以生成 1024 维 Embedding；
6. Knowledge、Web、Memory 上下文可以进入 Agent；
7. Docker Sandbox 可以执行命令并返回 Artifact；
8. SSE 可以实时显示并回放事件；
9. Worker 崩溃后能够通过 Lease 恢复。

## 2. 启动整个平台

```powershell
cd D:\gpt项目\AgentForge
.\scripts\bootstrap.ps1
```

打开：

- Web：`http://localhost:8000`
- OpenAPI：`http://localhost:8000/docs`

默认管理员账号位于 `.env` 中。

已有环境时：

```powershell
conda run -n agentforge agentforge sync
docker compose up -d --build
```

## 3. 真实模型配置检查

`.env` 应包含：

```dotenv
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
AGENTFORGE_DEEPSEEK_CHAT_MODEL=deepseek-chat

DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AGENTFORGE_DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
```

同步模型配置：

```powershell
conda run -n agentforge agentforge sync
```

同步后应看到：

- `deepseek-chat`：Chat 模型；
- `dashscope-embedding`：Embedding 模型；
- 示例 Agents 关联 `deepseek-chat`。

## 4. 准备 RAG 数据

在 Web 中：

1. 打开“知识库”。
2. 创建知识库，例如“平台设计资料”。
3. 上传 Markdown、文本或 PDF。
4. 等待文档状态变为 `ready`。
5. 后续运行会自动检索工作区中的知识库。

文档入库会调用 DashScope Embedding，并保存为 1024 维 pgvector。

## 5. Web 演示脚本

### 第一步：启动研究任务

1. 打开“任务工作台”。
2. 选择 `multi-agent-research · 最新版本`。
3. 输入：

```json
{
  "topic": "如何构建可靠的异步多 Agent 平台？"
}
```

4. 点击“创建运行”。

### 第二步：展示实时执行

在运行详情页展示：

- DAG 状态图；
- `run.started`；
- `node.ready`；
- `node.started`；
- `tool.started` 和 `tool.succeeded`；
- `node.succeeded`；
- 最终 `run.succeeded`。

### 第三步：展示产物

运行成功后展示：

- `research-report.md`；
- `chart.svg`；
- Agent citations；
- `memory_used`。

### 第四步：展示 Memory

打开“Memory”：

1. 查看自动提取的摘要；
2. 编辑一条 Memory；
3. 再次运行相似主题；
4. 在 Agent 输出中展示 `memory_used`。

### 第五步：展示运行控制

可以使用：

- 暂停；
- 恢复；
- 取消；
- 重试失败节点；
- Attempt 历史。

## 6. CLI 一键演示

`agentforge demo` 会：

- 创建并运行 `multi-agent-research`；
- 使用 DeepSeek Chat；
- 使用 DashScope Embedding；
- 自动使用 Redis DB 1 启动进程内 Scheduler/Worker；
- 与正在运行的 Compose Worker 隔离。

命令：

```powershell
conda run -n agentforge agentforge demo `
  --topic "如何构建可靠的异步多 Agent 平台？" `
  --timeout 300
```

预期：

```text
status: succeeded
```

并输出 `report`、`analysis` 和 `chart`。

## 7. 自定义 CLI 运行

```powershell
conda run -n agentforge agentforge sync

conda run -n agentforge agentforge run multi-agent-research `
  --input '{"topic":"事件驱动 Agent 的可靠性设计"}' `
  --wait
```

`--wait` 同样会使用隔离 Redis 队列。

## 8. Worker 故障恢复演示

准备隔离数据库和 Redis DB：

```powershell
$env:AGENTFORGE_DATABASE_URL='postgresql+asyncpg://agentforge:agentforge@localhost:5432/agentforge_recovery'
$env:AGENTFORGE_REDIS_URL='redis://localhost:6379/13'
$env:AGENTFORGE_WORKER_LEASE_SECONDS='2'

conda run --no-capture-output -n agentforge python benchmarks/recovery_check.py
```

脚本会验证：

1. Worker 抢占节点；
2. Worker 崩溃；
3. Redis 消息没有 ACK；
4. Lease 到期；
5. Scheduler 将节点恢复为 `ready`；
6. 替换 Worker 接管；
7. Capability 只执行一次。

## 9. ML/DL 训练实验演示

启动可选训练服务：

```powershell
docker compose --profile training up -d --build
conda run -n agentforge agentforge sync
```

Web 演示步骤：

1. 进入“数据集”，创建表格或图片数据集；
2. 上传 CSV，或上传按类别目录组织的 ZIP；
3. 查看 DatasetVersion checksum、split checksum、划分算法和随机种子；
4. 创建 Experiment，平台自动创建 DummyClassifier/DummyRegressor Baseline；
5. 通过 Agent 或 API 在 Recipe 允许参数范围内提交候选 Job；
6. 查看 reserved/consumed total 与 GPU 时间、validation metrics 和 DecisionRecord；
7. 在 Recipe 页面核对允许参数与资源范围，在实验报告页查看 validation leaderboard 和 selection reason；
8. Selection 完成后，由人工会话查看 final test summary；Agent API Key 不返回 test metrics。
7. 执行 validation-only selection；
8. 等待平台 final test evaluation，test 指标只用于人工审核；
9. 在模型仓库把 Candidate 人工提升为 Production。

关键约束：

- 预处理只在 train split fit，validation/test 只 transform；
- Agent 不能访问 test metrics；
- 候选未达到最低 improvement threshold 时保留 Baseline；
- `required_gpu` 不可用即失败，只有 `preferred_gpu` 允许自动回退 CPU；
- Checkpoint 用于保存和追踪，v0.2 不支持自动续训。

## 10. 演示排障

### DeepSeek 返回 400

检查：

- `DEEPSEEK_BASE_URL=https://api.deepseek.com`
- 模型为 `deepseek-chat`
- Key 未过期
- 已执行 `agentforge sync`

### Embedding 维度错误

确认：

- `DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4`
- 数据库迁移已到 `0003_embedding_1024`
- 旧向量已在迁移时清理

```powershell
conda run -n agentforge alembic current
```

### Compose 和 CLI 演示冲突

新版 `agentforge demo` 和 `agentforge run --wait` 会自动使用 Redis DB 1，不需要停止 Compose Worker。
