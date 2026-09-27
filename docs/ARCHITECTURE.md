# AgentForge Architecture

> 中文版本：[ARCHITECTURE.zh-CN.md](ARCHITECTURE.zh-CN.md)


## English

### Design Goals

AgentForge separates workflow control, agent reasoning, capability execution, and presentation. The outer workflow is a durable DAG owned by AgentForge. Agent reasoning is an internal LangGraph state machine. This makes scheduling observable and recoverable while preserving a mature model/tool loop.

### Runtime Topology

- `gateway`: FastAPI REST API, Jinja/HTMX Chinese UI, SSE, authentication, artifacts.
- `scheduler`: leases runs, computes ready nodes, enforces concurrency, recovers expired leases, finalizes runs.
- `worker`: consumes Redis Streams messages, claims node attempts, executes Agent/Capability nodes.
- `postgres`: source of truth for definitions, runs, nodes, ordered events, Memory, vectors, and outbox.
- `redis`: node dispatch stream, per-run event fan-out, and cancellation flags.
- `docker`: isolated execution for untrusted code; Local Sandbox remains available for deterministic tests.

### Durable Scheduling

Creating a run inserts all node attempts in `pending` state. The Scheduler chooses only ready nodes that fit workflow and workspace concurrency. A node and its transactional outbox record are committed together. Workers claim nodes with leases and acknowledge Redis messages only after the attempt reaches a terminal state.

The platform provides at-least-once execution. Idempotency keys are derived from `(run_id, node_id, attempt)`. Non-idempotent external capabilities must declare that property and should implement provider-side idempotency. Expired Worker leases are returned to `ready`; the same attempt resumes while the Redis pending-entry remains available for the replacement consumer.

### Agent Loop

Each Agent node compiles a LangGraph state machine:

```text
START -> reason -> act -> reason -> ... -> END
```

`reason` calls an OpenAI-compatible model. `act` invokes only capabilities allowed by the Agent and node definition. Every Agent step inherits the platform cancellation key and stores usage and context metrics in the node result.

### Capability Runtime

Every tool is a Capability with a name, JSON Schema, side-effect level, idempotency flag, and async invocation method. Built-ins include Web Search, RAG, Memory, workspace files, Sandbox execution, and Artifact creation. MCP tools are registered with the namespace `mcp.<server>.<tool>`. Skills may contribute instructions or register trusted Python capabilities.

### Model Providers

AgentForge stores provider-neutral `ModelProfile` records. v0.1 supports `openai_compatible` and `fake`. Profiles have a `chat` or `embedding` role. The bundled environment creates `deepseek-chat` from `DEEPSEEK_API_KEY` and `dashscope-embedding` from `DASHSCOPE_API_KEY`; embeddings are stored as 1024-dimensional pgvector values.

### Context and Data

RAG ingestion extracts text, chunks documents, computes embeddings, and stores vectors in pgvector. Retrieval merges vector and lexical ranks using Reciprocal Rank Fusion. Memory is extracted after successful runs, deduplicated, reviewable, and retrieved for subsequent Agent contexts. Artifacts are stored behind an ArtifactStore interface; v0.1 uses a local volume and reserves S3-compatible extension.

### Reliability Semantics

- Run states: `pending`, `queued`, `running`, `paused`, `succeeded`, `failed`, `cancelling`, `cancelled`.
- Node states: `pending`, `ready`, `running`, `retry_wait`, `paused`, `succeeded`, `failed`, `skipped`, `cancelled`.
- Events have a monotonic per-run sequence and are persisted before best-effort Redis fan-out.
- SSE replay reads PostgreSQL, so a Redis loss does not lose audit history.
- Pause stops new dispatch; running nodes checkpoint and finish. Cancel writes a Redis cancellation key; Workers stop active tasks and remove containers.

## 中文架构说明

### 设计目标

AgentForge 将工作流控制面、Agent 推理、能力执行和界面分离。外层工作流是平台自研的持久化 DAG，Agent 内部才使用 LangGraph。这样既能清晰展示调度、恢复、并发和事件机制，也能复用成熟的模型工具循环。

### 运行拓扑

- `gateway`：FastAPI、中文 Jinja/HTMX 界面、SSE、认证和 Artifact 下载。
- `scheduler`：抢占运行、计算就绪节点、限制并发、恢复过期 Lease、最终确定 Run。
- `worker`：消费 Redis Streams，抢占 NodeRun，执行 Agent 或 Capability。
- `postgres`：保存定义、运行、节点、有序事件、Memory、向量和 Outbox。
- `redis`：节点分发、每 Run 事件扇出和取消标记。
- `docker`：隔离子进程代码；测试可使用 Local Sandbox。

### 调度与恢复

创建 Run 时所有节点先写入 `pending`。Scheduler 仅选择满足依赖且符合并发上限的节点。节点状态与 Outbox 在同一事务提交。Worker 通过 Lease 抢占节点，直到节点进入终态才 ACK Redis 消息。

平台语义是 at-least-once。幂等键由 `(run_id, node_id, attempt)` 构成。Worker 崩溃后，Lease 到期由 Scheduler 回收为 `ready`；替换 Worker 可继续处理 Redis Pending Entry。

### 安全边界

Docker Sandbox 默认非 root、只读根目录、受限 CPU/内存/PID、无网络，工作目录独立。MCP stdio 与 Python Skill 属于受信任代码，只能由管理员配置或启用。工作区凭据使用 Fernet 加密，API 和事件输出不会返回明文。
