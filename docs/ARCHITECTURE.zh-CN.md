# AgentForge 架构设计

> 英文版本：[ARCHITECTURE.md](ARCHITECTURE.md)


## 1. 架构目标

AgentForge 将系统拆成四个清晰边界：

1. **控制面**：负责 DAG、状态机、并发、重试、暂停、恢复和审计事件。
2. **Agent 运行时**：负责单个 Agent 内部的推理与工具循环。
3. **能力运行时**：统一模型、MCP、RAG、Memory、Skill 和 Sandbox。
4. **接入层**：通过 FastAPI、SSE 和中文 Web 工作台对外提供服务。

外层工作流由 AgentForge 自研实现，Agent 节点内部使用 LangGraph。这样不会把调度和恢复逻辑隐藏在 Agent 框架里，同时仍能复用成熟的模型工具生态。

## 2. 运行拓扑

```text
浏览器 / API 客户端
        │
        ▼
FastAPI Gateway ─────────────── PostgreSQL
        │                       业务状态、事件、向量、Outbox
        │
        ▼
   DAG Scheduler ────────────── Redis Streams
        │                       节点分发、事件扇出、取消标记
        ▼
 Execution Workers
   │           │            │
   ▼           ▼            ▼
LangGraph   Capability   Docker Sandbox
 Agents       Runtime
              │
      MCP / RAG / Memory / Skills
```

### 主要进程

| 进程 | 职责 |
| --- | --- |
| `gateway` | REST API、Jinja 页面、登录、SSE、Artifact 下载 |
| `scheduler` | 计算就绪节点、执行并发限制、恢复过期 Lease、完成 Run |
| `worker` | 消费 Redis Streams，抢占 NodeRun 并执行 Agent/Capability |
| `postgres` | 不可变配置、Run、NodeRun、事件、Memory、向量和 Outbox |
| `redis` | 低延迟节点派发、事件广播和取消信号 |
| `docker` | 隔离运行 Agent 生成的代码 |

## 3. DAG 控制面

### 工作流真源

Workflow 和 Agent 以代码仓库中的 Python DSL 或 YAML 为真源。执行 `agentforge sync` 后：

- 校验 DAG 是否存在重复节点、缺失依赖、环和非法引用；
- 生成不可变 Workflow Version；
- 相同版本如果内容改变，会拒绝覆盖；
- Web 只查看、发布和运行版本，不做双向同步。

### 调度流程

1. 创建 Run 时，所有 NodeRun 先进入 `pending`。
2. Scheduler 找出依赖已经满足的节点。
3. 仅派发符合 `maxConcurrency` 的节点。
4. NodeRun 与 Transactional Outbox 在同一 PostgreSQL 事务提交。
5. Outbox Publisher 将消息写入 Redis Streams。
6. Worker 抢占 NodeRun 并写入 Lease。
7. 节点进入终态后才 ACK Redis 消息。
8. Scheduler 检查全部节点状态并完成 Run。

### 状态机

Run 状态：

```text
pending → queued → running → succeeded
                    │  ├─→ failed
                    │  ├─→ paused → queued
                    │  └─→ cancelling → cancelled
```

Node 状态：

```text
pending → ready → running → succeeded
                    │  ├─→ retry_wait → ready
                    │  ├─→ failed
                    │  └─→ cancelled
                    └─→ skipped
```

### 并发与恢复

- PostgreSQL 是事实源，Redis 只负责低延迟传输。
- 节点执行语义是 **at-least-once**。
- 幂等键为 `(run_id, node_id, attempt)`。
- Worker 崩溃后，Lease 到期由 Scheduler 回收为 `ready`。
- Redis Pending Entry 可由替换 Worker 使用 `XAUTOCLAIM` 接管。
- 非幂等外部 Capability 必须在定义中声明，并自行实现供应商侧幂等。
- Worker 和 Scheduler 统一按 `Run → NodeRun` 顺序加锁，避免死锁。

## 4. Agent 运行时

每个 Agent 节点都会编译一个 LangGraph 状态图：

```text
START → reason → act → reason → ... → END
```

`reason` 负责模型推理；`act` 只允许执行 Agent 与节点声明范围内的 Capability。默认限制为一个工具步骤，工具执行完成后进入最终推理。

### `tool_hint` 的确定性执行

为了兼容不同模型供应商且保持流程稳定，Workflow 节点可以配置：

```yaml
with:
  tool_hint: knowledge.search
```

平台会先直接执行该 Capability，再把结果作为 `tool_result` 注入 Agent 上下文。模型不负责猜测工具名，因此不会因为不同模型对函数名限制不同而导致流程不稳定。

### Agent 预算

- `max_steps`：最大推理回合。
- `max_tool_steps`：最大工具调用回合。
- `timeout_seconds`：节点超时。
- 模型返回的 Token 使用量、步骤数和上下文规模写入 NodeRun metrics。

## 5. Capability Runtime

所有可执行能力统一实现以下契约：

- 名称与描述；
- JSON Schema 输入校验；
- 副作用等级；
- 幂等标记；
- 异步 `invoke(context, arguments)`；
- Artifact 返回。

内置能力：

| 能力 | 作用 |
| --- | --- |
| `web.search` | Web 搜索适配器 |
| `knowledge.search` | 工作区知识库检索 |
| `memory.search` / `memory.write` | 长期 Memory |
| `workspace.read/write/list` | Sandbox 文件操作 |
| `sandbox.execute` | 隔离命令执行 |
| `artifact.write` | 生成可下载产物 |
| `mcp.<server>.<tool>` | MCP 工具动态注册 |

MCP 支持 stdio 和 Streamable HTTP。Python Skill 与 MCP stdio 属于受信任代码边界，只允许管理员配置。

## 6. 模型 Provider 与上下文

### 模型角色

`ModelProfile` 分为：

- `chat`：Agent 推理与工具调用；
- `embedding`：RAG 与 Memory 向量。

默认配置：

- `deepseek-chat` → `https://api.deepseek.com`
- `dashscope-embedding` → `text-embedding-v4`
- DashScope Embedding 在当前配置下返回 **1024 维向量**。

工作区凭据使用 Fernet 加密保存，API 和日志不会返回明文。

### RAG

1. 提取 Markdown、文本、PDF、CSV 或 JSON 内容。
2. 按配置进行分块。
3. 调用工作区 Embedding Profile。
4. 将向量写入 PostgreSQL pgvector。
5. 查询时融合向量排名与词法排名，使用 RRF 生成最终结果。
6. 检索结果带有 `[source:...]` 引用标记。

### Memory

- Run 成功后自动提取事实摘要。
- Memory 记录包含来源 Run、置信度和作用域。
- 用户可在 Web 中查看、编辑和删除。
- 后续 Run 根据工作区、Agent 与查询相关性注入 Memory。

## 7. Sandbox 与 Artifact

- 生产和开发默认使用 Docker Sandbox。
- 测试使用 Local Sandbox，保证确定性。
- Docker 容器使用非 root 用户、只读根文件系统、CPU/内存/PID 限制。
- 默认禁用网络；需要联网时必须显式开启。
- 每个节点尝试使用独立工作目录。
- Compose 中通过 `AGENTFORGE_HOST_WORKSPACE_ROOT` 将宿主机工作区正确挂载给 Docker Sandbox。
- 输出文件通过 ArtifactStore 持久化；v0.1 使用本地卷，并预留 S3 兼容接口。

## 8. 事件、SSE 与可观测性

- 每个 Run 的事件序号单调递增。
- 事件先写入 PostgreSQL，再尽力广播到 Redis。
- SSE 的断线重放从 PostgreSQL 读取，因此 Redis 重启不会丢失审计历史。
- 事件覆盖 Run、Node、Agent、Tool、MCP、Sandbox 和 Artifact。
- 日志使用 JSON 结构和 request/correlation id。

## 9. 安全边界

- 所有 Repository 查询强制 Workspace 隔离。
- Web 使用 HttpOnly JWT Cookie、CSRF 和 Argon2。
- API Key 只保存哈希。
- 工作区密钥加密存储。
- 日志和事件统一脱敏。
- MCP stdio 与 Python Skill 必须经过管理员信任边界。
