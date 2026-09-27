# AgentForge API 中文文档

> 英文版本：[API.md](API.md)


## 1. 基本约定

- API 前缀：`/api/v1`
- 开发环境 OpenAPI：`http://localhost:8000/docs`
- 浏览器认证：登录 `/login` 后使用 HttpOnly JWT Cookie。
- 程序认证：请求头 `Authorization: Bearer af_...`
- 浏览器表单写操作：必须携带 `agentforge_csrf` Cookie 对应的 CSRF Token。
- 错误响应采用 Problem Details 风格，包含 `type`、`title`、`status`、`detail` 和 `instance`。

## 2. 认证与工作区

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/me` | 当前主体、工作区、角色和认证方式 |
| GET | `/api/v1/workspaces` | 当前用户可访问的工作区 |
| POST | `/api/v1/workspaces` | 创建工作区 |

程序调用推荐在工作区设置页创建 API Key。Key 只在创建时显示一次，数据库只保存哈希。

## 3. Workflow 与 Agent

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/workspaces/{workspace_id}/workflows` | Workflow 定义列表 |
| GET | `/api/v1/workspaces/{workspace_id}/workflows/{workflow_id}/versions` | 不可变版本列表 |
| POST | `/api/v1/workspaces/{workspace_id}/workflows/sync` | 校验并发布 Workflow |
| GET | `/api/v1/workspaces/{workspace_id}/agents` | 已同步 Agent 列表 |
| GET | `/api/v1/workspaces/{workspace_id}/capabilities` | 内置与 MCP 能力列表 |

### Workflow YAML 契约

```yaml
apiVersion: agentforge/v1
kind: Workflow
metadata:
  name: example
  version: 1.0.0
spec:
  inputSchema:
    type: object
    properties:
      topic:
        type: string
    required: [topic]
  outputSchema: {}
  maxConcurrency: 4
  nodes:
    - id: research
      type: agent
      uses: repository-researcher
      needs: []
      with:
        query: $.input.topic
        tool_hint: knowledge.search
      retry:
        maxAttempts: 3
        baseDelaySeconds: 1
      timeoutSeconds: 300
  edges: []
  output:
    report: $.nodes.research.output.content
```

- `with` 中的 `$` 表达式使用 JMESPath。
- `when` 使用 JSON Logic。
- 外层 DAG 不允许环。
- `tool_hint` 会确定性执行对应 Capability，再将结果交给 Agent 做最终推理。

## 4. Run API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/workspaces/{workspace_id}/runs` | 创建异步运行，返回 `202` |
| GET | `/api/v1/runs/{run_id}` | Run 快照 |
| GET | `/api/v1/runs/{run_id}/nodes` | 每个节点的最新 Attempt |
| GET | `/api/v1/runs/{run_id}/events` | SSE 事件流和历史重放 |
| POST | `/api/v1/runs/{run_id}/pause` | 暂停后续节点派发 |
| POST | `/api/v1/runs/{run_id}/resume` | 恢复运行 |
| POST | `/api/v1/runs/{run_id}/cancel` | 取消运行和活动沙箱 |
| POST | `/api/v1/runs/{run_id}/retry?node_id=...` | 重试失败节点或 Run |

### 创建 Run

```json
POST /api/v1/workspaces/{workspace_id}/runs
{
  "workflow_version_id": "UUID",
  "input": {
    "topic": "多 Agent 系统可靠性"
  },
  "metadata": {
    "source": "api"
  }
}
```

响应：

```json
{
  "run_id": "UUID",
  "status": "queued",
  "event_url": "/api/v1/runs/UUID/events"
}
```

### Run 状态

`pending`、`queued`、`running`、`paused`、`succeeded`、`failed`、`cancelling`、`cancelled`。

### Node 状态

`pending`、`ready`、`running`、`retry_wait`、`paused`、`succeeded`、`failed`、`skipped`、`cancelled`。

## 5. SSE 事件

使用方式：

```javascript
const source = new EventSource('/api/v1/runs/RUN_ID/events?after=0');
source.addEventListener('node.succeeded', event => {
  console.log(JSON.parse(event.data));
});
```

事件结构：

```json
{
  "id": "UUID",
  "run_id": "UUID",
  "node_run_id": "UUID",
  "seq": 12,
  "type": "node.succeeded",
  "payload": {},
  "schema_version": 1,
  "created_at": "2026-09-14T12:00:00Z"
}
```

- `seq` 在同一 Run 内单调递增。
- 支持 `Last-Event-ID` 和 `after` 参数。
- 权威来源是 PostgreSQL，Redis 重启不会导致审计事件丢失。

常见事件：`run.queued`、`run.started`、`node.ready`、`node.started`、`node.succeeded`、`node.failed`、`node.recovered`、`tool.started`、`tool.succeeded`、`run.succeeded`、`run.failed`、`run.cancelled`。

## 6. 模型配置

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/workspaces/{workspace_id}/models` | Chat/Embedding Profile 列表 |
| POST | `/api/v1/workspaces/{workspace_id}/models` | 创建模型 Profile |

```json
{
  "name": "deepseek-chat",
  "kind": "chat",
  "provider": "openai_compatible",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com",
  "api_key": "只写入请求，不会回传",
  "max_context_tokens": 64000,
  "is_default": true
}
```

`kind` 可选值：

- `chat`：Agent 推理。
- `embedding`：RAG 和 Memory。

## 7. MCP、Skill 与知识库

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET/POST | `/api/v1/workspaces/{workspace_id}/mcp-servers` | MCP Server 配置 |
| GET | `/api/v1/workspaces/{workspace_id}/skills` | 已同步 Skill |
| GET/POST | `/api/v1/workspaces/{workspace_id}/knowledge-bases` | 知识库 |
| POST | `/api/v1/knowledge-bases/{knowledge_base_id}/documents` | 上传并索引文档 |

知识库文档支持 Markdown、文本、PDF、CSV 和 JSON。索引使用工作区默认 `embedding` Profile。

## 8. Memory 与 Artifact

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/workspaces/{workspace_id}/memories` | Memory 列表 |
| POST | `/api/v1/workspaces/{workspace_id}/memories` | 创建 Memory |
| PATCH | `/api/v1/memories/{memory_id}` | 编辑 Memory |
| DELETE | `/api/v1/memories/{memory_id}` | 软删除 Memory |
| GET | `/api/v1/workspaces/{workspace_id}/artifacts` | Artifact 列表 |
| GET | `/api/v1/artifacts/{artifact_id}` | 下载 Artifact |

Artifact 查询可附加 `run_id`：

```text
GET /api/v1/workspaces/{workspace_id}/artifacts?run_id={run_id}
```

## 9. API Key

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/workspaces/{workspace_id}/api-keys` | Key 元数据列表 |
| POST | `/api/v1/workspaces/{workspace_id}/api-keys` | 创建 Key，明文只返回一次 |

## 10. cURL 示例

```bash
curl -X POST "http://localhost:8000/api/v1/workspaces/WORKSPACE_ID/runs" \
  -H "Authorization: Bearer af_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_version_id": "VERSION_UUID",
    "input": {"topic": "多 Agent 系统可靠性"},
    "metadata": {"source": "curl"}
  }'
```
