# AgentForge API

> 中文版本：[API.zh-CN.md](API.zh-CN.md)


Base path: `/api/v1`. Interactive OpenAPI is available at `/docs` in development.

## Authentication

- Browser: HttpOnly JWT session cookie created by `POST /login`.
- Programmatic: `Authorization: Bearer af_...` workspace API key.
- Mutating browser form requests require the CSRF token from the `agentforge_csrf` cookie.

## Resources

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/me` | Current principal and workspace |
| GET/POST | `/workspaces` | List/create workspaces |
| GET | `/workspaces/{id}/workflows` | List workflow definitions |
| GET | `/workspaces/{id}/workflows/{workflow_id}/versions` | Immutable versions |
| POST | `/workspaces/{id}/workflows/sync` | Validate and publish a workflow document |
| GET | `/workspaces/{id}/agents` | Synced Agent definitions |
| GET | `/workspaces/{id}/capabilities` | Built-in and MCP tools |
| POST | `/workspaces/{id}/runs` | Create an asynchronous run |
| GET | `/runs/{run_id}` | Run snapshot |
| GET | `/runs/{run_id}/nodes` | Latest attempt for every node |
| GET | `/runs/{run_id}/events` | SSE event stream with replay |
| POST | `/runs/{run_id}/pause` | Pause future dispatch |
| POST | `/runs/{run_id}/resume` | Resume a paused run |
| POST | `/runs/{run_id}/cancel` | Cancel run and active sandbox |
| POST | `/runs/{run_id}/retry?node_id=...` | Retry a failed run or node |
| GET/POST | `/workspaces/{id}/models` | Chat/Embedding model profiles |
| GET/POST | `/workspaces/{id}/mcp-servers` | MCP servers |
| GET/POST | `/workspaces/{id}/knowledge-bases` | Knowledge bases |
| POST | `/knowledge-bases/{id}/documents` | Upload/index a file |
| GET/POST | `/workspaces/{id}/memories` | Review/create Memory |
| DELETE | `/memories/{id}` | Soft-delete Memory |
| GET | `/workspaces/{id}/artifacts` | List artifacts |
| GET | `/artifacts/{id}` | Download artifact |
| GET/POST | `/workspaces/{id}/api-keys` | Workspace API keys |

## Create Run

```json
POST /api/v1/workspaces/{workspace_id}/runs
{
  "workflow_version_id": "UUID",
  "input": {"topic": "durable agents"},
  "metadata": {"source": "demo"}
}
```

Response: `202 Accepted`

```json
{
  "run_id": "UUID",
  "status": "queued",
  "event_url": "/api/v1/runs/UUID/events"
}
```

## Event Contract

Every event contains `id`, `run_id`, `node_run_id`, `seq`, `type`, `payload`, `created_at`, and `schema_version`.

Important event types: `run.queued`, `run.started`, `node.ready`, `node.started`, `node.succeeded`, `node.failed`, `node.recovered`, `tool.started`, `tool.succeeded`, `run.succeeded`, `run.failed`, `run.cancelled`.

SSE supports `Last-Event-ID` and the `after` query parameter. PostgreSQL is the authoritative replay source.

## Workflow Contract

```yaml
apiVersion: agentforge/v1
kind: Workflow
metadata:
  name: example
  version: 1.0.0
spec:
  inputSchema: {type: object}
  maxConcurrency: 4
  nodes:
    - id: research
      type: agent
      uses: repository-researcher
      needs: []
      with:
        query: $.input.topic
      retry: {maxAttempts: 3, baseDelaySeconds: 1}
      timeoutSeconds: 300
  edges: []
  output:
    result: $.nodes.research.output
```

`with` maps JMESPath values. `when` uses JSON Logic. Cyclic DAGs are rejected.

## 中文 API 说明

所有接口位于 `/api/v1`。浏览器使用 HttpOnly JWT Cookie；程序调用使用工作区 API Key。创建运行总是异步返回 `202`。SSE 使用持久化事件表回放，因此客户端断开或 Redis 重启不会丢失审计事件。

工作流使用 `agentforge/v1` 协议。Web 首发只查看和运行版本，代码仓库中的 Python/YAML 是定义真源。输入映射使用 JMESPath，条件使用 JSON Logic，不允许执行任意 Python 表达式。
