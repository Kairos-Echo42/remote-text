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
| GET/POST | `/workspaces/{id}/datasets` | ML datasets |
| GET/POST | `/datasets/{id}/versions` | Immutable dataset versions and split manifests |
| GET | `/workspaces/{id}/training/recipes` | Trusted Recipe catalog and allowed hyperparameters |
| GET | `/workspaces/{id}/training/baselines` | Trusted BaselineStrategy catalog |
| GET/POST | `/workspaces/{id}/experiments` | Training experiments and budgets |
| GET | `/experiments/{id}/report` | Budget, baseline, validation leaderboard and human final-test report |
| GET | `/experiments/{id}/leaderboard` | Validation-only ranking |
| GET | `/experiments/{id}/budget-reservations` | Job-level reservation and reconciliation records |
| GET | `/experiments/{id}/jobs` | Jobs for an experiment |
| POST | `/experiments/{id}/jobs` | Submit a budgeted Job batch |
| POST | `/experiments/{id}/select-best` | Validation-only model selection |
| POST | `/experiments/{id}/finalize` | Wait for platform final test evaluation |
| GET | `/experiments/{id}/decisions` | DecisionRecord audit |
| GET | `/training/jobs/{id}` | TrainingJob state and resource accounting |
| GET | `/training/jobs/{id}/metrics` | Persisted metrics; test data is human-session-only and rejected for Agent API keys |
| GET | `/training/jobs/{id}/events` | Training lifecycle events |
| GET | `/training/jobs/{id}/manifest` | Reproducibility Manifest |
| GET | `/training/jobs/{id}/checkpoints` | Checkpoint checksum and validation metadata |
| POST | `/training/jobs/{id}/cancel` | Graceful SIGTERM then SIGKILL cancellation |
| GET | `/workspaces/{id}/model-versions` | Candidate/Production/Archived model registry |
| POST | `/model-versions/{id}/promote` | Manual promotion without test-metric ordering |
| POST | `/model-versions/{id}/archive` | Archive a model version |

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

## Training Experiment Contract

- Every Experiment has an immutable BaselineStrategy; the default uses DummyClassifier or DummyRegressor.
- Preprocessing is fit on train only. Validation/test are transform-only.
- Model selection reads validation metrics only. Final test metrics are produced after selection and never enter Agent context.
- `ExperimentBudget` tracks reserved and consumed total/GPU seconds using transactional admission and later reconciliation.
- RecipeRegistry is platform-trusted and immutable. Agents cannot register or modify Recipes.
- Artifact checksums live on Artifact records. Lineage edges store input/output artifacts, operation, timestamp and actor.
- `DecisionRecord` records accepted and rejected decisions; action, result, reason and timestamp are always required.

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
