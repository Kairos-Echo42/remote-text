# AgentForge

AgentForge is an asynchronous, Python-based multi-agent runtime platform. It combines a custom durable DAG control plane with LangGraph-powered agent loops, a unified Capability Runtime, hybrid RAG, reviewable long-term memory, MCP, Skills, Docker Sandbox, and a FastAPI + Jinja web workbench.

> v0.1 is an independent implementation inspired by the architectural ideas of DeerFlow. It does not copy DeerFlow source code.

## Capabilities

| Area | v0.1 implementation |
| --- | --- |
| Orchestration | `agentforge/v1` Workflow DSL, DAG validation, static fan-out/fan-in, JSON Logic conditions |
| Scheduling | PostgreSQL outbox, Redis Streams, leases, retries, pause/resume/cancel, crash recovery |
| Agents | LangGraph plan-act-observe loop, typed capabilities, token and step budgets |
| Models | OpenAI-compatible Chat Completions and Embeddings, plus deterministic Fake providers |
| Context | Hybrid pgvector + lexical RAG, citation markers, durable reviewable Memory |
| Training | Recipe-driven sklearn/PyTorch experiments, budgets, baseline strategies, validation-only selection, checkpoints, model registry |
| Extensibility | MCP stdio/Streamable HTTP, declarative and code-backed Skills |
| Execution | Docker Sandbox with non-root, read-only root, resource limits, network disabled by default |
| Gateway | Native REST API, SSE replay, workspace API keys, OpenAPI |
| Web | Chinese server-rendered workbench, run center, DAG visualization, event timeline, artifacts |
| Quality | Unit/integration tests, benchmark, worker crash recovery check, Ruff, Mypy |

## Architecture

```text
Browser / API Client
        |
        v
FastAPI Gateway ---- PostgreSQL (metadata, events, vector, checkpoints)
        |                    |
        |                    +---- Redis Streams (dispatch, fan-out)
        v
DAG Scheduler
        |
        v
Execution Workers
   |         |          |
   v         v          v
LangGraph  Capability  Docker/Local
 Agents      Runtime      Sandbox
              |
      MCP | RAG | Memory | Skills
```

The outer DAG is owned by AgentForge. Each Agent node uses LangGraph internally. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick Start

Prerequisites: Windows PowerShell, Conda, and Docker Desktop.

```powershell
cd D:\gpt项目\AgentForge
.\scripts\bootstrap.ps1
```

If the Conda environment is already prepared:

```powershell
.\scripts\bootstrap.ps1 -SkipEnvironment
```

The bootstrap script:

1. Detects the Docker CLI, including the configured Docker Desktop path.
2. creates `.env` from `.env.example`;
3. creates the `agentforge` Conda environment with Python 3.12;
4. installs the project and development dependencies;
5. starts PostgreSQL, Redis, migrations, Gateway, Scheduler, and Workers;
6. waits for `GET /health/ready`.

Open [http://localhost:8000](http://localhost:8000). The default local credentials are defined in `.env`.

### Local Demo Without Compose Services

With PostgreSQL and Redis running:

```powershell
conda run -n agentforge agentforge sync
conda run -n agentforge agentforge demo --timeout 180
```

The research workflow runs parallel researchers, analysis, a sandbox chart, and a Markdown report. Configure a Chat provider and an Embedding provider, or use Fake providers in automated tests.

## Provider Configuration

The checked-in examples use DeepSeek for Agent reasoning and DashScope for embeddings. Real credentials belong only in the Git-ignored `.env`:

```dotenv
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
AGENTFORGE_DEEPSEEK_CHAT_MODEL=deepseek-chat

DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AGENTFORGE_DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
```

`agentforge bootstrap` or `agentforge sync` creates/updates:

- `deepseek-chat`: default Chat + tool-calling profile;
- `dashscope-embedding`: default 1024-dimensional embedding profile.

After changing provider credentials:

```powershell
conda run -n agentforge agentforge sync
docker compose up -d --build gateway worker scheduler
```

If the full Compose stack is already running, use the Web workbench or API to create runs. `agentforge demo` automatically uses Redis DB 1 for its in-process Scheduler/Worker so it can coexist with the Compose stack.

## Training Profile

The ML/DL training subsystem is optional and uses a separate CUDA-capable image:

```powershell
docker compose --profile training up -d --build
# Or, from a clean Windows checkout:
.\scripts\bootstrap.ps1 -WithTraining
```

It provides:

- tabular classification/regression with sklearn baselines and PyTorch MLP;
- small image classification with a from-scratch CNN;
- mandatory explainable baselines through trusted BaselineStrategy implementations;
- platform-enforced job, round, wall-clock and GPU budgets with reservation accounting;
- train-only preprocessing, validation-only selection and platform-only final test evaluation;
- DatasetVersion/split/Recipe/config/image/environment reproducibility manifests;
- immutable RecipeRegistry, Artifact Lineage, Checkpoint checksums and candidate model bundles;
- `required_gpu`, `preferred_gpu` and `cpu_only` execution policies.

## CLI

```text
agentforge validate examples/workflows/multi-agent-research.yaml
agentforge bootstrap
agentforge sync
agentforge run multi-agent-research --input '{"topic":"test"}' --wait
agentforge worker
agentforge scheduler
agentforge serve
agentforge demo
```

## API Example

```bash
curl -X POST http://localhost:8000/api/v1/workspaces/WORKSPACE_ID/runs \
  -H "Authorization: Bearer af_..." \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_version_id": "VERSION_UUID",
    "input": {"topic": "durable multi-agent systems"}
  }'
```

The response is `202 Accepted` with `run_id` and `event_url`. Run events use Server-Sent Events and support `Last-Event-ID` replay.

## Development Checks

```powershell
conda run --no-capture-output -n agentforge ruff check --no-cache src tests benchmarks
conda run --no-capture-output -n agentforge mypy --cache-dir $env:TEMP\agentforge-mypy src/agentforge
conda run --no-capture-output -n agentforge pytest -q -p no:cacheprovider
```

## Documentation

English:

- [Architecture](docs/ARCHITECTURE.md)
- [Training](docs/TRAINING.md)
- [API](docs/API.md)
- [Demo guide](docs/DEMO.md)
- [Benchmark results](docs/BENCHMARKS.md)
- [Architecture decisions](docs/adr/)

中文：

- [中文文档总索引](docs/README.zh-CN.md)
- [架构设计](docs/ARCHITECTURE.zh-CN.md)
- [API 文档](docs/API.zh-CN.md)
- [演示指南](docs/DEMO.zh-CN.md)
- [基准结果](docs/BENCHMARKS.zh-CN.md)
- [架构决策](docs/adr/README.zh-CN.md)

## Scope Boundaries

v0.1 intentionally excludes OIDC/self-registration, enterprise RBAC, Kubernetes, MinIO, Office document conversion, a drag-and-drop workflow editor, node-level approvals, cross-region HA, and distributed placement policies.
