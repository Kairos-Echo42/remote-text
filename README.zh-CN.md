# AgentForge 中文文档

AgentForge 是一个基于 Python 的异步 Multi-Agent Runtime Platform。它将自研持久化 DAG 控制面、LangGraph Agent 内循环、统一 Capability Runtime、混合 RAG、可审阅长期 Memory、MCP、Skill 插件、Docker Sandbox 和 FastAPI + Jinja 中文工作台组合在一起。

> v0.1 是借鉴 DeerFlow 架构思想的独立实现，不复制 DeerFlow 源码。

## 核心能力

| 模块 | v0.1 实现 |
| --- | --- |
| DAG 编排 | `agentforge/v1` Workflow DSL、环与依赖校验、静态 fan-out/fan-in、JSON Logic 条件 |
| 调度可靠性 | PostgreSQL Outbox、Redis Streams、Lease、重试、暂停/恢复/取消、崩溃接管 |
| Agent | LangGraph plan-act-observe、结构化输出、步数与 Token 预算 |
| 模型 | OpenAI 兼容 Chat Completions/Embeddings，测试使用确定性 Fake Provider |
| 上下文 | pgvector + 全文混合检索、引用、自动提取且可审阅的长期 Memory |
| 扩展 | MCP stdio/Streamable HTTP、声明式与代码型 Skills |
| 沙箱 | Docker 非 root、只读根文件系统、资源限制、默认禁用网络、Local Provider 测试 |
| Gateway | 原生 REST API、SSE 断线重放、Workspace API Key、OpenAPI |
| Web | 中文工作台、运行中心、DAG 状态图、实时事件、产物下载 |
| 质量 | 单元/集成测试、并发基准、Worker 恢复验证、Ruff、Mypy |

## 快速开始

前置要求：Windows PowerShell、Conda、Docker Desktop。

```powershell
cd D:\gpt项目\AgentForge
.\scripts\bootstrap.ps1
```

脚本会自动识别 Docker CLI、创建 `.env`、创建 Python 3.12 Conda 环境、安装依赖、执行迁移并启动全部服务。访问 [http://localhost:8000](http://localhost:8000)。

已有 PostgreSQL 与 Redis 时，可直接运行示例研究工作流：

```powershell
conda run -n agentforge agentforge sync
conda run -n agentforge agentforge demo --timeout 180
```

## 模型配置

示例 Agents 默认使用 DeepSeek 进行推理，DashScope 生成 Embedding。真实密钥只写入被 Git 忽略的 `.env`：

```dotenv
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
AGENTFORGE_DEEPSEEK_CHAT_MODEL=deepseek-chat

DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AGENTFORGE_DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
```

执行 `agentforge bootstrap` 或 `agentforge sync` 后，平台会自动创建：

- `deepseek-chat`：默认 Chat/工具调用模型；
- `dashscope-embedding`：默认 1024 维 Embedding 模型。

修改密钥后重新同步并重建容器：

```powershell
conda run -n agentforge agentforge sync
docker compose up -d --build gateway worker scheduler
```

Compose 服务运行时，建议通过 Web 工作台或 API 创建任务。`agentforge demo` 会自动使用 Redis DB 1 启动进程内 Scheduler/Worker，因此不会和 Compose Worker 争夺同一队列。

## 中文文档

- [中文文档总索引](docs/README.zh-CN.md)
- [架构设计](docs/ARCHITECTURE.zh-CN.md)
- [API 文档](docs/API.zh-CN.md)
- [演示指南](docs/DEMO.zh-CN.md)
- [基准结果](docs/BENCHMARKS.zh-CN.md)
- [架构决策中文索引](docs/adr/README.zh-CN.md)

## 边界

v0.1 暂不包含 OIDC/公开注册、企业级细粒度 RBAC、Kubernetes、MinIO、Office 文档转换、拖拽式工作流编辑器、节点审批、跨地域高可用和分布式放置策略。