# ADR 0001：自研 DAG 控制面与 LangGraph Agent 内循环

- 状态：已接受
- 日期：2026-09-13

## 背景

平台需要同时满足两个目标：

1. Workflow 调度、并发、恢复和审计必须可检查、可持久化；
2. 单 Agent 的推理与工具循环需要成熟生态支持。

如果整个系统都交给 LangGraph，平台级调度会隐藏在 Agent Graph 中，不便于观察和运维。如果所有 Agent 循环都自研，又会重复实现大量模型协议和状态管理能力。

## 决策

AgentForge 自己负责外层 DAG 控制面，包括：

- Workflow Version；
- NodeRun；
- 依赖计算；
- 并发限制；
- Lease；
- Retry；
- Outbox；
- 事件；
- 暂停、恢复、取消；
- Worker 崩溃恢复。

LangGraph 只用于单个 Agent 节点内部。

平台状态与 LangGraph Checkpoint 通过 `node_run_id` 关联。

## 影响

### 正面

- Workflow 行为可在数据库和 Web 页面中直接观察。
- DAG 调度和恢复不依赖 Agent 框架内部实现。
- Agent 仍能复用 LangGraph 的工具循环和状态迁移。
- 排障和维护时可以明确区分“平台问题”和“Agent 问题”。

### 代价

- 需要维护平台状态与 LangGraph Checkpoint 的关联。
- Agent 节点同时存在平台级 NodeRun 和内部 Graph State。
- 两条状态链需要统一测试。