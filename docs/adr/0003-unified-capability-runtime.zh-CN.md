# ADR 0003：统一 Capability Runtime

- 状态：已接受
- 日期：2026-09-13

## 背景

如果 MCP、RAG、Memory、Sandbox、HTTP 和自定义 Skill 各自实现一套工具协议，会产生：

- 不同的参数校验；
- 不同的权限过滤；
- 不同的错误模型；
- 不同的审计事件；
- 不同的重试和幂等语义。

## 决策

所有可执行工具实现统一的异步 Capability 契约：

- 名称与描述；
- JSON Schema；
- 副作用等级；
- Idempotency；
- `async invoke(context, arguments)`；
- 统一 Artifact 返回；
- 统一 Tool Event。

MCP 作为一种 Capability Adapter，而不是平行执行模型。

## 影响

### 正面

- Agent 的工具绑定、权限过滤、Schema 校验和审计集中管理。
- Workflow 和 UI 可以用同一方式展示所有工具。
- MCP、RAG、Memory、Sandbox 可以相互组合。
- 测试可以统一使用 Capability Registry。

### 代价

- 外部协议需要适配层。
- 高吞吐 Capability 可能需要绕过标准结果对象以降低开销。
- 非幂等能力必须显式声明并接受重试风险。