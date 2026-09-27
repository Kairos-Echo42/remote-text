# ADR 0002：PostgreSQL 事实源与 Redis 分发

- 状态：已接受
- 日期：2026-09-13

## 背景

Runtime 需要：

- 持久化 Workflow、Run 和 NodeRun；
- 有序审计事件；
- 低延迟 Worker 分发；
- 可恢复的任务队列；
- 向量检索；
- 取消信号。

只用 PostgreSQL 会让高频派发和事件扇出不够轻量；只用 Redis 又会把易失或可重建的数据误当成事实源。

## 决策

PostgreSQL 作为唯一事实源，保存：

- Workflow 和 Agent 定义；
- Run、NodeRun；
- 有序 RunEvent；
- Memory；
- pgvector；
- Transactional Outbox。

Redis Streams 负责：

- 节点派发；
- 每 Run 事件扇出；
- 取消标记；
- Pending Entry 和 Consumer Group。

## 影响

### 正面

- NodeRun 与 Outbox 可以原子提交，避免“状态成功但消息丢失”。
- SSE 断线重放可以直接读取 PostgreSQL。
- Redis 重启或清空不会丢失审计历史。
- 不再需要把 Redis 描述成事实数据库。

### 代价

- 消费者必须接受 at-least-once 语义。
- 节点执行必须使用幂等键。
- Redis 消息可能重复，Worker 必须在事务中再次检查 NodeRun 状态。