# AgentForge Benchmarks

> 中文版本：[BENCHMARKS.zh-CN.md](BENCHMARKS.zh-CN.md)


## 并发执行基准

脚本：`benchmarks/concurrency_benchmark.py`

场景：8 个独立 Mock Agent，每个执行 200ms；`maxConcurrency=4`；4 个 Worker；真实 PostgreSQL 和 Redis。

```powershell
$env:AGENTFORGE_DATABASE_URL='postgresql+asyncpg://agentforge:agentforge@localhost:5432/agentforge_benchmark'
$env:AGENTFORGE_REDIS_URL='redis://localhost:6379/14'
conda run --no-capture-output -n agentforge python benchmarks/concurrency_benchmark.py
```

2026-09-13 本地实测：

```json
{
  "nodes": 8,
  "max_concurrency": 4,
  "executor_calls": 8,
  "unique_executor_calls": 8,
  "node_span_seconds": 0.6862,
  "workflow_wall_seconds": 1.7148,
  "serial_estimate_seconds": 1.6,
  "parallel_ratio": 0.4288,
  "passed": true
}
```

- `node_span_seconds`：从第一个节点开始到最后一个节点结束，用于衡量 DAG 并发收益。
- `workflow_wall_seconds`：从调度开始到终态轮询观察到成功，包含调度与最终化开销。
- 验收门槛基于并发执行窗口，避免 Windows 上数据库提交与事件循环抖动造成 CI 不稳定。
- 8 次执行对应 8 个唯一节点；没有重复提交。

## Worker 崩溃恢复

脚本：`benchmarks/recovery_check.py`

本地实测：

```json
{
  "run_status": "succeeded",
  "node_status": "succeeded",
  "executor_calls": 1,
  "passed": true
}
```

验证点：Worker 抢占后崩溃、Redis 消息未 ACK、Lease 到期、Scheduler 回收节点、替换 Worker 接管、业务能力只执行一次。


## Training Reliability Checks

The training subsystem uses deterministic Fake Backend tests before any GPU benchmark:

- mandatory baseline creation;
- train-only preprocessor fit and transform-only validation/test;
- validation-only selection and platform final test evaluation;
- reservation accounting under `max_total_seconds` and `max_gpu_seconds`;
- Recipe range rejection and immutable RecipeRegistry;
- DecisionRecord persistence for budget and test-metric policy violations;
- Model Bundle checksums, Checkpoint checksum and lineage.

GPU throughput and multi-GPU scaling are intentionally outside v0.2.1.
