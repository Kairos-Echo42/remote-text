# AgentForge 中文基准结果

> 英文版本：[BENCHMARKS.md](BENCHMARKS.md)


## 1. 基准目标

基准只测量平台并发调度和 Worker 执行能力，不调用真实大模型。这样可以：

- 排除模型网络延迟；
- 排除 DeepSeek/DashScope 成本；
- 在 CI 或本地稳定复现；
- 专注验证 DAG 并发、重复提交和 Lease 恢复。

## 2. 并发执行基准

脚本：

```text
benchmarks/concurrency_benchmark.py
```

场景：

- 8 个独立 Mock Agent；
- 每个 Agent 固定执行 200ms；
- `maxConcurrency=4`；
- 4 个 Worker；
- 使用真实 PostgreSQL 和 Redis。

运行方式：

```powershell
$env:AGENTFORGE_DATABASE_URL='postgresql+asyncpg://agentforge:agentforge@localhost:5432/agentforge_benchmark'
$env:AGENTFORGE_REDIS_URL='redis://localhost:6379/14'

conda run --no-capture-output -n agentforge `
  python benchmarks/concurrency_benchmark.py
```

本地实测结果：

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

指标解释：

| 指标 | 含义 |
| --- | --- |
| `node_span_seconds` | 从第一个节点开始到最后一个节点结束 |
| `workflow_wall_seconds` | 从调度开始到观察到 Run 终态 |
| `serial_estimate_seconds` | 8 × 200ms 的串行估算 |
| `parallel_ratio` | 并发执行窗口 / 串行估算 |
| `executor_calls` | 实际执行次数 |
| `unique_executor_calls` | 去重执行次数 |

验收标准：

- `parallel_ratio < 0.6`；
- 8 个节点只执行 8 次；
- 没有重复提交；
- Run 最终状态为 `succeeded`。

为什么不用 `workflow_wall_seconds` 做硬门槛：

- Windows Docker 和事件循环会产生约 0.5–1 秒抖动；
- 该时间包含调度轮询和最终化开销；
- `node_span_seconds` 才直接反映并发执行收益。

## 3. Worker 崩溃恢复基准

脚本：

```text
benchmarks/recovery_check.py
```

运行方式：

```powershell
$env:AGENTFORGE_DATABASE_URL='postgresql+asyncpg://agentforge:agentforge@localhost:5432/agentforge_recovery'
$env:AGENTFORGE_REDIS_URL='redis://localhost:6379/13'
$env:AGENTFORGE_WORKER_LEASE_SECONDS='2'

conda run --no-capture-output -n agentforge `
  python benchmarks/recovery_check.py
```

本地实测：

```json
{
  "run_status": "succeeded",
  "node_status": "succeeded",
  "executor_calls": 1,
  "passed": true
}
```

验证过程：

1. Worker A 抢占节点；
2. Worker A 模拟崩溃且不 ACK；
3. Redis 消息保持 Pending；
4. NodeRun Lease 到期；
5. Scheduler 将节点恢复为 `ready`；
6. Worker B 通过 Pending Entry 接管；
7. Capability 最终只执行一次；
8. Run 和 NodeRun 均为 `succeeded`。

## 4. 真实模型端到端验证

真实 DeepSeek + DashScope 验证不放在自动基准脚本中，使用：

```powershell
conda run -n agentforge agentforge demo `
  --topic "多 Agent 平台可靠性设计" `
  --timeout 300
```

验收：

- Run 状态为 `succeeded`；
- `kb_researcher` 和 `web_researcher` 成功；
- 生成 `research-report.md`；
- 生成 `chart.svg`；
- Web/API 可查询到 citations；
- 后续 Run 可读取长期 Memory。

## 5. 工程检查

```powershell
conda run --no-capture-output -n agentforge `
  ruff check --no-cache src tests benchmarks

$cache = Join-Path $env:TEMP "agentforge-mypy"
conda run --no-capture-output -n agentforge `
  mypy --cache-dir $cache src/agentforge

conda run --no-capture-output -n agentforge `
  pytest -q -p no:cacheprovider
```

当前自动测试结果：`18 passed`。


## 训练可靠性验证

在 GPU 性能基准之前，训练子系统先使用确定性 Fake Backend 验证：

- 每个 Experiment 自动创建 Baseline；
- 预处理只在 train split fit，validation/test 只 transform；
- 模型选择只使用 validation，final test evaluation 由 Platform 在 selection 后执行；
- `max_total_seconds` 与 `max_gpu_seconds` 的预留和实际消耗核销；
- Recipe 参数范围拒绝与不可变 RecipeRegistry；
- 超预算和 test metric 越权请求产生 DecisionRecord；
- Model Bundle checksum、Checkpoint checksum 和 Lineage 完整性。

v0.2.1 不提供多 GPU 或吞吐扩展基准。
