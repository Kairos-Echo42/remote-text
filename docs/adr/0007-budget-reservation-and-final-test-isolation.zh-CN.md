# ADR 0007：预算预留与 Final Test 隔离

- 状态：已接受
- 日期：2026-09-27

## 背景

并发 Agent 提交可能造成 GPU 时间超卖。即使模型选择规则要求使用 validation，test 指标也容易被误用为搜索信号。

## 决策

提交 Job 时，在创建 BudgetReservation 的同一数据库事务中，按 `max_job_seconds` 硬上限预留 total/GPU 预算。Job 完成后按实际消耗核销并释放未使用部分；`preferred_gpu` 回退 CPU 时立即释放 GPU 预留。

Test metrics 始终是 Platform 所有的 final evaluation metadata，只在 selection 后生成，与 validation metrics 分区保存，仅供人工审核；Agent API Key 必须被隐藏或拒绝。

## 影响

- 并发提交不能超卖总时间或 GPU 预算。
- 预算拒绝、模型选择和最终评估都有完整审计。
- Test metrics 不能影响排序、Promotion、后续搜索或 Agent 上下文。
