# AgentForge 演示指南

> 中文版本：[DEMO.zh-CN.md](DEMO.zh-CN.md)


## 演示目标

用一次真实运行证明以下模块已经连通：

1. FastAPI Gateway 与中文 Web 工作台；
2. PostgreSQL 不可变 Workflow Version；
3. DAG Scheduler 与 4 路并发限制；
4. Knowledge、Web、Memory Context 构建；
5. LangGraph Agent 与 Capability Runtime；
6. Local/Docker Sandbox 命令执行；
7. Artifact 报告与 SVG 图表；
8. SSE 持久化事件和运行恢复语义。

## 启动

```powershell
cd D:\gpt项目\AgentForge
.\scripts\bootstrap.ps1
```

打开 `http://localhost:8000`，使用 `.env` 中的管理员账号登录。

## CLI 一键演示

```powershell
conda run -n agentforge agentforge sync
conda run -n agentforge agentforge demo --topic "如何构建可靠的异步多 Agent 平台？" --timeout 180
```

预期结果：

- Run 状态为 `succeeded`；
- 输出包含 `report`、`analysis` 和 `chart`；
- Artifact 包含 `research-report.md` 和 `chart.svg`；
- `planner → kb_researcher/web_researcher → analyst → chart_run → writer → report_artifact` 依序完成；
- 两个研究节点处于同一并发波次；
- 成功后自动提取长期 Memory。

再次运行同一主题，将看到 `memory_used` 中包含上一轮报告事实。

## Web 演示脚本

1. 在“运行中心”打开最近的 Run。
2. 展示 DAG 状态图和实时事件时间线。
3. 在“知识库”上传 Markdown 或 PDF，下次运行会自动检索。
4. 在“Memory”查看、添加或删除长期事实。
5. 对长时间沙箱节点点击“暂停”和“恢复”。
6. 对失败 Run 点击“重试”，展示 Attempt 历史。
7. 从“产物”区域下载 Markdown 报告和 SVG 图表。

## 故障恢复演示

隔离环境准备 PostgreSQL 与 Redis 后运行：

```powershell
$env:AGENTFORGE_DATABASE_URL='postgresql+asyncpg://agentforge:agentforge@localhost:5432/agentforge_recovery'
$env:AGENTFORGE_REDIS_URL='redis://localhost:6379/13'
$env:AGENTFORGE_WORKER_LEASE_SECONDS='2'
conda run --no-capture-output -n agentforge python benchmarks/recovery_check.py
```

该检查模拟 Worker 抢占后崩溃、不 ACK Redis、Lease 到期、Scheduler 回收节点、替换 Worker 接管，并验证 Capability 只执行一次。
