# ADR 0006：Recipe 驱动实验与无泄漏模型选择

- 状态：已接受
- 日期：2026-09-27

## 背景

允许 Agent 生成任意训练代码会带来安全、复现和审查风险。允许 test 指标进入搜索则会造成数据泄漏和过拟合。

## 决策

RecipeRegistry 是平台受信任且不可变的组件。Agent 只能选择已注册 Recipe，以及 Recipe 声明范围内的超参数。预处理只在 train split 上 fit，validation/test 只能 transform。模型选择仅使用 validation 指标。Final test evaluation 由 Platform 在 selection 完成后执行。

## 影响

- Agent 决策保持可审计和可复现。
- Test 指标不能影响排序、搜索、Promotion 或 Agent 上下文。
- 新增模型类型必须提供经过审查的新 Recipe version。
- 超预算和策略违规通过 DecisionRecord 持久记录。