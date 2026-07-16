# 架构整改状态

当前审计列出的整改项均已完成。以下内容记录现有边界及其验收依据，
避免后续改动重新把独立职责堆回编排入口。

## 1. 过大的编排模块已拆分

原先集中在少数大文件中的职责已按稳定边界拆开：

- `subbake-adapters/src/llm_backends/` 分离 HTTP 传输与重试、各协议请求/响应转换，
  以及原生工具调用的 continuation 和结果回传；`llm_backends.rs` 只保留适配器组装
  与 `LlmBackend` 编排。
- `subbake-agent/src/decision/` 分离决策数据模型、提示词和意图路由；
  `tool_runner.rs` 负责注册工具的实际执行。
- Agent 会话生命周期、计划持久化、profile 解析与切换、对话呈现和 undo 分别由
  `session_controller.rs`、`plan_coordinator.rs`、`profile_coordinator.rs`、
  `presentation.rs` 和 `undo.rs` 管理；`engine.rs` 只协调这些协作者。
- `subbake-agent/src/tui/` 分离 typed protocol、输入路由、渲染、历史/observer、
  terminal RAII 和 worker 生命周期；`tui.rs` 保留应用状态与主事件循环。
- 翻译管线的术语预处理、恢复/持久化、翻译阶段编排和复审阶段编排分别位于
  `pipeline/terminology.rs`、`pipeline/persistence.rs`、
  `pipeline/translation_runner.rs` 和 `pipeline/review_runner.rs`。

提取后的模块仅暴露 crate 内所需接口；跨 crate 的公共类型仍从所属 crate 根模块
统一导出。完整 workspace 测试（包括真实 PTY 测试）和 Clippy 严格检查均通过。

## 2. 配置模型已收敛

配置文件现使用显式 `version = 1` 和 backend、translation、output、storage 分组，
不再读取旧扁平字段。配置文件、profile 和 CLI 覆盖统一转换为 `SettingsOverrides`，
再由 `ConfigurationResolver` 按内置默认值、defaults、profile、CLI 的固定顺序解析为
`ResolvedSettings`。

profile 写入复用同一分组序列化模型；翻译命令、provider check 和交互式 Agent 均使用
相同解析规则。新增配置项只需在所属分组及其覆盖类型中定义。

## 3. 交互式终端的真实 PTY 验证已自动化

`subbake-agent` 现有 Unix PTY 集成测试会启动真实 `SubBakeTui`，并由测试端模拟终端
响应 Crossterm 的键盘增强能力查询和 DSR 光标位置查询。测试覆盖 Shift+Tab、
profile picker 与创建表单、typed plan approval、处理中 Esc 协作取消，以及取消后的
worker 状态恢复。

退出时测试比较同一 PTY 的前后 `stty -g`，并检查 keyboard enhancement 与 alternate
screen 的进入/退出序列成对出现；子进程必须在硬超时内正常退出，以验证 worker 已
完成 join。该测试随 Unix 上的 `cargo test --workspace` 默认执行；Windows ConPTY
行为不在本项验收范围内。
