# 未完成的架构整改项

本文档只记录仍未完成的整改项；已完成的历史审计结论已移除，避免把
旧状态误当作当前架构保证。

## 1. 拆分仍然过大的编排模块

`subbake-adapters/src/llm_backends.rs`、
`subbake-agent/src/decision.rs`、`subbake-agent/src/engine.rs` 与
`subbake-agent/src/tui.rs` 仍分别承担多项职责。虽然部分 reducer、工具执行器
和协议转换已经抽出，但主要编排文件依然过长，修改一项流程时容易影响无关逻辑。

后续应按稳定边界继续拆分，而不是仅移动代码：

- 将各 LLM 协议的请求/响应转换、原生工具调用和 HTTP 重试拆到独立模块；
- 将 Agent 决策、计划执行、配置切换和对话呈现分为独立协作者；
- 将 TUI 的键盘路由、渲染、worker 生命周期和 picker/form 进一步解耦。

验收：每个提取后的模块只拥有一种主要职责，公共类型位于所属 crate 的明确边界，
原编排入口只负责组装和顺序控制。

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
