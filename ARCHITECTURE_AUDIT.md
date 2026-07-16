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

## 3. 交互式终端的真实 PTY 验证待补

单元测试覆盖了新的 `InteractionState` 转换，二进制也可在伪终端启动；但当前自动化
执行环境不会响应终端 DSR 光标位置查询，因此无法完成一次正常的全交互退出验证。

后续应在支持 DSR 的真实 PTY 中手工或自动验证：启动/退出的 raw-mode 与 alternate
screen 清理、Shift+Tab、picker/form、处理中的 Esc 取消、以及 worker 退出后的状态恢复。

验收：上述流程正常退出且终端状态恢复，无悬挂 worker 或残留 alternate screen。
