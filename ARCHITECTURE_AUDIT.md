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

## 2. 错误模型仍有字符串化和边界泄漏

LLM 边界现已使用结构化调用错误区分取消、超时、认证、限流、传输、拒绝、无效响应
和 continuation 错配；但其他 agent/adapter 路径仍会过早将错误转为展示文本或
`io::Error`，损失可供 CLI、TUI 与策略判断的类别信息。

后续应补齐 crate 边界的结构化错误类型和转换规则：core 保持领域错误，adapters
保留外部错误上下文，CLI/TUI 仅在最终展示层格式化。持久化的兼容数据需继续维持
已有 shape，若要改变必须版本化或提供兼容读取。

验收：取消、配置、认证、限流、输入校验和外部 I/O 可被调用方可靠区分；终端文案
不再作为控制流依据。

## 3. 配置模型的最终收敛尚未完成

默认值已集中到 core，旧 `final_review` 已在配置边界规范化；但配置文件兼容层、CLI
覆盖、profile 更新和运行时展示仍跨多个模块维护字段映射。

后续应以一个明确的兼容解码层读取扁平配置，再转换为 backend、translation、storage
和 output 等拥有者的类型；profile 写入也应复用同一映射。

验收：新增配置项只在其拥有者及一个兼容映射处出现；优先级不依赖赋值顺序，并有
defaults/profile/CLI 三层覆盖测试。

## 4. 交互式终端的真实 PTY 验证待补

单元测试覆盖了新的 `InteractionState` 转换，二进制也可在伪终端启动；但当前自动化
执行环境不会响应终端 DSR 光标位置查询，因此无法完成一次正常的全交互退出验证。

后续应在支持 DSR 的真实 PTY 中手工或自动验证：启动/退出的 raw-mode 与 alternate
screen 清理、Shift+Tab、picker/form、处理中的 Esc 取消、以及 worker 退出后的状态恢复。

验收：上述流程正常退出且终端状态恢复，无悬挂 worker 或残留 alternate screen。
