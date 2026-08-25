# SubBake

SubBake 是一个使用 Rust 编写的字幕翻译与音视频转写命令行工具，同时提供交互式终端代理。它面向需要批量处理、断点续跑和稳定输出校验的字幕工作流。

> 项目目前处于预览阶段，命令和配置仍可能调整。建议在重要任务中保留原始文件，并先用少量内容验证配置。

## 主要功能

- 翻译 SRT、VTT、ASS/SSA、TTML/DFXP 和 TXT 文件
- 处理媒体容器中的文本字幕轨，并通过 whisper.cpp 转写音视频
- 支持批量翻译、缓存、失败重试、断点续跑、术语表和翻译记忆
- 提供 `economy`、`turbo` 和 `cinema` 三种翻译策略
- 提供字幕质量检查、离线对照评估和带差异预览的字幕润色
- 提供带计划确认、会话恢复和撤销功能的终端代理

## 安装

### Linux x64 预编译包

[Releases](https://github.com/heyifan142857/subbake-rust/releases) 提供两种 Linux x64 预编译包：musl 包不依赖 glibc，GNU 包适用于 glibc 2.35 或更新版本。选择其中一种，并下载对应压缩包和 `SHA256SUMS`。以下以 musl 包为例：

```bash
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf subbake-*-x86_64-unknown-linux-musl.tar.gz
install -Dm755 subbake-*-x86_64-unknown-linux-musl/sbake "$HOME/.local/bin/sbake"
sbake --version
```

预编译包只包含 `sbake`。媒体处理需要系统提供 FFmpeg；终端代理的命令沙箱需要 bubblewrap；位图字幕 OCR 需要 Tesseract。whisper.cpp 及其模型可以由 SubBake 管理。

### 从源码安装

源码构建需要 Rust 1.88 或更新版本：

```bash
git clone https://github.com/heyifan142857/subbake-rust.git
cd subbake-rust
cargo install --path crates/subbake-cli
```

如果不需要交互式终端代理，可以只安装非交互式 CLI：

```bash
cargo install --path crates/subbake-cli --no-default-features
```

## 快速开始

```bash
# 启动交互式终端代理
sbake

# 翻译单个字幕
sbake translate episode.srt --target-lang Chinese

# 批量翻译目录中的字幕
sbake batch ./subtitles --target-lang Chinese

# 安装 whisper.cpp 和基础模型
sbake whisper install
sbake whisper model base

# 转写音视频，或在转写后继续翻译
sbake transcribe interview.mp4 --model base
sbake pipeline interview.mp4 --transcribe-model base --target-lang Chinese

# 检查字幕时间轴与可读性
sbake qa episode.srt
```

`translate` 只处理字幕、文本文件或媒体中的文本字幕轨，不会隐式转写音视频。需要语音识别时请使用 `transcribe` 或 `pipeline`。完整参数可通过 `sbake --help` 和 `sbake <COMMAND> --help` 查看。

## 配置

SubBake 默认读取 `~/.config/subbake/config.toml`，也会识别当前目录中的 `subbake.toml` 或 `.subbake.toml`。下面是一份最小配置示例：

```toml
version = 2
default_profile = "default"

[backends.primary]
id = "provider-id"
model = "translation-model"
api_format = "openai_chat"
base_url = "https://api.example.com/v1"
api_key_env = "SUBBAKE_PROVIDER_API_KEY"

[profiles.default]
translator = "primary"

[profiles.default.translation]
source_language = "Auto"
target_language = "Chinese"
mode = "turbo"
```

API 密钥建议通过环境变量提供，不要写入仓库：

```bash
export SUBBAKE_PROVIDER_API_KEY="your-api-key"
sbake provider check --profile default
```

支持的接口格式包括 OpenAI Chat、OpenAI Responses、Anthropic Messages 和 Gemini Generate Content。具体字段、profile、审校模型及本地转写配置见[使用文档](docs/usage.md)。

## 翻译模式

| 模式 | 适用场景 |
| --- | --- |
| `economy` | 优先减少请求和成本，适合大批量处理 |
| `turbo` | 平衡速度与一致性，也是默认模式 |
| `cinema` | 使用更多上下文、术语预检和审校，适合质量优先的任务 |

显式的配置项和命令行参数会覆盖模式默认值。三种模式都保留输出对齐、格式标记保护、术语约束、缓存隔离和最终校验。

## 平台

| 平台 | 状态 |
| --- | --- |
| Linux x64 | 主要支持平台，提供 GNU 和 musl 预编译包及完整功能 |
| Windows x64 | 实验性支持，不提供终端代理的命令沙箱 |
| macOS arm64 / Intel | 实验性支持，不提供终端代理的命令沙箱 |

目前不提供 Windows 或 macOS 预编译包。更具体的平台边界见[兼容性说明](docs/compatibility.md)。

## 文档

- [使用文档](docs/usage.md)：安装、配置、命令和常见问题
- [更新记录](CHANGELOG.md)
- [兼容性说明](docs/compatibility.md)

## 许可证

[GNU General Public License v3.0](LICENSE)（GPL-3.0-only）
