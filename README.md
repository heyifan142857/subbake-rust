# SubBake

SubBake 是一个使用 Rust 编写的字幕翻译与音视频转写 CLI，也提供交互式终端 Agent。

本项目从原有的 Python 项目 [heyifan142857/subbake](https://github.com/heyifan142857/subbake) 迁移而来。它重新设计了部分核心逻辑与命令，利用并发翻译提升处理速度，并在类型安全、资源占用、错误处理和单文件部署方面做了改进。

> 当前版本仍处于早期开发阶段，命令和配置格式可能继续调整。

## 功能

- 翻译 SRT、ASS、VTT 等字幕与文本文件
- 直接翻译 MKV、MP4/M4V/MOV 和 WebM 中的文本字幕轨
- 并发分批翻译、审校、缓存、失败重试与断点续跑
- 批量处理目录中的字幕文件
- 通过本地 whisper.cpp 转写音视频
- 将转写与翻译组合成完整流水线
- 支持 OpenAI、Anthropic、Gemini 及兼容接口
- 提供带计划确认、会话恢复、历史记录和撤销功能的终端 Agent
- 支持术语表与翻译记忆，并兼容部分旧版运行数据
- 提供无参考译文的字幕 QA、可恢复批处理清单和资源预算

## 安装

需要较新的 Rust 工具链；处理音视频时还需安装 FFmpeg。

```bash
git clone https://github.com/heyifan142857/subbake-rust.git
cd subbake-rust
cargo install --path crates/subbake-cli
```

如需本地转写，再安装 whisper.cpp 和模型：

```bash
sbake whisper install
sbake whisper model base
```

默认安装 CPU 版本，也可用 `--variant cuda|metal|vulkan|openblas` 选择加速构建。安装后即可运行：

```bash
sbake transcribe movie.mp4 --model base --language Auto
sbake pipeline movie.mp4 --transcribe-model base --target-language zh-Hans
```

## 配置

SubBake 会读取 `~/.config/subbake/config.toml` 或项目目录下的 `.subbake.toml`。建议通过环境变量保存 API Key：

```toml
version = 2
default_profile = "default"

[backends.openai]
id = "openai"
model = "gpt-4.1-mini"
api_format = "openai_chat"
api_key_env = "OPENAI_API_KEY"
timeout_seconds = 120

[profiles.default]
translator = "openai"

[profiles.default.translation]
mode = "turbo"
source_language = "English"
target_language = "Simplified Chinese"
```

```bash
export OPENAI_API_KEY="your-api-key"
sbake provider check --profile default
```

使用 `--config` 指定其他配置文件，使用 `--profile` 切换 profile。翻译模式可选
`economy`、`turbo` 或 `cinema`。

## 使用

```bash
# 交互式 Agent
sbake

# 翻译字幕、文本文件或媒体中的文本字幕轨
sbake translate episode.srt --target-lang Chinese
sbake translate movie.mkv --subtitle-stream 7 --target-lang Chinese

# 批量翻译
sbake batch ./subtitles

# 转写，或转写后翻译
sbake transcribe episode.mp4
sbake pipeline episode.mp4

# 恢复最近的 Agent 会话
sbake resume
```

`translate` 不会自动转写音视频；没有文本字幕轨时请使用 `pipeline`。完整选项请运行
`sbake --help` 或 `sbake <COMMAND> --help`。

## 开发

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## License

GNU General Public License v3.0（GPL-3.0-only）
