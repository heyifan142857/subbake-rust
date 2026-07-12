# SubBake

SubBake 是一个使用 Rust 编写的字幕翻译与音视频转写 CLI，也提供交互式终端 Agent。

本项目从作者原有的 Python 项目 [heyifan142857/subbake](https://github.com/heyifan142857/subbake) 迁移而来。它重新设计了部分核心逻辑与命令，利用并发翻译提升处理速度，并在类型安全、资源占用、错误处理和单文件部署方面做了改进。

> 当前版本仍处于早期开发阶段，命令和配置格式可能继续调整。

## 功能

- 翻译 SRT、ASS、VTT 等字幕与文本文件
- 并发分批翻译、审校、缓存、失败重试与断点续跑
- 批量处理目录中的字幕文件
- 通过 Whisper API 或本地 whisper.cpp 转写音视频
- 将转写与翻译组合成完整流水线
- 支持 OpenAI、Anthropic、Gemini 及兼容接口
- 提供带计划确认、会话恢复、历史记录和撤销功能的终端 Agent
- 支持术语表与翻译记忆，并兼容部分旧版运行数据

## 安装

需要 Rust 2024 edition 对应的较新 Rust 工具链：

```bash
git clone https://github.com/heyifan142857/subbake-rust.git
cd subbake-rust
cargo install --path crates/subbake-cli
```

安装后使用 `sbake` 命令。音视频处理还需要 FFmpeg；本地转写可通过内置命令管理 whisper.cpp：

```bash
sbake whisper install
sbake whisper model base
```

## 配置

SubBake 会依次查找 `~/.config/subbake/config.toml` 和项目目录下的 `.subbake.toml`。建议通过环境变量保存 API Key：

```toml
default_profile = "openai"

[profiles.openai]
provider = "openai"
model = "gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"
source_language = "English"
target_language = "Simplified Chinese"
translation_concurrency = 4
review_concurrency = 2
```

```bash
export OPENAI_API_KEY="your-api-key"
sbake provider check --profile openai
```

也可以使用 `--config` 指定配置文件，或使用 `--profile` 切换配置档案。

## 使用

直接启动交互式 Agent：

```bash
sbake
```

翻译单个字幕：

```bash
sbake translate episode.srt
```

批量翻译目录：

```bash
sbake batch ./subtitles
```

转写音视频：

```bash
sbake transcribe episode.mp4
```

先转写，再翻译：

```bash
sbake pipeline episode.mp4
```

恢复最近一次 Agent 会话：

```bash
sbake resume
```

查看完整命令：

```bash
sbake --help
```

`translate` 只处理字幕或文本文件。音视频输入请明确使用 `transcribe` 或 `pipeline`。

## 开发

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## License

GNU General Public License v3.0（GPL-3.0-only）
