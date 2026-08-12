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

默认安装包含交互式 Agent 和全部 CLI 命令。如只需要非交互式 CLI（包括翻译、
批处理、转写、流水线和运行时管理），可以关闭默认的 `agent` feature：

```bash
cargo install --path crates/subbake-cli --no-default-features
```

如需本地转写，可以让 SubBake 安装 whisper.cpp 和模型：

```bash
sbake whisper install
sbake whisper model base
```

默认安装 CPU 版本，也可用 `--variant cuda|metal|vulkan|openblas` 选择加速构建。安装后即可运行：

```bash
sbake transcribe movie.mp4 --model base --language Auto
sbake pipeline movie.mp4 --transcribe-model base --target-language zh-Hans
```

安装器不是必需的。如果机器上已有 `whisper-cli` 和 GGML/GGUF 模型，可在配置中
直接指定可执行文件和模型目录，SubBake 不会再要求使用内置安装流程。

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

[profiles.default.transcription]
model = "large-v3-turbo"

[profiles.default.storage]
whisper_binary_path = "/opt/whisper.cpp/build/bin/whisper-cli"
whisper_models_dir = "/opt/whisper.cpp/models"
```

```bash
export OPENAI_API_KEY="your-api-key"
sbake provider check --profile default
```

使用 `--config` 指定其他配置文件，使用 `--profile` 切换 profile。翻译模式可选
`economy`、`turbo` 或 `cinema`。翻译模型由 `[backends.<名称>].model` 指定；如需只为
某个 profile 覆盖模型，可写入 `[profiles.<名称>.backend].model`。交互式界面中输入
`/config` 也可以编辑翻译模型、Whisper 模型、`whisper-cli` 路径和模型目录。

## 翻译模式

三种模式是不同的处理策略，不只是速度档位：

| 模式 | 主要目标 | 翻译策略 | 媒体流水线 |
| --- | --- | --- | --- |
| Economy | 降低请求数和成本 | 使用较大的自包含 batch，默认关闭全文术语预检、在线术语和模型审校 | 缓存转录块，达到配置的 batch 或 token 阈值后再翻译 |
| Turbo | 平衡延迟、吞吐与一致性 | 高并发翻译，使用相邻原文、已确认译文和轻量人名/术语对齐 | 每个稳定的 10 分钟核心块完成后立即开始翻译 |
| Cinema | 优先保证全片质量与一致性 | 场景感知分批、全文术语预检、在线术语和完整审校 | 等待完整转录后再翻译，以便使用全文上下文 |

显式启用全文术语预检时，Turbo 和 Economy 也会自动回退到完整转录后再翻译。
命令行可通过 `--mode economy|turbo|cinema` 选择模式；配置文件使用
`[profiles.<名称>.translation]` 下的 `mode`。显式配置项和 CLI 参数会覆盖模式默认值。

## 转写与流水线

超过 12 分钟的音频会自动按 10 分钟核心窗口转录；相邻窗口各保留 30 秒重叠，
避免边界切断对白。合并后若有效字幕明显未覆盖媒体尾部，任务会失败且不会写入残缺结果。

增量流水线分别保存转录 chunk 和翻译 group，恢复时可独立跳过已完成工作。术语与人名
始终按字幕顺序提交，避免并发导致译名漂移。只有全片覆盖和源文/译文 ID 完整性校验
都通过后，最终字幕才会原子发布。

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

交互式 Agent 另有数据驱动的确定性场景、可选真实模型回归、属性测试和 fuzz
入口。评测用例格式、运行方法及质量门槛见
[`docs/agent-evaluation.md`](docs/agent-evaluation.md)。

## License

GNU General Public License v3.0（GPL-3.0-only）
