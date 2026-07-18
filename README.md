# SubBake

SubBake 是一个使用 Rust 编写的字幕翻译与音视频转写 CLI，也提供交互式终端 Agent。

本项目从原有的 Python 项目 [heyifan142857/subbake](https://github.com/heyifan142857/subbake) 迁移而来。它重新设计了部分核心逻辑与命令，利用并发翻译提升处理速度，并在类型安全、资源占用、错误处理和单文件部署方面做了改进。

> 当前版本仍处于早期开发阶段，命令和配置格式可能继续调整。

## 功能

- 翻译 SRT、ASS、VTT 等字幕与文本文件
- 并发分批翻译、审校、缓存、失败重试与断点续跑
- 批量处理目录中的字幕文件
- 通过本地 whisper.cpp 转写音视频
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
sbake whisper versions
sbake whisper model list
sbake whisper model base
```

`whisper install` 默认安装可移植的 CPU 构建；可通过
`--variant cuda|metal|vulkan|openblas` 显式选择加速源码构建。Whisper 运行路径默认位于
`.subbake/whisper`；配置 `storage.runtime_dir` 后改为 `<runtime_dir>/whisper`，也可用
`storage.whisper_binary_path` 和 `storage.whisper_models_dir` 分别覆盖。

`whisper versions` 会从 whisper.cpp 的 GitHub Releases 拉取版本列表，并标出当前经过
SubBake 固定校验、可安装的版本。`whisper model list` 会从官方 Hugging Face 仓库拉取
模型目录，同时标记已经下载到本地的模型。安装 CLI 和模型后即可转写：

```bash
sbake transcribe movie.mp4 --model base --language Auto
sbake pipeline movie.mp4 --transcribe-model base --target-language zh-Hans
```

也可以直接启动 `sbake`，对 Agent 说“安装 Whisper”。Agent 会先请求批准安装
whisper-cli，安装完成后拉取模型列表并让你选择；只有你明确选择模型后才会继续下载。

转写未指定模型时：唯一已安装模型会自动使用；多个模型中存在完整的 `small` 时优先
使用它。Agent 在没有完整 `small` 时按 `small* > base* > medium* >
large-v3-turbo* > large-v3* > large-v2* > large-v1* > tiny*` 自动选择，并明确
报告所选模型；同一家族按多语言完整版、q8、q5、英文专用版排序。CLI 遇到无法唯一
决定的多个模型时会列出候选并要求 `--model`。可用
`[defaults.transcription].model` 固定首选模型。

转写前，除 WAV 外的 MP3、M4A、FLAC、OGG 和视频输入都会先通过 FFmpeg 统一转换为
16 kHz、单声道、16-bit PCM WAV。`PREPARE_AUDIO` 会按媒体时长显示转换进度；中间
WAV 存放在运行目录的唯一临时目录中，并在转写成功、失败或取消后自动清理。
Whisper 推理默认使用可用并行度的一半（最多 16 个线程），并通过 `TRANSCRIBE`
进度条持续报告 whisper-cli 的实际完成百分比。

## 配置

SubBake 会依次查找 `~/.config/subbake/config.toml` 和项目目录下的 `.subbake.toml`。建议通过环境变量保存 API Key：

```toml
version = 2
default_profile = "turbo"

[backends.fast]
id = "openai"
model = "gpt-4.1-mini"
api_format = "openai_chat"
api_key_env = "OPENAI_API_KEY"

[backends.reviewer]
id = "anthropic"
model = "claude-sonnet-4-5"
api_format = "anthropic_messages"
api_key_env = "ANTHROPIC_API_KEY"

[profiles.turbo]
translator = "fast"

[profiles.turbo.translation]
mode = "turbo"
source_language = "English"
target_language = "Simplified Chinese"

[profiles.cinema]
translator = "fast"
reviewer = "reviewer"

[profiles.cinema.translation]
mode = "cinema"

[defaults.output]
bilingual = true
bilingual_order = "target_first" # target_first 或 source_first

[defaults.transcription]
model = "small-q8_0" # 可选；未设置时根据已安装模型选择

```

```bash
export OPENAI_API_KEY="your-api-key"
sbake provider check --profile openai
```

也可以使用 `--config` 指定配置文件，或使用 `--profile` 切换完整运行档案。
配置格式当前版本为 `2`，并兼容读取既有版本 `1`。v2 将可复用的 provider
backend 与运行 profile 分开，Cinema profile 可选配置独立 reviewer；未配置时会
回退到 translator 并在结果中标出。`translation.mode` 可选 `economy`、`turbo`
或 `cinema`，高级批次、并发、审校设置仍可覆盖模式默认值。

## 使用

直接启动交互式 Agent：

```bash
sbake
```

翻译单个字幕：

```bash
sbake translate episode.srt

# 最小 token、最低延迟或最高质量
sbake translate episode.srt --mode economy
sbake translate episode.srt --mode turbo
sbake translate episode.srt --mode cinema --profile cinema
```

提交可等待的异步翻译（`overnight`）：

```bash
# OpenAI Batch；仅支持 openai_chat / openai_responses，且必须是 Economy 模式
sbake overnight submit episode.srt --mode economy --profile openai
# 输出的 Manifest 路径是后续操作的唯一凭据（不含 API key）
sbake overnight status .subbake/.../overnight/batch_xxx.json --profile openai
sbake overnight collect .subbake/.../overnight/batch_xxx.json --profile openai
```

`collect` 会重新校验输入字幕的签名，避免把延迟完成的结果写入已经修改过的文件。

离线对照译文回归：

```bash
sbake evaluate candidate.zh-Hans.translated.srt reference.srt --json
```

该命令给出可复现的 chrF 与机械 MQM 结构发现（序号、数字、格式、空译文），用于
版本回归比较；语义与风格仍应由人工抽检。

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
