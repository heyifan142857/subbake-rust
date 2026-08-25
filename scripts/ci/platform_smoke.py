#!/usr/bin/env python3
"""Run the opt-in real FFmpeg/whisper.cpp platform smoke test."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "target" / "platform-smoke-logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"smoke-{sys.platform}.log"


def run(*arguments: os.PathLike[str] | str) -> subprocess.CompletedProcess[str]:
    command = [str(argument) for argument in arguments]
    rendered = subprocess.list2cmdline(command)
    print(f"+ {rendered}", flush=True)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"+ {rendered}\n")
        log.write(result.stdout)
        log.write(result.stderr)
        log.write(f"exit={result.returncode}\n")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    return result


def generate_speech(destination: Path) -> None:
    if sys.platform == "win32":
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$voice.SetOutputToWaveFile('{destination}'); "
            "$voice.Speak('Hello from SubBake cross platform testing.'); "
            "$voice.Dispose()"
        )
        run("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script)
    elif sys.platform == "darwin":
        run("say", "-o", destination, "Hello from SubBake cross platform testing.")
    elif sys.platform.startswith("linux"):
        if shutil.which("espeak-ng") is None:
            raise RuntimeError("espeak-ng is not available on PATH")
        run("espeak-ng", "-w", destination, "Hello from SubBake cross platform testing.")
    else:
        raise RuntimeError(f"unsupported smoke-test platform: {sys.platform}")


def main() -> None:
    binary = ROOT / "target" / "debug" / ("sbake.exe" if sys.platform == "win32" else "sbake")
    if not binary.is_file():
        raise RuntimeError(f"SubBake binary is missing: {binary}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available on PATH")

    with tempfile.TemporaryDirectory(prefix="subbake-platform-smoke-") as temporary:
        work = Path(temporary)
        runtime = work / "runtime"
        source_audio = work / (
            "speech.aiff" if sys.platform == "darwin" else "speech.wav"
        )
        compressed_audio = work / "speech.mp3"
        output = work / "speech.srt"

        generate_speech(source_audio)
        run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", source_audio, compressed_audio)
        run(binary, "whisper", "install", "--variant", "cpu", "--runtime-dir", runtime)
        run(binary, "whisper", "status", "--runtime-dir", runtime)
        run(binary, "whisper", "model", "tiny.en", "--runtime-dir", runtime)
        run(
            binary,
            "transcribe",
            compressed_audio,
            "--language",
            "en",
            "--model",
            "tiny.en",
            "--no-vad",
            "--runtime-dir",
            runtime,
            "--output",
            output,
        )

        subtitle = output.read_text(encoding="utf-8-sig")
        if not re.search(r"\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}", subtitle):
            raise RuntimeError("transcription did not produce a parseable SRT cue")
        text_lines = [
            line.strip()
            for line in subtitle.splitlines()
            if line.strip() and " --> " not in line and not line.strip().isdigit()
        ]
        if not text_lines:
            raise RuntimeError("transcription produced an SRT cue without text")

        run(binary, "whisper", "uninstall", "--runtime-dir", runtime)
        managed_binary = runtime / "whisper" / "bin" / (
            "whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"
        )
        if managed_binary.exists():
            raise RuntimeError("managed whisper binary survived uninstall")


if __name__ == "__main__":
    main()
