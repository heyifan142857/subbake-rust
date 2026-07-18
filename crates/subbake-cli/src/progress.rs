use std::io::{self, IsTerminal, Write};
use std::sync::Mutex;
use std::time::Instant;

use subbake_core::{ProgressEvent, ProgressSink, ProgressUnit, TaskState};

const DOWNLOAD_BAR_WIDTH: usize = 20;

pub struct CliProgress {
    tty: bool,
    started: Instant,
    last: Mutex<Option<(String, u64, TaskState)>>,
}
impl CliProgress {
    pub fn new() -> Self {
        Self {
            tty: io::stderr().is_terminal(),
            started: Instant::now(),
            last: Mutex::new(None),
        }
    }
}
impl ProgressSink for CliProgress {
    fn emit(&self, event: ProgressEvent) {
        let key = (event.stage.clone(), event.current, event.state);
        if !self.tty
            && self
                .last
                .lock()
                .ok()
                .is_some_and(|last| last.as_ref() == Some(&key))
        {
            return;
        }
        if let Ok(mut last) = self.last.lock() {
            *last = Some(key);
        }
        let count = format_progress_count(&event);
        let resumed = if event.resumed > 0 {
            format!(" · resumed {}", event.resumed)
        } else {
            String::new()
        };
        let activity = event
            .translation
            .as_ref()
            .map_or_else(String::new, |detail| {
                format!(
                    " · {}/{} batches · active {} · buffered {} · retry {} · TM {} · cache {}",
                    detail.batches_committed,
                    detail.batches_total,
                    detail.requests_in_flight,
                    detail.requests_buffered,
                    detail.requests_retrying,
                    detail.translation_memory_hits,
                    detail.cache_hits,
                )
            });
        let line = format!(
            "{} {count}{activity} · {:.1}s · {}/{} tok{resumed}",
            event.stage,
            self.started.elapsed().as_secs_f32(),
            event.usage.input_tokens,
            event.usage.output_tokens
        );
        if self.tty
            && !matches!(
                event.state,
                TaskState::Completed | TaskState::Failed | TaskState::Cancelled
            )
        {
            eprint!("\r\x1b[2K{line}");
            let _ = io::stderr().flush();
        } else {
            if self.tty {
                eprint!("\r\x1b[2K");
            }
            eprintln!("{line}");
        }
    }
}

fn format_progress_count(event: &ProgressEvent) -> String {
    match (event.unit, event.total) {
        (ProgressUnit::Bytes, Some(total)) if total > 0 => {
            let current = event.current.min(total);
            let ratio = current as f64 / total as f64;
            let filled =
                ((ratio * DOWNLOAD_BAR_WIDTH as f64).floor() as usize).min(DOWNLOAD_BAR_WIDTH);
            let bar = format!(
                "{}{}",
                "█".repeat(filled),
                "░".repeat(DOWNLOAD_BAR_WIDTH - filled)
            );
            format!(
                "[{bar}] {:>5.1}% · {}/{}",
                ratio * 100.0,
                format_bytes(current),
                format_bytes(total)
            )
        }
        (ProgressUnit::Bytes, None) => format_bytes(event.current),
        (ProgressUnit::Duration, Some(total)) if total > 0 => {
            let current = event.current.min(total);
            let ratio = current as f64 / total as f64;
            let filled =
                ((ratio * DOWNLOAD_BAR_WIDTH as f64).floor() as usize).min(DOWNLOAD_BAR_WIDTH);
            let bar = format!(
                "{}{}",
                "█".repeat(filled),
                "░".repeat(DOWNLOAD_BAR_WIDTH - filled)
            );
            format!(
                "[{bar}] {:>5.1}% · {}/{}",
                ratio * 100.0,
                format_duration(current),
                format_duration(total)
            )
        }
        (ProgressUnit::Duration, None) => format_duration(event.current),
        (ProgressUnit::Percent, Some(total)) if total > 0 => {
            let current = event.current.min(total);
            let ratio = current as f64 / total as f64;
            let filled =
                ((ratio * DOWNLOAD_BAR_WIDTH as f64).floor() as usize).min(DOWNLOAD_BAR_WIDTH);
            let bar = format!(
                "{}{}",
                "█".repeat(filled),
                "░".repeat(DOWNLOAD_BAR_WIDTH - filled)
            );
            format!("[{bar}] {:>5.1}%", ratio * 100.0)
        }
        (_, Some(total)) => format!("{}/{total}", event.current),
        (_, None) => event.current.to_string(),
    }
}

fn format_duration(milliseconds: u64) -> String {
    let total_seconds = milliseconds / 1_000;
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0_usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

#[cfg(test)]
mod tests {
    use subbake_core::{ProgressEvent, ProgressUnit, TaskKind};

    use super::{format_bytes, format_progress_count};

    #[test]
    fn byte_progress_renders_a_bar_percentage_and_sizes() {
        let event = ProgressEvent::running(
            TaskKind::Download,
            "DOWNLOAD_MODEL",
            5 * 1024 * 1024,
            Some(10 * 1024 * 1024),
            ProgressUnit::Bytes,
        );

        assert_eq!(
            format_progress_count(&event),
            "[██████████░░░░░░░░░░]  50.0% · 5.0 MiB/10.0 MiB"
        );
    }

    #[test]
    fn byte_progress_without_a_total_still_uses_a_readable_size() {
        let event = ProgressEvent::running(
            TaskKind::Download,
            "DOWNLOAD_MODEL",
            1536,
            None,
            ProgressUnit::Bytes,
        );

        assert_eq!(format_progress_count(&event), "1.5 KiB");
        assert_eq!(format_bytes(512), "512 B");
    }

    #[test]
    fn non_byte_progress_keeps_the_existing_count_shape() {
        let event = ProgressEvent::running(
            TaskKind::Translation,
            "TRANSLATE",
            3,
            Some(7),
            ProgressUnit::Batches,
        );

        assert_eq!(format_progress_count(&event), "3/7");
    }

    #[test]
    fn duration_progress_renders_media_time_and_percentage() {
        let event = ProgressEvent::running(
            TaskKind::Transcription,
            "PREPARE_AUDIO",
            90_000,
            Some(180_000),
            ProgressUnit::Duration,
        );

        assert_eq!(
            format_progress_count(&event),
            "[██████████░░░░░░░░░░]  50.0% · 1:30/3:00"
        );
    }

    #[test]
    fn percent_progress_renders_a_bar() {
        let event = ProgressEvent::running(
            TaskKind::Transcription,
            "TRANSCRIBE",
            25,
            Some(100),
            ProgressUnit::Percent,
        );

        assert_eq!(
            format_progress_count(&event),
            "[█████░░░░░░░░░░░░░░░]  25.0%"
        );
    }
}
