use subbake_core::{ProgressEvent, ProgressUnit, TaskState};

pub(super) fn format_progress(event: &ProgressEvent, elapsed: std::time::Duration) -> String {
    let state = match event.state {
        TaskState::Cancelling => "Cancelling",
        TaskState::Resuming => "Resuming",
        _ => event.stage.as_str(),
    };
    let counts = match (event.unit, event.total) {
        (ProgressUnit::Duration, Some(total)) if total > 0 => {
            let current = event.current.min(total);
            format!(
                "{} {:>5.1}% · {}/{}",
                progress_bar(current, total),
                current as f64 / total as f64 * 100.0,
                format_duration(current),
                format_duration(total)
            )
        }
        (ProgressUnit::Percent, Some(total)) if total > 0 => {
            let current = event.current.min(total);
            format!(
                "{} {:>5.1}%",
                progress_bar(current, total),
                current as f64 / total as f64 * 100.0
            )
        }
        (_, Some(total)) if total > 0 => format!(
            "{} {}/{}",
            progress_bar(event.current.min(total), total),
            event.current,
            total
        ),
        (ProgressUnit::Duration, _) => {
            format!(
                "{} {}",
                spinner_frame(elapsed),
                format_duration(event.current)
            )
        }
        _ => format!("{} {}", spinner_frame(elapsed), event.current),
    };
    let resumed = if event.resumed > 0 {
        format!(" · resumed {}", event.resumed)
    } else {
        String::new()
    };
    let eta = event
        .total
        .and_then(|total| {
            let completed = event.current.saturating_sub(event.resumed);
            (completed >= 2 && event.current < total).then(|| {
                let seconds = elapsed.as_secs_f64() / completed as f64
                    * total.saturating_sub(event.current) as f64;
                format!(" · ETA {:.0}s", seconds)
            })
        })
        .unwrap_or_default();
    let activity = event
        .translation
        .as_ref()
        .map_or_else(String::new, |detail| {
            format!(
                " · {}/{} batches · active {} · buffered {} · retry {} · TM {}",
                detail.batches_committed,
                detail.batches_total,
                detail.requests_in_flight,
                detail.requests_buffered,
                detail.requests_retrying,
                detail.translation_memory_hits,
            )
        });
    format!(
        "{state} {counts}{activity} · {:.1}s{eta} · {}/{} tok{resumed}",
        elapsed.as_secs_f32(),
        event.usage.input_tokens,
        event.usage.output_tokens
    )
}

fn progress_bar(current: u64, total: u64) -> String {
    let width = 10_u64;
    let filled = current.saturating_mul(width) / total;
    format!(
        "[{}{}]",
        "█".repeat(filled as usize),
        "─".repeat((width - filled) as usize)
    )
}

fn format_duration(milliseconds: u64) -> String {
    let seconds = milliseconds / 1_000;
    let hours = seconds / 3_600;
    let minutes = seconds % 3_600 / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

pub(super) fn spinner_frame(elapsed: std::time::Duration) -> char {
    const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    FRAMES[(elapsed.as_millis() as usize / 80) % FRAMES.len()]
}
