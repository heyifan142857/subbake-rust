use subbake_core::{ProgressEvent, TaskState};

pub(super) fn format_progress(event: &ProgressEvent, elapsed: std::time::Duration) -> String {
    let state = match event.state {
        TaskState::Cancelling => "Cancelling",
        TaskState::Resuming => "Resuming",
        _ => event.stage.as_str(),
    };
    let counts = event
        .total
        .map(|total| {
            let width = 10u64;
            let filled = (event.current.min(total) * width)
                .checked_div(total)
                .unwrap_or(width);
            format!(
                "[{}{}] {}/{}",
                "█".repeat(filled as usize),
                "─".repeat((width - filled) as usize),
                event.current,
                total
            )
        })
        .unwrap_or_else(|| format!("{} {}", spinner_frame(elapsed), event.current));
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

pub(super) fn spinner_frame(elapsed: std::time::Duration) -> char {
    const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    FRAMES[(elapsed.as_millis() as usize / 80) % FRAMES.len()]
}
