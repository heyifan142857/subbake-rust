use std::io::{self, IsTerminal, Write};
use std::sync::Mutex;
use std::time::Instant;

use subbake_core::{ProgressEvent, ProgressSink, TaskState};

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
        let count = event.total.map_or_else(
            || event.current.to_string(),
            |total| format!("{}/{}", event.current, total),
        );
        let resumed = if event.resumed > 0 {
            format!(" · resumed {}", event.resumed)
        } else {
            String::new()
        };
        let line = format!(
            "{} {count} · {:.1}s · {}/{} tok{resumed}",
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
