use std::collections::BTreeMap;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use subbake_core::{CancellationGuard, CancellationToken};

static NEXT_REGISTRATION: AtomicU64 = AtomicU64::new(1);
static ACTIVE_TOKENS: OnceLock<Mutex<BTreeMap<u64, CancellationToken>>> = OnceLock::new();
static SIGNAL_HANDLER: OnceLock<Result<(), String>> = OnceLock::new();

/// Owns the process-level cancellation bridge for one non-interactive command.
///
/// The OS handler is installed once per process. Each active command registers
/// its own generation token, which keeps repeated in-process CLI tests and
/// independent concurrent callers isolated while still making Ctrl+C cancel all
/// work owned by this process.
pub(crate) struct CliCancellation {
    guard: CancellationGuard,
    registration: u64,
}

impl CliCancellation {
    pub(crate) fn new() -> io::Result<Self> {
        let mut active = active_tokens()
            .lock()
            .map_err(|_| io::Error::other("CLI cancellation registry is poisoned"))?;
        // Hold the registry lock while installing the handler and publishing the
        // token. A signal arriving during setup blocks in the handler until the
        // token and its original generation are both visible.
        ensure_signal_handler()?;
        let token = CancellationToken::default();
        let guard = token.guard();
        let registration = NEXT_REGISTRATION.fetch_add(1, Ordering::Relaxed);
        active.insert(registration, token);
        drop(active);
        Ok(Self {
            guard,
            registration,
        })
    }

    pub(crate) fn guard(&self) -> &CancellationGuard {
        &self.guard
    }
}

impl Drop for CliCancellation {
    fn drop(&mut self) {
        if let Ok(mut active) = active_tokens().lock() {
            active.remove(&self.registration);
        }
    }
}

fn active_tokens() -> &'static Mutex<BTreeMap<u64, CancellationToken>> {
    ACTIVE_TOKENS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn ensure_signal_handler() -> io::Result<()> {
    match SIGNAL_HANDLER
        .get_or_init(|| ctrlc::set_handler(request_cancellation).map_err(|error| error.to_string()))
    {
        Ok(()) => Ok(()),
        Err(message) => Err(io::Error::other(message.clone())),
    }
}

fn request_cancellation() {
    eprintln!("Cancellation requested; stopping current operation…");
    if let Ok(active) = active_tokens().lock() {
        for token in active.values() {
            token.cancel();
        }
    }
}
