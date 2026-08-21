use subbake_core::{CancellationGuard, CancellationToken};

/// Owns the process-level cancellation bridge for one non-interactive command.
/// Dropping it closes and joins the signal listener before command shutdown.
pub(crate) struct CliCancellation {
    guard: CancellationGuard,
    #[cfg(unix)]
    signal_ids: Vec<signal_hook::SigId>,
    #[cfg(unix)]
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    #[cfg(unix)]
    listener: Option<std::thread::JoinHandle<()>>,
}

impl CliCancellation {
    pub(crate) fn new() -> std::io::Result<Self> {
        let token = CancellationToken::default();
        // Capture the generation before the listener starts so a very early
        // SIGINT cannot be lost by creating the guard afterwards.
        let guard = token.guard();

        #[cfg(unix)]
        {
            use std::sync::Arc;
            use std::sync::atomic::{AtomicBool, Ordering};

            use signal_hook::consts::{SIGINT, SIGTERM};

            let requested = Arc::new(AtomicBool::new(false));
            let shutdown = Arc::new(AtomicBool::new(false));
            let sigint = signal_hook::flag::register(SIGINT, requested.clone())?;
            let sigterm = match signal_hook::flag::register(SIGTERM, requested.clone()) {
                Ok(id) => id,
                Err(error) => {
                    signal_hook::low_level::unregister(sigint);
                    return Err(error);
                }
            };
            let listener_shutdown = shutdown.clone();
            let listener = match std::thread::Builder::new()
                .name("subbake-sigint".to_owned())
                .spawn(move || {
                    while !listener_shutdown.load(Ordering::Acquire) {
                        if requested.swap(false, Ordering::AcqRel) {
                            eprintln!("Cancellation requested; stopping current operation…");
                            token.cancel();
                        }
                        std::thread::park_timeout(std::time::Duration::from_millis(10));
                    }
                }) {
                Ok(listener) => listener,
                Err(error) => {
                    signal_hook::low_level::unregister(sigint);
                    signal_hook::low_level::unregister(sigterm);
                    return Err(error);
                }
            };
            Ok(Self {
                guard,
                signal_ids: vec![sigint, sigterm],
                shutdown,
                listener: Some(listener),
            })
        }

        #[cfg(not(unix))]
        {
            let _ = token;
            Ok(Self { guard })
        }
    }

    pub(crate) fn guard(&self) -> &CancellationGuard {
        &self.guard
    }
}

#[cfg(unix)]
impl Drop for CliCancellation {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;

        for id in self.signal_ids.drain(..) {
            signal_hook::low_level::unregister(id);
        }
        self.shutdown.store(true, Ordering::Release);
        if let Some(listener) = self.listener.take() {
            listener.thread().unpark();
            let _ = listener.join();
        }
    }
}
