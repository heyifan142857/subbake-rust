use std::io;
use std::sync::mpsc;
use std::thread;

use subbake_core::CancellationGuard;

use super::{TuiAction, TuiInteraction, TuiObserver};
use crate::error::AgentResult;

pub(super) type WorkerRequest = (TuiAction, CancellationGuard);

pub(super) struct TuiWorker {
    request_tx: Option<mpsc::Sender<WorkerRequest>>,
    response_rx: mpsc::Receiver<AgentResult<TuiInteraction>>,
    join: Option<thread::JoinHandle<()>>,
}

impl TuiWorker {
    pub(super) fn spawn<F>(mut process: F, mut observer: TuiObserver) -> io::Result<Self>
    where
        F: FnMut(TuiAction, CancellationGuard, &mut TuiObserver) -> AgentResult<TuiInteraction>
            + Send
            + 'static,
    {
        let (request_tx, request_rx) = mpsc::channel::<WorkerRequest>();
        let (response_tx, response_rx) = mpsc::channel::<AgentResult<TuiInteraction>>();
        let join = thread::Builder::new()
            .name("subbake-agent-worker".to_owned())
            .spawn(move || {
                while let Ok((action, guard)) = request_rx.recv() {
                    let result = process(action, guard, &mut observer);
                    if response_tx.send(result).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self {
            request_tx: Some(request_tx),
            response_rx,
            join: Some(join),
        })
    }

    pub(super) fn sender(&self) -> io::Result<&mpsc::Sender<WorkerRequest>> {
        self.request_tx
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "agent worker stopped"))
    }

    pub(super) fn try_recv(&self) -> Result<AgentResult<TuiInteraction>, mpsc::TryRecvError> {
        self.response_rx.try_recv()
    }

    pub(super) fn shutdown(&mut self) -> io::Result<()> {
        self.request_tx.take();
        self.join.take().map_or(Ok(()), |join| {
            join.join()
                .map_err(|_| io::Error::other("agent worker panicked"))
        })
    }
}

impl Drop for TuiWorker {
    fn drop(&mut self) {
        self.request_tx.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}
