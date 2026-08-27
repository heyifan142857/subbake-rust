//! Typed active-turn input shared by the TUI and the agent decision loop.
//!
//! Steering interrupts only an in-flight model request. Tool cancellation
//! remains owned by the operation cancellation token, so a follow-up cannot
//! accidentally abort a mutating side effect halfway through.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

use subbake_core::{CancellationGuard, CancellationToken};

#[derive(Debug, Default)]
struct TurnSteeringState {
    pending: Mutex<VecDeque<String>>,
    model_interrupt: CancellationToken,
}

/// A cloneable handle for sending instructions to the active agent turn.
#[derive(Debug, Clone, Default)]
pub struct TurnSteering {
    state: Arc<TurnSteeringState>,
}

impl TurnSteering {
    /// Submit a non-empty instruction and interrupt any in-flight model call.
    /// Returns `false` when the input contains only whitespace.
    pub fn submit(&self, text: impl Into<String>) -> bool {
        let text = text.into();
        let text = text.trim();
        if text.is_empty() {
            return false;
        }
        self.pending().push_back(text.to_owned());
        self.state.model_interrupt.cancel();
        true
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.pending().is_empty()
    }

    pub(crate) fn drain(&self) -> Vec<String> {
        self.pending().drain(..).collect()
    }

    pub(crate) fn model_interrupt_guard(&self) -> CancellationGuard {
        self.state.model_interrupt.guard()
    }

    fn pending(&self) -> MutexGuard<'_, VecDeque<String>> {
        self.state
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steering_is_fifo_and_interrupts_only_existing_model_guards() {
        let steering = TurnSteering::default();
        let before = steering.model_interrupt_guard();

        assert!(!steering.submit("   "));
        assert!(steering.submit(" first "));
        assert!(steering.submit("second"));
        assert!(before.is_cancelled());
        assert!(!steering.model_interrupt_guard().is_cancelled());
        assert_eq!(steering.drain(), ["first", "second"]);
        assert!(!steering.has_pending());
    }
}
