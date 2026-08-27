use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicU64>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }

    pub fn guard(&self) -> CancellationGuard {
        CancellationGuard {
            generation: self.0.load(Ordering::Acquire),
            token: self.clone(),
            linked: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CancellationGuard {
    token: CancellationToken,
    generation: u64,
    linked: Vec<(CancellationToken, u64)>,
}

impl CancellationGuard {
    pub fn never() -> Self {
        CancellationToken::default().guard()
    }

    pub fn is_cancelled(&self) -> bool {
        self.token.0.load(Ordering::Acquire) != self.generation
            || self
                .linked
                .iter()
                .any(|(token, generation)| token.0.load(Ordering::Acquire) != *generation)
    }

    /// Return a guard cancelled when either source guard is cancelled.
    pub fn combined_with(mut self, other: &Self) -> Self {
        self.linked.push((other.token.clone(), other.generation));
        self.linked.extend(other.linked.iter().cloned());
        self
    }

    pub fn check(&self) -> CoreResult<()> {
        if self.is_cancelled() {
            Err(CoreError::Cancelled)
        } else {
            Ok(())
        }
    }
}

impl Default for CancellationGuard {
    fn default() -> Self {
        Self::never()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generations_only_cancel_existing_guards() {
        let token = CancellationToken::default();
        let old = token.guard();
        assert!(!old.is_cancelled());
        token.cancel();
        assert!(old.is_cancelled());
        assert!(!token.guard().is_cancelled());
    }

    #[test]
    fn combined_guard_observes_either_cancellation_source() {
        let operation = CancellationToken::default();
        let steering = CancellationToken::default();
        let combined = operation.guard().combined_with(&steering.guard());

        assert!(!combined.is_cancelled());
        steering.cancel();
        assert!(combined.is_cancelled());
        assert!(!operation.guard().is_cancelled());
    }
}
