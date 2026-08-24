use crate::LlmCallError;
use crate::entities::ConcurrencyStrategy;
use crate::error::{CoreError, CoreResult};
use crate::ports::GenerationResponse;

/// Mutable execution accounting kept separate from pipeline stage orchestration.
pub(super) struct PipelineAccounting {
    translation_memory_hits: usize,
    cache_hits: usize,
    provider_requests: usize,
    provider_tokens: usize,
    adaptive_translation_concurrency: usize,
    translation_window_was_rate_limited: bool,
}

impl PipelineAccounting {
    pub(super) fn new(configured_translation_concurrency: usize) -> Self {
        Self {
            translation_memory_hits: 0,
            cache_hits: 0,
            provider_requests: 0,
            provider_tokens: 0,
            adaptive_translation_concurrency: configured_translation_concurrency.clamp(1, 2),
            translation_window_was_rate_limited: false,
        }
    }

    pub(super) fn reserve_requests(
        &mut self,
        additional: usize,
        max_requests: Option<usize>,
        max_tokens: Option<usize>,
    ) -> CoreResult<()> {
        if additional == 0 {
            return Ok(());
        }
        if let Some(limit) = max_requests
            && self.provider_requests.saturating_add(additional) > limit
        {
            return Err(CoreError::ResourceBudgetExceeded(format!(
                "request limit is {limit}; {} request(s) already used and {additional} more required",
                self.provider_requests
            )));
        }
        if let Some(limit) = max_tokens
            && self.provider_tokens >= limit
        {
            return Err(CoreError::ResourceBudgetExceeded(format!(
                "token limit is {limit}; {} token(s) already used",
                self.provider_tokens
            )));
        }
        self.provider_requests = self.provider_requests.saturating_add(additional);
        Ok(())
    }

    pub(super) fn record_response_tokens(
        &mut self,
        responses: &[Result<GenerationResponse, LlmCallError>],
    ) {
        for response in responses
            .iter()
            .filter_map(|response| response.as_ref().ok())
        {
            self.record_tokens(response.usage.total_tokens);
        }
    }

    pub(super) fn record_tokens(&mut self, tokens: usize) {
        self.provider_tokens = self.provider_tokens.saturating_add(tokens);
    }

    pub(super) fn record_cache_hit(&mut self) {
        self.cache_hits = self.cache_hits.saturating_add(1);
    }

    pub(super) const fn cache_hits(&self) -> usize {
        self.cache_hits
    }

    pub(super) fn set_translation_memory_hits(&mut self, hits: usize) {
        self.translation_memory_hits = hits;
    }

    pub(super) const fn translation_memory_hits(&self) -> usize {
        self.translation_memory_hits
    }

    pub(super) fn effective_translation_concurrency(
        &self,
        strategy: ConcurrencyStrategy,
        configured: usize,
    ) -> usize {
        if matches!(strategy, ConcurrencyStrategy::AdaptiveQueued { .. }) {
            self.adaptive_translation_concurrency
        } else {
            configured.max(1)
        }
    }

    pub(super) fn note_translation_window_success(
        &mut self,
        strategy: ConcurrencyStrategy,
        configured: usize,
    ) {
        if matches!(strategy, ConcurrencyStrategy::AdaptiveQueued { .. }) {
            if std::mem::take(&mut self.translation_window_was_rate_limited) {
                return;
            }
            self.adaptive_translation_concurrency = self
                .adaptive_translation_concurrency
                .saturating_add(1)
                .min(configured.max(1));
        }
    }

    pub(super) fn note_translation_rate_limit(&mut self, strategy: ConcurrencyStrategy) {
        if matches!(strategy, ConcurrencyStrategy::AdaptiveQueued { .. }) {
            self.translation_window_was_rate_limited = true;
            self.adaptive_translation_concurrency = self
                .adaptive_translation_concurrency
                .saturating_div(2)
                .max(1);
        }
    }
}
