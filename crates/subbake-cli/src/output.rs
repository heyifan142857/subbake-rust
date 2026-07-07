use subbake_core::entities::{BatchPlanEntry, PipelineResult};

pub fn result_json(result: &PipelineResult) -> String {
    let output_path = result
        .output_path
        .as_ref()
        .map(|path| quote_json(&path.to_string_lossy()))
        .unwrap_or_else(|| "null".to_owned());
    let glossary_path = result
        .glossary_path
        .as_ref()
        .map(|path| quote_json(&path.to_string_lossy()))
        .unwrap_or_else(|| "null".to_owned());
    let planned_batches = result
        .planned_batches
        .iter()
        .map(batch_json)
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "{{\"output_path\":{output_path},\"batches_translated\":{},\"review_batches\":{},\"usage\":{{\"input_tokens\":{},\"output_tokens\":{},\"total_tokens\":{}}},\"dry_run\":{},\"planned_batches\":[{}],\"glossary_path\":{glossary_path}}}",
        result.batches_translated,
        result.review_batches,
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.total_tokens,
        result.dry_run,
        planned_batches
    )
}

fn batch_json(batch: &BatchPlanEntry) -> String {
    format!(
        "{{\"index\":{},\"size\":{},\"first_id\":{},\"last_id\":{}}}",
        batch.index,
        batch.size,
        quote_json(&batch.first_id),
        quote_json(&batch.last_id)
    )
}

fn quote_json(value: &str) -> String {
    let mut output = String::new();
    output.push('"');
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ch => output.push(ch),
        }
    }
    output.push('"');
    output
}

#[cfg(test)]
mod tests {
    use subbake_core::entities::Usage;

    use super::*;

    #[test]
    fn result_json_escapes_paths() {
        let result = PipelineResult {
            output_path: Some("quote\"path.txt".into()),
            batches_translated: 0,
            review_batches: 0,
            usage: Usage::default(),
            dry_run: true,
            planned_batches: Vec::new(),
            cache_hits: 0,
            resumed_translation_batches: 0,
            resumed_review_batches: 0,
            translation_memory_hits: 0,
            state_path: None,
            glossary_path: None,
            agent_repairs: Vec::new(),
        };

        assert!(result_json(&result).contains("quote\\\"path.txt"));
    }
}
