#![no_main]

use libfuzzer_sys::fuzz_target;
use subbake_agent::tools::{ALL_TOOL_SPECS, validate_tool_call};

fuzz_target!(|data: &[u8]| {
    let Ok(value) = serde_json::from_slice(data) else {
        return;
    };
    for spec in ALL_TOOL_SPECS {
        let _ = validate_tool_call(spec.name, &value);
    }
});
