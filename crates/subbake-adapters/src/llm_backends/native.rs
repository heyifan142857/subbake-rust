use std::collections::HashMap;

use serde_json::{Value, json};
use subbake_core::error::{CoreError, CoreResult};
use subbake_core::ports::{
    ChatMessage, ModelToolCall, ModelToolResult, ToolChoice, ToolDefinition,
};

use crate::providers::ApiFormat;

use super::protocols;

#[derive(Debug)]
pub(super) struct ProtocolContinuation {
    pub(super) format: ApiFormat,
    system: Option<String>,
    pub(super) history: Vec<Value>,
    pub(super) call_ids: HashMap<String, Option<String>>,
}

pub(super) fn start(format: ApiFormat, messages: &[ChatMessage]) -> ProtocolContinuation {
    let (system, history) = match format {
        ApiFormat::OpenaiChat => (
            None,
            messages
                .iter()
                .map(protocols::openai_message)
                .collect::<Vec<_>>(),
        ),
        ApiFormat::OpenaiResponses => (
            None,
            messages
                .iter()
                .map(|message| {
                    json!({"role":message.role,"content":[{"type":"input_text","text":message.content}]})
                })
                .collect(),
        ),
        ApiFormat::AnthropicMessages => (
            Some(system_text(messages)),
            messages
                .iter()
                .filter(|message| message.role != "system")
                .map(|message| {
                    json!({"role":message.role,"content":[{"type":"text","text":message.content}]})
                })
                .collect(),
        ),
        ApiFormat::GeminiGenerateContent => (
            Some(system_text(messages)),
            messages
                .iter()
                .filter(|message| message.role != "system")
                .map(|message| {
                    json!({"role":if message.role == "assistant" {"model"} else {"user"},"parts":[{"text":message.content}]})
                })
                .collect(),
        ),
    };
    ProtocolContinuation {
        format,
        system,
        history,
        call_ids: HashMap::new(),
    }
}

pub(super) fn append_results(
    continuation: &mut ProtocolContinuation,
    results: &[ModelToolResult],
) -> CoreResult<()> {
    for result in results {
        if !continuation.call_ids.contains_key(&result.id) {
            return Err(CoreError::DataInvariant(format!(
                "native tool result references unknown call `{}`",
                result.id
            )));
        }
    }
    match continuation.format {
        ApiFormat::OpenaiChat => {
            for result in results {
                let wire_id = wire_id(continuation, result);
                continuation.history.push(json!({
                    "role":"tool",
                    "tool_call_id":wire_id,
                    "content":result.output,
                }));
            }
        }
        ApiFormat::OpenaiResponses => {
            for result in results {
                let wire_id = wire_id(continuation, result);
                continuation.history.push(json!({
                    "type":"function_call_output",
                    "call_id":wire_id,
                    "output":result.output,
                }));
            }
        }
        ApiFormat::AnthropicMessages => {
            let content = results
                .iter()
                .map(|result| {
                    json!({
                        "type":"tool_result",
                        "tool_use_id":wire_id(continuation, result),
                        "content":result.output,
                        "is_error":result.is_error,
                    })
                })
                .collect::<Vec<_>>();
            continuation
                .history
                .push(json!({"role":"user","content":content}));
        }
        ApiFormat::GeminiGenerateContent => {
            let parts = results
                .iter()
                .map(|result| {
                    let response = if result.is_error {
                        json!({"error":result.output})
                    } else {
                        json!({"result":result.output})
                    };
                    let mut function_response = json!({
                        "name":result.name,
                        "response":response,
                    });
                    if let Some(id) = continuation
                        .call_ids
                        .get(&result.id)
                        .and_then(|value| value.as_deref())
                    {
                        function_response["id"] = Value::String(id.to_owned());
                    }
                    json!({"functionResponse":function_response})
                })
                .collect::<Vec<_>>();
            continuation
                .history
                .push(json!({"role":"user","parts":parts}));
        }
    }
    Ok(())
}

pub(super) fn request_body(
    format: ApiFormat,
    model: &str,
    continuation: &ProtocolContinuation,
    tools: &[ToolDefinition],
    choice: &ToolChoice,
) -> Value {
    match format {
        ApiFormat::OpenaiChat => json!({
            "model":model,
            "messages":continuation.history,
            "tools":tools.iter().map(openai_chat_tool).collect::<Vec<_>>(),
            "tool_choice":openai_chat_tool_choice(choice),
            "parallel_tool_calls":false,
        }),
        ApiFormat::OpenaiResponses => json!({
            "model":model,
            "input":continuation.history,
            "tools":tools.iter().map(openai_responses_tool).collect::<Vec<_>>(),
            "tool_choice":openai_responses_tool_choice(choice),
            "parallel_tool_calls":false,
        }),
        ApiFormat::AnthropicMessages => json!({
            "model":model,
            "max_tokens":4096,
            "system":continuation.system.as_deref().unwrap_or(""),
            "messages":continuation.history,
            "tools":tools.iter().map(anthropic_tool).collect::<Vec<_>>(),
            "tool_choice":anthropic_tool_choice(choice),
        }),
        ApiFormat::GeminiGenerateContent => json!({
            "systemInstruction":{"parts":[{"text":continuation.system.as_deref().unwrap_or("")}]},
            "contents":continuation.history,
            "tools":[{"functionDeclarations":tools.iter().map(gemini_tool).collect::<Vec<_>>()}],
            "toolConfig":{"functionCallingConfig":gemini_tool_choice(choice)},
        }),
    }
}

pub(super) fn parse_response(
    format: ApiFormat,
    body: &Value,
    continuation: &mut ProtocolContinuation,
) -> CoreResult<(Option<String>, Vec<ModelToolCall>)> {
    let mut calls = Vec::new();
    let text = match format {
        ApiFormat::OpenaiChat => parse_openai_chat(body, continuation, &mut calls)?,
        ApiFormat::OpenaiResponses => parse_openai_responses(body, continuation, &mut calls)?,
        ApiFormat::AnthropicMessages => parse_anthropic(body, continuation, &mut calls)?,
        ApiFormat::GeminiGenerateContent => parse_gemini(body, continuation, &mut calls)?,
    };
    Ok((text.filter(|value| !value.is_empty()), calls))
}

fn parse_openai_chat(
    body: &Value,
    continuation: &mut ProtocolContinuation,
    calls: &mut Vec<ModelToolCall>,
) -> CoreResult<Option<String>> {
    let mut message = body["choices"][0]["message"].clone();
    for (index, call) in message["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let id = call["id"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("openai_chat_call_{index}"));
        continuation.call_ids.insert(id.clone(), Some(id.clone()));
        calls.push(ModelToolCall {
            id,
            name: required_wire_string(call, &["function", "name"])?.to_owned(),
            arguments: parse_wire_arguments(&call["function"]["arguments"])?,
        });
    }
    let mut text = message["content"].as_str().map(str::to_owned);
    if calls.is_empty()
        && let Some(content) = text.as_deref()
        && let Some(dsml_calls) = parse_dsml_tool_calls(content)?
    {
        for call in &dsml_calls {
            continuation
                .call_ids
                .insert(call.id.clone(), Some(call.id.clone()));
        }
        *calls = dsml_calls;
        text = None;
        message["content"] = Value::Null;
        message["tool_calls"] = Value::Array(
            calls
                .iter()
                .map(|call| {
                    json!({
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments.to_string(),
                        }
                    })
                })
                .collect(),
        );
    }
    continuation.history.push(message);
    Ok(text)
}

fn parse_openai_responses(
    body: &Value,
    continuation: &mut ProtocolContinuation,
    calls: &mut Vec<ModelToolCall>,
) -> CoreResult<Option<String>> {
    let output = body["output"].as_array().ok_or_else(|| {
        CoreError::InvalidBackendResponse("OpenAI Responses output is missing".to_owned())
    })?;
    for (index, item) in output.iter().enumerate() {
        continuation.history.push(item.clone());
        if item["type"].as_str() != Some("function_call") {
            continue;
        }
        let wire_id = item["call_id"].as_str().ok_or_else(|| {
            CoreError::InvalidBackendResponse("function call is missing call_id".to_owned())
        })?;
        let id = wire_id.to_owned();
        continuation
            .call_ids
            .insert(id.clone(), Some(wire_id.to_owned()));
        calls.push(ModelToolCall {
            id,
            name: item["name"]
                .as_str()
                .ok_or_else(|| {
                    CoreError::InvalidBackendResponse(format!(
                        "function call {} is missing name",
                        index + 1
                    ))
                })?
                .to_owned(),
            arguments: parse_wire_arguments(&item["arguments"])?,
        });
    }
    Ok(protocols::response_text(ApiFormat::OpenaiResponses, body))
}

fn parse_anthropic(
    body: &Value,
    continuation: &mut ProtocolContinuation,
    calls: &mut Vec<ModelToolCall>,
) -> CoreResult<Option<String>> {
    let content = body["content"].as_array().ok_or_else(|| {
        CoreError::InvalidBackendResponse("Anthropic response content is missing".to_owned())
    })?;
    let mut texts = Vec::new();
    for (index, block) in content.iter().enumerate() {
        match block["type"].as_str() {
            Some("text") => {
                if let Some(value) = block["text"].as_str() {
                    texts.push(value.to_owned());
                }
            }
            Some("tool_use") => {
                let id = block["id"]
                    .as_str()
                    .ok_or_else(|| {
                        CoreError::InvalidBackendResponse(format!(
                            "tool use {} is missing id",
                            index + 1
                        ))
                    })?
                    .to_owned();
                continuation.call_ids.insert(id.clone(), Some(id.clone()));
                calls.push(ModelToolCall {
                    id,
                    name: block["name"]
                        .as_str()
                        .ok_or_else(|| {
                            CoreError::InvalidBackendResponse(format!(
                                "tool use {} is missing name",
                                index + 1
                            ))
                        })?
                        .to_owned(),
                    arguments: block["input"].clone(),
                });
            }
            _ => {}
        }
    }
    continuation
        .history
        .push(json!({"role":"assistant","content":content}));
    Ok((!texts.is_empty()).then(|| texts.join("\n")))
}

fn parse_gemini(
    body: &Value,
    continuation: &mut ProtocolContinuation,
    calls: &mut Vec<ModelToolCall>,
) -> CoreResult<Option<String>> {
    let content = body["candidates"][0]["content"].clone();
    let parts = content["parts"].as_array().ok_or_else(|| {
        CoreError::InvalidBackendResponse("Gemini response parts are missing".to_owned())
    })?;
    let mut texts = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        if let Some(value) = part["text"].as_str() {
            texts.push(value.to_owned());
        }
        let Some(function_call) = part.get("functionCall") else {
            continue;
        };
        let wire_id = function_call["id"].as_str().map(str::to_owned);
        let id = wire_id
            .clone()
            .unwrap_or_else(|| format!("gemini_call_{index}"));
        continuation.call_ids.insert(id.clone(), wire_id);
        calls.push(ModelToolCall {
            id,
            name: function_call["name"]
                .as_str()
                .ok_or_else(|| {
                    CoreError::InvalidBackendResponse(format!(
                        "function call {} is missing name",
                        index + 1
                    ))
                })?
                .to_owned(),
            arguments: function_call["args"].clone(),
        });
    }
    continuation.history.push(content);
    Ok((!texts.is_empty()).then(|| texts.join("\n")))
}

fn parse_dsml_tool_calls(content: &str) -> CoreResult<Option<Vec<ModelToolCall>>> {
    const PREFIXES: [&str; 2] = ["｜｜DSML｜｜", "||DSML||"];
    let Some(prefix) = PREFIXES
        .iter()
        .find(|prefix| content.contains(&format!("<{prefix}tool_calls>")))
    else {
        return Ok(None);
    };
    let invoke_open = format!("<{prefix}invoke ");
    let invoke_close = format!("</{prefix}invoke>");
    let parameter_open = format!("<{prefix}parameter ");
    let parameter_close = format!("</{prefix}parameter>");
    let mut remainder = content;
    let mut calls = Vec::new();
    while let Some(start) = remainder.find(&invoke_open) {
        remainder = &remainder[start + invoke_open.len()..];
        let header_end = remainder.find('>').ok_or_else(|| {
            CoreError::InvalidBackendResponse(
                "DSML tool invocation has an unterminated header".to_owned(),
            )
        })?;
        let name = dsml_attribute(&remainder[..header_end], "name").ok_or_else(|| {
            CoreError::InvalidBackendResponse("DSML tool invocation is missing name".to_owned())
        })?;
        remainder = &remainder[header_end + 1..];
        let body_end = remainder.find(&invoke_close).ok_or_else(|| {
            CoreError::InvalidBackendResponse(format!(
                "DSML tool invocation `{name}` is unterminated"
            ))
        })?;
        let body = &remainder[..body_end];
        let mut parameters = serde_json::Map::new();
        let mut parameter_remainder = body;
        while let Some(parameter_start) = parameter_remainder.find(&parameter_open) {
            parameter_remainder = &parameter_remainder[parameter_start + parameter_open.len()..];
            let parameter_header_end = parameter_remainder.find('>').ok_or_else(|| {
                CoreError::InvalidBackendResponse(format!(
                    "DSML parameter for `{name}` has an unterminated header"
                ))
            })?;
            let header = &parameter_remainder[..parameter_header_end];
            let parameter_name = dsml_attribute(header, "name").ok_or_else(|| {
                CoreError::InvalidBackendResponse(format!(
                    "DSML parameter for `{name}` is missing name"
                ))
            })?;
            let is_string = dsml_attribute(header, "string").as_deref() == Some("true");
            parameter_remainder = &parameter_remainder[parameter_header_end + 1..];
            let value_end = parameter_remainder.find(&parameter_close).ok_or_else(|| {
                CoreError::InvalidBackendResponse(format!(
                    "DSML parameter `{parameter_name}` is unterminated"
                ))
            })?;
            let raw_value = decode_dsml_entities(parameter_remainder[..value_end].trim());
            let value = if is_string {
                Value::String(raw_value)
            } else {
                serde_json::from_str(&raw_value).unwrap_or(Value::String(raw_value))
            };
            let parameter_name = if name == "translate_file" && parameter_name == "file_path" {
                "path".to_owned()
            } else {
                parameter_name
            };
            parameters.insert(parameter_name, value);
            parameter_remainder = &parameter_remainder[value_end + parameter_close.len()..];
        }
        let index = calls.len() + 1;
        calls.push(ModelToolCall {
            id: format!("dsml_call_{index}"),
            name,
            arguments: Value::Object(parameters),
        });
        remainder = &remainder[body_end + invoke_close.len()..];
    }
    if calls.is_empty() {
        return Err(CoreError::InvalidBackendResponse(
            "DSML tool_calls block contains no invocation".to_owned(),
        ));
    }
    Ok(Some(calls))
}

fn system_text(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .filter(|message| message.role == "system")
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn wire_id<'a>(continuation: &'a ProtocolContinuation, result: &'a ModelToolResult) -> &'a str {
    continuation
        .call_ids
        .get(&result.id)
        .and_then(|value| value.as_deref())
        .unwrap_or(&result.id)
}

fn openai_chat_tool(tool: &ToolDefinition) -> Value {
    json!({"type":"function","function":{"name":tool.name,"description":tool.description,"parameters":tool.input_schema}})
}

fn openai_responses_tool(tool: &ToolDefinition) -> Value {
    json!({"type":"function","name":tool.name,"description":tool.description,"parameters":tool.input_schema})
}

fn anthropic_tool(tool: &ToolDefinition) -> Value {
    json!({"name":tool.name,"description":tool.description,"input_schema":tool.input_schema})
}

fn gemini_tool(tool: &ToolDefinition) -> Value {
    json!({"name":tool.name,"description":tool.description,"parameters":tool.input_schema})
}

fn openai_chat_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Specific(name) => json!({"type":"function","function":{"name":name}}),
        ToolChoice::None => json!("none"),
    }
}

fn openai_responses_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Specific(name) => json!({"type":"function","name":name}),
        ToolChoice::None => json!("none"),
    }
}

fn anthropic_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({"type":"auto"}),
        ToolChoice::Required => json!({"type":"any"}),
        ToolChoice::Specific(name) => json!({"type":"tool","name":name}),
        ToolChoice::None => json!({"type":"none"}),
    }
}

fn gemini_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({"mode":"AUTO"}),
        ToolChoice::Required => json!({"mode":"ANY"}),
        ToolChoice::Specific(name) => json!({"mode":"ANY","allowedFunctionNames":[name]}),
        ToolChoice::None => json!({"mode":"NONE"}),
    }
}

fn dsml_attribute(header: &str, name: &str) -> Option<String> {
    let marker = format!(r#"{name}="#);
    let value = header.split_once(&marker)?.1.trim_start();
    let quote = value.chars().next()?;
    if !matches!(quote, '\'' | '"') {
        return None;
    }
    let value = &value[quote.len_utf8()..];
    let end = value.find(quote)?;
    Some(decode_dsml_entities(&value[..end]))
}

fn decode_dsml_entities(value: &str) -> String {
    value
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn required_wire_string<'a>(value: &'a Value, path: &[&str]) -> CoreResult<&'a str> {
    let mut current = value;
    for part in path {
        current = &current[*part];
    }
    current.as_str().ok_or_else(|| {
        CoreError::InvalidBackendResponse(format!("native tool call is missing {}", path.join(".")))
    })
}

fn parse_wire_arguments(value: &Value) -> CoreResult<Value> {
    if let Some(text) = value.as_str() {
        serde_json::from_str(text).map_err(|error| {
            CoreError::InvalidBackendResponse(format!("invalid native tool arguments: {error}"))
        })
    } else {
        Ok(value.clone())
    }
}
