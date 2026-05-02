//! Prompt formatting — chat templates for each expert model.
//!
//! Each model expects messages formatted differently:
//! - Qwen3: <|im_start|>role\ncontent<|im_end|>
//! - SmolLM3: <|im_start|>role\ncontent<|im_end|> (same as Qwen/ChatML)
//! - Phi-4: <|role|>content<|end|>

use crate::tokenizer::Tokenizer;

/// A chat message with role and content.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self { role: "system".to_string(), content: content.to_string() }
    }
    pub fn user(content: &str) -> Self {
        Self { role: "user".to_string(), content: content.to_string() }
    }
    pub fn assistant(content: &str) -> Self {
        Self { role: "assistant".to_string(), content: content.to_string() }
    }
}

/// Format variant for each model.
#[derive(Debug, Clone, Copy)]
pub enum ChatFormat {
    /// Qwen3 / SmolLM3: <|im_start|>role\ncontent<|im_end|>\n
    ChatML,
    /// Phi-4: <|role|>content<|end|>\n
    Phi,
}

/// Format messages into a prompt string for the given model.
pub fn format_prompt(messages: &[Message], format: ChatFormat) -> String {
    let mut prompt = String::new();

    match format {
        ChatFormat::ChatML => {
            for msg in messages {
                prompt.push_str(&format!(
                    "<|im_start|>{}\n{}<|im_end|>\n",
                    msg.role, msg.content
                ));
            }
            // Add assistant prefix for generation.
            prompt.push_str("<|im_start|>assistant\n");
        }
        ChatFormat::Phi => {
            for msg in messages {
                prompt.push_str(&format!(
                    "<|{}|>{}<|end|>\n",
                    msg.role, msg.content
                ));
            }
            prompt.push_str("<|assistant|>");
        }
    }

    prompt
}

/// Format a prompt with the Strategos system message.
pub fn format_strategos_prompt(
    user_input: &str,
    context_summary: Option<&str>,
    format: ChatFormat,
) -> String {
    let mut messages = Vec::new();

    // System message — defines Strategos's role.
    let mut system = String::from(
        "You are Strategos, an intelligent assistant. \
         You route queries to the appropriate expert, call tools when needed, \
         and maintain conversation context. Be concise and helpful."
    );

    if let Some(ctx) = context_summary {
        system.push_str("\n\nConversation context:\n");
        system.push_str(ctx);
    }

    messages.push(Message::system(&system));
    messages.push(Message::user(user_input));

    format_prompt(&messages, format)
}

/// Format a tool-calling prompt for Qwen3 (the router).
pub fn format_router_prompt(user_input: &str, available_tools: &[ToolDef]) -> String {
    let mut messages = Vec::new();

    let mut system = String::from(
        "You are a routing assistant. Given the user's input, decide which expert should handle it.\n\n\
         Available experts:\n\
         - communicator: General conversation, responses, explanations\n\
         - thinker: Deep reasoning, analysis, complex questions\n\
         - memory: Look up previous conversations or stored knowledge\n\n\
         Respond with a JSON object: {\"expert\": \"communicator|thinker|memory\", \"reason\": \"...\"}\n"
    );

    if !available_tools.is_empty() {
        system.push_str("\nAvailable tools:\n");
        for tool in available_tools {
            system.push_str(&format!("- {}: {}\n", tool.name, tool.description));
        }
    }

    messages.push(Message::system(&system));
    messages.push(Message::user(user_input));

    format_prompt(&messages, ChatFormat::ChatML)
}

/// A tool definition for the router.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
}

/// Tokenize a formatted prompt string.
pub fn tokenize_prompt(prompt: &str, tokenizer: &Tokenizer) -> Vec<u32> {
    tokenizer.encode(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_format() {
        let messages = vec![
            Message::system("You are helpful."),
            Message::user("Hello"),
        ];
        let prompt = format_prompt(&messages, ChatFormat::ChatML);
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn phi_format() {
        let messages = vec![
            Message::system("You are helpful."),
            Message::user("Hello"),
        ];
        let prompt = format_prompt(&messages, ChatFormat::Phi);
        assert!(prompt.contains("<|system|>You are helpful.<|end|>"));
        assert!(prompt.contains("<|user|>Hello<|end|>"));
        assert!(prompt.ends_with("<|assistant|>"));
    }
}
