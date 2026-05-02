//! MoE Router — decides which expert handles each input.
//!
//! The router receives text input, tokenizes it, and determines:
//! - Is this a question needing reasoning? → Phi-4 (Thinker)
//! - Is this needing a response? → SmolLM3 (Communicator)
//! - Is this a tool call or routing decision? → Qwen3 handles directly
//! - Is this needing memory? → RAG lookup, then route result
//!
//! For now, routing is rule-based (keyword matching).
//! Eventually Qwen3 itself makes the routing decision via tool calling.

/// Which expert should handle this input.
#[derive(Debug, Clone, PartialEq)]
pub enum Expert {
    /// Qwen3 handles directly (tool calls, simple routing).
    Router,
    /// SmolLM3 generates a response.
    Communicator,
    /// Phi-4 does deep reasoning first.
    Thinker,
    /// Look up memory/context first, then route.
    Memory,
}

/// A routing decision with context.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub expert: Expert,
    pub reason: String,
    pub input_tokens: Vec<u32>,
}

/// Rule-based router (v1). Will be replaced by Qwen3 tool-calling.
pub struct Router {
    // Keyword patterns for routing.
}

impl Router {
    pub fn new() -> Self {
        Self {}
    }

    /// Route an input text to the appropriate expert.
    pub fn route(&self, text: &str) -> Expert {
        let lower = text.to_lowercase();

        // Questions needing reasoning.
        if lower.contains("why ") || lower.contains("how does")
            || lower.contains("explain") || lower.contains("analyze")
            || lower.contains("reason") || lower.contains("think about")
            || lower.contains("what if") || lower.contains("compare")
        {
            return Expert::Thinker;
        }

        // Memory queries.
        if lower.contains("remember") || lower.contains("last time")
            || lower.contains("you said") || lower.contains("we discussed")
            || lower.contains("earlier") || lower.contains("history")
        {
            return Expert::Memory;
        }

        // Tool/command patterns.
        if lower.contains("search for") || lower.contains("look up")
            || lower.contains("find ") || lower.contains("run ")
            || lower.contains("execute") || lower.contains("call ")
        {
            return Expert::Router;
        }

        // Default: generate a response.
        Expert::Communicator
    }
}

/// Context manager — tracks conversation state.
pub struct Context {
    /// Full conversation history (token IDs).
    pub history: Vec<(String, Vec<u32>)>, // (role, tokens)
    /// Summary of older context (compressed).
    pub summary: Option<String>,
    /// Turn counter.
    pub turn: usize,
}

impl Context {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            summary: None,
            turn: 0,
        }
    }

    /// Add a message to the conversation.
    pub fn push(&mut self, role: &str, text: &str, tokens: Vec<u32>) {
        self.history.push((role.to_string(), tokens));
        self.turn += 1;
    }

    /// Get the last N turns as token sequences.
    pub fn recent_tokens(&self, n: usize) -> Vec<&[u32]> {
        self.history.iter()
            .rev()
            .take(n * 2) // user + assistant per turn
            .rev()
            .map(|(_, tokens)| tokens.as_slice())
            .collect()
    }

    /// Total tokens in history.
    pub fn total_tokens(&self) -> usize {
        self.history.iter().map(|(_, t)| t.len()).sum()
    }
}

/// The Strategos brain — orchestrates experts + context.
pub struct Strategos {
    pub router: Router,
    pub context: Context,
}

impl Strategos {
    pub fn new() -> Self {
        Self {
            router: Router::new(),
            context: Context::new(),
        }
    }

    /// Process user input: route → expert → response.
    /// Returns (expert_used, response_text).
    /// For now, returns the routing decision without actual inference.
    pub fn process(&mut self, input: &str) -> (Expert, String) {
        let expert = self.router.route(input);

        let response = match &expert {
            Expert::Thinker => format!("[Phi-4 would reason about: {}]", input),
            Expert::Communicator => format!("[SmolLM3 would respond to: {}]", input),
            Expert::Router => format!("[Qwen3 would handle tool call: {}]", input),
            Expert::Memory => format!("[Memory lookup for: {}]", input),
        };

        self.context.push("user", input, Vec::new());
        self.context.push("assistant", &response, Vec::new());

        (expert, response)
    }
}
