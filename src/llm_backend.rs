//! The one language-model backend Said.ai will call.
//!
//! Transcription stays on local Parakeet. Nothing in the window, menu bar, or
//! settings reads this module. Change [`LLM_BACKEND`] when the owner picks a
//! provider; `rewrite` is the only call site.

/// Which backend rewrites text to follow Talk to AI instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmBackend {
    /// No provider is selected yet. `rewrite` returns the original text and
    /// does not make a network request.
    Unconfigured,
}

/// The owner's choice. This is the only place that selects a provider.
pub const LLM_BACKEND: LlmBackend = LlmBackend::Unconfigured;

/// Result of asking the configured backend to apply instructions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rewrite {
    pub text: String,
    /// `true` only when a model request was actually made.
    pub called_model: bool,
}

/// Apply `instructions` to `text`. With no instructions, or while the backend
/// is [`LlmBackend::Unconfigured`], the text is returned unchanged and no
/// model is called.
pub fn rewrite(text: &str, instructions: &[String]) -> Rewrite {
    if instructions.is_empty() {
        return Rewrite {
            text: text.to_string(),
            called_model: false,
        };
    }
    match LLM_BACKEND {
        LlmBackend::Unconfigured => Rewrite {
            text: text.to_string(),
            called_model: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unconfigured_backend_does_not_call_a_model() {
        assert_eq!(LLM_BACKEND, LlmBackend::Unconfigured);
        let rewrite = rewrite(
            "hello there",
            &["translate this to Spanish".to_string()],
        );
        assert_eq!(rewrite.text, "hello there");
        assert!(!rewrite.called_model);
    }

    #[test]
    fn empty_instructions_skip_the_backend() {
        let rewrite = rewrite("hello there", &[]);
        assert_eq!(rewrite.text, "hello there");
        assert!(!rewrite.called_model);
    }
}
