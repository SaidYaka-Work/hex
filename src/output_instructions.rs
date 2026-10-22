//! Spoken instructions that change how Said.ai's text comes out.
//!
//! Talk to AI transcribes the utterance with Parakeet and stores it here.
//! Short-term instructions live until the process exits or the person
//! replaces or clears them. Long-term instructions are rewritten to
//! `output-instructions.json` under Application Support and survive restarts.
//! Talk itself never calls a language model; [`finish_talk`] only asks
//! [`crate::llm_backend`] when an instruction is active, and that backend
//! currently performs no request.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::llm_backend;

const MAX_INSTRUCTIONS: usize = 8;
const MAX_CHARS: usize = 500;

static REVISION: AtomicU64 = AtomicU64::new(0);
static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
static PERSIST_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstructionScope {
    /// Active until the person changes it, clears it, or quits.
    Session,
    /// Kept across restarts until the person clears it.
    Persistent,
}

impl InstructionScope {
    pub fn label(self) -> &'static str {
        match self {
            Self::Session => "For now",
            Self::Persistent => "Always",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveInstruction {
    pub scope: InstructionScope,
    pub text: String,
}

/// What the home screen says after a Talk to AI utterance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstructionEvent {
    SessionUpdated,
    PersistentUpdated,
    ClearedSession,
    ClearedPersistent,
    ClearedAll,
    Applied,
    WaitingForModel,
    NoTextYet,
}

impl InstructionEvent {
    pub fn label(self) -> &'static str {
        match self {
            Self::SessionUpdated => "That applies for now.",
            Self::PersistentUpdated => "That is saved for next time.",
            Self::ClearedSession => "Cleared the instructions for this session.",
            Self::ClearedPersistent => "Cleared the saved instructions.",
            Self::ClearedAll => "Cleared every instruction.",
            Self::Applied => "Applied that to the current text.",
            Self::WaitingForModel => {
                "Heard that. The language model is not configured yet, so the text stays as dictated."
            }
            Self::NoTextYet => "Heard that. There is no text to change yet.",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstructionOutcome {
    /// Text to paste. Memory-only updates paste nothing.
    pub paste: Option<String>,
    pub event: InstructionEvent,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct PersistentFile {
    instructions: Vec<String>,
}

struct Store {
    session: Vec<String>,
    persistent: Vec<String>,
    last_dictation: Option<String>,
    event: Option<InstructionEvent>,
    loaded: bool,
}

impl Store {
    fn new() -> Self {
        Self {
            session: Vec::new(),
            persistent: Vec::new(),
            last_dictation: None,
            event: None,
            loaded: false,
        }
    }
}

fn store() -> &'static Mutex<Store> {
    STORE.get_or_init(|| Mutex::new(Store::new()))
}

fn lock() -> std::sync::MutexGuard<'static, Store> {
    store().lock().unwrap_or_else(|error| error.into_inner())
}

fn bump() {
    REVISION.fetch_add(1, Ordering::Release);
}

pub fn revision() -> u64 {
    REVISION.load(Ordering::Acquire)
}

pub fn last_event() -> Option<InstructionEvent> {
    let mut state = lock();
    ensure_loaded(&mut state);
    state.event
}

pub fn active() -> Vec<ActiveInstruction> {
    let mut state = lock();
    ensure_loaded(&mut state);
    let mut lines = Vec::new();
    for text in &state.session {
        lines.push(ActiveInstruction {
            scope: InstructionScope::Session,
            text: text.clone(),
        });
    }
    for text in &state.persistent {
        lines.push(ActiveInstruction {
            scope: InstructionScope::Persistent,
            text: text.clone(),
        });
    }
    lines
}

/// Remember a Talk transcript and return the text to paste.
///
/// Parakeet already produced `transcript`. This does not interpret it as an
/// instruction. When no instruction is active the transcript is returned as
/// itself. When one is active, the configured backend may rewrite it; the
/// unconfigured backend returns the transcript and makes no request.
pub fn finish_talk(transcript: &str) -> String {
    let text = transcript.trim();
    if text.is_empty() {
        return String::new();
    }
    let mut state = lock();
    ensure_loaded(&mut state);
    state.last_dictation = Some(bound(text));
    let instructions = standing(&state);
    drop(state);
    llm_backend::rewrite(text, &instructions).text
}

/// Record a Talk to AI utterance. The transcript is the instruction, not the
/// text to paste.
pub fn accept(transcript: &str, selected: Option<&str>) -> InstructionOutcome {
    let spoken = transcript.trim();
    let normalized = normalize(spoken);
    let mut state = lock();
    ensure_loaded(&mut state);
    if normalized.is_empty() {
        state.event = Some(InstructionEvent::NoTextYet);
        bump();
        return InstructionOutcome {
            paste: None,
            event: InstructionEvent::NoTextYet,
        };
    }
    if let Some(event) = clear_command(&normalized) {
        match event {
            InstructionEvent::ClearedAll => {
                state.session.clear();
                state.persistent.clear();
                save_persistent(&state.persistent);
            }
            InstructionEvent::ClearedPersistent => {
                state.persistent.clear();
                save_persistent(&state.persistent);
            }
            InstructionEvent::ClearedSession => state.session.clear(),
            InstructionEvent::SessionUpdated
            | InstructionEvent::PersistentUpdated
            | InstructionEvent::Applied
            | InstructionEvent::WaitingForModel
            | InstructionEvent::NoTextYet => {}
        }
        state.event = Some(event);
        bump();
        return InstructionOutcome {
            paste: None,
            event,
        };
    }

    let one_shot = is_one_shot(&normalized);
    let persistent = wants_persistent(&normalized);
    let session = wants_session(&normalized) || (!one_shot && !persistent);
    let append = wants_append(&normalized);
    let phrase = bound(spoken);
    if persistent {
        store_phrase(&mut state.persistent, phrase.clone(), append);
        save_persistent(&state.persistent);
    }
    if session {
        store_phrase(&mut state.session, phrase.clone(), append);
    }

    let source = selected
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| state.last_dictation.clone());
    let outcome = if one_shot {
        match source {
            Some(source) => {
                let rewrite = llm_backend::rewrite(&source, &[phrase.clone()]);
                if rewrite.called_model {
                    InstructionOutcome {
                        paste: Some(rewrite.text),
                        event: InstructionEvent::Applied,
                    }
                } else {
                    if !session && !persistent {
                        store_phrase(&mut state.session, phrase, false);
                    }
                    InstructionOutcome {
                        paste: None,
                        event: InstructionEvent::WaitingForModel,
                    }
                }
            }
            None => {
                if !session && !persistent {
                    store_phrase(&mut state.session, phrase, false);
                }
                InstructionOutcome {
                    paste: None,
                    event: InstructionEvent::NoTextYet,
                }
            }
        }
    } else if persistent {
        InstructionOutcome {
            paste: None,
            event: InstructionEvent::PersistentUpdated,
        }
    } else {
        InstructionOutcome {
            paste: None,
            event: InstructionEvent::SessionUpdated,
        }
    };
    state.event = Some(outcome.event);
    bump();
    outcome
}

fn standing(state: &Store) -> Vec<String> {
    state
        .session
        .iter()
        .chain(state.persistent.iter())
        .cloned()
        .collect()
}

fn store_phrase(list: &mut Vec<String>, phrase: String, append: bool) {
    if append {
        if !list.iter().any(|item| item == &phrase) {
            list.push(phrase);
        }
    } else {
        list.clear();
        list.push(phrase);
    }
    if list.len() > MAX_INSTRUCTIONS {
        let drop_count = list.len() - MAX_INSTRUCTIONS;
        list.drain(0..drop_count);
    }
}

fn ensure_loaded(state: &mut Store) {
    if state.loaded {
        return;
    }
    state.loaded = true;
    state.persistent = load_persistent();
}

fn persist_path() -> PathBuf {
    if let Some(path) = PERSIST_PATH
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone()
    {
        return path;
    }
    crate::app_paths::support_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("said-ai"))
        .join("output-instructions.json")
}

fn load_persistent() -> Vec<String> {
    let path = persist_path();
    let Ok(bytes) = fs::read(&path) else {
        return Vec::new();
    };
    serde_json::from_slice::<PersistentFile>(&bytes)
        .map(|file| file.instructions.into_iter().map(|item| bound(&item)).collect())
        .unwrap_or_default()
}

fn save_persistent(instructions: &[String]) {
    let path = persist_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = PersistentFile {
        instructions: instructions.to_vec(),
    };
    let Ok(bytes) = serde_json::to_vec_pretty(&file) else {
        return;
    };
    atomic_write(&path, &bytes);
}

fn atomic_write(path: &Path, bytes: &[u8]) {
    let temporary = path.with_extension("json.tmp");
    if fs::write(&temporary, bytes).is_ok() {
        if fs::rename(&temporary, path).is_err() {
            let _ = fs::write(path, bytes);
            let _ = fs::remove_file(&temporary);
        }
    }
}

fn bound(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= MAX_CHARS {
        return trimmed.to_string();
    }
    trimmed.chars().take(MAX_CHARS).collect()
}

fn normalize(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character.is_ascii_punctuation() {
                ' '
            } else {
                character.to_ascii_lowercase()
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_phrase(text: &str, phrase: &str) -> bool {
    format!(" {text} ").contains(&format!(" {phrase} "))
}

fn clear_command(text: &str) -> Option<InstructionEvent> {
    let text = text
        .strip_prefix("please ")
        .unwrap_or(text)
        .strip_suffix(" please")
        .unwrap_or(text);
    const ALL: &[&str] = &[
        "clear all",
        "clear everything",
        "clear all instructions",
        "forget everything",
        "forget all",
        "forget all instructions",
        "reset everything",
        "reset all",
    ];
    const PERSISTENT: &[&str] = &[
        "clear preferences",
        "clear saved",
        "clear saved instructions",
        "forget my preferences",
        "clear always",
        "clear long term",
        "clear long term memory",
    ];
    const SESSION: &[&str] = &[
        "clear",
        "clear it",
        "clear that",
        "clear this",
        "clear instructions",
        "clear the instructions",
        "clear instruction",
        "clear session",
        "clear my instructions",
        "forget that",
        "forget it",
        "forget this",
        "never mind",
        "nevermind",
        "back to normal",
        "write normally",
        "talk normally",
        "cancel that",
        "stop that",
        "stop writing",
        "stop translating",
        "reset",
    ];
    if ALL.contains(&text) {
        Some(InstructionEvent::ClearedAll)
    } else if PERSISTENT.contains(&text) {
        Some(InstructionEvent::ClearedPersistent)
    } else if SESSION.contains(&text) {
        Some(InstructionEvent::ClearedSession)
    } else {
        None
    }
}

fn is_one_shot(text: &str) -> bool {
    const VERBS: &[&str] = &[
        "translate",
        "rewrite",
        "rephrase",
        "summarize",
        "shorten",
        "expand",
        "convert",
        "turn",
        "make",
        "fix",
    ];
    const TARGETS: &[&str] = &[
        "this",
        "that",
        "it",
        "the last",
        "my last",
        "the previous",
        "what i just said",
    ];
    VERBS.iter().any(|verb| contains_phrase(text, verb))
        && TARGETS.iter().any(|target| contains_phrase(text, target))
}

fn wants_persistent(text: &str) -> bool {
    ["always", "from now on", "every time", "permanently", "remember to", "keep doing"]
        .iter()
        .any(|phrase| contains_phrase(text, phrase))
}

fn wants_session(text: &str) -> bool {
    ["for now", "for this session", "until i say", "until i tell you", "let's", "lets"]
        .iter()
        .any(|phrase| contains_phrase(text, phrase))
}

fn wants_append(text: &str) -> bool {
    text.starts_with("also ") || contains_phrase(text, "and also")
}

#[cfg(test)]
static TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
fn isolated(test: impl FnOnce()) {
    let _test = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let directory = std::env::temp_dir().join(format!(
        "said-ai-instructions-{}-{}",
        std::process::id(),
        REVISION.load(Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    *PERSIST_PATH
        .lock()
        .unwrap_or_else(|error| error.into_inner()) =
        Some(directory.join("output-instructions.json"));
    {
        let mut state = lock();
        *state = Store::new();
    }
    test();
    {
        let mut state = lock();
        *state = Store::new();
    }
    *PERSIST_PATH
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = None;
    let _ = fs::remove_dir_all(&directory);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn talk_pastes_parakeet_text_when_nothing_is_active() {
        isolated(|| {
            assert_eq!(finish_talk("  hello there  "), "hello there");
            assert!(active().is_empty());
        });
    }

    #[test]
    fn session_instruction_shows_until_cleared_and_does_not_rewrite_talk_yet() {
        isolated(|| {
            let outcome = accept("for now let's write formally", None);
            assert_eq!(outcome.event, InstructionEvent::SessionUpdated);
            assert!(outcome.paste.is_none());
            assert_eq!(
                active(),
                vec![ActiveInstruction {
                    scope: InstructionScope::Session,
                    text: "for now let's write formally".into(),
                }]
            );
            assert_eq!(
                finish_talk("this is a note"),
                "this is a note",
                "the unconfigured backend must not rewrite Talk"
            );
            assert_eq!(accept("clear", None).event, InstructionEvent::ClearedSession);
            assert!(active().is_empty());
        });
    }

    #[test]
    fn persistent_instruction_survives_a_reload() {
        isolated(|| {
            let outcome = accept("always write formally", None);
            assert_eq!(outcome.event, InstructionEvent::PersistentUpdated);
            let path = persist_path();
            assert!(path.exists());
            {
                let mut state = lock();
                *state = Store::new();
            }
            assert_eq!(
                active(),
                vec![ActiveInstruction {
                    scope: InstructionScope::Persistent,
                    text: "always write formally".into(),
                }]
            );
            assert_eq!(
                accept("clear preferences", None).event,
                InstructionEvent::ClearedPersistent
            );
            assert!(active().is_empty());
            {
                let mut state = lock();
                *state = Store::new();
            }
            assert!(active().is_empty());
        });
    }

    #[test]
    fn a_later_utterance_replaces_the_session_instruction() {
        isolated(|| {
            accept("for now let's write formally", None);
            accept("for now let's write casually", None);
            assert_eq!(active().len(), 1);
            assert_eq!(active()[0].text, "for now let's write casually");
            accept("also keep the greeting", None);
            assert_eq!(active().len(), 2);
        });
    }

    #[test]
    fn translate_this_uses_the_last_dictation_and_waits_without_a_model() {
        isolated(|| {
            finish_talk("good morning");
            let outcome = accept("translate this to Spanish", None);
            assert!(outcome.paste.is_none());
            assert_eq!(outcome.event, InstructionEvent::WaitingForModel);
            assert_eq!(active()[0].scope, InstructionScope::Session);
            assert_eq!(active()[0].text, "translate this to Spanish");
        });
    }

    #[test]
    fn translate_this_prefers_selected_text() {
        isolated(|| {
            finish_talk("ignored");
            let outcome = accept("make this shorter", Some("a selected sentence"));
            assert_eq!(outcome.event, InstructionEvent::WaitingForModel);
            assert!(outcome.paste.is_none());
        });
    }

    #[test]
    fn clear_all_removes_both_memories() {
        isolated(|| {
            accept("for now let's write formally", None);
            accept("always write formally", None);
            assert_eq!(active().len(), 2);
            assert_eq!(accept("clear everything", None).event, InstructionEvent::ClearedAll);
            assert!(active().is_empty());
        });
    }
}
