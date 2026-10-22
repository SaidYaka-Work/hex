//! Spoken instructions that change how Said.ai's text comes out.
//!
//! Talk stores the Parakeet transcript and pastes it unchanged. Talk to AI
//! classifies the utterance with the configured model when `GEMINI_API_KEY`
//! is set, on a thread that does not own dictation output. Phrase lists are
//! the offline fallback. One-shots rewrite the selection or the last Talk
//! transcript and are not stored. Session and persistent instructions are
//! included in that rewrite.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use serde::{Deserialize, Serialize};

use crate::llm_backend::{self, ModelClass, ModelError, RewriteNotice};

const MAX_INSTRUCTIONS: usize = 8;
const MAX_CHARS: usize = 500;

static REVISION: AtomicU64 = AtomicU64::new(0);
static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
static PERSIST_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
static WORKER: OnceLock<Sender<AiJob>> = OnceLock::new();

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
    RequestFailed,
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
                "Heard that. AI is not configured, so the text stays as dictated."
            }
            Self::RequestFailed => {
                "Heard that. Gemini didn't answer, so the text stays as dictated."
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

struct AiJob {
    transcript: String,
    selected: Option<String>,
    cancelled: Arc<dyn Fn() -> bool + Send + Sync>,
    paste: Box<dyn Fn(String) + Send>,
}

fn global_store() -> &'static Mutex<Store> {
    STORE.get_or_init(|| Mutex::new(Store::new()))
}

fn bump() {
    REVISION.fetch_add(1, Ordering::Release);
}

pub fn revision() -> u64 {
    REVISION.load(Ordering::Acquire)
}

fn with_store<T>(f: impl FnOnce(&mut Store) -> T) -> T {
    #[cfg(test)]
    if local_store_active() {
        return with_local_store(f);
    }
    let mut state = global_store()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    ensure_loaded(&mut state);
    f(&mut state)
}

pub fn last_event() -> Option<InstructionEvent> {
    with_store(|state| state.event)
}

pub fn active() -> Vec<ActiveInstruction> {
    with_store(|state| {
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
    })
}

/// Remember a Talk transcript and return it unchanged.
///
/// Talk never calls the language model. A cancelled capture does not become
/// the text a later one-shot rewrites.
#[cfg(test)]
pub fn finish_talk(transcript: &str) -> String {
    finish_talk_unless_cancelled(transcript, false)
}

pub fn finish_talk_unless_cancelled(transcript: &str, cancelled: bool) -> String {
    let text = transcript.trim();
    if cancelled || text.is_empty() {
        return String::new();
    }
    let stored = bound(text);
    with_store(|state| {
        state.last_dictation = Some(stored.clone());
    });
    stored
}

/// Classify and apply a Talk to AI utterance on the caller thread.
///
/// The dictation output threads use [`enqueue`] instead, so a Gemini call
/// cannot hold up Talk.
#[cfg(test)]
pub fn accept(transcript: &str, selected: Option<&str>) -> InstructionOutcome {
    accept_with(transcript, selected, &|| false)
}

/// Hand a Talk to AI transcript to the side thread.
///
/// `paste` runs only when a rewrite returns text and the capture was not
/// cancelled. It may run on the side thread.
pub fn enqueue(
    transcript: String,
    selected: Option<String>,
    cancelled: Arc<dyn Fn() -> bool + Send + Sync>,
    paste: impl Fn(String) + Send + 'static,
) {
    let job = AiJob {
        transcript,
        selected,
        cancelled,
        paste: Box::new(paste),
    };
    if worker_sender().send(job).is_err() {
        tracing::warn!("Talk to AI worker is gone");
    }
}

fn worker_sender() -> Sender<AiJob> {
    WORKER
        .get_or_init(|| {
            let (sender, receiver) = mpsc::channel::<AiJob>();
            thread::Builder::new()
                .name("said-ai".into())
                .spawn(move || {
                    while let Ok(job) = receiver.recv() {
                        if (job.cancelled)() {
                            continue;
                        }
                        let outcome = accept_with(
                            &job.transcript,
                            job.selected.as_deref(),
                            job.cancelled.as_ref(),
                        );
                        if (job.cancelled)() {
                            continue;
                        }
                        if let Some(text) = outcome.paste.filter(|text| !text.trim().is_empty()) {
                            (job.paste)(text);
                        }
                    }
                })
                .expect("Talk to AI worker");
            sender
        })
        .clone()
}

fn accept_with(
    transcript: &str,
    selected: Option<&str>,
    cancelled: &dyn Fn() -> bool,
) -> InstructionOutcome {
    if cancelled() {
        return InstructionOutcome {
            paste: None,
            event: InstructionEvent::NoTextYet,
        };
    }
    let spoken = transcript.trim();
    if spoken.is_empty() {
        return remember(InstructionEvent::NoTextYet, None);
    }
    let choice = match llm_backend::classify_utterance(spoken, cancelled) {
        Ok(class) => choice_from_model(class, spoken),
        Err(ModelError::MissingKey | ModelError::Failed) => choice_from_fallback(spoken),
    };
    if cancelled() {
        return InstructionOutcome {
            paste: None,
            event: InstructionEvent::NoTextYet,
        };
    }
    apply_choice(choice, selected, cancelled)
}

fn choice_from_model(class: ModelClass, spoken: &str) -> Choice {
    match class {
        ModelClass::Session(text) => Choice::Remember {
            persistent: false,
            text: filled(text, spoken),
            append: false,
        },
        ModelClass::Persistent(text) => Choice::Remember {
            persistent: true,
            text: filled(text, spoken),
            append: false,
        },
        ModelClass::ClearSession => Choice::Clear(InstructionEvent::ClearedSession),
        ModelClass::ClearPersistent => Choice::Clear(InstructionEvent::ClearedPersistent),
        ModelClass::ClearAll => Choice::Clear(InstructionEvent::ClearedAll),
        ModelClass::ApplyOnce(text) => Choice::ApplyOnce(filled(text, spoken)),
        ModelClass::Ignore => Choice::Ignore,
    }
}

fn choice_from_fallback(spoken: &str) -> Choice {
    let normalized = normalize(spoken);
    if let Some(event) = clear_command(&normalized) {
        return Choice::Clear(event);
    }
    let phrase = bound(spoken);
    if is_one_shot(&normalized) {
        return Choice::ApplyOnce(phrase);
    }
    let persistent = wants_persistent(&normalized) && !wants_session(&normalized);
    Choice::Remember {
        persistent,
        text: phrase,
        append: wants_append(&normalized),
    }
}

fn filled(text: String, spoken: &str) -> String {
    if text.trim().is_empty() {
        bound(spoken)
    } else {
        bound(&text)
    }
}

enum Choice {
    Clear(InstructionEvent),
    Remember {
        persistent: bool,
        text: String,
        append: bool,
    },
    ApplyOnce(String),
    Ignore,
}

fn apply_choice(
    choice: Choice,
    selected: Option<&str>,
    cancelled: &dyn Fn() -> bool,
) -> InstructionOutcome {
    match choice {
        Choice::Ignore => remember(InstructionEvent::NoTextYet, None),
        Choice::Clear(event) => with_store(|state| {
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
                _ => {}
            }
            state.event = Some(event);
            bump();
            InstructionOutcome { paste: None, event }
        }),
        Choice::Remember {
            persistent,
            text,
            append,
        } => with_store(|state| {
            if persistent {
                store_phrase(&mut state.persistent, text, append);
                save_persistent(&state.persistent);
            } else {
                store_phrase(&mut state.session, text, append);
            }
            let event = if persistent {
                InstructionEvent::PersistentUpdated
            } else {
                InstructionEvent::SessionUpdated
            };
            state.event = Some(event);
            bump();
            InstructionOutcome { paste: None, event }
        }),
        Choice::ApplyOnce(phrase) => apply_once(phrase, selected, cancelled),
    }
}

fn apply_once(
    phrase: String,
    selected: Option<&str>,
    cancelled: &dyn Fn() -> bool,
) -> InstructionOutcome {
    let source = with_store(|state| {
        selected
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| state.last_dictation.clone())
    });
    let Some(source) = source else {
        return remember(InstructionEvent::NoTextYet, None);
    };
    let mut instructions = with_store(|state| standing(state));
    if !instructions.iter().any(|item| item == &phrase) {
        instructions.push(phrase);
    }
    if cancelled() {
        return InstructionOutcome {
            paste: None,
            event: InstructionEvent::NoTextYet,
        };
    }
    let rewrite = llm_backend::rewrite(&source, &instructions, cancelled);
    if cancelled() || !rewrite.called_model {
        let event = match rewrite.notice {
            RewriteNotice::MissingKey | RewriteNotice::None => InstructionEvent::WaitingForModel,
            RewriteNotice::Failed => InstructionEvent::RequestFailed,
        };
        return remember(event, None);
    }
    remember(InstructionEvent::Applied, Some(rewrite.text))
}

fn remember(event: InstructionEvent, paste: Option<String>) -> InstructionOutcome {
    with_store(|state| {
        state.event = Some(event);
        bump();
    });
    InstructionOutcome { paste, event }
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
    #[cfg(test)]
    if let Some(path) = LOCAL_PATH.with(|cell| cell.borrow().clone()) {
        return path;
    }
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
        .map(|file| {
            file.instructions
                .into_iter()
                .map(|item| bound(&item))
                .collect()
        })
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
    if fs::write(&temporary, bytes).is_ok() && fs::rename(&temporary, path).is_err() {
        let _ = fs::write(path, bytes);
        let _ = fs::remove_file(&temporary);
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
        .filter(|character| *character != '\'' && *character != '\u{2019}')
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
    let text = text.strip_prefix("please ").unwrap_or(text);
    let text = text.strip_suffix(" please").unwrap_or(text);
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
    const OPERATIONS: &[&str] = &[
        "translate",
        "rewrite",
        "rephrase",
        "summarize",
        "shorten",
        "expand",
        "convert",
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
    if OPERATIONS.iter().any(|verb| contains_phrase(text, verb))
        && TARGETS.iter().any(|target| contains_phrase(text, target))
    {
        return true;
    }
    // "make it sound friendly" is a style. "make this shorter" edits the
    // current text once.
    if ["make", "fix", "turn"]
        .iter()
        .any(|verb| contains_phrase(text, verb))
        && TARGETS.iter().any(|target| contains_phrase(text, target))
        && !contains_phrase(text, "sound")
        && !contains_phrase(text, "like")
    {
        return true;
    }
    false
}

fn wants_persistent(text: &str) -> bool {
    [
        "always",
        "from now on",
        "every time",
        "permanently",
        "remember to",
        "keep doing",
    ]
    .iter()
    .any(|phrase| contains_phrase(text, phrase))
}

fn wants_session(text: &str) -> bool {
    [
        "for now",
        "for this session",
        "until i say",
        "until i tell you",
        "lets",
    ]
    .iter()
    .any(|phrase| contains_phrase(text, phrase))
}

fn wants_append(text: &str) -> bool {
    text.starts_with("also ") || contains_phrase(text, "and also")
}

#[cfg(test)]
thread_local! {
    static LOCAL_STORE: std::cell::RefCell<Option<Store>> = const { std::cell::RefCell::new(None) };
    static LOCAL_PATH: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn local_store_active() -> bool {
    LOCAL_STORE.with(|cell| cell.borrow().is_some())
}

#[cfg(test)]
fn with_local_store<T>(f: impl FnOnce(&mut Store) -> T) -> T {
    LOCAL_STORE.with(|cell| {
        let mut slot = cell.borrow_mut();
        let store = slot.as_mut().expect("local instruction store");
        ensure_loaded(store);
        f(store)
    })
}

#[cfg(test)]
fn isolated(test: impl FnOnce()) {
    let _backend = crate::llm_backend::begin_missing_key();
    isolated_inner(test);
}

#[cfg(test)]
fn isolated_with(_backend: crate::llm_backend::TestGuard, test: impl FnOnce()) {
    isolated_inner(test);
}

#[cfg(test)]
fn isolated_inner(test: impl FnOnce()) {
    let directory = std::env::temp_dir().join(format!(
        "said-ai-instructions-{}-{}",
        std::process::id(),
        REVISION.load(Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    LOCAL_PATH.with(|cell| *cell.borrow_mut() = Some(directory.join("output-instructions.json")));
    LOCAL_STORE.with(|cell| *cell.borrow_mut() = Some(Store::new()));
    test();
    LOCAL_STORE.with(|cell| *cell.borrow_mut() = None);
    LOCAL_PATH.with(|cell| *cell.borrow_mut() = None);
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
    fn a_cancelled_talk_does_not_replace_the_last_transcript() {
        isolated(|| {
            finish_talk("keep this");
            assert_eq!(finish_talk_unless_cancelled("dropped", true), "");
            assert_eq!(
                with_store(|state| state.last_dictation.clone()).as_deref(),
                Some("keep this")
            );
        });
    }

    #[test]
    fn session_instruction_shows_until_cleared_and_does_not_rewrite_talk() {
        isolated(|| {
            let outcome = accept("for now let's write formally", None);
            assert_eq!(outcome.event, InstructionEvent::SessionUpdated);
            assert!(outcome.paste.is_none());
            assert_eq!(active()[0].text, "for now let's write formally");
            assert_eq!(finish_talk("this is a note"), "this is a note");
            assert_eq!(
                accept("clear", None).event,
                InstructionEvent::ClearedSession
            );
            assert!(active().is_empty());
        });
    }

    #[test]
    fn please_clear_clears_instead_of_being_saved() {
        isolated(|| {
            accept("for now let's write formally", None);
            assert_eq!(
                accept("please clear", None).event,
                InstructionEvent::ClearedSession
            );
            assert!(active().is_empty());
        });
    }

    #[test]
    fn lets_with_an_apostrophe_is_a_session_instruction() {
        isolated(|| {
            let outcome = accept("let's write formally", None);
            assert_eq!(outcome.event, InstructionEvent::SessionUpdated);
            assert_eq!(active()[0].scope, InstructionScope::Session);
        });
    }

    #[test]
    fn make_it_sound_friendly_is_not_a_one_shot() {
        isolated(|| {
            finish_talk("good morning");
            let outcome = accept("make it sound friendly", None);
            assert_eq!(outcome.event, InstructionEvent::SessionUpdated);
            assert!(outcome.paste.is_none());
            assert_eq!(active()[0].text, "make it sound friendly");
        });
    }

    #[test]
    fn persistent_instruction_survives_a_reload() {
        isolated(|| {
            let outcome = accept("always write formally", None);
            assert_eq!(outcome.event, InstructionEvent::PersistentUpdated);
            let path = persist_path();
            assert!(path.exists());
            LOCAL_STORE.with(|cell| *cell.borrow_mut() = Some(Store::new()));
            assert_eq!(active()[0].text, "always write formally");
            assert_eq!(
                accept("clear preferences", None).event,
                InstructionEvent::ClearedPersistent
            );
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
    fn translate_this_is_not_stored_when_the_model_is_missing() {
        isolated(|| {
            finish_talk("good morning");
            let outcome = accept("translate this to Spanish", None);
            assert!(outcome.paste.is_none());
            assert_eq!(outcome.event, InstructionEvent::WaitingForModel);
            assert!(active().is_empty());
        });
    }

    #[test]
    fn translate_this_prefers_selected_text_and_is_not_stored() {
        isolated(|| {
            finish_talk("ignored");
            let outcome = accept("make this shorter", Some("a selected sentence"));
            assert_eq!(outcome.event, InstructionEvent::WaitingForModel);
            assert!(outcome.paste.is_none());
            assert!(active().is_empty());
        });
    }

    #[test]
    fn clear_all_removes_both_memories() {
        isolated(|| {
            accept("for now let's write formally", None);
            accept("always write formally", None);
            assert_eq!(active().len(), 2);
            assert_eq!(
                accept("clear everything", None).event,
                InstructionEvent::ClearedAll
            );
            assert!(active().is_empty());
        });
    }

    fn spanish_and_style(
        request: &crate::llm_backend::CapturedRequest,
    ) -> Result<crate::llm_backend::MockReply, String> {
        let body = if request.prompt.contains("Classify this spoken") {
            let utterance = request
                .prompt
                .rsplit_once("Utterance:\n")
                .map(|(_, utterance)| utterance)
                .unwrap_or(request.prompt.as_str());
            let (action, instruction) = if utterance.contains("always write formally") {
                ("persistent", "always write formally")
            } else if utterance.contains("for now") {
                ("session", "for now let's write formally")
            } else {
                ("apply_once", "translate this to Spanish")
            };
            let text = serde_json::json!({
                "action": action,
                "instruction": instruction,
            })
            .to_string();
            serde_json::json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": text }]
                    }
                }]
            })
            .to_string()
        } else {
            assert!(request.prompt.contains("always write formally"));
            assert!(request.prompt.contains("for now let's write formally"));
            assert!(request.prompt.contains("translate this to Spanish"));
            assert!(request.prompt.contains("good morning"));
            r#"{"candidates":[{"content":{"parts":[{"text":"Buenos días."}]}}]}"#.to_string()
        };
        Ok(crate::llm_backend::MockReply { status: 200, body })
    }

    #[test]
    fn a_one_shot_rewrites_with_standing_instructions_and_does_not_touch_talk() {
        isolated_with(
            crate::llm_backend::begin_mock("test-key", spanish_and_style),
            || {
                accept("for now let's write formally", None);
                accept("always write formally", None);
                finish_talk("good morning");
                let outcome = accept("translate this to Spanish", None);
                assert_eq!(outcome.paste.as_deref(), Some("Buenos días."));
                assert_eq!(outcome.event, InstructionEvent::Applied);
                assert_eq!(active().len(), 2);
                assert_eq!(finish_talk("next note"), "next note");
            },
        );
    }
}
