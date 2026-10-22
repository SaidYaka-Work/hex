//! Spoken instructions that change how Said.ai's text comes out.
//!
//! Talk stores the Parakeet transcript and pastes it unchanged. Talk to AI
//! asks the configured model, on a thread that does not own dictation output,
//! for one JSON object that both classifies the utterance and rewrites text.
//! Phrase lists are the offline fallback. One-shots and ordinary speech are
//! not stored. Session and persistent instructions are included when a source
//! is rewritten.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use serde::{Deserialize, Serialize};

use crate::llm_backend::{self, ModelAction, ModelError, ModelTurn};

const MAX_INSTRUCTIONS: usize = 8;
const MAX_CHARS: usize = 500;
/// Selected text included in one model call. Longer selections are cut here.
const SOURCE_CAP: usize = 4_000;

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
                "Heard that. AI is unavailable right now, so the text stays as dictated."
            }
            Self::RequestFailed => {
                "Heard that. AI is unavailable right now, so the text stays as dictated."
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
    #[serde(default)]
    selection_notice_shown: bool,
}

struct Store {
    session: Vec<String>,
    persistent: Vec<String>,
    last_dictation: Option<String>,
    event: Option<InstructionEvent>,
    loaded: bool,
    selection_notice_shown: bool,
    selection_notice_pending: bool,
}

impl Store {
    fn new() -> Self {
        Self {
            session: Vec::new(),
            persistent: Vec::new(),
            last_dictation: None,
            event: None,
            loaded: false,
            selection_notice_shown: false,
            selection_notice_pending: false,
        }
    }
}

/// Focus observed around a Talk to AI call.
///
/// `Unreadable` is Wayland, or a platform read that failed. It does not block
/// a paste. A confirmed app at the start, then a different app or no app,
/// does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FocusRead {
    /// Wayland, or a focus read that failed. Windows always reports `None` or `App`.
    #[cfg_attr(target_os = "windows", allow(dead_code))]
    Unreadable,
    None,
    App(String),
}

pub fn focus_allows_paste(start: &FocusRead, now: &FocusRead) -> bool {
    let FocusRead::App(start) = start else {
        return true;
    };
    match now {
        FocusRead::Unreadable => true,
        FocusRead::None => false,
        FocusRead::App(now) => now == start,
    }
}

struct AiJob {
    transcript: String,
    selected: Option<String>,
    focus: FocusRead,
    focus_now: Arc<dyn Fn() -> FocusRead + Send + Sync>,
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

/// Drop one chip. Persistent removals are written; session removals are not.
pub fn dismiss(scope: InstructionScope, text: &str) {
    with_store(|state| {
        let changed = match scope {
            InstructionScope::Session => {
                let before = state.session.len();
                state.session.retain(|item| item != text);
                state.session.len() != before
            }
            InstructionScope::Persistent => {
                let before = state.persistent.len();
                state.persistent.retain(|item| item != text);
                let changed = state.persistent.len() != before;
                if changed {
                    save_state(state);
                }
                changed
            }
        };
        if changed {
            state.event = None;
            bump();
        }
    });
}

/// The one-time sentence shown the first time selected text is sent.
pub fn take_selection_notice() -> Option<&'static str> {
    with_store(|state| {
        if !state.selection_notice_pending {
            return None;
        }
        state.selection_notice_pending = false;
        Some("Selected text is sent to our AI service.")
    })
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
/// The dictation output threads use [`enqueue`] instead, so a model call
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
    focus: FocusRead,
    focus_now: Arc<dyn Fn() -> FocusRead + Send + Sync>,
    cancelled: Arc<dyn Fn() -> bool + Send + Sync>,
    paste: impl Fn(String) + Send + 'static,
) {
    let job = AiJob {
        transcript,
        selected,
        focus,
        focus_now,
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
                        if !focus_allows_paste(&job.focus, &(job.focus_now)()) {
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

const SYSTEM_PROMPT: &str = "\
You edit dictation for a voice app. Reply with one JSON object and no other text. \
Fields: action, instruction, append, text. \
action is one of: session, persistent, clear_session, clear_persistent, clear_all, apply_once, rewrite, ignore. \
session: a temporary style, such as \"for now write formally\". \
persistent: a style to keep, such as \"always write formally\". Set append true so other saved preferences stay. \
also sets append true. \
apply_once: edit Source once, such as \"translate this to Spanish\". Do not store it. \
rewrite: ordinary speech. Rewrite the utterance with the active instructions into text. \
clear_session, clear_persistent, and clear_all forget instructions. \
ignore: nothing to paste. \
instruction is the style, or empty. \
text is the paste-ready rewrite of Source when Source is present, or of the utterance when action is rewrite. \
Leave text empty when nothing should be pasted. Never put the instruction itself in text.";

fn accept_with(
    transcript: &str,
    selected: Option<&str>,
    cancelled: &dyn Fn() -> bool,
) -> InstructionOutcome {
    if cancelled() {
        return quiet();
    }
    let spoken = transcript.trim();
    if spoken.is_empty() {
        return remember(InstructionEvent::NoTextYet, None);
    }
    let selected_text = selected
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(clip_source);
    let source = selected_text.clone().or_else(|| {
        with_store(|state| state.last_dictation.clone()).map(|text| clip_source(&text))
    });
    let instructions = with_store(|state| standing(state));
    let user = user_prompt(&instructions, source.as_deref(), spoken);
    let decision = match llm_backend::complete(SYSTEM_PROMPT, &user, cancelled) {
        Ok(turn) => {
            if selected_text.is_some() {
                mark_selection_notice();
            }
            decision_from_turn(turn, spoken, source.is_some())
        }
        Err(ModelError::Cancelled) => return quiet(),
        Err(ModelError::MissingKey) => {
            decision_from_fallback(spoken, InstructionEvent::WaitingForModel)
        }
        Err(ModelError::Failed) => {
            if selected_text.is_some() {
                mark_selection_notice();
            }
            decision_from_fallback(spoken, InstructionEvent::RequestFailed)
        }
    };
    if cancelled() {
        return quiet();
    }
    apply_decision(decision)
}

fn quiet() -> InstructionOutcome {
    InstructionOutcome {
        paste: None,
        event: InstructionEvent::NoTextYet,
    }
}

fn user_prompt(instructions: &[String], source: Option<&str>, spoken: &str) -> String {
    let mut prompt = String::from("Active instructions:\n");
    if instructions.is_empty() {
        prompt.push_str("(none)\n");
    } else {
        for instruction in instructions {
            prompt.push_str("- ");
            prompt.push_str(instruction);
            prompt.push('\n');
        }
    }
    prompt.push_str("\nSource:\n");
    prompt.push_str(source.unwrap_or("(none)"));
    prompt.push_str("\n\nUtterance:\n");
    prompt.push_str(spoken);
    prompt
}

fn decision_from_turn(turn: ModelTurn, spoken: &str, has_source: bool) -> Decision {
    let paste_text = nonempty(&turn.text);
    match turn.action {
        ModelAction::Session => Decision {
            store: Some(StoreOp {
                persistent: false,
                text: filled(turn.instruction, spoken),
                append: turn.append,
            }),
            paste: has_source.then_some(paste_text).flatten(),
            event: InstructionEvent::SessionUpdated,
        },
        ModelAction::Persistent => Decision {
            store: Some(StoreOp {
                persistent: true,
                text: filled(turn.instruction, spoken),
                // A persistent phrase adds. It does not wipe other saved preferences.
                append: true,
            }),
            paste: has_source.then_some(paste_text).flatten(),
            event: InstructionEvent::PersistentUpdated,
        },
        ModelAction::ClearSession => Decision::clear(InstructionEvent::ClearedSession),
        ModelAction::ClearPersistent => Decision::clear(InstructionEvent::ClearedPersistent),
        ModelAction::ClearAll => Decision::clear(InstructionEvent::ClearedAll),
        ModelAction::ApplyOnce => {
            let applied = paste_text.is_some();
            Decision {
                store: None,
                paste: paste_text,
                event: if applied {
                    InstructionEvent::Applied
                } else {
                    InstructionEvent::NoTextYet
                },
            }
        }
        ModelAction::Rewrite => {
            let failed = paste_text.is_none();
            Decision {
                store: None,
                paste: Some(paste_text.unwrap_or_else(|| bound(spoken))),
                event: if failed {
                    InstructionEvent::RequestFailed
                } else {
                    InstructionEvent::Applied
                },
            }
        }
        ModelAction::Ignore => Decision {
            store: None,
            paste: None,
            event: InstructionEvent::NoTextYet,
        },
    }
}

fn decision_from_fallback(spoken: &str, notice: InstructionEvent) -> Decision {
    let normalized = normalize(spoken);
    if let Some(event) = clear_command(&normalized) {
        return Decision::clear(event);
    }
    let phrase = bound(spoken);
    if is_one_shot(&normalized) {
        return Decision {
            store: None,
            paste: None,
            event: notice,
        };
    }
    let persistent = wants_persistent(&normalized) && !wants_session(&normalized);
    let append = wants_append(&normalized);
    let style =
        persistent || wants_session(&normalized) || contains_phrase(&normalized, "sound") || append;
    if !style {
        return Decision {
            store: None,
            paste: Some(phrase),
            event: notice,
        };
    }
    Decision {
        store: Some(StoreOp {
            persistent,
            text: phrase,
            append: persistent || append,
        }),
        paste: None,
        event: if persistent {
            InstructionEvent::PersistentUpdated
        } else {
            InstructionEvent::SessionUpdated
        },
    }
}

fn filled(text: String, spoken: &str) -> String {
    if text.trim().is_empty() {
        bound(spoken)
    } else {
        bound(&text)
    }
}

fn mark_selection_notice() {
    with_store(|state| {
        if state.selection_notice_shown {
            return;
        }
        state.selection_notice_shown = true;
        state.selection_notice_pending = true;
        save_state(state);
        bump();
    });
}

fn clip_source(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= SOURCE_CAP {
        trimmed.to_string()
    } else {
        trimmed.chars().take(SOURCE_CAP).collect()
    }
}

fn nonempty(text: &str) -> Option<String> {
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

struct StoreOp {
    persistent: bool,
    text: String,
    append: bool,
}

struct Decision {
    store: Option<StoreOp>,
    paste: Option<String>,
    event: InstructionEvent,
}

impl Decision {
    fn clear(event: InstructionEvent) -> Self {
        Self {
            store: None,
            paste: None,
            event,
        }
    }
}

fn apply_decision(decision: Decision) -> InstructionOutcome {
    let Decision {
        store,
        paste,
        event,
    } = decision;
    with_store(|state| {
        match event {
            InstructionEvent::ClearedAll => {
                state.session.clear();
                state.persistent.clear();
                save_state(state);
            }
            InstructionEvent::ClearedPersistent => {
                state.persistent.clear();
                save_state(state);
            }
            InstructionEvent::ClearedSession => state.session.clear(),
            _ => {}
        }
        if let Some(op) = store {
            if op.persistent {
                store_phrase(&mut state.persistent, op.text, op.append);
                save_state(state);
            } else {
                store_phrase(&mut state.session, op.text, op.append);
            }
        }
        state.event = Some(event);
        bump();
        InstructionOutcome { paste, event }
    })
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
    let file = load_file();
    state.selection_notice_shown = file.selection_notice_shown;
    state.persistent = file
        .instructions
        .into_iter()
        .map(|item| bound(&item))
        .collect();
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

fn load_file() -> PersistentFile {
    let path = persist_path();
    let Ok(bytes) = fs::read(&path) else {
        return PersistentFile::default();
    };
    serde_json::from_slice::<PersistentFile>(&bytes).unwrap_or_default()
}

fn save_state(state: &Store) {
    let path = persist_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = PersistentFile {
        instructions: state.persistent.clone(),
        selection_notice_shown: state.selection_notice_shown,
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
        let utterance = request
            .prompt
            .rsplit_once("Utterance:\n")
            .map(|(_, utterance)| utterance)
            .unwrap_or(request.prompt.as_str());
        let (action, instruction, append, text) = if utterance.contains("always write formally") {
            ("persistent", "always write formally", true, "")
        } else if utterance.contains("for now") {
            ("session", "for now let's write formally", false, "")
        } else if utterance.contains("hello team") {
            ("rewrite", "", false, "Hello, team.")
        } else {
            assert!(request.prompt.contains("always write formally"));
            assert!(request.prompt.contains("for now let's write formally"));
            assert!(request.prompt.contains("good morning"));
            (
                "apply_once",
                "translate this to Spanish",
                false,
                "Buenos días.",
            )
        };
        let content = serde_json::json!({
            "action": action,
            "instruction": instruction,
            "append": append,
            "text": text,
        })
        .to_string();
        Ok(crate::llm_backend::MockReply {
            status: 200,
            body: crate::llm_backend::chat_json(&content),
        })
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

    #[test]
    fn ordinary_speech_is_pasted_and_not_stored_when_the_model_is_missing() {
        isolated(|| {
            let outcome = accept("hello team", None);
            assert_eq!(outcome.paste.as_deref(), Some("hello team"));
            assert_eq!(outcome.event, InstructionEvent::WaitingForModel);
            assert!(active().is_empty());
        });
    }

    #[test]
    fn ordinary_speech_is_rewritten_and_not_stored() {
        isolated_with(
            crate::llm_backend::begin_mock("test-key", spanish_and_style),
            || {
                accept("for now let's write formally", None);
                let outcome = accept("hello team", None);
                assert_eq!(outcome.paste.as_deref(), Some("Hello, team."));
                assert_eq!(outcome.event, InstructionEvent::Applied);
                assert_eq!(active().len(), 1);
            },
        );
    }

    #[test]
    fn always_adds_a_saved_preference_instead_of_replacing_it() {
        isolated(|| {
            accept("always write formally", None);
            accept("always be brief", None);
            let saved: Vec<_> = active()
                .into_iter()
                .filter(|item| item.scope == InstructionScope::Persistent)
                .map(|item| item.text)
                .collect();
            assert_eq!(
                saved,
                vec![
                    "always write formally".to_string(),
                    "always be brief".to_string()
                ]
            );
        });
    }

    #[test]
    fn a_model_persistent_instruction_still_appends_when_append_is_false() {
        isolated_with(
            crate::llm_backend::begin_mock("test-key", |request| {
                let utterance = request
                    .prompt
                    .rsplit_once("Utterance:\n")
                    .map(|(_, utterance)| utterance)
                    .unwrap_or("");
                let instruction = if utterance.contains("brief") {
                    "be brief"
                } else {
                    "write formally"
                };
                let content = serde_json::json!({
                    "action": "persistent",
                    "instruction": instruction,
                    "append": false,
                    "text": "",
                })
                .to_string();
                Ok(crate::llm_backend::MockReply {
                    status: 200,
                    body: crate::llm_backend::chat_json(&content),
                })
            }),
            || {
                accept("always write formally", None);
                accept("always be brief", None);
                assert_eq!(active().len(), 2);
            },
        );
    }

    #[test]
    fn selected_text_is_capped_and_announced_once() {
        isolated_with(
            crate::llm_backend::begin_mock("test-key", |request| {
                let source = request
                    .prompt
                    .split_once("Source:\n")
                    .and_then(|(_, rest)| rest.split_once("\n\nUtterance:\n"))
                    .map(|(source, _)| source)
                    .unwrap_or("");
                if source.chars().count() > 100 {
                    assert!(source.chars().count() <= 4_000);
                    assert!(source.chars().count() > 3_000);
                }
                let content = serde_json::json!({
                    "action": "apply_once",
                    "instruction": "shorten",
                    "append": false,
                    "text": "short",
                })
                .to_string();
                Ok(crate::llm_backend::MockReply {
                    status: 200,
                    body: crate::llm_backend::chat_json(&content),
                })
            }),
            || {
                let long = "word ".repeat(2_000);
                let outcome = accept("make this shorter", Some(&long));
                assert_eq!(outcome.paste.as_deref(), Some("short"));
                assert_eq!(
                    take_selection_notice(),
                    Some("Selected text is sent to our AI service.")
                );
                assert!(take_selection_notice().is_none());
                accept("make this shorter", Some("again"));
                assert!(take_selection_notice().is_none());
            },
        );
    }

    #[test]
    fn a_chip_can_be_dismissed() {
        isolated(|| {
            accept("for now let's write formally", None);
            accept("also keep the greeting", None);
            dismiss(InstructionScope::Session, "also keep the greeting");
            assert_eq!(active().len(), 1);
            assert_eq!(active()[0].text, "for now let's write formally");
        });
    }

    #[test]
    fn focus_change_blocks_paste_and_wayland_does_not() {
        assert!(focus_allows_paste(
            &FocusRead::Unreadable,
            &FocusRead::App("other".into())
        ));
        assert!(focus_allows_paste(
            &FocusRead::App("notes".into()),
            &FocusRead::App("notes".into())
        ));
        assert!(!focus_allows_paste(
            &FocusRead::App("notes".into()),
            &FocusRead::App("browser".into())
        ));
        assert!(!focus_allows_paste(
            &FocusRead::App("notes".into()),
            &FocusRead::None
        ));
        assert!(focus_allows_paste(
            &FocusRead::App("notes".into()),
            &FocusRead::Unreadable
        ));
    }
}
