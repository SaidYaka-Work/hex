//! The one language-model slot Said.ai calls.
//!
//! Switching provider or model is a string change, not a code change. The
//! baked default talks to Google's OpenAI-compatible endpoint with
//! `gemini-3.5-flash-lite` and reads the key from `GEMINI_API_KEY`. The same
//! client reaches OpenRouter by pointing the slot at
//! `https://openrouter.ai/api/v1` and a model id such as
//! `google/gemini-3.5-flash-lite`.
//!
//! Resolution order, outside tests:
//! 1. Baked [`LLM_SLOT`].
//! 2. When `SAID_AI_CONFIG_URL` is an https URL, a small JSON document is
//!    fetched once on a background thread and cached as `ai-slot.json`.
//!    Remote config can change `base_url` and `model` only. It cannot change
//!    which environment variable holds the key. The host must be on
//!    [`ALLOWED_HOSTS`], and the document must carry an ed25519 signature
//!    under [`SLOT_PUBLIC_KEY`]. An unsigned, invalid, or off-list document
//!    is ignored. The cache is checked with the same rules.
//! 3. `SAID_AI_BASE_URL` and `SAID_AI_MODEL` override that result. The base
//!    URL still has to be an allowlisted host. `SAID_AI_KEY_VARIABLE` is a
//!    local override of the key's variable name and is never taken from the
//!    network.
//!
//! The key itself stays behind [`api_key`]. Reading it does not load remote
//! config, so the home can ask whether a key is present without blocking.
//! A small proxy at [`SAID_PROXY_HOST`] would keep both the key and the model
//! id out of the binary. That proxy is not running yet; the host is only an
//! allowlist placeholder. The private key that matches [`SLOT_PUBLIC_KEY`] is
//! not in this repo, so a remote document stays ignored until the owner
//! replaces the public key and signs the config. Tests never read a real key
//! and never hit the network.

use std::io::{Read, Write};
#[cfg(not(target_os = "windows"))]
use std::path::Path;
use std::path::PathBuf;
use std::process::{Command, Stdio};
#[cfg(test)]
use std::sync::Mutex;
#[cfg(not(test))]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine;
use ed25519_dalek::{Signature, VerifyingKey};
use serde::Deserialize;

/// Baked Talk to AI slot. Strings only; see the module docs for overrides.
pub struct LlmSlot {
    /// OpenAI-compatible origin, without `/chat/completions`.
    pub base_url: &'static str,
    pub model: &'static str,
    /// Environment variable that holds the bearer token.
    pub key_variable: &'static str,
}

pub const LLM_SLOT: LlmSlot = LlmSlot {
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
    model: "gemini-3.5-flash-lite",
    key_variable: "GEMINI_API_KEY",
};

/// Future owner-controlled proxy. Remote config may name this host. Nothing
/// is deployed here yet; a proxy would hold the key and the model id.
pub const SAID_PROXY_HOST: &str = "ai.saidyaka.com";

/// Hosts a slot base URL may use. Matched exactly, with no port and no userinfo.
const ALLOWED_HOSTS: &[&str] = &[
    "generativelanguage.googleapis.com",
    "openrouter.ai",
    SAID_PROXY_HOST,
];

/// Ed25519 public key for remote slot documents. The matching private key is
/// not in the repository. Unsigned or invalid documents are ignored.
const SLOT_PUBLIC_KEY: [u8; 32] = [
    0xee, 0x8d, 0xf2, 0x24, 0xe7, 0x5a, 0xe0, 0xb4, 0x34, 0xc8, 0x29, 0x5a, 0x8f, 0x44, 0x4f, 0xa8,
    0xf9, 0x21, 0x96, 0xc6, 0x16, 0xa5, 0x29, 0x16, 0x60, 0x57, 0x53, 0x18, 0xee, 0x3c, 0x7c, 0x25,
];

const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
#[cfg(not(test))]
const CONFIG_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const TEMPERATURE: f32 = 0.2;
const MIN_OUTPUT_TOKENS: u32 = 1500;
const MAX_OUTPUT_TOKENS: u32 = 8192;

static THINKING: AtomicBool = AtomicBool::new(false);

/// What one chat-completions call decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelTurn {
    pub action: ModelAction,
    pub instruction: String,
    pub append: bool,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelAction {
    Session,
    Persistent,
    ClearSession,
    ClearPersistent,
    ClearAll,
    ApplyOnce,
    Rewrite,
    Ignore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    MissingKey,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedSlot {
    base_url: String,
    model: String,
}

#[derive(Clone, Debug, Deserialize)]
struct SlotFile {
    base_url: String,
    model: String,
    /// Present on older caches. Remote config must not choose the key variable.
    #[serde(default)]
    key_variable: Option<String>,
    #[serde(default)]
    signature: String,
}

/// `true` while a Talk to AI request is in flight in this process.
///
/// Other processes (the Linux settings window) also see a fresh marker file
/// written beside the settings, so the home can shimmer without sharing memory.
#[cfg(test)]
pub fn is_thinking() -> bool {
    THINKING.load(Ordering::Acquire)
}

#[cfg(not(test))]
pub fn is_thinking() -> bool {
    THINKING.load(Ordering::Acquire) || thinking_file_fresh()
}

/// Fetch remote slot JSON in the background, once. Tests do not.
pub fn warm_slot() {
    #[cfg(not(test))]
    {
        static STARTED: OnceLock<()> = OnceLock::new();
        STARTED.get_or_init(|| {
            let _ = thread::Builder::new()
                .name("said-ai-slot".into())
                .spawn(|| {
                    let _ = resolved_slot();
                });
        });
    }
}

/// `true` when the key variable holds a usable value.
///
/// This reads the environment only. It does not fetch or wait for remote config.
pub fn api_key_present() -> bool {
    api_key().is_some()
}

/// One JSON chat completion. A missing key, a failed call, or cancellation
/// makes no further request.
pub fn complete(
    system: &str,
    user: &str,
    cancelled: &dyn Fn() -> bool,
) -> Result<ModelTurn, ModelError> {
    if cancelled() {
        return Err(ModelError::Cancelled);
    }
    let Some(key) = api_key() else {
        return Err(ModelError::MissingKey);
    };
    if cancelled() {
        return Err(ModelError::Cancelled);
    }
    let prepared = prepare_request(&key, system, user, true);
    set_thinking(true);
    let first = dispatch(&prepared, cancelled);
    let result = match first {
        Ok(response) if response.status == 400 || response.status == 422 => {
            let plain = prepare_request(&key, system, user, false);
            dispatch(&plain, cancelled)
        }
        other => other,
    };
    set_thinking(false);
    if cancelled() {
        return Err(ModelError::Cancelled);
    }
    match result {
        Ok(response) if response.status == 200 => {
            parse_turn(&response.body).map_err(|_| ModelError::Failed)
        }
        Ok(_) => Err(ModelError::Failed),
        Err(DispatchError::Cancelled) => Err(ModelError::Cancelled),
        Err(DispatchError::Failed) => Err(ModelError::Failed),
    }
}

fn set_thinking(on: bool) {
    THINKING.store(on, Ordering::Release);
    #[cfg(not(test))]
    {
        let path = thinking_path();
        if on {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, b"1");
        } else {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(not(test))]
fn thinking_path() -> PathBuf {
    support_dir().join("ai-thinking")
}

#[cfg(not(test))]
fn thinking_file_fresh() -> bool {
    let Ok(meta) = std::fs::metadata(thinking_path()) else {
        return false;
    };
    meta.modified()
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age < REQUEST_TIMEOUT + CONFIG_TIMEOUT)
}

fn baked_slot() -> ResolvedSlot {
    ResolvedSlot {
        base_url: LLM_SLOT.base_url.to_string(),
        model: LLM_SLOT.model.to_string(),
    }
}

fn resolved_slot() -> ResolvedSlot {
    #[cfg(test)]
    {
        baked_slot()
    }
    #[cfg(not(test))]
    {
        static SLOT: OnceLock<ResolvedSlot> = OnceLock::new();
        SLOT.get_or_init(load_slot).clone()
    }
}

#[cfg(not(test))]
fn load_slot() -> ResolvedSlot {
    let mut slot = fetch_remote_slot().unwrap_or_else(baked_slot);
    if let Some(base_url) = env_https("SAID_AI_BASE_URL")
        && base_url_allowed(&base_url)
    {
        slot.base_url = base_url;
    }
    if let Some(model) = env_token("SAID_AI_MODEL") {
        slot.model = model;
    }
    slot
}

#[cfg(not(test))]
fn fetch_remote_slot() -> Option<ResolvedSlot> {
    let url = std::env::var("SAID_AI_CONFIG_URL").ok()?;
    let url = url.trim();
    if !url.starts_with("https://") || !safe_token(url) {
        return read_cached_slot();
    }
    match fetch_https(url, CONFIG_TIMEOUT) {
        Some(body) => accept_remote_slot(&body).or_else(read_cached_slot),
        None => read_cached_slot(),
    }
}

/// Accept a remote or cached slot document. `key_variable` is ignored.
/// The signature covers `base_url + "\n" + model`.
fn accept_remote_slot(body: &str) -> Option<ResolvedSlot> {
    let file: SlotFile = serde_json::from_str(body).ok()?;
    let _ = file.key_variable;
    let base_url = file.base_url.trim().trim_end_matches('/').to_string();
    let model = file.model.trim().to_string();
    if !base_url_allowed(&base_url) || !safe_token(&model) {
        return None;
    }
    if !signature_valid(&base_url, &model, file.signature.trim()) {
        return None;
    }
    let slot = ResolvedSlot { base_url, model };
    #[cfg(not(test))]
    write_cached_slot(&slot, file.signature.trim());
    Some(slot)
}

fn base_url_allowed(url: &str) -> bool {
    let Some(host) = https_host(url) else {
        return false;
    };
    ALLOWED_HOSTS.contains(&host) && safe_token(url)
}

fn https_host(url: &str) -> Option<&str> {
    let rest = url.strip_prefix("https://")?;
    if rest.contains('@') || rest.is_empty() {
        return None;
    }
    let host = rest.split(['/', '?', '#']).next()?;
    if host.is_empty() || host.contains(':') {
        return None;
    }
    Some(host)
}

fn signature_valid(base_url: &str, model: &str, signature: &str) -> bool {
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(signature) else {
        return false;
    };
    let Ok(signature) = Signature::from_slice(&bytes) else {
        return false;
    };
    let Ok(key) = VerifyingKey::from_bytes(&SLOT_PUBLIC_KEY) else {
        return false;
    };
    let message = format!("{base_url}\n{model}");
    key.verify_strict(message.as_bytes(), &signature).is_ok()
}

#[cfg(not(test))]
fn cached_slot_path() -> PathBuf {
    support_dir().join("ai-slot.json")
}

#[cfg(not(test))]
fn read_cached_slot() -> Option<ResolvedSlot> {
    let bytes = std::fs::read(cached_slot_path()).ok()?;
    let body = String::from_utf8(bytes).ok()?;
    // Re-check the host and the signature. Do not trust a file that only
    // starts with https.
    accept_remote_slot(&body)
}

#[cfg(not(test))]
fn write_cached_slot(slot: &ResolvedSlot, signature: &str) {
    let path = cached_slot_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let bytes = serde_json::json!({
        "base_url": slot.base_url,
        "model": slot.model,
        "signature": signature,
    })
    .to_string()
    .into_bytes();
    let temporary = path.with_extension("json.tmp");
    if std::fs::write(&temporary, bytes).is_ok() {
        let _ = std::fs::rename(&temporary, path);
    }
}

#[cfg(not(test))]
fn support_dir() -> PathBuf {
    crate::app_paths::support_dir().unwrap_or_else(|_| std::env::temp_dir().join("said-ai"))
}

#[cfg(not(test))]
fn env_https(name: &str) -> Option<String> {
    let value = env_token(name)?;
    let value = value.trim_end_matches('/').to_string();
    value.starts_with("https://").then_some(value)
}

#[cfg(not(test))]
fn env_token(name: &str) -> Option<String> {
    let value = std::env::var(name).ok()?;
    let value = value.trim();
    safe_token(value).then_some(value.to_string())
}

fn safe_token(value: &str) -> bool {
    !value.is_empty()
        && value.chars().all(|character| {
            !character.is_whitespace()
                && !character.is_control()
                && character != '"'
                && character != '\\'
        })
}

fn usable_key(raw: &str) -> Option<String> {
    safe_token(raw.trim()).then(|| raw.trim().to_string())
}

fn key_variable_name() -> String {
    #[cfg(not(test))]
    if let Some(name) = env_token("SAID_AI_KEY_VARIABLE") {
        return name;
    }
    LLM_SLOT.key_variable.to_string()
}

fn api_key() -> Option<String> {
    #[cfg(test)]
    {
        match test_key_override() {
            Some(overridden) => overridden.and_then(|key| usable_key(&key)),
            // Unit tests never read a developer key, so a loaded shell cannot
            // send a request while the suite runs.
            None => None,
        }
    }
    #[cfg(not(test))]
    {
        std::env::var(key_variable_name())
            .ok()
            .and_then(|value| usable_key(&value))
    }
}

fn chat_completions_url(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
}

#[derive(Clone)]
struct PreparedRequest {
    url: String,
    prompt: String,
    arguments: Vec<String>,
    config: String,
}

struct TransportResponse {
    status: u16,
    body: String,
}

enum DispatchError {
    Cancelled,
    Failed,
}

fn output_token_budget(system: &str, user: &str) -> u32 {
    let chars = system.chars().count() + user.chars().count();
    let estimated = chars.div_ceil(4) as u32;
    estimated
        .saturating_mul(3)
        .clamp(MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
}

fn prepare_request(api_key: &str, system: &str, user: &str, json_object: bool) -> PreparedRequest {
    let slot = resolved_slot();
    let url = chat_completions_url(&slot.base_url);
    let mut body = serde_json::json!({
        "model": slot.model,
        "temperature": TEMPERATURE,
        "max_tokens": output_token_budget(system, user),
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user }
        ]
    });
    if json_object {
        body["response_format"] = serde_json::json!({ "type": "json_object" });
    }
    let body = body.to_string();
    let config = format!(
        "url = \"{}\"\nheader = \"Content-Type: application/json\"\nheader = \"Authorization: Bearer {}\"\ndata-binary = \"{}\"\n",
        curl_quoted(&url),
        curl_quoted(api_key),
        curl_quoted(&body),
    );
    PreparedRequest {
        url,
        prompt: user.to_string(),
        arguments: curl_arguments(REQUEST_TIMEOUT),
        config,
    }
}

fn curl_arguments(timeout: Duration) -> Vec<String> {
    [
        "--disable",
        "--silent",
        "--show-error",
        "--max-time",
        &timeout.as_secs().to_string(),
        "--write-out",
        "\n%{http_code}",
        "--proto",
        "=https",
        "--proto-redir",
        "-all",
        "--config",
        "-",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn curl_quoted(value: &str) -> String {
    let mut quoted = String::new();
    for character in value.chars() {
        match character {
            '\\' => quoted.push_str("\\\\"),
            '"' => quoted.push_str("\\\""),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            character => quoted.push(character),
        }
    }
    quoted
}

fn dispatch(
    prepared: &PreparedRequest,
    cancelled: &dyn Fn() -> bool,
) -> Result<TransportResponse, DispatchError> {
    debug_assert!(!prepared.url.is_empty() && !prepared.prompt.is_empty());
    #[cfg(test)]
    if let Some(response) = dispatch_test(prepared) {
        return response.map_err(|_| DispatchError::Failed);
    }
    dispatch_curl(prepared, cancelled)
}

fn dispatch_curl(
    prepared: &PreparedRequest,
    cancelled: &dyn Fn() -> bool,
) -> Result<TransportResponse, DispatchError> {
    if cancelled() {
        return Err(DispatchError::Cancelled);
    }
    let mut command = Command::new(curl_program());
    command
        .args(&prepared.arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = command.spawn().map_err(|_| DispatchError::Failed)?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prepared.config.as_bytes())
            .map_err(|_| DispatchError::Failed)?;
    }
    let mut stdout = child.stdout.take().ok_or(DispatchError::Failed)?;
    let reader = thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = stdout
            .by_ref()
            .take((MAX_RESPONSE_BYTES + 1) as u64)
            .read_to_end(&mut buffer);
        buffer
    });
    let started = Instant::now();
    let status = loop {
        if cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            return Err(DispatchError::Cancelled);
        }
        if started.elapsed() > REQUEST_TIMEOUT + Duration::from_secs(5) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            return Err(DispatchError::Failed);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => thread::sleep(Duration::from_millis(20)),
            Err(_) => {
                let _ = reader.join();
                return Err(DispatchError::Failed);
            }
        }
    };
    let buffer = reader.join().map_err(|_| DispatchError::Failed)?;
    if cancelled() {
        return Err(DispatchError::Cancelled);
    }
    if buffer.len() > MAX_RESPONSE_BYTES {
        return Err(DispatchError::Failed);
    }
    if !status.success() && buffer.is_empty() {
        return Err(DispatchError::Failed);
    }
    let stdout = String::from_utf8(buffer).map_err(|_| DispatchError::Failed)?;
    let (body, http_status) = split_http_status(&stdout);
    let http_status = http_status.ok_or(DispatchError::Failed)?;
    Ok(TransportResponse {
        status: http_status,
        body,
    })
}

#[cfg(not(test))]
fn fetch_https(url: &str, timeout: Duration) -> Option<String> {
    let config = format!("url = \"{}\"\n", curl_quoted(url));
    let prepared = PreparedRequest {
        url: url.to_string(),
        prompt: url.to_string(),
        arguments: curl_arguments(timeout),
        config,
    };
    let response = dispatch_curl(&prepared, &|| false).ok()?;
    (response.status == 200).then_some(response.body)
}

fn split_http_status(stdout: &str) -> (String, Option<u16>) {
    match stdout.rsplit_once('\n') {
        Some((body, status)) => {
            let status = status.trim();
            if status.len() == 3 && status.chars().all(|character| character.is_ascii_digit()) {
                (body.to_string(), status.parse().ok())
            } else {
                (stdout.to_string(), None)
            }
        }
        None => (stdout.to_string(), None),
    }
}

fn message_content(body: &str) -> Result<String, ()> {
    let value: serde_json::Value = serde_json::from_str(body).map_err(|_| ())?;
    if value.get("error").is_some() {
        return Err(());
    }
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(|content| content.as_str())
        .ok_or(())?;
    let text = strip_fences(content).trim();
    if text.is_empty() {
        Err(())
    } else {
        Ok(text.to_string())
    }
}

fn strip_fences(text: &str) -> &str {
    let trimmed = text.trim();
    let trimmed = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed)
        .trim();
    trimmed.strip_suffix("```").unwrap_or(trimmed).trim()
}

fn parse_turn(body: &str) -> Result<ModelTurn, ()> {
    let text = message_content(body)?;
    let start = text.find('{').ok_or(())?;
    let end = text.rfind('}').ok_or(())?;
    let value: serde_json::Value = serde_json::from_str(&text[start..=end]).map_err(|_| ())?;
    let action = value
        .get("action")
        .and_then(|action| action.as_str())
        .ok_or(())?;
    let instruction = value
        .get("instruction")
        .and_then(|instruction| instruction.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let append = value
        .get("append")
        .and_then(|append| append.as_bool())
        .unwrap_or(false);
    let text = value
        .get("text")
        .and_then(|text| text.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let action = match action {
        "session" => ModelAction::Session,
        "persistent" => ModelAction::Persistent,
        "clear_session" => ModelAction::ClearSession,
        "clear_persistent" => ModelAction::ClearPersistent,
        "clear_all" => ModelAction::ClearAll,
        "apply_once" => ModelAction::ApplyOnce,
        "rewrite" => ModelAction::Rewrite,
        "ignore" => ModelAction::Ignore,
        _ => return Err(()),
    };
    Ok(ModelTurn {
        action,
        instruction,
        append,
        text,
    })
}

fn curl_program() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        PathBuf::from("curl.exe")
    }
    #[cfg(not(target_os = "windows"))]
    {
        let system = Path::new("/usr/bin/curl");
        if cfg!(target_os = "macos") || system.exists() {
            system.into()
        } else {
            PathBuf::from("curl")
        }
    }
}

#[cfg(test)]
mod test_hook {
    use super::{PreparedRequest, TransportResponse};
    use std::sync::{Mutex, MutexGuard};

    pub struct Fixture {
        pub key: Option<String>,
        pub handler: fn(&PreparedRequest) -> Result<TransportResponse, ()>,
        pub requests: Vec<PreparedRequest>,
    }

    static EXCLUSION: Mutex<()> = Mutex::new(());
    static FIXTURE: Mutex<Option<Fixture>> = Mutex::new(None);

    pub struct TestGuard {
        _exclusion: MutexGuard<'static, ()>,
    }

    impl Drop for TestGuard {
        fn drop(&mut self) {
            *FIXTURE.lock().unwrap_or_else(|error| error.into_inner()) = None;
        }
    }

    pub fn begin(
        key: Option<String>,
        handler: fn(&PreparedRequest) -> Result<TransportResponse, ()>,
    ) -> TestGuard {
        let exclusion = EXCLUSION.lock().unwrap_or_else(|error| error.into_inner());
        *FIXTURE.lock().unwrap_or_else(|error| error.into_inner()) = Some(Fixture {
            key,
            handler,
            requests: Vec::new(),
        });
        TestGuard {
            _exclusion: exclusion,
        }
    }

    pub fn key_override() -> Option<Option<String>> {
        FIXTURE
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .map(|fixture| fixture.key.clone())
    }

    pub fn dispatch(prepared: &PreparedRequest) -> Option<Result<TransportResponse, ()>> {
        let mut fixture = FIXTURE.lock().unwrap_or_else(|error| error.into_inner());
        let fixture = fixture.as_mut()?;
        let result = (fixture.handler)(prepared);
        fixture.requests.push(prepared.clone());
        Some(result)
    }

    pub fn requests() -> Vec<PreparedRequest> {
        FIXTURE
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .map(|fixture| fixture.requests.clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
fn test_key_override() -> Option<Option<String>> {
    test_hook::key_override()
}

#[cfg(test)]
fn dispatch_test(prepared: &PreparedRequest) -> Option<Result<TransportResponse, ()>> {
    test_hook::dispatch(prepared)
}

#[cfg(test)]
pub(crate) struct CapturedRequest {
    pub url: String,
    pub prompt: String,
    pub arguments: Vec<String>,
    pub config: String,
}

#[cfg(test)]
pub(crate) struct MockReply {
    pub status: u16,
    pub body: String,
}

#[cfg(test)]
pub(crate) struct TestGuard {
    _inner: test_hook::TestGuard,
}

#[cfg(test)]
fn refuse_transport(prepared: &PreparedRequest) -> Result<TransportResponse, ()> {
    let _ = prepared;
    Err(())
}

#[cfg(test)]
type MockHandler = fn(&CapturedRequest) -> Result<MockReply, String>;

#[cfg(test)]
static MOCK: Mutex<MockHandler> = Mutex::new(refuse_mock);

#[cfg(test)]
fn refuse_mock(request: &CapturedRequest) -> Result<MockReply, String> {
    let _ = request;
    Err("no mock".into())
}

#[cfg(test)]
fn forward_mock(prepared: &PreparedRequest) -> Result<TransportResponse, ()> {
    let captured = CapturedRequest {
        url: prepared.url.clone(),
        prompt: prepared.prompt.clone(),
        arguments: prepared.arguments.clone(),
        config: prepared.config.clone(),
    };
    let handler = *MOCK.lock().unwrap_or_else(|error| error.into_inner());
    match handler(&captured) {
        Ok(reply) => Ok(TransportResponse {
            status: reply.status,
            body: reply.body,
        }),
        Err(_) => Err(()),
    }
}

#[cfg(test)]
pub(crate) fn begin_missing_key() -> TestGuard {
    TestGuard {
        _inner: test_hook::begin(None, refuse_transport),
    }
}

#[cfg(test)]
pub(crate) fn begin_mock(
    key: &'static str,
    handler: fn(&CapturedRequest) -> Result<MockReply, String>,
) -> TestGuard {
    let guard = TestGuard {
        _inner: test_hook::begin(Some(key.to_string()), forward_mock),
    };
    // Install the handler only after the suite lock is held, so parallel
    // tests cannot observe each other's transport.
    *MOCK.lock().unwrap_or_else(|error| error.into_inner()) = handler;
    guard
}

#[cfg(test)]
pub(crate) fn captured_requests() -> Vec<CapturedRequest> {
    test_hook::requests()
        .into_iter()
        .map(|prepared| CapturedRequest {
            url: prepared.url,
            prompt: prepared.prompt,
            arguments: prepared.arguments,
            config: prepared.config,
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn chat_json(content: &str) -> String {
    serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": content }
        }]
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn running() -> bool {
        false
    }

    fn reply_ok(request: &CapturedRequest) -> Result<MockReply, String> {
        assert!(request.url.ends_with("/chat/completions"));
        assert!(request.url.contains(LLM_SLOT.model) || request.config.contains(LLM_SLOT.model));
        assert!(
            request
                .url
                .starts_with("https://generativelanguage.googleapis.com/v1beta/openai/")
        );
        assert!(!request.url.contains("test-key"));
        assert!(request.arguments.iter().all(|argument| {
            !argument.contains("test-key") && argument != "--noproxy" && argument != "--proxy"
        }));
        assert!(request.config.contains("Authorization: Bearer test-key"));
        assert!(request.config.contains("gemini-3.5-flash-lite"));
        assert!(request.config.contains("temperature"));
        assert!(request.config.contains("max_tokens"));
        assert!(request.config.contains("json_object"));
        assert!(is_thinking());
        Ok(MockReply {
            status: 200,
            body: chat_json(
                r#"{"action":"rewrite","instruction":"","append":false,"text":"hola"}"#,
            ),
        })
    }

    #[test]
    fn chat_url_joins_google_and_openrouter() {
        assert_eq!(
            chat_completions_url("https://generativelanguage.googleapis.com/v1beta/openai"),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://openrouter.ai/api/v1"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }

    #[test]
    fn missing_key_does_not_call_the_transport() {
        let _guard = begin_missing_key();
        let error = complete("system", "hello there", &running).unwrap_err();
        assert_eq!(error, ModelError::MissingKey);
        assert!(captured_requests().is_empty());
        assert!(!is_thinking());
    }

    #[test]
    fn complete_reads_chat_completions_json() {
        let _guard = begin_mock("test-key", reply_ok);
        let turn = complete("system", "hello \"there\"", &running).unwrap();
        assert_eq!(turn.action, ModelAction::Rewrite);
        assert_eq!(turn.text, "hola");
        assert!(!is_thinking());
        let requests = captured_requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].prompt.contains("hello \"there\""));
        assert!(requests[0].config.contains("system"));
    }

    #[test]
    fn fences_around_json_are_stripped() {
        let _guard = begin_mock("test-key", |_| {
            Ok(MockReply {
                status: 200,
                body: chat_json(
                    "```json\n{\"action\":\"ignore\",\"instruction\":\"\",\"append\":false,\"text\":\"\"}\n```",
                ),
            })
        });
        let turn = complete("system", "hello", &running).unwrap();
        assert_eq!(turn.action, ModelAction::Ignore);
    }

    #[test]
    fn a_failed_response_is_a_failure() {
        let _guard = begin_mock("test-key", |_| {
            Ok(MockReply {
                status: 400,
                body: r#"{"error":{"message":"no"}}"#.into(),
            })
        });
        assert_eq!(
            complete("system", "hello there", &running),
            Err(ModelError::Failed)
        );
        assert!(!is_thinking());
    }

    #[test]
    fn an_unreadable_body_is_a_failure() {
        let _guard = begin_mock("test-key", |_| {
            Ok(MockReply {
                status: 200,
                body: chat_json("not json"),
            })
        });
        assert_eq!(
            complete("system", "hello", &running),
            Err(ModelError::Failed)
        );
    }

    #[test]
    fn cancellation_does_not_parse_a_body() {
        let _guard = begin_mock("test-key", reply_ok);
        assert_eq!(
            complete("system", "hello", &|| true),
            Err(ModelError::Cancelled)
        );
        assert!(captured_requests().is_empty());
    }

    #[test]
    fn output_budget_scales_with_input_and_has_a_floor() {
        assert_eq!(output_token_budget("hi", "there"), MIN_OUTPUT_TOKENS);
        let long = "word ".repeat(8_000);
        let budget = output_token_budget("system", &long);
        assert!(budget > MIN_OUTPUT_TOKENS);
        assert!(budget <= MAX_OUTPUT_TOKENS);
    }

    #[test]
    fn remote_slot_rejects_other_hosts_and_unsigned_documents() {
        assert!(base_url_allowed(
            "https://generativelanguage.googleapis.com/v1beta/openai"
        ));
        assert!(base_url_allowed("https://openrouter.ai/api/v1"));
        assert!(base_url_allowed(&format!("https://{SAID_PROXY_HOST}/v1")));
        assert!(!base_url_allowed("https://evil.example/v1"));
        assert!(!base_url_allowed(
            "https://generativelanguage.googleapis.com.evil.example/v1"
        ));
        assert!(!base_url_allowed(
            "https://user:pass@generativelanguage.googleapis.com/v1"
        ));
        assert!(!base_url_allowed(
            "https://generativelanguage.googleapis.com:8443/v1"
        ));
        assert!(!base_url_allowed("http://openrouter.ai/api/v1"));
        let unsigned = r#"{"base_url":"https://openrouter.ai/api/v1","model":"google/gemini-3.5-flash-lite","key_variable":"AWS_SECRET_ACCESS_KEY"}"#;
        assert!(accept_remote_slot(unsigned).is_none());
    }

    #[test]
    fn a_signed_allowlisted_slot_is_accepted_without_its_key_variable() {
        let signature = "sNYztw5Q5HcyLEe4L22tM0ARBz9brDXPED0KyuhrflXlbtsL2tXaok/VNxPc4Nonm+4/3vdnTeJE68mSOAdZDQ==";
        let body = format!(
            r#"{{"base_url":"https://openrouter.ai/api/v1","model":"google/gemini-3.5-flash-lite","key_variable":"AWS_SECRET_ACCESS_KEY","signature":"{signature}"}}"#
        );
        let slot = accept_remote_slot(&body).expect("signed slot");
        assert_eq!(slot.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(slot.model, "google/gemini-3.5-flash-lite");
        assert_eq!(key_variable_name(), LLM_SLOT.key_variable);
    }

    #[test]
    fn a_rejected_json_mode_is_retried_once_without_it() {
        use std::sync::atomic::AtomicUsize;
        static TRIES: AtomicUsize = AtomicUsize::new(0);
        TRIES.store(0, Ordering::SeqCst);
        let _guard = begin_mock("test-key", |_| {
            let try_index = TRIES.fetch_add(1, Ordering::SeqCst);
            if try_index == 0 {
                Ok(MockReply {
                    status: 400,
                    body: r#"{"error":"response_format"}"#.into(),
                })
            } else {
                Ok(MockReply {
                    status: 200,
                    body: chat_json(
                        "note {\"action\":\"ignore\",\"instruction\":\"\",\"append\":false,\"text\":\"\"} thanks",
                    ),
                })
            }
        });
        let turn = complete("system", "hello", &running).unwrap();
        assert_eq!(turn.action, ModelAction::Ignore);
        assert_eq!(TRIES.load(Ordering::SeqCst), 2);
        let requests = captured_requests();
        assert!(requests[0].config.contains("json_object"));
        assert!(!requests[1].config.contains("json_object"));
    }
}
