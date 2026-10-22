//! The one language-model slot Said.ai calls.
//!
//! Change [`LLM_BACKEND`] to switch provider, and its `model` field to switch
//! the Flash id. The key is `GEMINI_API_KEY`. It is read from the environment
//! at call time and is never written into the binary, onto disk, into argv,
//! or into the request URL.
//!
//! Talk does not call this module. Talk to AI classifies an utterance here,
//! then may ask for a rewrite. Both calls share one transport.

use std::io::{Read, Write};
#[cfg(not(target_os = "windows"))]
use std::path::Path;
use std::path::PathBuf;
use std::process::{Command, Stdio};
#[cfg(test)]
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

/// Which provider the single slot uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmProvider {
    /// Google Gemini `generateContent`.
    GeminiFlash,
}

/// The owner's provider choice. This is the only configuration site.
pub struct LlmBackend {
    pub provider: LlmProvider,
    /// Fastest current Flash text model.
    ///
    /// Google's models page, updated 2026-10-21, calls Gemini 3.5 Flash-Lite
    /// "our fastest, most cost-effective 3.5 model for high-throughput
    /// execution" and points new projects at it or at Gemini 3.8 Flash.
    /// 3.8 Flash is the newer, more capable model and defaults to medium
    /// thinking, so it is slower. There is no text 3.8 Flash-Lite, only a
    /// speech model.
    pub model: &'static str,
}

pub const LLM_BACKEND: LlmBackend = LlmBackend {
    provider: LlmProvider::GeminiFlash,
    model: "gemini-3.5-flash-lite",
};

const GEMINI_API_KEY_VARIABLE: &str = "GEMINI_API_KEY";
const GENERATE_CONTENT_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_RESPONSE_BYTES: usize = 64 * 1024;

/// Result of asking the configured backend to apply instructions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rewrite {
    pub text: String,
    /// `true` only when the model returned rewritten text.
    pub called_model: bool,
    pub notice: RewriteNotice,
}

/// Why a model call did not produce text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewriteNotice {
    None,
    MissingKey,
    Failed,
}

/// What Gemini, or the offline fallback's caller, decided an utterance means.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelClass {
    Session(String),
    Persistent(String),
    ClearSession,
    ClearPersistent,
    ClearAll,
    ApplyOnce(String),
    Ignore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    MissingKey,
    Failed,
}

/// `true` when `GEMINI_API_KEY` is set to a usable value.
pub fn gemini_key_present() -> bool {
    std::env::var(GEMINI_API_KEY_VARIABLE)
        .ok()
        .and_then(|value| usable_key(&value))
        .is_some()
}

/// Ask the configured provider to classify one Talk to AI utterance.
///
/// A missing key or a failed call returns [`ModelError`] and makes no further
/// request. The caller then uses the offline phrase lists.
pub fn classify_utterance(
    transcript: &str,
    cancelled: &dyn Fn() -> bool,
) -> Result<ModelClass, ModelError> {
    if cancelled() {
        return Err(ModelError::Failed);
    }
    let Some(key) = api_key() else {
        return Err(ModelError::MissingKey);
    };
    let prompt = format!(
        "Classify this spoken instruction for a dictation app. Reply with one JSON object and no other text.\n\
         Fields: action, instruction.\n\
         action is one of: session, persistent, clear_session, clear_persistent, clear_all, apply_once, ignore.\n\
         session: a temporary style, such as \"for now let's write formally\" or \"make it sound friendly\".\n\
         persistent: a style to keep, such as \"always write formally\" or \"from now on be brief\".\n\
         apply_once: change the current selection or the last transcript once, such as \"translate this to Spanish\".\n\
         clear_session, clear_persistent, and clear_all forget instructions.\n\
         ignore: not an instruction.\n\
         instruction is the style or edit, or an empty string for clear and ignore.\n\n\
         Utterance:\n{transcript}"
    );
    let prepared = prepare_request(&key, &prompt);
    match dispatch(&prepared, cancelled) {
        Ok(response) if response.status == 200 => {
            parse_classification(&response.body).map_err(|_| ModelError::Failed)
        }
        Ok(_) => Err(ModelError::Failed),
        Err(DispatchError::Cancelled) => Err(ModelError::Failed),
        Err(DispatchError::Failed) => Err(ModelError::Failed),
    }
}

/// Apply `instructions` to `text`. With no instructions, or without
/// `GEMINI_API_KEY`, the text is returned unchanged and no request is made.
pub fn rewrite(text: &str, instructions: &[String], cancelled: &dyn Fn() -> bool) -> Rewrite {
    let instructions: Vec<&str> = instructions
        .iter()
        .map(|instruction| instruction.trim())
        .filter(|instruction| !instruction.is_empty())
        .collect();
    if instructions.is_empty() || cancelled() {
        return unchanged(text, RewriteNotice::None);
    }
    let Some(key) = api_key() else {
        return unchanged(text, RewriteNotice::MissingKey);
    };
    let prepared = prepare_request(&key, &rewrite_prompt(text, &instructions));
    if cancelled() {
        return unchanged(text, RewriteNotice::None);
    }
    match dispatch(&prepared, cancelled) {
        Ok(response) if response.status == 200 => match parse_generate_content(&response.body) {
            Ok(rewritten) => Rewrite {
                text: rewritten,
                called_model: true,
                notice: RewriteNotice::None,
            },
            Err(_) => unchanged(text, RewriteNotice::Failed),
        },
        Ok(_) => unchanged(text, RewriteNotice::Failed),
        Err(DispatchError::Cancelled) => unchanged(text, RewriteNotice::None),
        Err(DispatchError::Failed) => unchanged(text, RewriteNotice::Failed),
    }
}

fn unchanged(text: &str, notice: RewriteNotice) -> Rewrite {
    Rewrite {
        text: text.to_string(),
        called_model: false,
        notice,
    }
}

fn usable_key(raw: &str) -> Option<String> {
    let key = raw.trim();
    if key.is_empty()
        || key.chars().any(|character| {
            character.is_whitespace()
                || character.is_control()
                || character == '"'
                || character == '\\'
        })
    {
        return None;
    }
    Some(key.to_string())
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
        std::env::var(GEMINI_API_KEY_VARIABLE)
            .ok()
            .and_then(|value| usable_key(&value))
    }
}

fn rewrite_prompt(text: &str, instructions: &[&str]) -> String {
    let mut prompt = String::from(
        "Rewrite the text so it follows every instruction. Reply with only the rewritten text, with no quotes and no explanation.\n\nInstructions:\n",
    );
    for instruction in instructions {
        prompt.push_str("- ");
        prompt.push_str(instruction);
        prompt.push('\n');
    }
    prompt.push_str("\nText:\n");
    prompt.push_str(text.trim());
    prompt
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

fn prepare_request(api_key: &str, prompt: &str) -> PreparedRequest {
    let model = match LLM_BACKEND.provider {
        LlmProvider::GeminiFlash => LLM_BACKEND.model,
    };
    let url = format!("{GENERATE_CONTENT_URL}/{model}:generateContent");
    let body = serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [{ "text": prompt }]
        }]
    })
    .to_string();
    let config = format!(
        "url = \"{}\"\nheader = \"Content-Type: application/json\"\nheader = \"x-goog-api-key: {}\"\ndata-binary = \"{}\"\n",
        curl_quoted(&url),
        curl_quoted(api_key),
        curl_quoted(&body),
    );
    PreparedRequest {
        url,
        prompt: prompt.to_string(),
        arguments: curl_arguments(),
        config,
    }
}

fn curl_arguments() -> Vec<String> {
    [
        "--disable",
        "--silent",
        "--show-error",
        "--max-time",
        "20",
        "--write-out",
        "\n%{http_code}",
        "--noproxy",
        "*",
        "--proxy",
        "",
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

fn parse_generate_content(body: &str) -> Result<String, ()> {
    let value: serde_json::Value = serde_json::from_str(body).map_err(|_| ())?;
    if value.get("error").is_some() {
        return Err(());
    }
    let parts = value
        .pointer("/candidates/0/content/parts")
        .and_then(|parts| parts.as_array())
        .ok_or(())?;
    let mut text = String::new();
    for part in parts {
        if part.get("thought").and_then(|thought| thought.as_bool()) == Some(true) {
            continue;
        }
        if let Some(piece) = part.get("text").and_then(|piece| piece.as_str()) {
            text.push_str(piece);
        }
    }
    let text = text.trim();
    if text.is_empty() {
        Err(())
    } else {
        Ok(text.to_string())
    }
}

fn parse_classification(body: &str) -> Result<ModelClass, ()> {
    let text = parse_generate_content(body)?;
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
    Ok(match action {
        "session" => ModelClass::Session(instruction),
        "persistent" => ModelClass::Persistent(instruction),
        "clear_session" => ModelClass::ClearSession,
        "clear_persistent" => ModelClass::ClearPersistent,
        "clear_all" => ModelClass::ClearAll,
        "apply_once" => ModelClass::ApplyOnce(instruction),
        "ignore" => ModelClass::Ignore,
        _ => return Err(()),
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
    *MOCK.lock().unwrap_or_else(|error| error.into_inner()) = handler;
    TestGuard {
        _inner: test_hook::begin(Some(key.to_string()), forward_mock),
    }
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
mod tests {
    use super::*;

    fn running() -> bool {
        false
    }

    fn json_text(text: &str) -> String {
        format!(
            r#"{{"candidates":[{{"content":{{"parts":[{{"thought":true,"text":"hidden"}},{{"text":{}}}]}}}}]}}"#,
            serde_json::to_string(text).unwrap()
        )
    }

    fn reply_ok(request: &CapturedRequest) -> Result<MockReply, String> {
        assert!(request.url.contains(LLM_BACKEND.model));
        assert!(request.url.ends_with(":generateContent"));
        assert!(!request.url.contains("test-key"));
        assert!(request.arguments.iter().all(|argument| {
            !argument.contains("test-key") && !argument.contains("generativelanguage")
        }));
        assert!(request.config.contains("x-goog-api-key: test-key"));
        assert!(request.config.contains("\"hello\\\"there\"") || request.prompt.contains("hello"));
        Ok(MockReply {
            status: 200,
            body: json_text("hola"),
        })
    }

    #[test]
    fn missing_key_does_not_call_the_transport() {
        let _guard = begin_missing_key();
        let rewrite = rewrite(
            "hello there",
            &["translate this to Spanish".into()],
            &running,
        );
        assert_eq!(rewrite.text, "hello there");
        assert!(!rewrite.called_model);
        assert_eq!(rewrite.notice, RewriteNotice::MissingKey);
        assert!(captured_requests().is_empty());
        assert!(classify_utterance("translate this to Spanish", &running).is_err());
        assert!(captured_requests().is_empty());
    }

    #[test]
    fn empty_instructions_skip_the_backend() {
        let _guard = begin_mock("test-key", reply_ok);
        let rewrite = rewrite("hello there", &[], &running);
        assert_eq!(rewrite.text, "hello there");
        assert!(!rewrite.called_model);
        assert!(captured_requests().is_empty());
    }

    #[test]
    fn rewrite_uses_the_mocked_generate_content_response() {
        let _guard = begin_mock("test-key", reply_ok);
        let rewrite = rewrite(
            "hello \"there\"",
            &["translate this to Spanish".into(), "write formally".into()],
            &running,
        );
        assert_eq!(rewrite.text, "hola");
        assert!(rewrite.called_model);
        let requests = captured_requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].prompt.contains("translate this to Spanish"));
        assert!(requests[0].prompt.contains("write formally"));
        assert!(requests[0].prompt.contains("hello \"there\""));
        assert!(requests[0].url.contains("gemini-3.5-flash-lite"));
    }

    #[test]
    fn a_failed_response_returns_the_original_text() {
        let _guard = begin_mock("test-key", |_| {
            Ok(MockReply {
                status: 400,
                body: r#"{"error":{"message":"no"}}"#.into(),
            })
        });
        let rewrite = rewrite("hello there", &["translate this".into()], &running);
        assert_eq!(rewrite.text, "hello there");
        assert!(!rewrite.called_model);
        assert_eq!(rewrite.notice, RewriteNotice::Failed);
    }

    #[test]
    fn classification_reads_the_model_json() {
        let _guard = begin_mock("test-key", |_| {
            Ok(MockReply {
                status: 200,
                body: json_text(
                    r#"{"action":"apply_once","instruction":"translate this to Spanish"}"#,
                ),
            })
        });
        let class = classify_utterance("translate this to Spanish", &running).unwrap();
        assert_eq!(
            class,
            ModelClass::ApplyOnce("translate this to Spanish".into())
        );
    }

    #[test]
    fn an_unreadable_classification_is_a_failure() {
        let _guard = begin_mock("test-key", |_| {
            Ok(MockReply {
                status: 200,
                body: json_text("not json"),
            })
        });
        assert_eq!(
            classify_utterance("hello", &running),
            Err(ModelError::Failed)
        );
    }
}
