//! Click-to-record requests for the home cards.
//!
//! The window and the listener are not always the same process. Linux Settings
//! sends [`crate::desktop_host::DesktopAction::ToggleCapture`]; the service
//! queues the request here, and the listener applies it. macOS and Windows
//! queue directly. The active capture is also written beside the settings so
//! another process can light the orb.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
#[cfg(not(test))]
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HomeRequest {
    Talk,
    TalkToAi,
}

static QUEUE: Mutex<Vec<HomeRequest>> = Mutex::new(Vec::new());
static ACTIVE: AtomicU8 = AtomicU8::new(0);
static LEVEL: AtomicU32 = AtomicU32::new(0);
#[cfg(not(test))]
static LAST_LEVEL_WRITE: Mutex<Option<Instant>> = Mutex::new(None);

pub(crate) fn request(kind: HomeRequest) {
    QUEUE
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .push(kind);
}

pub(crate) fn take() -> Vec<HomeRequest> {
    std::mem::take(&mut *QUEUE.lock().unwrap_or_else(|error| error.into_inner()))
}

pub(crate) fn set_active(kind: Option<HomeRequest>) {
    let next = encode(kind);
    if ACTIVE.swap(next, Ordering::AcqRel) != next {
        #[cfg(not(test))]
        write_active(kind);
    }
}

pub(crate) fn active() -> Option<HomeRequest> {
    let local = decode(ACTIVE.load(Ordering::Acquire));
    if local.is_some() {
        local
    } else {
        #[cfg(not(test))]
        {
            read_active()
        }
        #[cfg(test)]
        None
    }
}

/// Root-mean-square of a capture chunk, for the orb. Other processes see a
/// throttled file.
pub(crate) fn publish_level(samples: &[f32]) {
    if samples.is_empty() {
        return;
    }
    let mut sum = 0.0_f32;
    for sample in samples {
        sum += sample * sample;
    }
    let rms = (sum / samples.len() as f32).sqrt().clamp(0.0, 1.0);
    LEVEL.store(rms.to_bits(), Ordering::Release);
    #[cfg(not(test))]
    write_level(rms);
}

pub(crate) fn level() -> f32 {
    let local = f32::from_bits(LEVEL.load(Ordering::Acquire));
    let local = if local.is_finite() {
        local.clamp(0.0, 1.0)
    } else {
        0.0
    };
    #[cfg(not(test))]
    if let Some(remote) = read_level() {
        return remote.max(local);
    }
    local
}

fn encode(kind: Option<HomeRequest>) -> u8 {
    match kind {
        None => 0,
        Some(HomeRequest::Talk) => 1,
        Some(HomeRequest::TalkToAi) => 2,
    }
}

fn decode(value: u8) -> Option<HomeRequest> {
    match value {
        1 => Some(HomeRequest::Talk),
        2 => Some(HomeRequest::TalkToAi),
        _ => None,
    }
}

#[cfg(not(test))]
fn support_dir() -> std::path::PathBuf {
    crate::app_paths::support_dir().unwrap_or_else(|_| std::env::temp_dir().join("said-ai"))
}

#[cfg(not(test))]
fn write_active(kind: Option<HomeRequest>) {
    let path = support_dir().join("capture-active");
    match kind {
        Some(HomeRequest::Talk) => {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, b"talk");
        }
        Some(HomeRequest::TalkToAi) => {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, b"ai");
        }
        None => {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(not(test))]
fn read_active() -> Option<HomeRequest> {
    let bytes = std::fs::read(support_dir().join("capture-active")).ok()?;
    match bytes.as_slice() {
        b"talk" => Some(HomeRequest::Talk),
        b"ai" => Some(HomeRequest::TalkToAi),
        _ => None,
    }
}

#[cfg(not(test))]
fn write_level(rms: f32) {
    let mut last = LAST_LEVEL_WRITE
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if last.is_some_and(|last| last.elapsed() < Duration::from_millis(80)) {
        return;
    }
    *last = Some(Instant::now());
    drop(last);
    let path = support_dir().join("mic-level");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, format!("{rms:.4}"));
}

#[cfg(not(test))]
fn read_level() -> Option<f32> {
    let path = support_dir().join("mic-level");
    let meta = std::fs::metadata(&path).ok()?;
    let fresh = meta
        .modified()
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age < Duration::from_millis(250));
    if !fresh {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    text.trim()
        .parse::<f32>()
        .ok()
        .map(|level| level.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_is_taken_once() {
        take();
        request(HomeRequest::Talk);
        request(HomeRequest::TalkToAi);
        assert_eq!(take(), vec![HomeRequest::Talk, HomeRequest::TalkToAi]);
        assert!(take().is_empty());
    }

    #[test]
    fn level_is_the_root_mean_square() {
        publish_level(&[0.0, 0.6, 0.0, 0.8]);
        let level = level();
        assert!(level > 0.4 && level < 0.6, "{level}");
    }
}
