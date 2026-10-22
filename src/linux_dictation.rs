use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread;
use std::time::{Duration, Instant};

use color_eyre::Result;
use color_eyre::eyre::eyre;

use crate::audio::{CaptureInstant, RecoveringAudioInput, RecoveringAudioInputEvent};
use crate::dictation::{DictationCapture, Finish, POST_RELEASE_TAIL};
use crate::events::{DictationPhase, EventLog, TranscriptPhase, VoiceEvent, VoiceState, now_ms};
use crate::feedback::{self, Tone};
use crate::linux_desktop::LinuxIndicator;
use crate::linux_input::{HotkeyEvent, HotkeyStatus, LinuxHotkeyMonitor};
use crate::linux_paste::LinuxPaster;
use crate::local_transcriber::LocalTranscriber;
use crate::transcription_engine::{PendingTranscript, TranscriptionEngine};

const UPDATE_INTERVAL: Duration = Duration::from_millis(20);
/// How long past the tail a release waits for audio that never arrives.
const TAIL_STALL_GRACE: Duration = Duration::from_millis(100);

struct Job {
    transcript: PendingTranscript,
    audio_ms: u64,
}

enum OutputJob {
    Dictation(Job),
    VoiceAction(Job),
    /// Pastes the newest completed dictation again.
    PasteLast,
    /// A Talk to AI rewrite that finished off this thread.
    DeferredPaste(String),
}

#[derive(Clone, Copy, PartialEq)]
enum OutputKind {
    Dictation,
    VoiceAction,
    PasteLast,
}

/// A release whose capture is still collecting its post-release tail.
struct Finishing {
    ended_at: CaptureInstant,
    /// The recording is a Voice Action instruction.
    voice: bool,
    deadline: Instant,
}

struct OutputWorker {
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Drop for OutputWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// History retention for a pasted transcript. Talk does not run mode processing.
#[derive(Clone)]
pub struct DictationOutput {
    pub history: Option<crate::history::History>,
    pub transformations: Arc<crate::personal_commands::TransformationClient>,
}

impl DictationOutput {
    pub fn from_settings(
        _settings: &crate::app_settings::AppSettings,
        history: Option<crate::history::History>,
    ) -> Self {
        Self {
            history,
            transformations: Arc::new(crate::personal_commands::TransformationClient::default()),
        }
    }

    /// Processes and pastes one transcript, then records it.
    fn deliver(
        &self,
        raw_text: String,
        paster: &mut LinuxPaster,
        audio_ms: u64,
        inference_ms: u64,
        started: Instant,
    ) -> Result<String> {
        let context = crate::linux_context::capture();
        // Talk is the Parakeet transcript. Mode and OpenCode rewrites stay off
        // this path; standing instructions apply only through the one backend.
        let text = crate::output_instructions::finish_talk_unless_cancelled(&raw_text, false);
        let processed = crate::dictation_processor::Processed {
            text: text.clone(),
            observation: None,
            transformations: Vec::new(),
        };
        if processed.text.trim().is_empty() {
            return Err(eyre!("text replacements produced empty output"));
        }
        if !crate::output_instructions::copy_for_self(&processed.text, &linux_focus()) {
            paster.paste(&processed.text)?;
        }
        if let Some(history) = &self.history {
            let draft = crate::dictation_modes::history_draft(
                raw_text,
                &processed,
                context.application,
                audio_ms,
                inference_ms,
                started.elapsed().as_millis() as u64,
            );
            if let Err(error) = history.record(draft) {
                tracing::warn!(%error, "could not retain Linux dictation history");
            }
        }
        Ok(processed.text)
    }
}

/// Runs hotkey dictation until `shutdown`. A prepared transcriber skips the
/// model load; otherwise the supplied transcription selection is loaded here.
pub fn run(
    event_path: &Path,
    device: Option<&str>,
    shutdown: &AtomicBool,
    settings: &crate::app_settings::AppSettings,
    output_settings: DictationOutput,
    transcriber: Option<LocalTranscriber>,
    hotkey_status: HotkeyStatus,
) -> Result<()> {
    crate::llm_backend::warm_slot();
    let transcriber =
        transcriber.map_or_else(|| LocalTranscriber::load(&settings.transcription), Ok)?;
    // The engine outlives the output worker, which waits on its transcripts.
    let engine = TranscriptionEngine::start(transcriber)?;
    // Drop audio/hotkeys and close the job queue before joining slow output on
    // every exit, including failures during startup or recording.
    let mut output = OutputWorker {
        stop: Arc::new(AtomicBool::new(false)),
        worker: None,
    };
    // A missing microphone falls back to the default one. A released
    // microphone opens only while a dictation records.
    let microphone = device.or(settings.microphone.as_deref());
    let release_while_idle = settings.release_microphone_while_idle;
    let mut input = if release_while_idle {
        RecoveringAudioInput::closed(None, 0, microphone)
    } else {
        RecoveringAudioInput::open(None, 0, microphone)
    };
    let binding = crate::linux_input::LinuxHotkey::from(&settings.dictation_hotkey);
    let hotkey_label = binding.label();
    let toggle = settings.shortcut_mode == crate::dictation::ShortcutMode::Toggle;
    let paste_with_shift = settings.paste_with_shift;
    let options = crate::linux_input::HotkeyOptions {
        double_tap: settings.double_tap_lock,
        // Double-tap-only needs the double-tap window and a non-modifier key.
        double_tap_only: settings.double_tap_lock
            && settings.double_tap_only
            && settings.dictation_hotkey.key.is_some(),
        toggle,
        paste_last: settings
            .paste_last_hotkey
            .as_ref()
            .map(crate::linux_input::LinuxHotkey::from),
        voice_action: Some(crate::linux_input::LinuxHotkey::from(&settings.edit_hotkey)),
    };
    let hotkey = LinuxHotkeyMonitor::start_with(binding, options, hotkey_status)?;
    let wayland_modifiers = hotkey.wayland_modifiers();
    let indicator = LinuxIndicator::new(settings.indicator_position);
    let suppressor =
        crate::linux_audio_control::AudioSuppressor::new(settings.recording_audio_behavior);
    let (jobs, job_receiver) = mpsc::sync_channel::<OutputJob>(2);
    let jobs_for_ai = jobs.clone();
    let ai_cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = Arc::clone(&ai_cancel);
    let (result_sender, results) = mpsc::channel();
    let worker_stop = output.stop.clone();
    output.worker = Some(thread::spawn(move || {
        let mut paster = LinuxPaster::new(worker_stop.clone(), paste_with_shift, wayland_modifiers);
        let mut last_dictation: Option<String> = None;
        while let Ok(job) = job_receiver.recv() {
            if worker_stop.load(Ordering::Acquire) {
                break;
            }
            let (job, kind) = match job {
                OutputJob::Dictation(job) => (job, OutputKind::Dictation),
                OutputJob::VoiceAction(job) => (job, OutputKind::VoiceAction),
                OutputJob::DeferredPaste(text) => {
                    let result = if crate::output_instructions::copy_for_self(&text, &linux_focus())
                    {
                        Ok(text)
                    } else {
                        paster
                            .as_mut()
                            .map_err(|error| eyre!("{error:#}"))
                            .and_then(|paster| paster.paste(&text).map(|()| text))
                            .map_err(|error| format!("{error:#}"))
                    };
                    if result_sender
                        .send((OutputKind::VoiceAction, result))
                        .is_err()
                    {
                        break;
                    }
                    continue;
                }
                OutputJob::PasteLast => {
                    let result = last_dictation
                        .clone()
                        .ok_or_else(|| eyre!("there is no completed dictation to paste"))
                        .and_then(|text| {
                            paster
                                .as_mut()
                                .map_err(|error| eyre!("{error:#}"))?
                                .paste(&text)
                        })
                        .map(|()| String::new())
                        .map_err(|error| format!("{error:#}"));
                    if result_sender.send((OutputKind::PasteLast, result)).is_err() {
                        break;
                    }
                    continue;
                }
            };
            let started = Instant::now();
            let result = (|| -> Result<String> {
                let transcribed = job.transcript.wait().map_err(|error| eyre!(error))?;
                let inference_ms = started.elapsed().as_millis() as u64;
                tracing::info!(
                    source = transcribed.source.label(),
                    finish_ms = transcribed.finish_ms,
                    audio_ms = job.audio_ms,
                    "transcribed Linux dictation"
                );
                let text = transcribed.text;
                if text.is_empty() {
                    return Err(eyre!("transcription was empty"));
                }
                let paster = paster.as_mut().map_err(|error| eyre!("{error:#}"))?;
                if kind == OutputKind::VoiceAction {
                    return run_voice_action(
                        &text,
                        paster,
                        jobs_for_ai.clone(),
                        worker_stop.clone(),
                        Arc::clone(&worker_cancel),
                    );
                }
                if worker_stop.load(Ordering::Acquire) {
                    return Err(eyre!("dictation stopped"));
                }
                output_settings.deliver(text, paster, job.audio_ms, inference_ms, started)
            })()
            .map_err(|error| format!("{error:#}"));
            tracing::info!(
                audio_ms = job.audio_ms,
                elapsed_ms = started.elapsed().as_millis(),
                "completed Linux dictation job"
            );
            if let Ok(text) = &result
                && kind == OutputKind::Dictation
            {
                last_dictation = Some(text.clone());
            }
            if result_sender.send((kind, result)).is_err() {
                break;
            }
        }
    }));

    let mut events = EventLog::create(event_path)?;
    let mut audio_ready = input.is_open();
    // A released microphone's sample rate is unknown until it opens; the
    // first chunk replaces this placeholder capture.
    let mut capture = DictationCapture::new(if audio_ready {
        input.sample_rate()
    } else {
        16_000
    });
    let mut device_label = if audio_ready {
        input.device_name().to_string()
    } else {
        microphone.unwrap_or("microphone").to_string()
    };
    // A press that arrived while the released microphone was opening;
    // recording starts at its first chunk.
    let mut start_pending = false;
    // The recording, or the pending one, is a Voice Action instruction.
    let mut recording_voice = false;
    let mut captured_through = CaptureInstant::ZERO;
    let mut finishing: Option<Finishing> = None;
    let mut pending = 0_usize;
    events.emit(&VoiceEvent::SessionStarted {
        timestamp_ms: now_ms(),
    })?;
    emit_state(&mut events, VoiceState::Listening, &device_label)?;
    println!(
        "HEX dictation is ready on {device_label}. Hold {hotkey_label} to dictate; Ctrl-C stops."
    );

    while !shutdown.load(Ordering::Relaxed) {
        if let Ok(error) = hotkey.errors.try_recv() {
            return Err(eyre!("Linux hotkey monitor stopped: {error}"));
        }
        // A release keeps recording through its tail; shortcut events wait,
        // so a press inside the tail starts after this clip is submitted.
        if let Some(release) = &finishing
            && (captured_through >= release.ended_at + POST_RELEASE_TAIL
                || Instant::now() >= release.deadline)
        {
            let ended_at = release.ended_at;
            let voice = release.voice;
            finishing = None;
            recording_voice = false;
            submit_capture(
                &mut capture,
                ended_at,
                &engine,
                &jobs,
                &mut events,
                &mut pending,
                voice,
            )?;
            suppressor.restore();
            if release_while_idle {
                input.close();
                audio_ready = false;
            }
            emit_state(
                &mut events,
                active_state(capture.is_recording(), pending),
                &device_label,
            )?;
        }
        for request in crate::home_capture::take() {
            if finishing.is_some() {
                continue;
            }
            if capture.is_recording() {
                finishing = Some(begin_finishing(&capture, captured_through, recording_voice));
                continue;
            }
            if start_pending {
                start_pending = false;
                recording_voice = false;
                suppressor.restore();
                input.close();
                audio_ready = false;
                continue;
            }
            let voice = matches!(request, crate::home_capture::HomeRequest::TalkToAi);
            ai_cancel.store(false, Ordering::Release);
            crate::output_instructions::clear_inflight_cancel();
            recording_voice = voice;
            if audio_ready {
                start_capture(&mut capture, captured_through, |_| {
                    if !recording_voice {
                        feedback::play_provisional_start();
                    }
                });
                stream_when_due(&mut capture, &engine, toggle);
            } else if release_while_idle {
                start_pending = true;
                input.request_open();
                feedback::prepare();
            } else {
                continue;
            }
            suppressor.suppress();
            events.dictation(DictationPhase::Started, "")?;
            emit_state(&mut events, VoiceState::Dictating, &device_label)?;
        }
        while finishing.is_none()
            && let Ok(action) = hotkey.events.try_recv()
        {
            match action {
                HotkeyEvent::Start | HotkeyEvent::VoiceStart
                    if !capture.is_recording() && !start_pending =>
                {
                    crate::home_capture::note_hotkey_origin();
                    ai_cancel.store(false, Ordering::Release);
                    crate::output_instructions::clear_inflight_cancel();
                    recording_voice = action == HotkeyEvent::VoiceStart;
                    if audio_ready {
                        // An instruction plays no start tone, as on Windows.
                        start_capture(&mut capture, captured_through, |_| {
                            if !recording_voice {
                                feedback::play_provisional_start();
                            }
                        });
                        stream_when_due(&mut capture, &engine, toggle);
                    } else if release_while_idle {
                        start_pending = true;
                        input.request_open();
                        feedback::prepare();
                    } else {
                        continue;
                    }
                    suppressor.suppress();
                    events.dictation(DictationPhase::Started, "")?;
                    emit_state(&mut events, VoiceState::Dictating, &device_label)?;
                }
                HotkeyEvent::Finish | HotkeyEvent::VoiceFinish | HotkeyEvent::Cancel
                    if start_pending =>
                {
                    if action == HotkeyEvent::Cancel {
                        ai_cancel.store(true, Ordering::Release);
                        crate::output_instructions::cancel_inflight();
                    }
                    recording_voice = false;
                    // Released before the opening microphone produced audio:
                    // nothing was captured.
                    start_pending = false;
                    suppressor.restore();
                    input.close();
                    audio_ready = false;
                    events.dictation(
                        if action == HotkeyEvent::Cancel {
                            DictationPhase::Cancelled
                        } else {
                            DictationPhase::Discarded
                        },
                        "",
                    )?;
                    emit_state(&mut events, active_state(false, pending), &device_label)?;
                }
                HotkeyEvent::Finish if capture.is_recording() && !recording_voice => {
                    finishing = Some(begin_finishing(&capture, captured_through, false));
                }
                HotkeyEvent::VoiceFinish if capture.is_recording() && recording_voice => {
                    finishing = Some(begin_finishing(&capture, captured_through, true));
                }
                HotkeyEvent::PasteLast if !capture.is_recording() => {
                    match jobs.try_send(OutputJob::PasteLast) {
                        Ok(()) | Err(TrySendError::Full(_)) => {}
                        Err(TrySendError::Disconnected(_)) => {
                            return Err(eyre!("dictation worker is unavailable"));
                        }
                    }
                }
                HotkeyEvent::Cancel if capture.is_recording() => {
                    ai_cancel.store(true, Ordering::Release);
                    crate::output_instructions::cancel_inflight();
                    recording_voice = false;
                    capture.cancel();
                    feedback::retract_start();
                    feedback::play(Tone::Cancel);
                    suppressor.restore();
                    if release_while_idle {
                        input.close();
                        audio_ready = false;
                    }
                    events.dictation(DictationPhase::Cancelled, "")?;
                    emit_state(
                        &mut events,
                        active_state(capture.is_recording(), pending),
                        &device_label,
                    )?;
                }
                HotkeyEvent::Cancel => {
                    ai_cancel.store(true, Ordering::Release);
                    crate::output_instructions::cancel_inflight();
                }
                _ => {}
            }
        }
        while let Ok((kind, result)) = results.try_recv() {
            // A repeated paste has no transcript and is not pending work.
            if kind == OutputKind::PasteLast {
                if let Err(error) = result {
                    events.dictation(DictationPhase::Failed(error), "")?;
                }
                continue;
            }
            pending = pending.saturating_sub(1);
            match result {
                Ok(text) => {
                    events.emit(&VoiceEvent::Transcript {
                        timestamp_ms: now_ms(),
                        phase: TranscriptPhase::Completed,
                        latency_ms: 0,
                        text: text.clone(),
                    })?;
                    events.dictation(DictationPhase::Pasted, text)?;
                }
                Err(error) => events.dictation(DictationPhase::Failed(error), "")?,
            }
            emit_state(
                &mut events,
                active_state(capture.is_recording(), pending),
                &device_label,
            )?;
        }

        crate::home_capture::set_active((capture.is_recording() || start_pending).then_some(
            if recording_voice {
                crate::home_capture::HomeRequest::TalkToAi
            } else {
                crate::home_capture::HomeRequest::Talk
            },
        ));
        let thinking = crate::llm_backend::is_thinking();
        let notice = !capture.is_recording()
            && !thinking
            && crate::output_instructions::cloud_notice_outstanding()
            && crate::output_instructions::last_event()
                == Some(crate::output_instructions::InstructionEvent::CloudNotice);
        reveal_settings_window(notice);
        indicator.update(
            capture.is_recording() || start_pending,
            pending,
            thinking,
            notice,
            &if notice {
                crate::desktop_i18n::tr("Open Said.ai to continue").to_string()
            } else {
                hud_caption(capture.is_recording() || start_pending, thinking)
            },
        );
        match input.recv_timeout(UPDATE_INTERVAL, !capture.is_recording()) {
            RecoveringAudioInputEvent::Chunk {
                samples,
                captured_through: chunk_captured_through,
            } => {
                captured_through = chunk_captured_through;
                if !audio_ready {
                    capture = DictationCapture::new(input.sample_rate());
                    audio_ready = true;
                    device_label = input.device_name().to_string();
                    if start_pending {
                        // The on-demand open finished while the shortcut is
                        // held: recording starts here, without pre-roll.
                        start_pending = false;
                        start_capture(&mut capture, captured_through, |_| {
                            if !recording_voice {
                                feedback::play_provisional_start();
                            }
                        });
                    }
                }
                capture.ingest(&samples, captured_through);
                if capture.is_recording() {
                    crate::home_capture::publish_level(&samples);
                }
                capture.become_intentional(captured_through);
                stream_when_due(&mut capture, &engine, toggle);
            }
            RecoveringAudioInputEvent::Timeout => {}
            RecoveringAudioInputEvent::Interrupted => {
                // A release still in its tail finishes with the audio it has.
                if let Some(release) = &mut finishing {
                    release.deadline = Instant::now();
                }
                audio_ready = false;
                if capture.is_recording() && finishing.is_none() {
                    recording_voice = false;
                    capture.cancel();
                    feedback::retract_start();
                    feedback::play(Tone::Cancel);
                    suppressor.restore();
                    events.dictation(DictationPhase::Cancelled, "")?;
                }
                events.dictation(
                    DictationPhase::Failed("microphone stream interrupted; reopening".into()),
                    "",
                )?;
                emit_state(
                    &mut events,
                    active_state(capture.is_recording(), pending),
                    &device_label,
                )?;
            }
            RecoveringAudioInputEvent::Reopened => {
                audio_ready = false;
                device_label = input.device_name().to_string();
            }
            RecoveringAudioInputEvent::OpenFailed(error) => {
                if !release_while_idle {
                    return Err(eyre!("could not open the microphone: {error}"));
                }
                // An on-demand open failed (device busy or unplugged): report
                // it and stay released rather than stopping the listener.
                start_pending = false;
                suppressor.restore();
                events.dictation(
                    DictationPhase::Failed(format!("could not open microphone: {error}")),
                    "",
                )?;
                emit_state(&mut events, active_state(false, pending), &device_label)?;
            }
        }
    }

    if capture.is_recording() {
        capture.cancel();
        events.dictation(DictationPhase::Cancelled, "")?;
    }
    indicator.update(false, pending, false, false, "");
    suppressor.restore();
    emit_state(&mut events, VoiceState::Stopping, &device_label)?;
    output.stop.store(true, Ordering::Release);
    drop(hotkey);
    drop(input);
    drop(jobs);
    if output
        .worker
        .take()
        .expect("dictation output worker exists")
        .join()
        .is_err()
    {
        return Err(eyre!("Linux dictation worker panicked"));
    }
    println!("Stopped.");
    Ok(())
}

/// Queues a Talk to AI instruction. Classification and rewrite run off this
/// thread; a one-shot paste comes back as [`OutputJob::DeferredPaste`].
fn run_voice_action(
    instruction: &str,
    paster: &mut LinuxPaster,
    jobs: mpsc::SyncSender<OutputJob>,
    stop: Arc<AtomicBool>,
    ai_cancel: Arc<AtomicBool>,
) -> Result<String> {
    let selected = paster.selected_text();
    let focus = linux_focus();
    crate::output_instructions::enqueue(
        instruction.to_string(),
        selected,
        focus,
        Arc::new(linux_focus),
        Arc::new(move || ai_cancel.load(Ordering::Acquire) || stop.load(Ordering::Acquire)),
        move |text| {
            let _ = jobs.send(OutputJob::DeferredPaste(text));
        },
    );
    Ok(String::new())
}

fn linux_focus() -> crate::output_instructions::FocusRead {
    if crate::linux_session::LinuxSession::detect().is_wayland() {
        return crate::output_instructions::FocusRead::Unreadable;
    }
    match crate::linux_context::capture().application {
        Some(application) => crate::output_instructions::FocusRead::App(application),
        None => crate::output_instructions::FocusRead::None,
    }
}

fn reveal_settings_window(notice: bool) {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if !notice {
        STARTED.store(false, Ordering::Release);
        return;
    }
    if crate::output_instructions::window_heartbeat_fresh() {
        return;
    }
    if STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    spawn_settings_window();
}

fn spawn_settings_window() {
    #[cfg(not(test))]
    if let Ok(exe) = std::env::current_exe() {
        // `app` is explicit so this process opens Settings. A dev binary
        // whose default command is the listener must not start a second one.
        let _ = std::process::Command::new(exe)
            .arg("app")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
}

fn hud_caption(recording: bool, thinking: bool) -> String {
    if !recording && !thinking {
        return String::new();
    }
    let state = if recording {
        crate::desktop_i18n::tr("Recording")
    } else {
        crate::desktop_i18n::tr("Thinking")
    };
    match crate::output_instructions::active().into_iter().next() {
        Some(instruction) => format!("{state} · {} · Esc", instruction.text),
        None => format!("{state} · Esc"),
    }
}

/// Applies the saved sound settings and preloads the tones.
pub(crate) fn initialize_feedback(settings: &crate::app_settings::AppSettings) {
    feedback::set_enabled(settings.sound_effects);
    feedback::set_volume(settings.sound_effect_volume.clamp(0.0, 1.0));
    if let Err(error) = feedback::preload() {
        tracing::warn!(%error, "recording sounds are unavailable; continuing without feedback");
    }
}

fn start_capture(
    capture: &mut DictationCapture,
    started_at: CaptureInstant,
    play: impl FnOnce(Tone),
) {
    capture.start(started_at);
    play(Tone::DictationStart);
}

/// Streams once the hold counts, so short taps and chords never reach the
/// model; a Toggle press is deliberate and streams at once.
fn stream_when_due(capture: &mut DictationCapture, engine: &TranscriptionEngine, toggle: bool) {
    if capture.is_recording() && (toggle || capture.is_intentional()) {
        capture.begin_stream(engine);
    }
}

fn begin_finishing(capture: &DictationCapture, ended_at: CaptureInstant, voice: bool) -> Finishing {
    let now = Instant::now();
    Finishing {
        ended_at,
        voice,
        // A short tap is discarded on its release alone and needs no tail.
        deadline: if capture.would_transcribe(ended_at) {
            now + POST_RELEASE_TAIL + TAIL_STALL_GRACE
        } else {
            now
        },
    }
}

fn submit_capture(
    capture: &mut DictationCapture,
    ended_at: CaptureInstant,
    engine: &TranscriptionEngine,
    jobs: &SyncSender<OutputJob>,
    events: &mut EventLog,
    pending: &mut usize,
    voice: bool,
) -> Result<()> {
    match capture.finish_with_tail(ended_at, POST_RELEASE_TAIL) {
        Finish::Discard => {
            // A short tap or chord withdraws the start tone of its press.
            feedback::retract_start();
            events.dictation(DictationPhase::Discarded, "")?
        }
        Finish::Transcribe(clip) if clip.is_near_silent() => {
            events.dictation(DictationPhase::Discarded, "")?
        }
        Finish::Transcribe(mut clip) => {
            feedback::play(Tone::DictationStop);
            let audio_ms = clip.duration_ms();
            let job = Job {
                transcript: engine.finish(clip.take_stream(), clip.into_transcription_samples())?,
                audio_ms,
            };
            let job = if voice {
                OutputJob::VoiceAction(job)
            } else {
                OutputJob::Dictation(job)
            };
            match jobs.try_send(job) {
                Ok(()) => {
                    *pending += 1;
                    events.dictation(DictationPhase::Transcribing, "")?;
                }
                Err(TrySendError::Full(_)) => events
                    .dictation(DictationPhase::Failed("dictation queue is full".into()), "")?,
                Err(TrySendError::Disconnected(_)) => {
                    return Err(eyre!("dictation worker is unavailable"));
                }
            }
        }
    }
    Ok(())
}

fn active_state(recording: bool, pending: usize) -> VoiceState {
    if recording {
        VoiceState::Dictating
    } else if pending > 0 {
        VoiceState::Transcribing
    } else {
        VoiceState::Listening
    }
}

fn emit_state(events: &mut EventLog, state: VoiceState, device: &str) -> Result<()> {
    events.emit(&VoiceEvent::State {
        timestamp_ms: now_ms(),
        state,
        device: device.into(),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcription_engine::Command;

    fn capture_time() -> CaptureInstant {
        CaptureInstant::from_nanos(60_000_000_000)
    }

    fn event_log() -> (EventLog, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "hex-linux-dictation-{}-{}.ndjson",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        (EventLog::create(&path).unwrap(), path)
    }

    /// Records `seconds` of speech-level audio in 100 ms chunks, streaming
    /// as the live loop would.
    fn record(
        capture: &mut DictationCapture,
        engine: &TranscriptionEngine,
        toggle: bool,
        seconds: u64,
    ) -> CaptureInstant {
        let start = capture_time();
        start_capture(capture, start, |_| {});
        stream_when_due(capture, engine, toggle);
        let mut through = start;
        for _ in 0..seconds * 10 {
            through = through + Duration::from_millis(100);
            let tone = (0..1_600)
                .map(|index| 0.1 * (index as f32 * 0.05).sin())
                .collect::<Vec<_>>();
            capture.ingest(&tone, through);
            capture.become_intentional(through);
            stream_when_due(capture, engine, toggle);
        }
        through
    }

    #[test]
    fn start_sound_precedes_audio_and_does_not_change_short_tap_discard() {
        let mut capture = DictationCapture::new(16_000);
        let started_at = capture_time();
        let mut sounds = Vec::new();

        start_capture(&mut capture, started_at, |tone| sounds.push(tone));

        assert!(capture.is_recording());
        assert_eq!(sounds, [Tone::DictationStart]);
        assert!(matches!(
            capture.finish(started_at + Duration::from_millis(100)),
            Finish::Discard
        ));
    }

    #[test]
    fn hold_streams_after_the_threshold_and_toggle_streams_from_the_press() {
        for toggle in [false, true] {
            let (engine, commands) = TranscriptionEngine::for_test();
            let mut capture = DictationCapture::new(16_000);
            let start = capture_time();
            start_capture(&mut capture, start, |_| {});
            stream_when_due(&mut capture, &engine, toggle);
            assert_eq!(capture.is_streaming(), toggle);
            capture.ingest(&[0.1; 1_600], start + Duration::from_millis(100));
            capture.become_intentional(start + Duration::from_millis(100));
            stream_when_due(&mut capture, &engine, toggle);
            assert_eq!(capture.is_streaming(), toggle, "toggle={toggle}");
            capture.ingest(&[0.1; 3_200], start + MINIMUM_HOLD);
            capture.become_intentional(start + MINIMUM_HOLD);
            stream_when_due(&mut capture, &engine, toggle);
            assert!(capture.is_streaming());
            let begun = commands
                .try_iter()
                .filter(|command| matches!(command, Command::Begin { .. }))
                .count();
            assert_eq!(begun, 1);
        }
    }

    const MINIMUM_HOLD: Duration = crate::dictation::MINIMUM_HOLD_DURATION;

    #[test]
    fn a_release_finishes_its_stream_with_the_complete_clip() {
        let (engine, commands) = TranscriptionEngine::for_test();
        let (mut events, path) = event_log();
        let (jobs, queued) = mpsc::sync_channel(2);
        let mut capture = DictationCapture::new(16_000);
        let released = record(&mut capture, &engine, false, 2);
        let mut pending = 0;
        // The release waits for its tail before it is submitted.
        let finishing = begin_finishing(
            &capture,
            released.checked_sub(POST_RELEASE_TAIL).unwrap(),
            false,
        );
        assert!(finishing.deadline > Instant::now());
        submit_capture(
            &mut capture,
            finishing.ended_at,
            &engine,
            &jobs,
            &mut events,
            &mut pending,
            false,
        )
        .unwrap();
        assert_eq!(pending, 1);
        let Ok(OutputJob::Dictation(job)) = queued.try_recv() else {
            panic!("a dictation job");
        };
        assert_eq!(job.audio_ms, 2_000);
        let received = commands.try_iter().collect::<Vec<_>>();
        let fed: usize = received
            .iter()
            .map(|command| match command {
                Command::Feed { samples, .. } => samples.len(),
                _ => 0,
            })
            .sum();
        assert!(matches!(
            received.first(),
            Some(Command::Begin { stream: 1 })
        ));
        assert!(matches!(
            received.last(),
            Some(Command::Finish { stream: Some(1), clip, .. }) if clip.len() == 32_000
        ));
        assert!(fed > 16_000 && fed <= 32_000);
        assert!(
            !received
                .iter()
                .any(|command| matches!(command, Command::Cancel { .. }))
        );
        drop(events);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn escape_and_short_taps_cancel_the_stream_and_submit_nothing() {
        let (engine, commands) = TranscriptionEngine::for_test();
        let (mut events, path) = event_log();
        let (jobs, queued) = mpsc::sync_channel(2);
        let mut pending = 0;

        let mut cancelled = DictationCapture::new(16_000);
        record(&mut cancelled, &engine, true, 1);
        cancelled.cancel();

        let mut short = DictationCapture::new(16_000);
        let start = capture_time();
        start_capture(&mut short, start, |_| {});
        stream_when_due(&mut short, &engine, true);
        short.ingest(&[0.1; 1_600], start + Duration::from_millis(100));
        let finishing = begin_finishing(&short, start + Duration::from_millis(100), false);
        assert!(
            finishing.deadline <= Instant::now(),
            "a short tap needs no tail"
        );
        submit_capture(
            &mut short,
            finishing.ended_at,
            &engine,
            &jobs,
            &mut events,
            &mut pending,
            false,
        )
        .unwrap();

        assert_eq!(pending, 0);
        assert!(queued.try_recv().is_err());
        let received = commands.try_iter().collect::<Vec<_>>();
        let count = |matches: fn(&Command) -> bool| received.iter().filter(|c| matches(c)).count();
        assert_eq!(count(|c| matches!(c, Command::Begin { .. })), 2);
        assert_eq!(count(|c| matches!(c, Command::Cancel { .. })), 2);
        assert_eq!(count(|c| matches!(c, Command::Finish { .. })), 0);
        drop(events);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_model_that_cannot_stream_transcribes_the_whole_clip_after_release() {
        let (engine, commands) = TranscriptionEngine::for_test_without_streaming();
        let (mut events, path) = event_log();
        let (jobs, queued) = mpsc::sync_channel(2);
        let mut capture = DictationCapture::new(16_000);
        let released = record(&mut capture, &engine, true, 2);
        assert!(!capture.is_streaming());
        let mut pending = 0;
        submit_capture(
            &mut capture,
            released.checked_sub(POST_RELEASE_TAIL).unwrap(),
            &engine,
            &jobs,
            &mut events,
            &mut pending,
            false,
        )
        .unwrap();
        assert!(queued.try_recv().is_ok());
        assert!(matches!(
            commands.try_iter().collect::<Vec<_>>().as_slice(),
            [Command::Finish { stream: None, clip, .. }] if clip.len() == 32_000
        ));
        drop(events);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn cancelled_capture_keeps_pending_output_visible() {
        assert_eq!(active_state(true, 2), VoiceState::Dictating);
        assert_eq!(active_state(false, 2), VoiceState::Transcribing);
        assert_eq!(active_state(false, 1), VoiceState::Transcribing);
        assert_eq!(active_state(false, 0), VoiceState::Listening);
        assert_eq!(active_state(true, 0), VoiceState::Dictating);
    }

    #[test]
    fn output_guard_cancels_and_joins_even_on_listener_error() {
        let stopped = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let result: Result<()> = {
            let mut output = OutputWorker {
                stop: stopped.clone(),
                worker: None,
            };
            let (_jobs, receiver) = mpsc::channel::<()>();
            let stop = stopped.clone();
            let completed = finished.clone();
            output.worker = Some(thread::spawn(move || {
                // The sender must close before the guard joins this worker.
                assert!(receiver.recv().is_err());
                while !stop.load(Ordering::Acquire) {
                    thread::yield_now();
                }
                completed.store(true, Ordering::Release);
            }));
            Err(eyre!("listener failure"))
        };
        assert!(result.is_err());
        assert!(stopped.load(Ordering::Acquire));
        assert!(finished.load(Ordering::Acquire));
    }
}
