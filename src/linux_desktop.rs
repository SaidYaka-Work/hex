//! The service-owned, focus-free recording HUD: a layer-shell surface on
//! Wayland, an override-redirect popup on X11. Settings never owns this
//! runtime.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

use color_eyre::eyre::Result;
use gtk::prelude::*;
use gtk_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};

use crate::app_settings::IndicatorPosition;
use crate::linux_session::LinuxSession;

const TICK: Duration = Duration::from_millis(16);
/// Distance between the HUD and the screen edge it sits on.
const EDGE_MARGIN: i32 = 16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum IndicatorState {
    #[default]
    Hidden,
    Recording,
    Processing,
    Thinking,
}

impl IndicatorState {
    fn for_capture(recording: bool, pending: usize, thinking: bool) -> Self {
        if recording {
            Self::Recording
        } else if thinking {
            Self::Thinking
        } else if pending > 0 {
            Self::Processing
        } else {
            Self::Hidden
        }
    }
}

#[derive(Default)]
struct IndicatorSlot {
    generation: u64,
    state: IndicatorState,
    caption: String,
}

impl IndicatorSlot {
    fn acquire(&mut self) -> u64 {
        self.generation += 1;
        self.state = IndicatorState::Hidden;
        self.caption.clear();
        self.generation
    }
    fn set(&mut self, generation: u64, state: IndicatorState, caption: &str) {
        if self.generation == generation {
            self.state = state;
            self.caption = caption.to_string();
        }
    }
}

#[derive(Default)]
struct State {
    indicator: IndicatorSlot,
    position: IndicatorPosition,
    initialized: bool,
    limitation: Option<String>,
}

struct Runtime {
    state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        let state = Arc::new(Mutex::new(State::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let owner_state = state.clone();
        let owner_stop = stop.clone();
        let worker = std::thread::spawn(move || {
            let result = std::panic::catch_unwind(|| run(&owner_state, &owner_stop));
            let error = match result {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(format!("Linux desktop integration unavailable: {error:#}")),
                Err(_) => Some("Linux desktop integration stopped unexpectedly".into()),
            };
            if let Some(error) = error {
                tracing::warn!(%error);
                let mut state = owner_state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                state.initialized = true;
                state.limitation = Some(error);
            }
        });
        Runtime {
            state,
            stop,
            worker: Mutex::new(Some(worker)),
        }
    })
}

pub(crate) fn hud_limitation() -> Option<String> {
    if !LinuxSession::detect().is_wayland() {
        return None;
    }
    RUNTIME.get().and_then(|runtime| {
        let state = runtime
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.initialized {
            state.limitation.clone()
        } else {
            Some("Recording HUD is starting; it may be unavailable on this compositor.".into())
        }
    })
}

pub(crate) fn shutdown() {
    if let Some(runtime) = RUNTIME.get() {
        runtime.stop.store(true, Ordering::Release);
        let mut worker = runtime
            .worker
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(worker) = worker.take_if(|worker| worker.is_finished()) {
            let _ = worker.join();
        }
    }
}

pub(crate) struct LinuxIndicator {
    generation: Option<u64>,
}

impl LinuxIndicator {
    pub(crate) fn new(position: IndicatorPosition) -> Self {
        let generation = (position != IndicatorPosition::Hidden).then(|| {
            let mut state = runtime()
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.position = position;
            state.indicator.acquire()
        });
        Self { generation }
    }
    pub(crate) fn update(&self, recording: bool, pending: usize, thinking: bool, caption: &str) {
        if let Some(generation) = self.generation {
            runtime()
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .indicator
                .set(
                    generation,
                    IndicatorState::for_capture(recording, pending, thinking),
                    caption,
                );
        }
    }
}

impl Drop for LinuxIndicator {
    fn drop(&mut self) {
        self.update(false, 0, false, "");
    }
}

fn run(state: &Mutex<State>, stop: &AtomicBool) -> Result<()> {
    gtk::init()?;
    let layer_shell = gtk_layer_shell::is_supported();
    let indicator = if layer_shell || !LinuxSession::detect().is_wayland() {
        Some(create_indicator(layer_shell)?)
    } else {
        None
    };
    {
        let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
        state.initialized = true;
        if indicator.is_none() {
            let limitation = "Recording HUD unavailable: this compositor does not support Wayland layer-shell. Dictation still works.";
            tracing::warn!("{limitation}");
            state.limitation = Some(limitation.into());
        }
    }
    let context = gtk::glib::MainContext::default();
    let mut displayed = IndicatorState::Hidden;
    let mut displayed_caption = String::new();
    while !stop.load(Ordering::Acquire) {
        let (desired, caption, position) = {
            let state = state.lock().unwrap_or_else(|error| error.into_inner());
            (
                state.indicator.state,
                state.indicator.caption.clone(),
                state.position,
            )
        };
        if desired != displayed || caption != displayed_caption {
            if let Some((window, label)) = &indicator {
                match desired {
                    IndicatorState::Hidden => window.hide(),
                    IndicatorState::Recording
                    | IndicatorState::Processing
                    | IndicatorState::Thinking => {
                        let fallback =
                            crate::desktop_i18n::tr(if desired == IndicatorState::Recording {
                                "Recording"
                            } else if desired == IndicatorState::Thinking {
                                "Thinking"
                            } else {
                                "Transcribing"
                            });
                        if caption.is_empty() {
                            label.set_text(fallback);
                        } else {
                            label.set_text(&caption);
                        }
                        if displayed == IndicatorState::Hidden {
                            place_indicator(window, layer_shell, position);
                        }
                        window.show_all();
                        if !layer_shell {
                            place_indicator(window, layer_shell, position);
                        }
                    }
                }
            }
            displayed = desired;
            displayed_caption = caption;
        }
        for _ in 0..32 {
            if !context.iteration(false) {
                break;
            }
        }
        std::thread::sleep(TICK);
    }
    if let Some((window, _)) = indicator {
        window.close();
    }
    Ok(())
}

/// Centers the HUD on the top or bottom edge of the primary monitor.
fn place_indicator(window: &gtk::Window, layer_shell: bool, position: IndicatorPosition) {
    let bottom = position == IndicatorPosition::Bottom;
    if layer_shell {
        window.set_anchor(Edge::Top, !bottom);
        window.set_anchor(Edge::Bottom, bottom);
        window.set_layer_shell_margin(if bottom { Edge::Bottom } else { Edge::Top }, EDGE_MARGIN);
        return;
    }
    let Some(display) = gtk::gdk::Display::default() else {
        return;
    };
    let Some(monitor) = display.primary_monitor().or_else(|| display.monitor(0)) else {
        return;
    };
    let area = monitor.workarea();
    let (width, height) = window.size();
    let y = if bottom {
        area.y() + area.height() - height - EDGE_MARGIN
    } else {
        area.y() + EDGE_MARGIN
    };
    window.move_(area.x() + (area.width() - width) / 2, y);
}

fn create_indicator(layer_shell: bool) -> Result<(gtk::Window, gtk::Label)> {
    // An X11 popup is override-redirect: it never takes focus and stays
    // above other windows.
    let window = gtk::Window::new(if layer_shell {
        gtk::WindowType::Toplevel
    } else {
        gtk::WindowType::Popup
    });
    window.set_decorated(false);
    window.set_accept_focus(false);
    window.set_focus_on_map(false);
    window.set_skip_taskbar_hint(true);
    window.set_skip_pager_hint(true);
    window.set_app_paintable(true);
    if let Some(screen) = GtkWindowExt::screen(&window)
        && let Some(visual) = screen.rgba_visual()
    {
        window.set_visual(Some(&visual));
    }
    if layer_shell {
        window.init_layer_shell();
        window.set_layer(Layer::Overlay);
        window.set_keyboard_mode(KeyboardMode::None);
        window.set_exclusive_zone(0);
    }
    window.style_context().add_class("hex-hud");
    let css = gtk::CssProvider::new();
    css.load_from_data(b"window.hex-hud { background: transparent; } .hex-hud label { background: #171717; color: #eeeeee; border: 1px solid #292929; border-radius: 18px; padding: 9px 18px; font: 600 13px sans-serif; }")?;
    if let Some(screen) = GtkWindowExt::screen(&window) {
        gtk::StyleContext::add_provider_for_screen(
            &screen,
            &css,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
    let label = gtk::Label::new(None);
    window.add(&label);
    window.connect_realize(|window| {
        if let Some(surface) = window.window() {
            surface.set_pass_through(true);
            surface.input_shape_combine_region(&gtk::cairo::Region::create(), 0, 0);
        }
    });
    Ok((window, label))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hud_remains_visible_until_all_output_finishes() {
        assert_eq!(
            IndicatorState::for_capture(true, 2, false),
            IndicatorState::Recording
        );
        assert_eq!(
            IndicatorState::for_capture(false, 2, false),
            IndicatorState::Processing
        );
        assert_eq!(
            IndicatorState::for_capture(false, 0, true),
            IndicatorState::Thinking
        );
        assert_eq!(
            IndicatorState::for_capture(false, 0, false),
            IndicatorState::Hidden
        );
    }
    #[test]
    fn old_listener_cannot_hide_or_update_a_restarted_hud() {
        let mut slot = IndicatorSlot::default();
        let old = slot.acquire();
        slot.set(old, IndicatorState::Processing, "");
        let current = slot.acquire();
        assert_eq!(slot.state, IndicatorState::Hidden);
        slot.set(current, IndicatorState::Recording, "Recording · Esc");
        slot.set(old, IndicatorState::Hidden, "");
        assert_eq!(slot.state, IndicatorState::Recording);
        assert_eq!(slot.caption, "Recording · Esc");
        slot.set(current, IndicatorState::Hidden, "");
        assert_eq!(slot.state, IndicatorState::Hidden);
    }
}
