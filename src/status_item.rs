use std::cell::RefCell;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use color_eyre::eyre::{Result, eyre};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObjectProtocol, Sel};
use objc2::{AnyThread, DefinedClass, MainThreadOnly, msg_send, sel};
use objc2_app_kit::{NSImage, NSMenu, NSMenuItem, NSStatusBar, NSStatusItem};
use objc2_foundation::{MainThreadMarker, NSData, NSObject, NSSize, NSString};

use crate::app_settings::AppSettings;
use crate::transcription_preparation::PreparationStatus;

const IDLE_ICON_PNG: &[u8] = include_bytes!("../assets/icons/menubar-idle@2x.png");
const RECORDING_ICON_PNG: &[u8] = include_bytes!("../assets/icons/menubar-recording@2x.png");
const STATUS_ICON_POINTS: f64 = 18.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatusItemAction {
    OpenSettings,
    PasteLast,
    StartMeeting,
    StopMeeting,
    CopyLastMeetingTranscript,
    OpenMeetings,
    CheckForUpdates,
    Quit,
}

#[derive(Debug)]
struct StatusItemTargetIvars {
    actions: SyncSender<StatusItemAction>,
}

objc2::define_class!(
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    #[ivars = StatusItemTargetIvars]
    struct StatusItemTarget;

    unsafe impl NSObjectProtocol for StatusItemTarget {}

    impl StatusItemTarget {
        #[unsafe(method(openSettings:))]
        fn open_settings(&self, _sender: &AnyObject) {
            let _ = self.ivars().actions.try_send(StatusItemAction::OpenSettings);
        }

        #[unsafe(method(pasteLast:))]
        fn paste_last(&self, _sender: &AnyObject) {
            let _ = self.ivars().actions.try_send(StatusItemAction::PasteLast);
        }

        #[unsafe(method(startMeeting:))]
        fn start_meeting(&self, _sender: &AnyObject) {
            let _ = self.ivars().actions.try_send(StatusItemAction::StartMeeting);
        }

        #[unsafe(method(stopMeeting:))]
        fn stop_meeting(&self, _sender: &AnyObject) {
            let _ = self.ivars().actions.try_send(StatusItemAction::StopMeeting);
        }

        #[unsafe(method(copyLastMeetingTranscript:))]
        fn copy_last_meeting_transcript(&self, _sender: &AnyObject) {
            let _ = self
                .ivars()
                .actions
                .try_send(StatusItemAction::CopyLastMeetingTranscript);
        }

        #[unsafe(method(openMeetings:))]
        fn open_meetings(&self, _sender: &AnyObject) {
            let _ = self.ivars().actions.try_send(StatusItemAction::OpenMeetings);
        }

        #[unsafe(method(checkForUpdates:))]
        fn check_for_updates(&self, _sender: &AnyObject) {
            let _ = self
                .ivars()
                .actions
                .try_send(StatusItemAction::CheckForUpdates);
        }

        #[unsafe(method(quit:))]
        fn quit(&self, _sender: &AnyObject) {
            let _ = self.ivars().actions.try_send(StatusItemAction::Quit);
        }
    }
);

impl StatusItemTarget {
    fn new(actions: SyncSender<StatusItemAction>, mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(StatusItemTargetIvars { actions });
        unsafe { msg_send![super(this), init] }
    }
}

struct StatusItemController {
    item: Retained<NSStatusItem>,
    idle_symbol: Retained<NSImage>,
    recording_symbol: Retained<NSImage>,
    meeting_status: Retained<NSMenuItem>,
    meeting_start: Retained<NSMenuItem>,
    meeting_stop: Retained<NSMenuItem>,
    meeting_copy_last: Retained<NSMenuItem>,
    meeting_open: Retained<NSMenuItem>,
    meeting_separator: Retained<NSMenuItem>,
    meeting_button: RefCell<(String, bool)>,
    _menu: Retained<NSMenu>,
    _target: Retained<StatusItemTarget>,
}

thread_local! {
    static STATUS_ITEM: RefCell<Option<StatusItemController>> = const { RefCell::new(None) };
}

pub fn install() -> Result<Receiver<StatusItemAction>> {
    let mtm =
        MainThreadMarker::new().ok_or_else(|| eyre!("status item requires the main thread"))?;
    let (actions, receiver) = sync_channel(8);
    let target = StatusItemTarget::new(actions, mtm);
    let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), &NSString::from_str("Said.ai"));
    let meeting_status = add_item(
        &menu,
        &target,
        "Recording meeting",
        sel!(openMeetings:),
        mtm,
    );
    let meeting_stop = add_item(&menu, &target, "Stop & Transcribe", sel!(stopMeeting:), mtm);
    let meeting_start = add_item(&menu, &target, "Record Meeting", sel!(startMeeting:), mtm);
    let meeting_copy_last = add_item(
        &menu,
        &target,
        "Copy Last Meeting Transcript",
        sel!(copyLastMeetingTranscript:),
        mtm,
    );
    let meeting_open = add_item(&menu, &target, "Meetings…", sel!(openMeetings:), mtm);
    let meeting_separator = NSMenuItem::separatorItem(mtm);
    menu.addItem(&meeting_separator);
    for item in [
        &meeting_status,
        &meeting_stop,
        &meeting_start,
        &meeting_copy_last,
        &meeting_open,
        &meeting_separator,
    ] {
        item.setHidden(true);
    }

    add_item(
        &menu,
        &target,
        "Paste Last Dictation",
        sel!(pasteLast:),
        mtm,
    );
    add_item(&menu, &target, "Settings…", sel!(openSettings:), mtm);
    menu.addItem(&NSMenuItem::separatorItem(mtm));
    if crate::sparkle::updates_managed() {
        // An item without an action is shown disabled.
        let managed = unsafe {
            NSMenuItem::initWithTitle_action_keyEquivalent(
                NSMenuItem::alloc(mtm),
                &NSString::from_str(crate::sparkle::MANAGED_UPDATES_LABEL),
                None,
                &NSString::new(),
            )
        };
        menu.addItem(&managed);
    } else {
        add_item(
            &menu,
            &target,
            "Check for Updates…",
            sel!(checkForUpdates:),
            mtm,
        );
    }
    menu.addItem(&NSMenuItem::separatorItem(mtm));
    add_item(&menu, &target, "Quit Said.ai", sel!(quit:), mtm);

    let item = NSStatusBar::systemStatusBar().statusItemWithLength(-2.0);
    let button = item
        .button(mtm)
        .ok_or_else(|| eyre!("status item button is unavailable"))?;
    let symbol = status_bar_icon(IDLE_ICON_PNG, "Said.ai")?;
    let recording_symbol = status_bar_icon(RECORDING_ICON_PNG, "Said.ai is recording a meeting")?;
    button.setImage(Some(&symbol));
    item.setMenu(Some(&menu));

    STATUS_ITEM.with(|status_item| {
        *status_item.borrow_mut() = Some(StatusItemController {
            item,
            idle_symbol: symbol,
            recording_symbol,
            meeting_status,
            meeting_start,
            meeting_stop,
            meeting_copy_last,
            meeting_open,
            meeting_separator,
            meeting_button: RefCell::new((String::new(), false)),
            _menu: menu,
            _target: target,
        });
    });
    Ok(receiver)
}

fn status_bar_icon(png: &[u8], accessibility_description: &str) -> Result<Retained<NSImage>> {
    let data = NSData::with_bytes(png);
    let image = NSImage::initWithData(NSImage::alloc(), &data)
        .ok_or_else(|| eyre!("could not decode the Said.ai status item icon"))?;
    image.setSize(NSSize::new(STATUS_ICON_POINTS, STATUS_ICON_POINTS));
    image.setTemplate(true);
    image.setAccessibilityDescription(Some(&NSString::from_str(accessibility_description)));
    Ok(image)
}

fn add_item(
    menu: &NSMenu,
    target: &StatusItemTarget,
    title: &str,
    action: Sel,
    mtm: MainThreadMarker,
) -> Retained<NSMenuItem> {
    let item = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &NSString::from_str(title),
            Some(action),
            &NSString::new(),
        )
    };
    unsafe { item.setTarget(Some(target)) };
    menu.addItem(&item);
    item
}

pub fn installed() -> bool {
    STATUS_ITEM.with(|controller| controller.borrow().is_some())
}

/// The menu no longer lists models. Parakeet English stays selected.
pub fn update_transcription(_settings: &AppSettings, _status: &PreparationStatus) {}

#[derive(Debug, Eq, PartialEq)]
struct MeetingMenu {
    status: Option<String>,
    button_title: String,
    recording: bool,
    can_start: bool,
    can_stop: bool,
}

fn meeting_menu(state: &crate::meeting_watcher::MeetingMenuState, enabled: bool) -> MeetingMenu {
    use crate::meeting_watcher::MeetingMenuState;
    match state {
        MeetingMenuState::Idle => MeetingMenu {
            status: None,
            button_title: String::new(),
            recording: false,
            can_start: enabled,
            can_stop: false,
        },
        MeetingMenuState::Starting => MeetingMenu {
            status: Some("Starting meeting recording…".into()),
            button_title: String::new(),
            recording: true,
            can_start: false,
            can_stop: false,
        },
        MeetingMenuState::Recording { started, title } => {
            let elapsed = crate::meeting::format_duration(started.elapsed().as_millis() as u64);
            MeetingMenu {
                status: Some(format!("Recording · {} · {elapsed}", menu_title(title))),
                button_title: format!(" {elapsed}"),
                recording: true,
                can_start: false,
                can_stop: true,
            }
        }
        MeetingMenuState::Transcribing => MeetingMenu {
            status: Some("Transcribing meeting…".into()),
            button_title: String::new(),
            recording: false,
            can_start: false,
            can_stop: false,
        },
    }
}

/// A meeting title short enough for one menu line.
fn menu_title(title: &str) -> String {
    const MAX_CHARS: usize = 48;
    let title = title.trim();
    if title.chars().count() <= MAX_CHARS {
        return title.into();
    }
    let cut = title.chars().take(MAX_CHARS - 1).collect::<String>();
    format!("{}…", cut.trim_end())
}

/// Mirrors the meeting recorder into the menu bar so an active recording
/// stays visible and stoppable after the Settings window closes.
pub fn update_meeting(state: &crate::meeting_watcher::MeetingMenuState, enabled: bool) {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    STATUS_ITEM.with(|controller| {
        let controller = controller.borrow();
        let Some(controller) = controller.as_ref() else {
            return;
        };
        let menu = meeting_menu(state, enabled);
        let visible = enabled || menu.status.is_some();
        controller.meeting_separator.setHidden(!visible);
        controller.meeting_copy_last.setHidden(!visible);
        controller.meeting_open.setHidden(!visible);
        controller.meeting_start.setHidden(!menu.can_start);
        controller.meeting_stop.setHidden(!menu.can_stop);
        controller.meeting_status.setHidden(menu.status.is_none());
        if let Some(status) = &menu.status {
            controller
                .meeting_status
                .setTitle(&NSString::from_str(status));
        }
        let mut button_state = controller.meeting_button.borrow_mut();
        if (*button_state != (menu.button_title.clone(), menu.recording))
            && let Some(button) = controller.item.button(mtm)
        {
            button.setTitle(&NSString::from_str(&menu.button_title));
            let symbol = if menu.recording {
                &controller.recording_symbol
            } else {
                &controller.idle_symbol
            };
            button.setImage(Some(symbol));
            *button_state = (menu.button_title, menu.recording);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meeting_menu_offers_one_action_per_state() {
        use crate::meeting_watcher::MeetingMenuState;
        let idle = meeting_menu(&MeetingMenuState::Idle, true);
        assert!(idle.can_start && !idle.can_stop && idle.status.is_none());
        assert!(!meeting_menu(&MeetingMenuState::Idle, false).can_start);
        let started = std::time::Instant::now() - std::time::Duration::from_secs(754);
        let title = "Zoom call — 2:30 PM".to_string();
        let recording = meeting_menu(&MeetingMenuState::Recording { started, title }, false);
        assert_eq!(
            recording.status.as_deref(),
            Some("Recording · Zoom call — 2:30 PM · 12:34")
        );
        let title = "A very long provisional title that keeps going past the limit".to_string();
        let long = meeting_menu(&MeetingMenuState::Recording { started, title }, false);
        assert_eq!(
            long.status.as_deref(),
            Some("Recording · A very long provisional title that keeps going… · 12:34")
        );
        assert_eq!(recording.button_title, " 12:34");
        assert!(recording.can_stop && !recording.can_start);
        let transcribing = meeting_menu(&MeetingMenuState::Transcribing, true);
        assert!(!transcribing.can_start && !transcribing.can_stop);
        assert_eq!(
            transcribing.status.as_deref(),
            Some("Transcribing meeting…")
        );
    }

    #[test]
    fn status_bar_icons_load_as_18pt_retina_templates() {
        for (png, description) in [
            (IDLE_ICON_PNG, "Said.ai"),
            (RECORDING_ICON_PNG, "Said.ai is recording a meeting"),
        ] {
            let icon = status_bar_icon(png, description).unwrap();
            assert!(icon.isTemplate());
            assert_eq!(
                icon.size(),
                NSSize::new(STATUS_ICON_POINTS, STATUS_ICON_POINTS)
            );
            assert_eq!(
                icon.accessibilityDescription().unwrap().to_string(),
                description
            );
            let rep = icon.representations().firstObject().unwrap();
            assert_eq!(rep.pixelsWide(), 36);
            assert_eq!(rep.pixelsHigh(), 36);
        }
    }
}
