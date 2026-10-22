//! The desktop design tokens: the only place raw colours, type sizes, line
//! heights, radii, and control heights are written down. Panes and components
//! name these tokens; `docs/design-system.md` explains when to use each one.
//!
//! Said.ai follows saidyaka.com: warm paper (`#f4efe6`), near-black ink,
//! terracotta (`#b4532a`, `#e8a87c` in the dark theme), white cards, and
//! pill-shaped buttons. Rows and menus use a smaller corner.

use std::cell::Cell;

use gpui::{FontWeight, Pixels, Styled, WindowAppearance, px};

// Surfaces: warm paper, with white cards sitting on it.
pub(crate) const SIDEBAR: u32 = 0xefe8dc;
pub(crate) const SIDEBAR_HOVER: u32 = 0xe7dfd2;
pub(crate) const CANVAS: u32 = 0xf4efe6;
pub(crate) const SURFACE: u32 = 0xfffdf8;
pub(crate) const SURFACE_HOVER: u32 = 0xf7f1e8;
pub(crate) const SURFACE_SELECTED: u32 = 0xe7dfd2;
pub(crate) const LINE: u32 = 0xe4dcd0;
pub(crate) const LINE_STRONG: u32 = 0xcfc4b4;
/// Modal backdrops.
pub(crate) const SCRIM: u32 = 0x161613cc;

// Ink, strongest to weakest. Tertiary still meets 4.5:1 on every surface.
pub(crate) const TEXT_PRIMARY: u32 = 0x161613;
pub(crate) const TEXT_SECONDARY: u32 = 0x3a3832;
pub(crate) const TEXT_TERTIARY: u32 = 0x5c564e;
pub(crate) const TEXT_DISABLED: u32 = 0xa39e94;

// Primary buttons use saidyaka.com terracotta. Ink stays the text colour.
// `ON_ACCENT` is one step off card white so the dark theme can map it to ink.
pub(crate) const ACCENT: u32 = 0xb4532a;
pub(crate) const ON_ACCENT: u32 = 0xfffef9;
pub(crate) const ACCENT_TEXT: u32 = 0x8c3f28;
pub(crate) const ACCENT_SURFACE: u32 = 0xf8ebe3;
pub(crate) const ACCENT_BORDER: u32 = 0xe7d3c6;
/// Warm end of the home orb. GPUI has no radial gradient, so the orb fades
/// from [`ACCENT`] toward this colour.
pub(crate) const ORB_WARM: u32 = 0xe8a87c;
/// Text selection highlight: ink at low opacity.
pub(crate) const SELECTION: u32 = 0x16161333;
pub(crate) const SUCCESS: u32 = 0x166534;
pub(crate) const SUCCESS_SURFACE: u32 = 0xe7f5ec;
pub(crate) const DANGER: u32 = 0x9f1239;
pub(crate) const DANGER_SURFACE: u32 = 0xfdecee;
pub(crate) const DANGER_BORDER: u32 = 0xf3c3cb;
/// Amber status, such as the AI-unavailable pill. Dark enough for the cream fill.
pub(crate) const WARNING: u32 = 0x92400e;
pub(crate) const WARNING_SURFACE: u32 = 0xfef3c7;

/// One step of the type scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TypeStep {
    pub(crate) size: f32,
    pub(crate) line_height: f32,
    /// `None` inherits the surrounding weight.
    pub(crate) weight: Option<FontWeight>,
}

/// Metadata, timestamps, tags, and section labels.
pub(crate) const CAPTION: TypeStep = TypeStep {
    size: 11.0,
    line_height: 14.0,
    weight: None,
};
/// Descriptions, control labels, and secondary copy.
pub(crate) const FOOTNOTE: TypeStep = TypeStep {
    size: 12.0,
    line_height: 16.0,
    weight: None,
};
/// The window default: values, inputs, list titles, and reading text.
pub(crate) const BODY: TypeStep = TypeStep {
    size: 13.0,
    line_height: 18.0,
    weight: None,
};
/// Detail, card, and dialog titles.
pub(crate) const HEADLINE: TypeStep = TypeStep {
    size: 15.0,
    line_height: 20.0,
    weight: Some(FontWeight::SEMIBOLD),
};
/// Pane titles.
pub(crate) const TITLE: TypeStep = TypeStep {
    size: 20.0,
    line_height: 26.0,
    weight: Some(FontWeight::SEMIBOLD),
};
/// The home wordmark. Serif, like the headlines on saidyaka.com.
pub(crate) const DISPLAY: TypeStep = TypeStep {
    size: 36.0,
    line_height: 48.0,
    weight: Some(FontWeight::MEDIUM),
};

/// The only text below the scale: the L/R badge in a sided modifier keycap.
pub(crate) const KEYCAP_SIDE_BADGE: TypeStep = TypeStep {
    size: 7.0,
    line_height: 8.0,
    weight: Some(FontWeight::SEMIBOLD),
};

pub(crate) const MONOSPACE_FAMILY: &str = "SF Mono";

/// Bundled Instrument Sans. Body copy uses it once [`install_bundled_fonts`]
/// has registered the file.
pub(crate) const SANS_FAMILY: &str = "Instrument Sans";
/// Bundled Instrument Serif. The home wordmark is the only place that uses it.
pub(crate) const SERIF_FAMILY: &str = "Instrument Serif";

thread_local! {
    static DARK: Cell<bool> = const { Cell::new(false) };
}

/// Follows the system appearance for the current render.
pub(crate) fn follow_window(window: &gpui::Window) {
    if std::env::var_os("HEX_FORCE_DARK").is_some_and(|value| value == "1") {
        set_dark(true);
        return;
    }
    set_dark(matches!(
        window.appearance(),
        WindowAppearance::Dark | WindowAppearance::VibrantDark
    ));
}

pub(crate) fn set_dark(dark: bool) {
    DARK.with(|cell| cell.set(dark));
}

fn dark_theme() -> bool {
    DARK.with(Cell::get)
}

const DARK_SIDEBAR: u32 = 0x100e0c;
const DARK_SIDEBAR_HOVER: u32 = 0x241f1b;
const DARK_CANVAS: u32 = 0x0c0b0a;
const DARK_SURFACE: u32 = 0x151311;
const DARK_SURFACE_HOVER: u32 = 0x221e1a;
const DARK_LINE: u32 = 0x3a342e;
const DARK_LINE_STRONG: u32 = 0x564c40;
const DARK_SCRIM: u32 = 0x0c0b09cc;
const DARK_TEXT_PRIMARY: u32 = 0xf4efe6;
const DARK_TEXT_SECONDARY: u32 = 0xd5cec2;
const DARK_TEXT_TERTIARY: u32 = 0xb7aea0;
const DARK_TEXT_DISABLED: u32 = 0x7d756a;
const DARK_ACCENT: u32 = 0xe8a87c;
const DARK_ON_ACCENT: u32 = 0x161613;
const DARK_ACCENT_TEXT: u32 = 0xe8a87c;
const DARK_ACCENT_SURFACE: u32 = 0x2a1a12;
const DARK_ACCENT_BORDER: u32 = 0x6b4632;
const DARK_ORB_WARM: u32 = 0xf6d3b0;
const DARK_SELECTION: u32 = 0xf4efe633;
const DARK_SUCCESS: u32 = 0x86efac;
const DARK_SUCCESS_SURFACE: u32 = 0x14281c;
const DARK_DANGER: u32 = 0xfb7185;
const DARK_DANGER_SURFACE: u32 = 0x2a1218;
const DARK_DANGER_BORDER: u32 = 0x6b3040;
const DARK_WARNING: u32 = 0xfbbf24;
const DARK_WARNING_SURFACE: u32 = 0x3a2a14;

/// The colour to paint. Light values stay as written; a dark system appearance
/// maps each token. Call [`follow_window`] at the start of a render.
pub(crate) fn paint(light: u32) -> u32 {
    if !dark_theme() {
        return light;
    }
    match light {
        SIDEBAR => DARK_SIDEBAR,
        // SURFACE_SELECTED shares this value, so one arm covers both.
        SIDEBAR_HOVER => DARK_SIDEBAR_HOVER,
        CANVAS => DARK_CANVAS,
        SURFACE => DARK_SURFACE,
        SURFACE_HOVER => DARK_SURFACE_HOVER,
        LINE => DARK_LINE,
        LINE_STRONG => DARK_LINE_STRONG,
        SCRIM => DARK_SCRIM,
        TEXT_PRIMARY => DARK_TEXT_PRIMARY,
        TEXT_SECONDARY => DARK_TEXT_SECONDARY,
        TEXT_TERTIARY => DARK_TEXT_TERTIARY,
        TEXT_DISABLED => DARK_TEXT_DISABLED,
        ACCENT => DARK_ACCENT,
        ON_ACCENT => DARK_ON_ACCENT,
        ACCENT_TEXT => DARK_ACCENT_TEXT,
        ACCENT_SURFACE => DARK_ACCENT_SURFACE,
        ACCENT_BORDER => DARK_ACCENT_BORDER,
        ORB_WARM => DARK_ORB_WARM,
        SELECTION => DARK_SELECTION,
        SUCCESS => DARK_SUCCESS,
        SUCCESS_SURFACE => DARK_SUCCESS_SURFACE,
        DANGER => DARK_DANGER,
        DANGER_SURFACE => DARK_DANGER_SURFACE,
        DANGER_BORDER => DARK_DANGER_BORDER,
        WARNING => DARK_WARNING,
        WARNING_SURFACE => DARK_WARNING_SURFACE,
        _ => light,
    }
}

/// Registers Instrument Sans and Instrument Serif with GPUI.
pub(crate) fn install_bundled_fonts(cx: &gpui::App) {
    let fonts = vec![
        std::borrow::Cow::Borrowed(include_bytes!("../assets/fonts/InstrumentSans.ttf").as_slice()),
        std::borrow::Cow::Borrowed(
            include_bytes!("../assets/fonts/InstrumentSerif-Regular.ttf").as_slice(),
        ),
        std::borrow::Cow::Borrowed(
            include_bytes!("../assets/fonts/InstrumentSerif-Italic.ttf").as_slice(),
        ),
    ];
    if let Err(error) = cx.text_system().add_fonts(fonts) {
        tracing::warn!(%error, "could not install Said.ai fonts");
    }
}

/// SF Pro's ascender and descender as fractions of the font size.
const FONT_ASCENT: f32 = 0.952;
const FONT_DESCENT: f32 = 0.241;

/// The top inset that puts `smaller`'s first baseline on `larger`'s when both
/// lines start at the same top, such as a timestamp beside a row title.
pub(crate) fn baseline_offset(larger: TypeStep, smaller: TypeStep) -> f32 {
    // GPUI centres the font's ascent + descent in the line box.
    let baseline =
        |step: TypeStep| step.line_height / 2.0 + step.size * (FONT_ASCENT - FONT_DESCENT) / 2.0;
    baseline(larger) - baseline(smaller)
}

/// Applies type steps to any styled element. Weight is applied only by steps
/// that own one, so call `font_weight` after the step to emphasise text.
pub(crate) trait Typography: Styled + Sized {
    fn text_step(self, step: TypeStep) -> Self {
        let styled = self
            .text_size(px(step.size))
            .line_height(px(step.line_height));
        match step.weight {
            Some(weight) => styled.font_weight(weight),
            None => styled,
        }
    }

    fn text_caption(self) -> Self {
        self.text_step(CAPTION).font_family(SANS_FAMILY)
    }

    fn text_footnote(self) -> Self {
        self.text_step(FOOTNOTE).font_family(SANS_FAMILY)
    }

    fn text_body(self) -> Self {
        self.text_step(BODY).font_family(SANS_FAMILY)
    }

    fn text_headline(self) -> Self {
        self.text_step(HEADLINE).font_family(SANS_FAMILY)
    }

    fn text_title(self) -> Self {
        self.text_step(TITLE).font_family(SANS_FAMILY)
    }

    fn text_display(self) -> Self {
        self.text_step(DISPLAY).font_family(SERIF_FAMILY)
    }

    fn text_monospace(self) -> Self {
        self.font_family(MONOSPACE_FAMILY)
    }
}

impl<E: Styled> Typography for E {}

// Spacing: a 4 px grid. Prefer GPUI's spacing utilities (`*_1` 4, `*_2` 8,
// `*_3` 12, `*_4` 16, `*_5` 20, `*_6` 24, `*_8` 32); `SPACE_2` is named for
// arithmetic on control heights and negative icon-button margins.
pub(crate) const SPACE_2: f32 = 8.0;

// Radii.
pub(crate) const RADIUS_SMALL: Pixels = px(4.0);
/// Rows, menus, inputs, and dropdowns.
pub(crate) const RADIUS_CONTROL: Pixels = px(8.0);
/// Buttons only. Half of [`CONTROL_HEIGHT_LARGE`], so the home actions are pills.
pub(crate) const RADIUS_PILL: Pixels = px(22.0);
/// The two home action cards.
pub(crate) const RADIUS_CARD: Pixels = px(22.0);
pub(crate) const RADIUS_PANEL: Pixels = px(16.0);
pub(crate) const RADIUS_MODAL: Pixels = px(12.0);
pub(crate) const RADIUS_KNOB: Pixels = px(2.0);

/// The corner radius of an application icon or monogram of `size`.
pub(crate) fn icon_radius(size: f32) -> Pixels {
    px(size * 0.22)
}

// Control geometry.
pub(crate) const CONTROL_HEIGHT: f32 = 30.0;
pub(crate) const CONTROL_HEIGHT_SMALL: f32 = 24.0;
/// The home actions. Tall enough to read as the page's buttons, still a pill.
pub(crate) const CONTROL_HEIGHT_LARGE: f32 = 44.0;
pub(crate) const KEYCAP_WIDTH: f32 = 34.0;
/// The narrowest a settings row's title and description get before its
/// control wraps below them.
pub(crate) const SETTINGS_COPY_MIN_WIDTH: f32 = 180.0;
pub(crate) const TAG_HEIGHT: f32 = 18.0;
pub(crate) const MENU_ITEM_HEIGHT: f32 = 28.0;
pub(crate) const MENU_WIDTH: f32 = 180.0;
/// The gap between a control and the menu it opens.
pub(crate) const MENU_OFFSET: f32 = 4.0;
pub(crate) const NAV_ITEM_HEIGHT: f32 = 32.0;
pub(crate) const MULTILINE_INPUT_HEIGHT: f32 = 76.0;
pub(crate) const DISCLOSURE_WIDTH: f32 = 220.0;
pub(crate) const SEARCH_FIELD_WIDTH: f32 = 220.0;
/// The widest an inline field beside a row label grows.
pub(crate) const MODE_FIELD_WIDTH: f32 = 300.0;
pub(crate) const ICON_SIZE: f32 = 14.0;
pub(crate) const ICON_SIZE_SMALL: f32 = 11.0;
pub(crate) const TOGGLE_WIDTH: f32 = 28.0;
pub(crate) const TOGGLE_HEIGHT: f32 = 16.0;
/// Opacity of a control that cannot currently be used.
pub(crate) const DISABLED_OPACITY: f32 = 0.4;

// Pane geometry.
pub(crate) const SIDEBAR_WIDTH: f32 = 220.0;
/// Space above the sidebar's first item, clear of the window controls.
pub(crate) const TITLEBAR_INSET: f32 = 52.0;
pub(crate) const PANE_HEADER_HEIGHT: f32 = 70.0;
/// Horizontal inset shared by every pane header and body, so cards align
/// with the header rule.
pub(crate) const PANE_GUTTER: f32 = 32.0;
/// The one content width every pane is bounded to.
pub(crate) const PANE_CONTENT_WIDTH: f32 = 940.0;
/// The home column. The simplified window is narrower than this.
pub(crate) const HOME_COLUMN: f32 = 420.0;
/// The widest active-instruction row on the home.
pub(crate) const HOME_INSTRUCTION: f32 = 360.0;
/// The running Said.ai window. Preview keeps the classic shell size.
pub(crate) const HOME_WINDOW_WIDTH: f32 = 440.0;
pub(crate) const HOME_WINDOW_HEIGHT: f32 = 560.0;
/// The voice orb and the soft glow behind it.
pub(crate) const ORB_SIZE: f32 = 72.0;
pub(crate) const ORB_GLOW: f32 = 168.0;
/// A home action card.
pub(crate) const ACTION_CARD_HEIGHT: f32 = 116.0;
/// The settings sheet on the home. It fits inside [`HOME_WINDOW_WIDTH`].
pub(crate) const SETTINGS_SHEET: f32 = 400.0;
/// Tallest the sheet grows before its body scrolls, inside the home window.
pub(crate) const SETTINGS_SHEET_MAX_HEIGHT: f32 = 520.0;
/// The list column of a list+detail pane.
pub(crate) const PANE_LIST_WIDTH: f32 = 320.0;
/// The narrower list column for panes whose detail is an editor (Modes).
pub(crate) const PANE_LIST_WIDTH_COMPACT: f32 = 220.0;
pub(crate) const MODAL_WIDTH: f32 = 780.0;
pub(crate) const MODAL_HEIGHT: f32 = 520.0;
/// A determinate or indeterminate progress track.
pub(crate) const PROGRESS_HEIGHT: f32 = 3.0;
pub(crate) const PANEL_HEADER_HEIGHT: f32 = 38.0;
pub(crate) const SETTINGS_ROW_MIN_HEIGHT: f32 = 56.0;
/// Fixed width of a meeting row's inline Copy / Copied button.
#[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(dead_code))]
pub(crate) const MEETING_ROW_COPY_WIDTH: f32 = 58.0;

// Meeting capsule geometry.
#[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(dead_code))]
pub(crate) const MEETING_PANEL_WIDTH: f32 = 420.0;
#[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(dead_code))]
pub(crate) const MEETING_PANEL_HEIGHT: f32 = 64.0;
/// Expanded width of the floating meeting recording capsule.
#[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(dead_code))]
pub(crate) const RECORDING_PANEL_WIDTH: f32 = 548.0;
/// Compact width of the floating meeting recording capsule at rest.
#[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(dead_code))]
pub(crate) const RECORDING_COMPACT_WIDTH: f32 = 244.0;
/// Height of the floating meeting recording capsule window.
#[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(dead_code))]
pub(crate) const RECORDING_PANEL_HEIGHT: f32 = 48.0;
/// Fixed width of the recording capsule's Copy new / Copied button.
#[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(dead_code))]
pub(crate) const RECORDING_COPY_NEW_WIDTH: f32 = 108.0;
/// Fixed width of the recording capsule's Copy all / Copied button.
#[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(dead_code))]
pub(crate) const RECORDING_COPY_ALL_WIDTH: f32 = 76.0;
/// Fixed width of the ready capsule's Copy / Copied button.
#[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(dead_code))]
pub(crate) const READY_COPY_WIDTH: f32 = 68.0;

// Onboarding geometry.
/// The tour's one column: page art, copy, cards, and footer share its edges.
pub(crate) const ONBOARDING_WIDTH: f32 = 520.0;
/// The illustration band above a centred page's title.
pub(crate) const ONBOARDING_ART_HEIGHT: f32 = 148.0;
/// The shortcut drawn as a large keycap in onboarding art.
pub(crate) const HERO_KEYCAP_SIZE: f32 = 56.0;
/// An icon drawn as onboarding art, such as the finished tour's check.
pub(crate) const HERO_ICON_SIZE: f32 = 28.0;
/// How far a hero keycap stands above its base before it is pressed.
pub(crate) const HERO_KEYCAP_DEPTH: f32 = 3.0;
/// The measure of a page's centred body copy.
pub(crate) const ONBOARDING_BODY_WIDTH: f32 = 400.0;
/// A setup row's download progress, track and percentage together.
pub(crate) const ONBOARDING_PROGRESS_WIDTH: f32 = 140.0;
/// The tour's main button, centred under each page.
pub(crate) const ONBOARDING_PRIMARY_WIDTH: f32 = 180.0;
/// The round badge holding a setup row's icon.
pub(crate) const ONBOARDING_BADGE_SIZE: f32 = 28.0;
/// The tour's step dots; the current one stretches to the wide width.
pub(crate) const STEP_DOT_SIZE: f32 = 6.0;
pub(crate) const STEP_DOT_ACTIVE_WIDTH: f32 = 16.0;

// Motion presets (`duration` + `bounce`, SwiftUI mapping) and transition offsets.
pub(crate) use crate::motion::Spec as SpringSpec;
#[allow(dead_code)]
pub(crate) const SPRING_INSTANT: SpringSpec = crate::motion::presets::INSTANT;
pub(crate) const SPRING_SNAP: SpringSpec = crate::motion::presets::SNAP;
#[allow(dead_code)]
pub(crate) const SPRING_SNAP_EXIT: SpringSpec = crate::motion::presets::SNAP_EXIT;
pub(crate) const SPRING_SETTLE: SpringSpec = crate::motion::presets::SETTLE;
#[allow(dead_code)]
pub(crate) const SPRING_SETTLE_EXIT: SpringSpec = crate::motion::presets::SETTLE_EXIT;
pub(crate) const SPRING_REFLOW: SpringSpec = crate::motion::presets::REFLOW;
#[allow(dead_code)]
pub(crate) const SPRING_REFLOW_EXIT: SpringSpec = crate::motion::presets::REFLOW_EXIT;
#[allow(dead_code)]
pub(crate) const SPRING_GLIDE: SpringSpec = crate::motion::presets::GLIDE;
#[allow(dead_code)]
pub(crate) const SPRING_GLIDE_EXIT: SpringSpec = crate::motion::presets::GLIDE_EXIT;
#[allow(dead_code)]
pub(crate) const SPRING_MORPH: SpringSpec = crate::motion::presets::MORPH;
#[allow(dead_code)]
pub(crate) const SPRING_MORPH_EXIT: SpringSpec = crate::motion::presets::MORPH_EXIT;
#[allow(dead_code)]
pub(crate) const SPRING_POP: SpringSpec = crate::motion::presets::POP;
#[allow(dead_code)]
pub(crate) const SPRING_POP_EXIT: SpringSpec = crate::motion::presets::POP_EXIT;
#[allow(dead_code)]
pub(crate) const SPRING_REDUCED_FADE: SpringSpec = crate::motion::presets::REDUCED_FADE;
/// Vertical travel (in px) of a pane's entrance transition when Reduce Motion is off.
pub(crate) const PANE_ENTER_OFFSET: f32 = 4.0;
/// Vertical travel (in px) of a dropdown menu opening from its trigger.
pub(crate) const MENU_ENTER_OFFSET: f32 = 4.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn luminance(hex: u32) -> f64 {
        let channel = |shift: u32| {
            let value = f64::from((hex >> shift) & 0xff) / 255.0;
            if value <= 0.039_28 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * channel(16) + 0.7152 * channel(8) + 0.0722 * channel(0)
    }

    fn contrast(foreground: u32, background: u32) -> f64 {
        let (light, dark) = {
            let (a, b) = (luminance(foreground), luminance(background));
            if a > b { (a, b) } else { (b, a) }
        };
        (light + 0.05) / (dark + 0.05)
    }

    #[test]
    fn readable_text_meets_aa_contrast_on_every_surface() {
        for background in [SIDEBAR, CANVAS, SURFACE, SURFACE_HOVER, SURFACE_SELECTED] {
            for text in [TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY] {
                let ratio = contrast(text, background);
                assert!(
                    ratio >= 4.5,
                    "{text:06x} on {background:06x} is {ratio:.2}:1"
                );
            }
        }
        for (text, surface) in [
            (ACCENT_TEXT, ACCENT_SURFACE),
            (SUCCESS, SUCCESS_SURFACE),
            (DANGER, DANGER_SURFACE),
            (WARNING, WARNING_SURFACE),
        ] {
            let ratio = contrast(text, surface);
            assert!(ratio >= 4.5, "{text:06x} on {surface:06x} is {ratio:.2}:1");
        }
        let on_accent = contrast(ON_ACCENT, ACCENT);
        assert!(
            on_accent >= 4.5,
            "accent label {ON_ACCENT:06x} on {ACCENT:06x} is {on_accent:.2}:1"
        );
    }

    #[test]
    fn dark_theme_keeps_aa_contrast() {
        set_dark(true);
        for background in [SIDEBAR, CANVAS, SURFACE, SURFACE_HOVER, SURFACE_SELECTED] {
            for text in [TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY] {
                let ratio = contrast(paint(text), paint(background));
                assert!(
                    ratio >= 4.5,
                    "dark {text:06x} on {background:06x} is {ratio:.2}:1"
                );
            }
        }
        for (text, surface) in [
            (ON_ACCENT, ACCENT),
            (ACCENT_TEXT, ACCENT_SURFACE),
            (SUCCESS, SUCCESS_SURFACE),
            (DANGER, DANGER_SURFACE),
            (WARNING, WARNING_SURFACE),
        ] {
            let ratio = contrast(paint(text), paint(surface));
            assert!(
                ratio >= 4.5,
                "dark {text:06x} on {surface:06x} is {ratio:.2}:1"
            );
        }
        set_dark(false);
    }

    #[test]
    fn type_scale_ascends() {
        let steps = [CAPTION, FOOTNOTE, BODY, HEADLINE, TITLE, DISPLAY];
        for pair in steps.windows(2) {
            assert!(pair[0].size < pair[1].size);
            assert!(pair[0].line_height < pair[1].line_height);
        }
        for step in steps {
            assert!(step.line_height >= step.size * 1.25);
        }
    }

    #[test]
    fn baseline_offset_lowers_smaller_text_onto_the_larger_baseline() {
        assert_eq!(baseline_offset(BODY, BODY), 0.0);
        let caption = baseline_offset(BODY, CAPTION);
        assert!((2.5..3.0).contains(&caption), "{caption}");
        assert!(baseline_offset(HEADLINE, CAPTION) > caption);
    }

    /// Raw styling values belong here. A pane that writes a literal colour,
    /// font size, line height, or radius, or truncates text by hand, bypasses
    /// the design system.
    #[test]
    fn desktop_sources_use_tokens_instead_of_raw_styles() {
        const SOURCES: &[(&str, &str)] = &[
            ("app_window.rs", include_str!("app_window.rs")),
            ("app_window/macos.rs", include_str!("app_window/macos.rs")),
            ("app_window/linux.rs", include_str!("app_window/linux.rs")),
            ("app_window/tour.rs", include_str!("app_window/tour.rs")),
            (
                "app_window/windows.rs",
                include_str!("app_window/windows.rs"),
            ),
            ("desktop_ui.rs", include_str!("desktop_ui.rs")),
            (
                "desktop_transcription_picker.rs",
                include_str!("desktop_transcription_picker.rs"),
            ),
            ("linux_app.rs", include_str!("linux_app.rs")),
            ("meeting_watcher.rs", include_str!("meeting_watcher.rs")),
            ("text_input.rs", include_str!("text_input.rs")),
        ];
        const FORBIDDEN: &[&str] = &[
            "rgb(0x",
            "rgba(0x",
            ".text_size(",
            ".line_height(px(",
            ".line_height(relative(",
            ".line_height(rems(",
            ".font_family(",
            ".rounded(px(",
            ".rounded_sm()",
            ".rounded_md()",
            ".rounded_lg()",
            ".text_xs()",
            ".text_sm()",
            ".text_base()",
            ".text_lg()",
            ".text_xl()",
            // One-line text goes through `desktop_ui::truncated_line`.
            ".truncate()",
        ];
        let mut violations = Vec::new();
        for (file, source) in SOURCES {
            for (index, line) in source.lines().enumerate() {
                for pattern in FORBIDDEN {
                    if line.contains(pattern) {
                        violations.push(format!("{file}:{}: {}", index + 1, line.trim()));
                    }
                }
            }
        }
        assert!(
            violations.is_empty(),
            "use design_tokens instead of raw styles:\n{}",
            violations.join("\n")
        );
    }
}
