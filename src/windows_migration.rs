//! One-time migration of the Windows beta's `windows-settings.json` into the
//! shared `settings.json`. Only [`legacy_settings`] reads this format.

use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};
use serde::Deserialize;

use crate::app_settings::{
    AppSettings, DictationMode, DictationPostProcessing, HotkeyBinding, IndicatorPosition,
    RecordingAudioBehavior, TextReplacement,
};
use crate::windows_input::WindowsHotkey;

#[derive(Deserialize)]
#[serde(default)]
struct LegacySettings {
    microphone: Option<String>,
    listen_on_launch: bool,
    dictation_hotkey: WindowsHotkey,
    shortcut_mode: crate::dictation::ShortcutMode,
    paste_last_hotkey: Option<WindowsHotkey>,
    voice_action_hotkey: Option<WindowsHotkey>,
    voice_action_model: Option<crate::dictation_processor::Model>,
    double_tap_lock: bool,
    double_tap_only: bool,
    while_dictating: RecordingAudioBehavior,
    release_microphone_while_idle: bool,
    feedback_volume: u8,
    ui_language: Option<String>,
    indicator_position: IndicatorPosition,
    history_retention: crate::history::HistoryRetention,
    text_replacements: Vec<TextReplacement>,
    dictation_post_processing: DictationPostProcessing,
    dictation_transformations: Vec<String>,
    modes: Vec<LegacyMode>,
    transcription: Option<crate::transcription_models::TranscriptionSelection>,
}

impl Default for LegacySettings {
    /// The Windows beta's defaults, for fields a file predates.
    fn default() -> Self {
        Self {
            microphone: None,
            listen_on_launch: true,
            dictation_hotkey: WindowsHotkey::default(),
            shortcut_mode: crate::dictation::ShortcutMode::default(),
            paste_last_hotkey: Some(WindowsHotkey::paste_last_default()),
            voice_action_hotkey: None,
            voice_action_model: None,
            double_tap_lock: true,
            double_tap_only: false,
            while_dictating: RecordingAudioBehavior::DoNothing,
            release_microphone_while_idle: false,
            feedback_volume: 50,
            ui_language: None,
            indicator_position: IndicatorPosition::default(),
            history_retention: crate::history::HistoryRetention::default(),
            text_replacements: Vec::new(),
            dictation_post_processing: DictationPostProcessing::default(),
            dictation_transformations: Vec::new(),
            modes: Vec::new(),
            transcription: None,
        }
    }
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct LegacyMode {
    name: String,
    applications: Vec<String>,
    websites: Vec<String>,
    corrections: Vec<TextReplacement>,
    post_processing: DictationPostProcessing,
    transformations: Vec<String>,
}

impl LegacySettings {
    fn apply_to(self, settings: &mut AppSettings) {
        settings.microphone = self.microphone;
        settings.enable_on_launch = self.listen_on_launch;
        settings.dictation_hotkey = HotkeyBinding::from(&self.dictation_hotkey);
        settings.shortcut_mode = self.shortcut_mode;
        settings.paste_last_hotkey = self.paste_last_hotkey.as_ref().map(HotkeyBinding::from);
        settings.voice_action.enabled = self.voice_action_hotkey.is_some();
        if let Some(hotkey) = &self.voice_action_hotkey {
            settings.edit_hotkey = HotkeyBinding::from(hotkey);
        }
        if let Some(model) = self.voice_action_model {
            settings.voice_action.model = Some(format!("{}/{}", model.provider, model.id));
            settings.voice_action.variant = model.variant;
        }
        settings.double_tap_lock = self.double_tap_lock;
        settings.double_tap_only = self.double_tap_only;
        settings.recording_audio_behavior = self.while_dictating;
        settings.release_microphone_while_idle = self.release_microphone_while_idle;
        // Muting keeps the remembered volume for when sounds return.
        settings.sound_effects = self.feedback_volume > 0;
        if self.feedback_volume > 0 {
            settings.sound_effect_volume = f32::from(self.feedback_volume.min(100)) / 100.0;
        }
        settings.ui_language = self.ui_language;
        settings.indicator_position = self.indicator_position;
        settings.history_retention = self.history_retention;
        let processing = &mut settings.dictation_processing;
        processing.default_mode.replacements = self.text_replacements;
        processing.default_mode.post_processing = self.dictation_post_processing;
        processing.default_mode.transformations = self.dictation_transformations;
        processing.modes = self
            .modes
            .into_iter()
            .map(|mode| DictationMode {
                name: mode.name,
                applications: mode.applications,
                browser_hosts: mode.websites,
                replacements: mode.corrections,
                transformations: mode.transformations,
                post_processing: mode.post_processing,
            })
            .collect();
        if let Some(transcription) = self.transcription {
            settings.transcription = transcription;
        }
    }
}

/// The legacy `windows-settings.json` in `directory`, converted, for a
/// one-time migration into `settings.json`.
pub(crate) fn legacy_settings(directory: &Path) -> Result<Option<(PathBuf, AppSettings)>> {
    let path = directory.join("windows-settings.json");
    if !path.exists() {
        return Ok(None);
    }
    let legacy: LegacySettings = serde_json::from_slice(&fs::read(&path)?)
        .wrap_err_with(|| format!("could not parse {}", path.display()))?;
    let mut settings = AppSettings::default();
    legacy.apply_to(&mut settings);
    Ok(Some((path, settings)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_directory(label: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "hex-windows-migration-{label}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn label(binding: &HotkeyBinding) -> String {
        binding.keycaps().join("+")
    }

    #[test]
    fn legacy_windows_settings_migrate_once_into_settings_json() {
        let directory = scratch_directory("once");
        fs::write(
            directory.join("windows-settings.json"),
            br#"{
                "schema_version": 1,
                "platform": "windows",
                "microphone": "Headset",
                "listen_on_launch": false,
                "dictation_hotkey": {"control": true, "windows": true, "alt": false, "shift": false, "key": null},
                "shortcut_mode": "toggle",
                "paste_last_hotkey": {"control": true, "windows": false, "alt": true, "shift": false, "key": "v"},
                "voice_action_hotkey": {"control": true, "windows": false, "alt": true, "shift": false, "key": "i"},
                "voice_action_model": {"providerID": "openai", "id": "gpt-5.6-sol", "variant": "fast"},
                "double_tap_lock": false,
                "while_dictating": "pause_media",
                "feedback_volume": 30,
                "ui_language": "pl",
                "indicator_position": "bottom",
                "history_retention": "day",
                "text_replacements": [{"matched_phrase": "open code", "output": "OpenCode"}],
                "dictation_transformations": ["lowercase"],
                "modes": [{"name": "Code", "applications": ["code"], "websites": ["github.com"],
                           "corrections": [{"matched_phrase": "rust", "output": "Rust"}]}],
                "transcription": {"model": "parakeet_v3", "language": "pl", "recognition_hints": ""}
            }"#,
        )
        .unwrap();
        let path = directory.join("settings.json");

        crate::app_settings::migrate_legacy_settings(&path).unwrap();

        assert!(!directory.join("windows-settings.json").exists());
        assert!(directory.join("windows-settings.json.migrated").is_file());
        let migrated: AppSettings = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(migrated.microphone.as_deref(), Some("Headset"));
        assert!(!migrated.enable_on_launch);
        assert_eq!(label(&migrated.dictation_hotkey), "Ctrl+Win");
        assert_eq!(
            label(migrated.paste_last_hotkey.as_ref().unwrap()),
            "Ctrl+Alt+V"
        );
        assert!(migrated.voice_action.enabled);
        assert_eq!(label(&migrated.edit_hotkey), "Ctrl+Alt+I");
        assert_eq!(
            migrated.voice_action.model.as_deref(),
            Some("openai/gpt-5.6-sol")
        );
        assert_eq!(migrated.voice_action.variant.as_deref(), Some("fast"));
        assert_eq!(
            migrated.shortcut_mode,
            crate::dictation::ShortcutMode::Toggle
        );
        assert!(!migrated.double_tap_lock);
        assert_eq!(
            migrated.recording_audio_behavior,
            RecordingAudioBehavior::PauseMedia
        );
        assert!(migrated.sound_effects);
        assert!((migrated.sound_effect_volume - 0.3).abs() < 1e-6);
        assert_eq!(migrated.ui_language.as_deref(), Some("pl"));
        assert_eq!(migrated.indicator_position, IndicatorPosition::Bottom);
        let processing = &migrated.dictation_processing;
        assert_eq!(processing.default_mode.replacements[0].output, "OpenCode");
        assert_eq!(processing.default_mode.transformations, ["lowercase"]);
        assert_eq!(processing.modes[0].browser_hosts, ["github.com"]);
        assert_eq!(processing.modes[0].replacements[0].output, "Rust");
        assert_eq!(migrated.transcription.language, "pl");

        // A second launch reads settings.json and leaves the backup alone.
        fs::write(directory.join("windows-settings.json"), b"{}").unwrap();
        crate::app_settings::migrate_legacy_settings(&path).unwrap();
        assert!(directory.join("windows-settings.json").is_file());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn an_english_beta_loads_as_parakeet_english() {
        let directory = scratch_directory("streaming");
        fs::write(directory.join("windows-settings.json"), br#"{"transcription": {"model": "parakeet_v3", "language": "en", "recognition_hints": ""}}"#).unwrap();
        let path = directory.join("settings.json");
        crate::app_settings::migrate_legacy_settings(&path).unwrap();
        let settings = AppSettings::load_from(&path).unwrap();
        assert_eq!(
            settings.transcription.model,
            crate::transcription_models::TranscriptionModelId::ParakeetUnifiedEnglish
        );
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn files_that_predate_a_setting_keep_the_beta_defaults() {
        let legacy: LegacySettings = serde_json::from_str(r#"{"platform":"windows"}"#).unwrap();
        let mut settings = AppSettings::default();
        legacy.apply_to(&mut settings);
        assert!(settings.enable_on_launch);
        assert_eq!(label(&settings.dictation_hotkey), "Ctrl+Space");
        assert_eq!(
            label(settings.paste_last_hotkey.as_ref().unwrap()),
            "Ctrl+Alt+V"
        );
        assert!(!settings.voice_action.enabled);
        assert!(settings.double_tap_lock);
        assert!(settings.sound_effects);
        assert_eq!(settings.sound_effect_volume, 0.5);
    }

    #[test]
    fn a_muted_beta_keeps_the_remembered_volume() {
        let legacy: LegacySettings = serde_json::from_str(r#"{"feedback_volume":0}"#).unwrap();
        let mut settings = AppSettings::default();
        legacy.apply_to(&mut settings);
        assert!(!settings.sound_effects);
        assert_eq!(settings.sound_effect_volume, 0.5);
    }
}
