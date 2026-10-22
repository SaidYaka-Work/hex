//! One-time migration of the Linux beta's `linux-settings.json` into the
//! shared `settings.json`. Only [`legacy_settings`] reads this format.

use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};
use serde::Deserialize;

use crate::app_settings::{AppSettings, HotkeyBinding};
use crate::linux_input::LinuxHotkey;

#[derive(Deserialize)]
#[serde(default)]
struct LegacySettings {
    dictation_hotkey: LinuxHotkey,
    double_tap_lock: bool,
    shortcut_mode: crate::dictation::ShortcutMode,
    /// Files that predate this preference always pasted with Ctrl-Shift-V.
    paste_with_shift: bool,
    sound_effect_volume: f32,
    transcription: crate::transcription_models::TranscriptionSelection,
}

impl Default for LegacySettings {
    fn default() -> Self {
        Self {
            dictation_hotkey: LinuxHotkey::default(),
            double_tap_lock: true,
            shortcut_mode: crate::dictation::ShortcutMode::Hold,
            paste_with_shift: true,
            sound_effect_volume: 0.5,
            transcription: crate::transcription_models::TranscriptionSelection::default(),
        }
    }
}

impl LegacySettings {
    fn apply_to(self, settings: &mut AppSettings) {
        settings.dictation_hotkey = HotkeyBinding::from(&self.dictation_hotkey);
        settings.double_tap_lock = self.double_tap_lock;
        settings.shortcut_mode = self.shortcut_mode;
        settings.paste_with_shift = self.paste_with_shift;
        // Muting keeps the remembered volume for when sounds return.
        let volume = self.sound_effect_volume;
        settings.sound_effects = volume > 0.0;
        if volume > 0.0 && volume <= 1.0 {
            settings.sound_effect_volume = volume;
        }
        settings.transcription = self.transcription;
    }
}

/// The legacy `linux-settings.json` in `directory`, converted, for a
/// one-time migration into `settings.json`.
pub(crate) fn legacy_settings(directory: &Path) -> Result<Option<(PathBuf, AppSettings)>> {
    let path = directory.join("linux-settings.json");
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
            "hex-linux-migration-{label}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn migrated(legacy: &str) -> AppSettings {
        let mut settings = AppSettings::default();
        serde_json::from_str::<LegacySettings>(legacy)
            .unwrap()
            .apply_to(&mut settings);
        settings
    }

    #[test]
    fn legacy_linux_settings_migrate_once_into_settings_json() {
        let directory = scratch_directory("once");
        fs::write(
            directory.join("linux-settings.json"),
            br#"{"schema_version":1,"platform":"x11","dictation_hotkey":{"control":true,"alt":false,"shift":false,"super_key":true,"key":"f12"},"double_tap_lock":false,"shortcut_mode":"toggle","sound_effect_volume":0.25}"#,
        )
        .unwrap();
        let path = directory.join("settings.json");

        crate::app_settings::migrate_legacy_settings(&path).unwrap();

        assert!(!directory.join("linux-settings.json").exists());
        assert!(directory.join("linux-settings.json.migrated").is_file());
        let migrated: AppSettings = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            migrated.dictation_hotkey.keycaps().join("+"),
            "Ctrl+Super+F12"
        );
        assert!(!migrated.double_tap_lock);
        assert_eq!(
            migrated.shortcut_mode,
            crate::dictation::ShortcutMode::Toggle
        );
        // Settings that predate the preference keep the terminal paste.
        assert!(migrated.paste_with_shift);
        assert!(migrated.sound_effects);
        assert_eq!(migrated.sound_effect_volume, 0.25);

        // A second launch reads settings.json and leaves the backup alone.
        fs::write(directory.join("linux-settings.json"), b"{}").unwrap();
        crate::app_settings::migrate_legacy_settings(&path).unwrap();
        assert!(directory.join("linux-settings.json").is_file());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn an_english_beta_moves_to_the_streaming_model_only_when_installed() {
        let directory = scratch_directory("streaming");
        fs::write(directory.join("linux-settings.json"), br#"{"transcription": {"model": "parakeet_v3", "language": "en", "recognition_hints": ""}}"#).unwrap();
        let path = directory.join("settings.json");
        crate::app_settings::migrate_legacy_settings(&path).unwrap();
        let settings = AppSettings::load_from(&path).unwrap();
        // Load pins every saved selection to Parakeet English, installed or not.
        // The first-run tour downloads that model.
        assert_eq!(
            settings.transcription.model,
            crate::transcription_models::TranscriptionModelId::ParakeetUnifiedEnglish
        );
        assert_eq!(settings.transcription.language, "en");
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn files_that_predate_a_setting_keep_the_beta_defaults() {
        let settings = migrated("{}");
        assert_eq!(settings.dictation_hotkey.keycaps().join("+"), "Alt+Space");
        assert!(settings.double_tap_lock);
        assert_eq!(settings.shortcut_mode, crate::dictation::ShortcutMode::Hold);
        assert!(settings.paste_with_shift);
        assert_eq!(
            settings.transcription,
            crate::transcription_models::TranscriptionSelection::default()
        );
        assert!(!migrated(r#"{"paste_with_shift":false}"#).paste_with_shift);
    }

    #[test]
    fn a_muted_beta_keeps_the_remembered_volume() {
        let settings = migrated(r#"{"sound_effect_volume":0.0}"#);
        assert!(!settings.sound_effects);
        assert_eq!(settings.sound_effect_volume, 0.5);
    }
}
