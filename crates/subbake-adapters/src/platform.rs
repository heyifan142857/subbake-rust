use std::ffi::OsString;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingSystem {
    Linux,
    MacOs,
    Windows,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    X86_64,
    Aarch64,
    Other,
}

/// Capabilities derived from the compiled target, kept in one place so
/// feature registration and adapter selection cannot drift independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilitySet {
    operating_system: OperatingSystem,
    architecture: Architecture,
}

impl CapabilitySet {
    pub fn current() -> Self {
        Self::from_target(std::env::consts::OS, std::env::consts::ARCH)
    }

    pub fn from_target(os: &str, architecture: &str) -> Self {
        let operating_system = match os {
            "linux" => OperatingSystem::Linux,
            "macos" => OperatingSystem::MacOs,
            "windows" => OperatingSystem::Windows,
            _ => OperatingSystem::Other,
        };
        let architecture = match architecture {
            "x86_64" => Architecture::X86_64,
            "aarch64" => Architecture::Aarch64,
            _ => Architecture::Other,
        };
        Self {
            operating_system,
            architecture,
        }
    }

    pub const fn operating_system(self) -> OperatingSystem {
        self.operating_system
    }

    pub const fn architecture(self) -> Architecture {
        self.architecture
    }

    pub const fn supports_command_sandbox(self) -> bool {
        matches!(self.operating_system, OperatingSystem::Linux)
    }

    pub const fn has_prebuilt_whisper(self) -> bool {
        matches!(
            (self.operating_system, self.architecture),
            (
                OperatingSystem::Linux,
                Architecture::X86_64 | Architecture::Aarch64
            ) | (OperatingSystem::Windows, Architecture::X86_64)
        )
    }
}

/// Native home/config path resolution shared by configuration and destructive
/// path guards. Environment access remains at the adapter edge.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlatformPaths;

impl PlatformPaths {
    pub fn home_dir() -> Option<PathBuf> {
        home_dir_with(CapabilitySet::current().operating_system(), |key| {
            std::env::var_os(key)
        })
    }

    pub fn canonical_home_dir() -> Option<PathBuf> {
        let home = Self::home_dir()?;
        Some(home.canonicalize().unwrap_or(home))
    }

    pub fn config_dir() -> Option<PathBuf> {
        config_dir_with(CapabilitySet::current().operating_system(), |key| {
            std::env::var_os(key)
        })
    }
}

fn home_dir_with(
    operating_system: OperatingSystem,
    lookup: impl Fn(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    if operating_system == OperatingSystem::Windows {
        if let Some(profile) = lookup("USERPROFILE") {
            return Some(PathBuf::from(profile));
        }
        let drive = lookup("HOMEDRIVE")?;
        let path = lookup("HOMEPATH")?;
        let mut home = PathBuf::from(drive);
        home.push(path);
        return Some(home);
    }
    lookup("HOME").map(PathBuf::from)
}

fn config_dir_with(
    operating_system: OperatingSystem,
    lookup: impl Fn(&str) -> Option<OsString> + Copy,
) -> Option<PathBuf> {
    if operating_system == OperatingSystem::Windows
        && let Some(path) = lookup("APPDATA")
    {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = lookup("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(path));
    }
    home_dir_with(operating_system, lookup).map(|home| home.join(".config"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(values: &[(&str, &str)], key: &str) -> Option<OsString> {
        values
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| OsString::from(value))
    }

    #[test]
    fn windows_paths_use_native_environment_fallbacks() {
        let values = [
            ("HOME", "/home/msys-user"),
            ("XDG_CONFIG_HOME", "/home/msys-user/.config"),
            ("USERPROFILE", r"C:\Users\Alice"),
            ("APPDATA", r"C:\Config"),
        ];

        assert_eq!(
            home_dir_with(OperatingSystem::Windows, |key| lookup(&values, key)),
            Some(PathBuf::from(r"C:\Users\Alice"))
        );
        assert_eq!(
            config_dir_with(OperatingSystem::Windows, |key| lookup(&values, key)),
            Some(PathBuf::from(r"C:\Config"))
        );
    }

    #[test]
    fn capabilities_are_derived_from_one_target_identity() {
        let linux = CapabilitySet::from_target("linux", "aarch64");
        let mac = CapabilitySet::from_target("macos", "aarch64");
        let windows = CapabilitySet::from_target("windows", "x86_64");

        assert!(linux.supports_command_sandbox());
        assert!(linux.has_prebuilt_whisper());
        assert!(!mac.supports_command_sandbox());
        assert!(!mac.has_prebuilt_whisper());
        assert!(windows.has_prebuilt_whisper());
    }

    #[cfg(windows)]
    #[test]
    fn windows_home_drive_and_path_form_an_absolute_native_path() {
        let values = [("HOMEDRIVE", "C:"), ("HOMEPATH", r"\Users\Alice")];

        assert_eq!(
            home_dir_with(OperatingSystem::Windows, |key| lookup(&values, key)),
            Some(PathBuf::from(r"C:\Users\Alice"))
        );
    }
}
