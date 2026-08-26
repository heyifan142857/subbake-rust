use std::fmt::{Debug, Formatter};
use std::fs;
use std::path::Path;

use toml_edit::{DocumentMut, Item, Table};

use crate::config::{CONFIG_VERSION, ConfigFile};
use crate::error::{AdapterError, AdapterResult, ConfigError};
use crate::fs::write_file_atomically_with_permissions;

#[derive(Clone, PartialEq, Eq)]
pub enum ConfigEditTarget {
    Defaults,
    Profile(String),
}

impl Debug for ConfigEditTarget {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Defaults => formatter.write_str("Defaults"),
            Self::Profile(name) => formatter.debug_tuple("Profile").field(name).finish(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigScalar {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
}

#[derive(Clone, PartialEq)]
pub struct ConfigFieldUpdate {
    pub path: Vec<&'static str>,
    pub value: Option<ConfigScalar>,
    pub secret: bool,
}

impl Debug for ConfigFieldUpdate {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("ConfigFieldUpdate");
        debug.field("path", &self.path);
        if self.secret && self.value.is_some() {
            debug.field("value", &"[REDACTED]");
        } else {
            debug.field("value", &self.value);
        }
        debug.field("secret", &self.secret).finish()
    }
}

#[derive(Debug, Clone)]
pub struct PreparedConfigUpdate {
    config: ConfigFile,
    content: String,
}

impl PreparedConfigUpdate {
    pub fn config(&self) -> &ConfigFile {
        &self.config
    }

    pub fn commit(self, path: &Path) -> AdapterResult<()> {
        atomic_replace(path, &self.content)
    }
}

pub fn prepare_config_update(
    path: &Path,
    target: &ConfigEditTarget,
    updates: &[ConfigFieldUpdate],
) -> AdapterResult<PreparedConfigUpdate> {
    let content = if path.exists() {
        fs::read_to_string(path).map_err(|source| {
            AdapterError::external_io("read configuration", Some(path.to_path_buf()), source)
        })?
    } else {
        format!("version = {CONFIG_VERSION}\n")
    };
    let mut document =
        content
            .parse::<DocumentMut>()
            .map_err(|error| AdapterError::ConfigurationFile {
                path: path.to_path_buf(),
                source: ConfigError::invalid(error.to_string()),
            })?;

    for update in updates {
        let mut full_path = match target {
            ConfigEditTarget::Defaults => vec!["defaults".to_owned()],
            ConfigEditTarget::Profile(name) => {
                vec!["profiles".to_owned(), name.clone()]
            }
        };
        full_path.extend(update.path.iter().map(|part| (*part).to_owned()));
        apply_update(&mut document, &full_path, update.value.as_ref())?;
    }

    let rendered = document.to_string();
    let config =
        ConfigFile::parse(&rendered).map_err(|source| AdapterError::ConfigurationFile {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(PreparedConfigUpdate {
        config,
        content: rendered,
    })
}

fn apply_update(
    document: &mut DocumentMut,
    path: &[String],
    value: Option<&ConfigScalar>,
) -> AdapterResult<()> {
    let Some((leaf, parents)) = path.split_last() else {
        return Err(ConfigError::invalid("configuration field path cannot be empty").into());
    };
    let mut table = document.as_table_mut();
    for part in parents {
        let item = table
            .entry(part)
            .or_insert_with(|| Item::Table(Table::new()));
        if !item.is_table() {
            return Err(ConfigError::invalid(format!(
                "configuration path `{}` is not a table",
                parents.join(".")
            ))
            .into());
        }
        table = item
            .as_table_mut()
            .ok_or_else(|| ConfigError::invalid("configuration table is unavailable"))?;
    }
    match value {
        Some(value) => {
            let mut replacement = scalar_item(value);
            if let (Some(existing), Some(replacement_value)) = (
                table.get(leaf).and_then(Item::as_value),
                replacement.as_value_mut(),
            ) {
                *replacement_value.decor_mut() = existing.decor().clone();
            }
            table.insert(leaf, replacement);
        }
        None => {
            table.remove(leaf);
        }
    }
    Ok(())
}

fn scalar_item(value: &ConfigScalar) -> Item {
    match value {
        ConfigScalar::String(value) => toml_edit::value(value.clone()),
        ConfigScalar::Integer(value) => toml_edit::value(*value),
        ConfigScalar::Float(value) => toml_edit::value(*value),
        ConfigScalar::Boolean(value) => toml_edit::value(*value),
    }
}

fn atomic_replace(path: &Path, content: &str) -> AdapterResult<()> {
    write_file_atomically_with_permissions(path, content.as_bytes(), private_permissions())
}

#[cfg(unix)]
fn private_permissions() -> Option<fs::Permissions> {
    use std::os::unix::fs::PermissionsExt;
    Some(fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn private_permissions() -> Option<fs::Permissions> {
    None
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn temporary_path(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("subbake-config-edit-{label}-{nonce}.toml"))
    }

    #[test]
    fn preserves_comments_and_updates_only_the_selected_profile() {
        let path = temporary_path("comments");
        fs::write(
            &path,
            r#"version = 3
# keep this comment
[defaults.translation]
target_language = "French"

[profiles.work.backend]
model = "before" # keep inline
id = "mock"
"#,
        )
        .expect("write config");
        let prepared = prepare_config_update(
            &path,
            &ConfigEditTarget::Profile("work".to_owned()),
            &[ConfigFieldUpdate {
                path: vec!["backend", "model"],
                value: Some(ConfigScalar::String("after".to_owned())),
                secret: false,
            }],
        )
        .expect("prepare update");
        prepared.commit(&path).expect("commit update");
        let text = fs::read_to_string(&path).expect("read config");
        assert!(text.contains("# keep this comment"));
        assert!(text.contains("model = \"after\" # keep inline"));
        assert!(text.contains("target_language = \"French\""));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn creates_private_current_version_configuration() {
        let path = temporary_path("create");
        let prepared = prepare_config_update(
            &path,
            &ConfigEditTarget::Defaults,
            &[ConfigFieldUpdate {
                path: vec!["agent", "max_steps"],
                value: Some(ConfigScalar::Integer(32)),
                secret: false,
            }],
        )
        .expect("prepare update");
        prepared.commit(&path).expect("commit update");
        let config = ConfigFile::load(&path).expect("load config");
        assert_eq!(config.version, CONFIG_VERSION);
        assert_eq!(config.defaults.agent.max_steps, Some(32));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).expect("metadata").permissions().mode() & 0o777,
                0o600
            );
        }
        let _ = fs::remove_file(path);
    }
}
