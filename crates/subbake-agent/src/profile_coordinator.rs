use std::path::{Path, PathBuf};

use subbake_adapters::{
    CONFIG_VERSION, ConfigEditTarget, ConfigFile, PreparedConfigUpdate, ResolvedSettings,
    SettingsOverrides, append_profile_snapshot, prepare_config_update,
};

use crate::config_editor::{ConfigChange, ConfigEditorSnapshot, build_snapshot};
use crate::error::AgentResult;
use crate::presentation::ProfileChoice;
use crate::session::AgentSession;

pub(crate) struct ProfileCoordinator<'a> {
    project_root: &'a Path,
    session: Option<&'a AgentSession>,
}

impl<'a> ProfileCoordinator<'a> {
    pub(crate) fn new(project_root: &'a Path, session: Option<&'a AgentSession>) -> Self {
        Self {
            project_root,
            session,
        }
    }

    pub(crate) fn load_config(&self) -> AgentResult<Option<(PathBuf, ConfigFile)>> {
        if let Some(path) = self
            .session
            .and_then(|session| session.config_path.as_deref())
            .map(PathBuf::from)
        {
            return Ok(Some((path.clone(), ConfigFile::load(&path)?)));
        }
        for path in [
            self.project_root.join("subbake.toml"),
            self.project_root.join(".subbake.toml"),
        ] {
            if path.exists() {
                return Ok(Some((path.clone(), ConfigFile::load(&path)?)));
            }
        }
        Ok(None)
    }

    pub(crate) fn active_settings(&self) -> AgentResult<ResolvedSettings> {
        let Some((_, config)) = self.load_config()? else {
            return Ok(ResolvedSettings::default());
        };
        self.settings_for_profile(
            &config,
            self.session.and_then(|session| session.profile.as_deref()),
        )
    }

    pub(crate) fn settings_for_profile(
        &self,
        config: &ConfigFile,
        profile: Option<&str>,
    ) -> AgentResult<ResolvedSettings> {
        config
            .resolve(profile, SettingsOverrides::default())
            .map(|(settings, _)| settings)
            .map_err(subbake_adapters::AdapterError::from)
            .map_err(Into::into)
    }

    pub(crate) fn create_snapshot(&self, name: &str) -> AgentResult<String> {
        let Some((path, config)) = self.load_config()? else {
            return Ok("No subbake config found. Create one before adding a profile.".to_owned());
        };
        let active = self.session.and_then(|session| session.profile.as_deref());
        let settings = self.settings_for_profile(&config, active)?;
        append_profile_snapshot(&path, name, &settings)?;
        Ok(format!(
            "Created profile `{name}` from the active settings. Inline credentials were not copied; review it, then select it with `/profile {name}`."
        ))
    }

    pub(crate) fn names(&self) -> AgentResult<Vec<String>> {
        let Some((_, config)) = self.load_config()? else {
            return Ok(Vec::new());
        };
        let mut profiles = config.profiles.keys().cloned().collect::<Vec<_>>();
        profiles.sort();
        Ok(profiles)
    }

    pub(crate) fn picker_choices(&self) -> AgentResult<Vec<ProfileChoice>> {
        let Some((_, config)) = self.load_config()? else {
            return Ok(Vec::new());
        };
        let active = self
            .session
            .and_then(|session| session.profile.as_deref())
            .or(config.default_profile.as_deref());
        let mut profiles = config
            .profiles
            .keys()
            .map(|name| {
                let settings = self.settings_for_profile(&config, Some(name))?;
                Ok(ProfileChoice {
                    name: name.clone(),
                    provider: settings.backend.id,
                    model: settings.backend.model,
                    active: active == Some(name.as_str()),
                    create: false,
                })
            })
            .collect::<AgentResult<Vec<_>>>()?;
        profiles.sort_by(|left, right| left.name.cmp(&right.name));
        profiles.push(ProfileChoice {
            name: "new profile…".to_owned(),
            provider: String::new(),
            model: "copy active settings without credentials".to_owned(),
            active: false,
            create: true,
        });
        Ok(profiles)
    }

    pub(crate) fn editor_snapshot(&self) -> AgentResult<ConfigEditorSnapshot> {
        let (path, config) = self.load_config()?.unwrap_or_else(|| {
            (
                self.project_root.join("subbake.toml"),
                ConfigFile {
                    version: CONFIG_VERSION,
                    default_profile: None,
                    defaults: SettingsOverrides::default(),
                    backends: std::collections::HashMap::new(),
                    profiles: std::collections::HashMap::new(),
                },
            )
        });
        let profiles = picker_choices_for(
            &config,
            self.session.and_then(|session| session.profile.as_deref()),
        )?;
        build_snapshot(
            path,
            &config,
            self.session.and_then(|session| session.profile.as_deref()),
            profiles,
        )
    }

    pub(crate) fn prepare_editor_update(
        &self,
        changes: Vec<ConfigChange>,
    ) -> AgentResult<(PathBuf, ConfigEditTarget, PreparedConfigUpdate)> {
        let snapshot = self.editor_snapshot()?;
        let updates = changes
            .into_iter()
            .map(ConfigChange::into_update)
            .collect::<AgentResult<Vec<_>>>()?;
        let prepared = prepare_config_update(&snapshot.path, &snapshot.target, &updates)?;
        Ok((snapshot.path, snapshot.target, prepared))
    }
}

fn picker_choices_for(
    config: &ConfigFile,
    requested_profile: Option<&str>,
) -> AgentResult<Vec<ProfileChoice>> {
    let active = config
        .selected_profile(requested_profile)
        .map_err(subbake_adapters::AdapterError::from)?;
    let mut profiles = config
        .profiles
        .keys()
        .map(|name| {
            let (settings, _) = config
                .resolve(Some(name), SettingsOverrides::default())
                .map_err(subbake_adapters::AdapterError::from)?;
            Ok(ProfileChoice {
                name: name.clone(),
                provider: settings.backend.id,
                model: settings.backend.model,
                active: active == Some(name.as_str()),
                create: false,
            })
        })
        .collect::<AgentResult<Vec<_>>>()?;
    profiles.sort_by(|left, right| left.name.cmp(&right.name));
    profiles.push(ProfileChoice {
        name: "new profile…".to_owned(),
        provider: String::new(),
        model: "copy active settings without credentials".to_owned(),
        active: false,
        create: true,
    });
    Ok(profiles)
}
