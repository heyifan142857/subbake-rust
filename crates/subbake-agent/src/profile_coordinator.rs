use std::path::{Path, PathBuf};

use subbake_adapters::{
    ConfigFile, SettingsOverrides, TranslationSettings, append_profile_snapshot,
};

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

    pub(crate) fn active_settings(&self) -> AgentResult<TranslationSettings> {
        let Some((_, config)) = self.load_config()? else {
            return Ok(TranslationSettings::default());
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
    ) -> AgentResult<TranslationSettings> {
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
}
