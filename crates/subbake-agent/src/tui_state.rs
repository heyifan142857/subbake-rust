use crate::engine::{ProfileChoice, SessionChoice};
use crate::tui::TuiAction;
use crate::{ConfigChange, ConfigEditorSnapshot, ConfigFieldId, ConfigFieldKind, ConfigSection};

pub(crate) const APPROVAL_OPTIONS: &[(&str, &str)] = &[
    ("approve", "execute the pending plan"),
    ("reject", "discard the pending plan"),
    (
        "tell agent what to do",
        "revise the plan with your instructions",
    ),
];
pub(crate) const COMMAND_APPROVAL_OPTIONS: &[(&str, &str)] = &[
    ("approve", "run the exact sandboxed command"),
    ("reject", "discard the pending command"),
];

pub(crate) struct TuiPicker {
    pub options: Vec<ProfileChoice>,
}

pub(crate) struct SessionPicker {
    pub options: Vec<SessionChoice>,
    pub cancel_exits: bool,
}

pub(crate) enum InputMode {
    Editing,
    BrowsingHistory { index: usize, draft: String },
    ChoosingProfile(TuiPicker),
    CreatingProfile,
    ChoosingConfigProfile(TuiPicker),
    CreatingConfigProfile,
    ConfigEditor,
    ChoosingSession(SessionPicker),
    AwaitingPlanDecision,
    AwaitingCommandDecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigFocus {
    Sections,
    Fields,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ConfigConfirm {
    Close,
    SwitchProfile(String),
}

#[derive(Debug, Clone)]
pub(crate) struct ConfigEditorState {
    pub snapshot: ConfigEditorSnapshot,
    pub section_index: usize,
    pub field_index: usize,
    pub focus: ConfigFocus,
    pub changes: Vec<ConfigChange>,
    pub editing_field: Option<ConfigFieldId>,
    pub confirm: Option<ConfigConfirm>,
}

impl ConfigEditorState {
    pub fn new(snapshot: ConfigEditorSnapshot) -> Self {
        Self {
            snapshot,
            section_index: 0,
            field_index: 0,
            focus: ConfigFocus::Sections,
            changes: Vec::new(),
            editing_field: None,
            confirm: None,
        }
    }

    pub fn section(&self) -> ConfigSection {
        ConfigSection::ALL[self.section_index.min(ConfigSection::ALL.len() - 1)]
    }

    pub fn field_ids(&self) -> Vec<ConfigFieldId> {
        self.snapshot
            .fields
            .iter()
            .filter(|field| field.id.section() == self.section())
            .map(|field| field.id)
            .collect()
    }

    pub fn selected_field(&self) -> Option<ConfigFieldId> {
        let fields = self.field_ids();
        fields
            .get(self.field_index.min(fields.len().saturating_sub(1)))
            .copied()
    }

    pub fn value(&self, id: ConfigFieldId) -> String {
        if let Some(change) = self.changes.iter().rev().find(|change| change.id == id) {
            return change.value.clone().unwrap_or_default();
        }
        self.snapshot
            .fields
            .iter()
            .find(|field| field.id == id)
            .map(|field| field.value.clone())
            .unwrap_or_default()
    }

    pub fn set_value(&mut self, id: ConfigFieldId, value: Option<String>) {
        self.changes.retain(|change| change.id != id);
        self.changes.push(ConfigChange { id, value });
    }

    pub fn is_dirty(&self) -> bool {
        !self.changes.is_empty()
    }

    pub fn cycle_selected(&mut self, backwards: bool) {
        let Some(id) = self.selected_field() else {
            return;
        };
        match id.kind() {
            ConfigFieldKind::Boolean => {
                let next = (self.value(id) != "true").to_string();
                self.set_value(id, Some(next));
            }
            ConfigFieldKind::Choice(options) if !options.is_empty() => {
                let current = self.value(id);
                let index = options
                    .iter()
                    .position(|option| *option == current)
                    .unwrap_or(0);
                let next = if backwards {
                    index.checked_sub(1).unwrap_or(options.len() - 1)
                } else {
                    (index + 1) % options.len()
                };
                self.set_value(id, Some(options[next].to_owned()));
            }
            _ => {}
        }
    }
}

pub(crate) enum InteractionState {
    Idle {
        input_mode: InputMode,
    },
    Processing {
        // Processing always renders an editing input; pickers and forms are
        // therefore structurally unavailable while a worker is active.
        input_mode: InputMode,
        plan_mode_rollback: Option<bool>,
        cancellation_requested: bool,
    },
}

impl Default for InteractionState {
    fn default() -> Self {
        Self::Idle {
            input_mode: InputMode::Editing,
        }
    }
}

impl InteractionState {
    pub fn input_mode(&self) -> &InputMode {
        match self {
            Self::Idle { input_mode } | Self::Processing { input_mode, .. } => input_mode,
        }
    }

    pub fn set_input_mode(&mut self, mode: InputMode) {
        match self {
            Self::Idle { input_mode } => *input_mode = mode,
            Self::Processing { .. } => {
                debug_assert!(false, "input modes cannot change while processing");
            }
        }
    }

    pub fn is_processing(&self) -> bool {
        matches!(self, Self::Processing { .. })
    }

    pub fn begin_processing(&mut self, plan_mode_rollback: Option<bool>) {
        *self = Self::Processing {
            input_mode: InputMode::Editing,
            plan_mode_rollback,
            cancellation_requested: false,
        };
    }

    pub fn request_cancellation(&mut self) -> bool {
        let Self::Processing {
            cancellation_requested,
            ..
        } = self
        else {
            return false;
        };
        if *cancellation_requested {
            return false;
        }
        *cancellation_requested = true;
        true
    }

    pub fn finish(&mut self) -> Option<bool> {
        let rollback = match self {
            Self::Idle { .. } => None,
            Self::Processing {
                plan_mode_rollback, ..
            } => *plan_mode_rollback,
        };
        *self = Self::default();
        rollback
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VerticalNavigation {
    Selection(usize),
    History,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ApprovalChoice {
    Submit(TuiAction),
    Revise,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProfilePickerChoice {
    Select(String),
    Create,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmptyModeChoice {
    Submit(TuiAction),
    RevisePlan,
    CreateProfile,
}

pub(crate) fn approval_choice(index: usize) -> ApprovalChoice {
    match index.min(APPROVAL_OPTIONS.len() - 1) {
        0 => ApprovalChoice::Submit(TuiAction::ApprovePlan),
        1 => ApprovalChoice::Submit(TuiAction::RejectPlan),
        _ => ApprovalChoice::Revise,
    }
}

pub(crate) fn vertical_navigation(mode: &InputMode, suggestion_count: usize) -> VerticalNavigation {
    match mode {
        InputMode::ChoosingProfile(picker) if !picker.options.is_empty() => {
            VerticalNavigation::Selection(picker.options.len())
        }
        InputMode::ChoosingConfigProfile(picker) if !picker.options.is_empty() => {
            VerticalNavigation::Selection(picker.options.len())
        }
        InputMode::ChoosingSession(picker) if !picker.options.is_empty() => {
            VerticalNavigation::Selection(picker.options.len())
        }
        InputMode::AwaitingPlanDecision if suggestion_count > 0 => {
            VerticalNavigation::Selection(suggestion_count)
        }
        InputMode::AwaitingCommandDecision if suggestion_count > 0 => {
            VerticalNavigation::Selection(suggestion_count)
        }
        InputMode::Editing if suggestion_count > 0 => {
            VerticalNavigation::Selection(suggestion_count)
        }
        InputMode::Editing | InputMode::BrowsingHistory { .. } => VerticalNavigation::History,
        InputMode::ChoosingProfile(_)
        | InputMode::ChoosingConfigProfile(_)
        | InputMode::ChoosingSession(_)
        | InputMode::AwaitingPlanDecision
        | InputMode::AwaitingCommandDecision
        | InputMode::CreatingProfile
        | InputMode::CreatingConfigProfile
        | InputMode::ConfigEditor => VerticalNavigation::Disabled,
    }
}

pub(crate) fn empty_mode_choice(mode: &InputMode, index: usize) -> Option<EmptyModeChoice> {
    match mode {
        InputMode::AwaitingPlanDecision => match approval_choice(index) {
            ApprovalChoice::Submit(action) => Some(EmptyModeChoice::Submit(action)),
            ApprovalChoice::Revise => Some(EmptyModeChoice::RevisePlan),
        },
        InputMode::AwaitingCommandDecision => Some(EmptyModeChoice::Submit(if index == 0 {
            TuiAction::ApproveCommand
        } else {
            TuiAction::RejectCommand
        })),
        InputMode::ChoosingProfile(picker) => match profile_picker_choice(picker, index)? {
            ProfilePickerChoice::Select(name) => {
                Some(EmptyModeChoice::Submit(TuiAction::SelectProfile(name)))
            }
            ProfilePickerChoice::Create => Some(EmptyModeChoice::CreateProfile),
        },
        InputMode::ChoosingConfigProfile(picker) => match profile_picker_choice(picker, index)? {
            ProfilePickerChoice::Select(name) => Some(EmptyModeChoice::Submit(
                TuiAction::SelectConfigProfile(name),
            )),
            ProfilePickerChoice::Create => Some(EmptyModeChoice::CreateProfile),
        },
        InputMode::ChoosingSession(picker) => picker
            .options
            .get(index.min(picker.options.len().saturating_sub(1)))
            .map(|session| EmptyModeChoice::Submit(TuiAction::SelectSession(session.id.clone()))),
        _ => None,
    }
}

pub(crate) fn history_up(
    history: &[String],
    input: &str,
    mode: &InputMode,
) -> Option<(InputMode, String)> {
    if history.is_empty() {
        return None;
    }
    let (index, draft) = match mode {
        InputMode::BrowsingHistory { index, draft } => (index.saturating_sub(1), draft.clone()),
        _ => (history.len() - 1, input.to_owned()),
    };
    Some((
        InputMode::BrowsingHistory { index, draft },
        history[index].clone(),
    ))
}

pub(crate) fn history_down(history: &[String], mode: &InputMode) -> Option<(InputMode, String)> {
    let InputMode::BrowsingHistory { index, draft } = mode else {
        return None;
    };
    if index + 1 < history.len() {
        let next = index + 1;
        Some((
            InputMode::BrowsingHistory {
                index: next,
                draft: draft.clone(),
            },
            history[next].clone(),
        ))
    } else {
        Some((InputMode::Editing, draft.clone()))
    }
}

pub(crate) fn profile_picker_choice(
    picker: &TuiPicker,
    index: usize,
) -> Option<ProfilePickerChoice> {
    let option = picker
        .options
        .get(index.min(picker.options.len().saturating_sub(1)))?;
    if option.create {
        Some(ProfilePickerChoice::Create)
    } else {
        Some(ProfilePickerChoice::Select(option.name.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processing_transitions_are_reduced_without_tui_state() {
        let mut phase = InteractionState::default();
        phase.begin_processing(Some(false));
        assert!(phase.request_cancellation());
        assert!(!phase.request_cancellation());
        assert_eq!(phase.finish(), Some(false));
        assert!(matches!(phase, InteractionState::Idle { .. }));
    }

    #[test]
    fn command_approval_picker_is_typed_and_has_no_plan_revision_action() {
        assert_eq!(
            empty_mode_choice(&InputMode::AwaitingCommandDecision, 0),
            Some(EmptyModeChoice::Submit(TuiAction::ApproveCommand))
        );
        assert_eq!(
            empty_mode_choice(&InputMode::AwaitingCommandDecision, 1),
            Some(EmptyModeChoice::Submit(TuiAction::RejectCommand))
        );
    }

    #[test]
    fn config_boolean_fields_are_edited_as_typed_changes() {
        let snapshot = ConfigEditorSnapshot {
            path: std::path::PathBuf::from("subbake.toml"),
            target: subbake_adapters::ConfigEditTarget::Defaults,
            active_profile: None,
            profiles: Vec::new(),
            fields: vec![crate::ConfigFieldView {
                id: crate::ConfigFieldId::AgentAutoApprove,
                value: "false".to_owned(),
                inherited: true,
                configured: true,
            }],
        };
        let mut editor = ConfigEditorState::new(snapshot);
        editor.section_index = ConfigSection::ALL
            .iter()
            .position(|section| *section == ConfigSection::Agent)
            .expect("agent section");
        editor.cycle_selected(false);
        assert_eq!(editor.value(crate::ConfigFieldId::AgentAutoApprove), "true");
        assert!(editor.is_dirty());
    }
}
