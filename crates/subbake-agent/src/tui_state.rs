use crate::engine::{ProfileChoice, SessionChoice};
use crate::tui::TuiAction;

pub(crate) const APPROVAL_OPTIONS: &[(&str, &str)] = &[
    ("approve", "execute the pending plan"),
    ("reject", "discard the pending plan"),
    (
        "tell agent what to do",
        "revise the plan with your instructions",
    ),
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
    ChoosingSession(SessionPicker),
    AwaitingPlanDecision,
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
        InputMode::ChoosingSession(picker) if !picker.options.is_empty() => {
            VerticalNavigation::Selection(picker.options.len())
        }
        InputMode::AwaitingPlanDecision if suggestion_count > 0 => {
            VerticalNavigation::Selection(suggestion_count)
        }
        InputMode::Editing if suggestion_count > 0 => {
            VerticalNavigation::Selection(suggestion_count)
        }
        InputMode::Editing | InputMode::BrowsingHistory { .. } => VerticalNavigation::History,
        InputMode::ChoosingProfile(_)
        | InputMode::ChoosingSession(_)
        | InputMode::AwaitingPlanDecision
        | InputMode::CreatingProfile => VerticalNavigation::Disabled,
    }
}

pub(crate) fn empty_mode_choice(mode: &InputMode, index: usize) -> Option<EmptyModeChoice> {
    match mode {
        InputMode::AwaitingPlanDecision => match approval_choice(index) {
            ApprovalChoice::Submit(action) => Some(EmptyModeChoice::Submit(action)),
            ApprovalChoice::Revise => Some(EmptyModeChoice::RevisePlan),
        },
        InputMode::ChoosingProfile(picker) => match profile_picker_choice(picker, index)? {
            ProfilePickerChoice::Select(name) => {
                Some(EmptyModeChoice::Submit(TuiAction::SelectProfile(name)))
            }
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
}
