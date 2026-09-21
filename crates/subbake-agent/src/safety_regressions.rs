//! Regression coverage for authorization, undo, and interrupted mutations.
use std::fs;
use std::path::Path;
use std::sync::Arc;

use serde_json::json;
use subbake_core::{CancellationGuard, CancellationToken, ProgressEvent, ProgressSink, TaskKind};

#[cfg(unix)]
use crate::FileGuard;
use crate::event::ToolCallDraft;
use crate::{AgentEngine, EngineObserver};

const SUBTITLE: &str = "1\n00:00:05,000 --> 00:00:07,000\nHello.\n";

fn engine(root: &Path) -> AgentEngine {
    fs::write(
        root.join("subbake.toml"),
        "version = 3\n[defaults.backend]\nid = \"mock\"\nmodel = \"mock-zh\"\n",
    )
    .expect("mock config");
    let mut engine = AgentEngine::new(root.to_path_buf()).expect("engine");
    engine.start_session().expect("session");
    engine
}

fn create_file(engine: &mut AgentEngine, path: &str) {
    engine.run_tool("apply_patch", &json!({"patch": format!("*** Begin Patch\n*** Add File: {path}\n+original\n*** End Patch")})).expect("create file");
}

fn creation_plan(engine: &mut AgentEngine) {
    engine.store_plan("create file", vec![ToolCallDraft {
        tool_name: "apply_patch".into(),
        arguments: json!({"patch": "*** Begin Patch\n*** Add File: output.txt\n+original\n*** End Patch"}),
    }]).expect("plan");
}

#[cfg(unix)]
#[test]
fn batch_uses_only_authorized_inputs_and_never_rescans_protected_paths() {
    let root = tempfile::tempdir().expect("project");
    let outside = tempfile::tempdir().expect("outside");
    fs::create_dir(root.path().join(".ssh")).expect("protected directory");
    fs::write(root.path().join(".ssh/protected.srt"), SUBTITLE).expect("protected subtitle");
    fs::write(outside.path().join("private.srt"), SUBTITLE).expect("outside subtitle");
    std::os::unix::fs::symlink(outside.path(), root.path().join("linked"))
        .expect("directory alias");
    let mut engine = engine(root.path());
    // An empty authorized selection is explicit, not permission to rediscover.
    let outcome = engine
        .run_tool("translate_series", &json!({"path":".", "recursive":true}))
        .expect("empty batch");
    let subbake_core::AgentToolOutcome::Translation(facts) = outcome else {
        panic!("translation outcome")
    };
    assert_eq!(facts.processed_files, 0);
    assert!(!outside.path().join("private.translated.srt").exists());
    assert!(!root.path().join(".ssh/protected.translated.srt").exists());
    fs::write(root.path().join("allowed.srt"), SUBTITLE).expect("allowed subtitle");
    engine
        .run_tool("translate_series", &json!({"path":".", "recursive":true}))
        .expect("selected batch");
    assert!(root.path().join("allowed.translated.srt").exists());
    assert!(!outside.path().join("private.translated.srt").exists());
    assert!(!root.path().join(".ssh/protected.translated.srt").exists());
    engine.undo_last().expect("undo authorized output");
    assert!(!root.path().join("allowed.translated.srt").exists());
}

struct CancelSecondFile(CancellationToken);
impl ProgressSink for CancelSecondFile {
    fn emit(&self, event: ProgressEvent) {
        if event.task == TaskKind::BatchTranslation && event.stage == "FILES" && event.current == 1
        {
            self.0.cancel();
        }
    }
}

#[test]
fn cancelled_batch_keeps_completed_output_undoable_after_restart() {
    let root = tempfile::tempdir().expect("project");
    fs::write(root.path().join("a.srt"), SUBTITLE).expect("first input");
    fs::write(root.path().join("b.srt"), SUBTITLE).expect("second input");
    let token = CancellationToken::default();
    let mut engine = engine(root.path()).with_progress(Arc::new(CancelSecondFile(token.clone())));
    engine.begin_operation(token.guard());
    let error = engine
        .run_tool("translate_series", &json!({"path":"."}))
        .expect_err("cancel second file");
    assert!(error.is_cancelled());
    assert!(root.path().join("a.translated.srt").exists());
    assert!(!root.path().join("b.translated.srt").exists());
    let mut resumed = AgentEngine::new(root.path().to_path_buf()).expect("engine");
    resumed.resume_session(None).expect("resume");
    resumed.undo_last().expect("undo completed file");
    assert!(!root.path().join("a.translated.srt").exists());
}

#[test]
fn undo_preserves_manual_edits_to_created_and_modified_files() {
    for original in [None, Some("before\n")] {
        let root = tempfile::tempdir().expect("project");
        let mut engine = engine(root.path());
        if let Some(content) = original {
            fs::write(root.path().join("output.txt"), content).expect("original file");
            engine.run_tool("apply_patch", &json!({"patch":"*** Begin Patch\n*** Update File: output.txt\n@@\n-before\n+original\n*** End Patch"})).expect("update file");
        } else {
            create_file(&mut engine, "output.txt");
        }
        fs::write(root.path().join("output.txt"), "user's later edits").expect("manual edit");
        assert!(
            engine
                .undo_last()
                .expect_err("conflict")
                .to_string()
                .contains("undo conflict")
        );
        assert_eq!(
            fs::read_to_string(root.path().join("output.txt")).expect("preserved"),
            "user's later edits"
        );
        fs::write(root.path().join("output.txt"), "original\n").expect("resolve conflict");
        engine
            .undo_last()
            .expect("unchanged post-state can be undone");
        match original {
            Some(content) => assert_eq!(
                fs::read_to_string(root.path().join("output.txt")).expect("restored"),
                content
            ),
            None => assert!(!root.path().join("output.txt").exists()),
        }
    }
}

#[test]
fn undo_checks_both_rename_paths_and_all_group_members_before_mutating() {
    let root = tempfile::tempdir().expect("project");
    let mut engine = engine(root.path());
    fs::write(root.path().join("old.txt"), "source").expect("source");
    engine
        .run_tool("rename_path", &json!({"from":"old.txt", "to":"new.txt"}))
        .expect("rename");
    fs::write(root.path().join("old.txt"), "replacement source").expect("new user file");
    assert!(engine.undo_last().is_err());
    assert_eq!(
        fs::read_to_string(root.path().join("old.txt")).expect("source preserved"),
        "replacement source"
    );
    assert_eq!(
        fs::read_to_string(root.path().join("new.txt")).expect("destination preserved"),
        "source"
    );
    engine.run_tool("apply_patch", &json!({"patch":"*** Begin Patch\n*** Add File: a.txt\n+a\n*** Add File: b.txt\n+b\n*** End Patch"})).expect("grouped patch");
    fs::write(root.path().join("a.txt"), "manual edit").expect("edit first member");
    assert!(engine.undo_last().is_err());
    assert_eq!(
        fs::read_to_string(root.path().join("b.txt")).expect("other group member intact"),
        "b\n"
    );
}

#[test]
fn old_sessions_are_rejected_and_excluded_from_session_discovery() {
    let root = tempfile::tempdir().expect("project");
    let mut engine = engine(root.path());
    create_file(&mut engine, "output.txt");
    for version in [1, 2] {
        let id = format!("old-{version}");
        let path = engine.session_store.path_for(&id).expect("path");
        fs::write(path, json!({"version":version}).to_string()).expect("old session");
        let error = engine
            .session_store
            .load(&id)
            .expect_err("old version rejected before decoding");
        assert!(
            matches!(error, crate::AgentError::UnsupportedSessionVersion { version: actual, .. } if actual == version)
        );
    }
    let sessions = engine
        .session_store
        .list(20)
        .expect("current sessions only");
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].version, 3);
    assert_eq!(sessions[0].id, engine.session.as_ref().expect("active").id);
}

#[test]
fn current_session_rejects_missing_undo_fingerprints_at_load_and_save() {
    let root = tempfile::tempdir().expect("project");
    let mut engine = engine(root.path());
    create_file(&mut engine, "output.txt");
    let mut session = engine.session.clone().expect("session");
    for event in &mut session.events {
        if event.kind == "file_operation" {
            event
                .data
                .as_object_mut()
                .expect("event object")
                .remove("post_state");
        }
    }
    assert!(engine.session_store.save(&session).is_err());
    let path = engine.session_store.path_for(&session.id).expect("path");
    fs::write(
        path,
        serde_json::to_vec(&session).expect("malformed session"),
    )
    .expect("write invalid data");
    assert!(engine.session_store.load(&session.id).is_err());
    assert!(root.path().join("output.txt").exists());
}

#[test]
fn current_plan_requires_an_explicit_execution_state() {
    let mut value = json!({"message":"plan", "tool_calls":[], "created_at":"2026-09-16T00:00:00Z"});
    assert!(serde_json::from_value::<crate::event::PendingPlan>(value.clone()).is_err());
    value["in_flight_call_id"] = serde_json::Value::Null;
    assert!(serde_json::from_value::<crate::event::PendingPlan>(value).is_ok());
}

#[cfg(unix)]
#[test]
fn private_backup_and_command_directories_reject_symlinks_before_writes() {
    for component in [
        ".subbake",
        ".subbake/agent",
        ".subbake/agent/backups",
        ".subbake/agent/command-runs",
    ] {
        let root = tempfile::tempdir().expect("project");
        let outside = tempfile::tempdir().expect("outside");
        fs::write(root.path().join("source.txt"), "original").expect("source");
        let guard = FileGuard::new(root.path().to_path_buf()).expect("guard before substitution");
        let link = root.path().join(component);
        fs::create_dir_all(link.parent().expect("parent")).expect("parents");
        std::os::unix::fs::symlink(outside.path(), &link).expect("storage alias");
        if component.ends_with("command-runs") {
            assert!(guard.create_command_staging().is_err());
        } else {
            assert!(FileGuard::new(root.path().to_path_buf()).is_err());
            assert!(
                guard
                    .replace_file(Path::new("source.txt"), "edited")
                    .is_err()
            );
            assert!(
                guard
                    .store_embedded_subtitle_undo(Path::new("source.txt"), b"subtitle", "srt")
                    .is_err()
            );
        }
        assert_eq!(
            fs::read_to_string(root.path().join("source.txt")).expect("unchanged source"),
            "original"
        );
        assert_eq!(
            fs::read_dir(outside.path())
                .expect("outside untouched")
                .count(),
            0
        );
    }
}

struct InterruptAfterSuccess;
impl EngineObserver for InterruptAfterSuccess {
    fn on_observation(&mut self, _: &str) {
        panic!("simulated crash after completion persisted");
    }
}
struct InterruptBeforeExecution;
impl EngineObserver for InterruptBeforeExecution {
    fn on_tool_call(&mut self, _: &str, _: &str, _: &serde_json::Value) {
        panic!("simulated crash during tool execution");
    }
}

#[test]
fn completed_plan_call_and_progress_survive_the_same_interruption() {
    let root = tempfile::tempdir().expect("project");
    let mut engine = engine(root.path()).with_observer(Box::new(InterruptAfterSuccess));
    creation_plan(&mut engine);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.approve_plan())).is_err()
    );
    let mut resumed = AgentEngine::new(root.path().to_path_buf()).expect("engine");
    resumed.resume_session(None).expect("resume");
    resumed.approve_plan().expect("finish without replay");
    assert_eq!(
        resumed
            .session_events()
            .iter()
            .filter(|event| event.kind == "tool_started")
            .count(),
        1
    );
    assert_eq!(
        fs::read_to_string(root.path().join("output.txt")).expect("output"),
        "original\n"
    );
}

#[test]
fn uncertain_plan_operation_is_not_replayed_after_restart() {
    let root = tempfile::tempdir().expect("project");
    let mut engine = engine(root.path()).with_observer(Box::new(InterruptBeforeExecution));
    creation_plan(&mut engine);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.approve_plan())).is_err()
    );
    let mut resumed = AgentEngine::new(root.path().to_path_buf()).expect("engine");
    resumed.resume_session(None).expect("resume");
    assert!(
        resumed
            .approve_plan()
            .expect_err("uncertain outcome")
            .to_string()
            .contains("interrupted or failed")
    );
    assert!(!root.path().join("output.txt").exists());
}

#[test]
fn failed_profile_backend_construction_preserves_session_and_active_backend() {
    let root = tempfile::tempdir().expect("project");
    let mut engine = engine(root.path());
    fs::write(
        root.path().join("subbake.toml"),
        r#"version = 3
[profiles.good.backend]
id = "mock"
model = "mock-zh"
[profiles.bad.backend]
id = "custom"
model = "unused"
api_format = "openai_chat"
base_url = "https://example.invalid/v1"
api_key_env = "SUBBAKE_TEST_MISSING_KEY_A8753"
"#,
    )
    .expect("profiles");
    engine.select_profile("good").expect("working profile");
    let error = engine.select_profile("bad").expect_err("missing key");
    assert!(error.to_string().contains("Missing API key"));
    assert_eq!(engine.active_profile(), Some("good"));
    assert_eq!(
        engine
            .profile_backend
            .as_ref()
            .expect("kept backend")
            .provider_name(),
        "echo-decision"
    );
    let mut resumed = AgentEngine::new(root.path().to_path_buf()).expect("engine");
    resumed.resume_session(None).expect("resume");
    assert_eq!(resumed.active_profile(), Some("good"));
}

#[test]
fn cancelled_plan_stops_before_marking_an_operation_in_flight() {
    let root = tempfile::tempdir().expect("project");
    let mut engine = engine(root.path());
    creation_plan(&mut engine);
    let token = CancellationToken::default();
    engine.begin_operation(token.guard());
    token.cancel();
    assert!(engine.approve_plan().expect_err("cancelled").is_cancelled());
    engine.begin_operation(CancellationGuard::never());
    engine
        .approve_plan()
        .expect("unstarted plan remains executable");
}

#[test]
fn independent_patches_have_independent_undo_groups() {
    let root = tempfile::tempdir().expect("project");
    let mut engine = engine(root.path());
    for prefix in ["first", "second"] {
        engine.run_tool("apply_patch", &json!({"patch":format!("*** Begin Patch\n*** Add File: {prefix}-a.txt\n+a\n*** Add File: {prefix}-b.txt\n+b\n*** End Patch")})).expect("patch group");
    }
    engine.undo_last().expect("undo only second patch");
    assert!(root.path().join("first-a.txt").exists());
    assert!(root.path().join("first-b.txt").exists());
    assert!(!root.path().join("second-a.txt").exists());
    assert!(!root.path().join("second-b.txt").exists());
}

#[test]
fn failed_profile_persistence_keeps_the_previous_backend_and_profile() {
    let root = tempfile::tempdir().expect("project");
    let mut engine = engine(root.path());
    fs::write(
        root.path().join("subbake.toml"),
        "version = 3\n[profiles.next.backend]\nid = \"mock\"\nmodel = \"mock-zh\"\n",
    )
    .expect("config");
    create_file(&mut engine, "unchanged.txt");
    let session = engine.session.as_ref().expect("session");
    let path = engine
        .session_store
        .path_for(&session.id)
        .expect("session path");
    fs::rename(&path, path.with_extension("saved")).expect("retain session");
    fs::create_dir(&path).expect("make persistence fail");
    assert!(
        engine
            .run_tool("switch_profile", &json!({"name":"next"}))
            .is_err()
    );
    assert_eq!(engine.active_profile(), None);
    assert!(engine.profile_backend.is_none());
}

#[cfg(unix)]
#[test]
fn undo_backup_validation_preserves_logical_project_aliases() {
    let parent = tempfile::tempdir().expect("parent");
    let real = parent.path().join("real");
    let alias = parent.path().join("alias");
    fs::create_dir(&real).expect("project");
    std::os::unix::fs::symlink(&real, &alias).expect("logical project alias");
    let mut engine = engine(&alias);
    fs::write(real.join("file.txt"), "before\n").expect("source");
    engine.run_tool("apply_patch", &json!({"patch":"*** Begin Patch\n*** Update File: file.txt\n@@\n-before\n+after\n*** End Patch"})).expect("edit through alias");
    engine
        .undo_last()
        .expect("restore using validated backup identity");
    assert_eq!(
        fs::read_to_string(real.join("file.txt")).expect("restored source"),
        "before\n"
    );
}
