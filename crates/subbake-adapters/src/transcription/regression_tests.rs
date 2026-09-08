use super::*;
use subbake_core::{CancellationToken, TranslationMode};

struct Fixture {
    _root: tempfile::TempDir,
    media: PathBuf,
    model: PathBuf,
    binary: PathBuf,
    settings: TranscriptionSettings,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().expect("temporary fixture");
        let media = root.path().join("audio.wav");
        let model = root.path().join("model.bin");
        let binary = root.path().join("not-an-executable");
        for path in [&media, &model, &binary] {
            fs::write(path, b"fixture").expect("fixture");
        }
        let settings = TranscriptionSettings {
            runtime_dir: Some(root.path().join("runtime")),
            language: Some("en".to_owned()),
            ..Default::default()
        };
        Self {
            _root: root,
            media,
            model,
            binary,
            settings,
        }
    }

    fn store(&self, settings: &TranscriptionSettings) -> source_store::SourceStore {
        source_store::SourceStore::new(
            &self.media,
            &self.model,
            &self.binary,
            "test-v1",
            settings,
            &CancellationGuard::never(),
        )
        .expect("store")
    }

    fn raw(&self, store: &source_store::SourceStore) -> PathBuf {
        let raw =
            store.recognition_path(&chunk_descriptor(0, transcription_chunks(10_000)[0]), false);
        let json = serde_json::json!({"model":{"multilingual":true},"result":{"language":"en"},"transcription":[{
            "text":" Hi. Hi.", "offsets":{"from":0,"to":10000}, "tokens":[
                {"id":1,"text":" Hi.","offsets":{"from":1000,"to":1500},"p":0.001},
                {"id":1,"text":" Hi.","offsets":{"from":8000,"to":8500},"p":0.001}
            ]
        }]});
        source_store::write(&raw, &json, &CancellationGuard::never()).expect("raw JSON");
        raw
    }

    fn run(
        &self,
        settings: &TranscriptionSettings,
        _format: TranscriptionFormat,
        observer: &mut dyn TranscriptionChunkObserver,
        cancellation: &CancellationGuard,
    ) -> AdapterResult<(
        SubtitleDocument,
        TranscriptionCleanupStats,
        TranscriptionReport,
    )> {
        let progress: SharedProgress = std::sync::Arc::new(NoopProgress);
        transcribe_prepared_audio_incremental(
            &self.binary,
            &self.model,
            &PreparedAudio::borrowed(&self.media, Some(10_000)),
            Some("en"),
            settings,
            cancellation,
            &progress,
            observer,
            10_000,
            &self.store(settings),
        )
    }
}

#[derive(Default)]
struct Capture(Vec<StableTranscriptionChunk>);
impl TranscriptionChunkObserver for Capture {
    fn stable(&mut self, chunk: StableTranscriptionChunk) -> AdapterResult<()> {
        self.0.push(chunk);
        Ok(())
    }
}

#[test]
fn all_modes_and_output_formats_share_stable_cues_without_running_whisper() {
    let fixture = Fixture::new();
    fixture.raw(&fixture.store(&fixture.settings));
    let mut expected = None;
    for mode in [
        TranslationMode::Economy,
        TranslationMode::Turbo,
        TranslationMode::Cinema,
    ] {
        let mut resolved = ResolvedSettings::default();
        resolved.translation.mode = mode;
        let mut settings = fixture.settings.clone();
        apply_whisper_configuration(&mut settings, &resolved);
        for format in [
            TranscriptionFormat::Srt,
            TranscriptionFormat::Vtt,
            TranscriptionFormat::Txt,
        ] {
            let mut capture = Capture::default();
            let (document, _, report) = fixture
                .run(&settings, format, &mut capture, &CancellationGuard::never())
                .expect("cached source");
            let stable = &capture.0[0].segments;
            assert_eq!(stable.len(), 2);
            assert_eq!(stable[0].start.as_deref(), Some("00:00:11,000"));
            assert_eq!(stable[1].start.as_deref(), Some("00:00:18,000"));
            assert_eq!(report.languages(), ["en"]);
            if let Some(expected) = &expected {
                assert_eq!(stable, expected);
            } else {
                expected = Some(stable.clone());
            }
            if format == TranscriptionFormat::Txt {
                assert!(document.segments.iter().all(|cue| cue.start.is_some()));
            }
        }
    }
}

#[test]
fn segmentation_reuses_raw_json_but_not_prepared_cues_and_resume_can_skip_raw() {
    let fixture = Fixture::new();
    let raw = fixture.raw(&fixture.store(&fixture.settings));
    let mut capture = Capture::default();
    fixture
        .run(
            &fixture.settings,
            TranscriptionFormat::Srt,
            &mut capture,
            &CancellationGuard::never(),
        )
        .expect("smart");
    assert!(!capture.0[0].resumed);
    let mut settings = fixture.settings.clone();
    settings.segmentation = Some(Segmentation::Segment);
    let (segment, _, _) = fixture
        .run(
            &settings,
            TranscriptionFormat::Srt,
            &mut DiscardObserver,
            &CancellationGuard::never(),
        )
        .expect("raw reuse");
    assert_eq!(segment.segments.len(), 1);
    assert_eq!(segment.segments[0].text, "Hi. Hi.");
    assert_eq!(segment.segments[0].start.as_deref(), Some("00:00:10,000"));
    fs::write(&raw, b"deliberately invalid fixture").expect("invalidate raw");
    settings.use_cache = false;
    let mut capture = Capture::default();
    fixture
        .run(
            &settings,
            TranscriptionFormat::Srt,
            &mut capture,
            &CancellationGuard::never(),
        )
        .expect("prepared reuse");
    assert!(capture.0[0].resumed);
    settings.resume = false;
    assert!(
        fixture
            .run(
                &settings,
                TranscriptionFormat::Srt,
                &mut DiscardObserver,
                &CancellationGuard::never()
            )
            .is_err()
    );
}

#[test]
fn recognition_identity_covers_contents_language_and_chunk_contract() {
    let fixture = Fixture::new();
    let descriptor = chunk_descriptor(0, transcription_chunks(10_000)[0]);
    let path = fixture
        .store(&fixture.settings)
        .recognition_path(&descriptor, false);
    let mut settings = fixture.settings.clone();
    settings.language = Some("Auto".to_owned());
    assert_ne!(
        path,
        fixture
            .store(&settings)
            .recognition_path(&descriptor, false)
    );
    assert_ne!(
        path,
        fixture
            .store(&fixture.settings)
            .recognition_path(&descriptor, true)
    );
    for changed in [&fixture.model, &fixture.binary, &fixture.media] {
        fs::write(changed, b"changed content").expect("change fixture");
        assert_ne!(
            path,
            fixture
                .store(&fixture.settings)
                .recognition_path(&descriptor, false)
        );
        fs::write(changed, b"fixture").expect("restore fixture");
    }
    let store = fixture.store(&fixture.settings);
    assert_ne!(
        store.segmented_path(&path, &fixture.settings, 0),
        store.segmented_path(&path, &fixture.settings, 10_000)
    );
}

#[test]
fn cancellation_does_not_publish_cached_chunks_or_new_artifacts() {
    let fixture = Fixture::new();
    fixture.raw(&fixture.store(&fixture.settings));
    let token = CancellationToken::default();
    let cancellation = token.guard();
    token.cancel();
    let mut capture = Capture::default();
    assert!(
        fixture
            .run(
                &fixture.settings,
                TranscriptionFormat::Srt,
                &mut capture,
                &cancellation
            )
            .is_err()
    );
    assert!(capture.0.is_empty());
    let path = fixture._root.path().join("must-not-exist/state.json");
    assert!(source_store::write(&path, &42, &cancellation).is_err());
    assert!(!path.exists());
}

#[test]
fn boundary_disagreement_is_reported_without_deleting_dialogue() {
    let fixture = Fixture::new();
    let store = fixture.store(&fixture.settings);
    let chunks = transcription_chunks(800_000);
    for (index, chunk) in chunks.iter().copied().enumerate() {
        let raw = store.recognition_path(&chunk_descriptor(index, chunk), true);
        // Both independently recognized original segments own a core, but their
        // estimated edges overlap. Keep both rather than guessing which is wrong.
        let (start, end) = if index == 0 {
            (597_000, 602_000)
        } else {
            (30_000, 34_000)
        };
        let json = serde_json::json!({"model":{"multilingual":true},"result":{"language":"en"},"transcription":[{
            "text":"Hello.", "offsets":{"from":start,"to":end}, "tokens":[]
        }]});
        source_store::write(&raw, &json, &CancellationGuard::never()).expect("raw");
    }
    let progress: SharedProgress = std::sync::Arc::new(NoopProgress);
    let (document, _, report) = transcribe_prepared_audio_incremental(
        &fixture.binary,
        &fixture.model,
        &PreparedAudio::borrowed(&fixture.media, Some(800_000)),
        Some("en"),
        &fixture.settings,
        &CancellationGuard::never(),
        &progress,
        &mut DiscardObserver,
        10_000,
        &store,
    )
    .expect("chunks");
    assert_eq!(document.segments.len(), 2);
    assert_eq!(report.chunks[1].boundary_overlap_ms, 2_000);
    assert_eq!(document.segments[1].start.as_deref(), Some("00:10:10,000"));
    assert_eq!(report.fallback_count(), 2);
}
