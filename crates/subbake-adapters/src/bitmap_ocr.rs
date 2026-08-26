//! OCR conversion for PGS, VobSub, and DVB bitmap subtitles.
//!
//! FFmpeg/ffprobe own container demuxing. This module renders each timed
//! bitmap state in pure Rust and delegates only character recognition to the
//! external Tesseract executable.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use libbitsub_core::dvb::DvbParser;
use libbitsub_core::pgs::{PgsParser, SubtitleComposition, SubtitleFrame};
use libbitsub_core::vobsub::VobSubParser;
use sha2::{Digest, Sha256};
use subbake_core::{
    CancellationGuard, OcrCueMetadata, OcrWordConfidence, ProgressEvent, ProgressUnit,
    SharedProgress, TaskKind, TaskState,
};

use crate::error::{AdapterError, AdapterResult};
use crate::process::ProcessSupervisor;

const OCR_SCALE: u32 = 2;
const OCR_PADDING: u32 = 16;
const MIN_VISIBLE_ALPHA: u8 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BitmapOcrOutcome {
    pub cue_count: usize,
    pub low_confidence_cues: usize,
    pub source_language: String,
    pub cues: Vec<OcrCueMetadata>,
}

#[derive(Debug)]
struct RenderedCue {
    start_ms: u64,
    end_ms: u64,
    fingerprint: [u8; 32],
    image_path: PathBuf,
}

#[derive(Debug)]
struct ActiveCue {
    start_ms: u64,
    end_ms: u64,
    fingerprint: [u8; 32],
    image_path: PathBuf,
}

#[derive(Debug, Clone)]
struct OcrText {
    text: String,
    confidence: Option<u8>,
    words: Vec<OcrWordConfidence>,
}

pub(crate) fn convert_sup_to_srt(
    sup_path: &Path,
    srt_path: &Path,
    source_language: &str,
    stream_language: Option<&str>,
    tesseract: &Path,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
) -> AdapterResult<BitmapOcrOutcome> {
    cancellation.check().map_err(AdapterError::from)?;
    let bytes = fs::read(sup_path).map_err(|source| {
        AdapterError::external_io(
            "read extracted PGS subtitle",
            Some(sup_path.to_path_buf()),
            source,
        )
    })?;
    let mut parser = PgsParser::new();
    let parsed = parser.parse(&bytes);
    if parsed == 0 {
        return Err(AdapterError::invalid_input(format!(
            "PGS subtitle contains no decodable display sets: {}",
            sup_path.display()
        )));
    }

    convert_decoder_to_srt(
        &mut parser,
        srt_path,
        source_language,
        stream_language,
        tesseract,
        cancellation,
        progress,
    )
}

pub(crate) fn convert_vobsub_mks_to_srt(
    mks_path: &Path,
    srt_path: &Path,
    source_language: &str,
    stream_language: Option<&str>,
    tesseract: &Path,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
) -> AdapterResult<BitmapOcrOutcome> {
    cancellation.check().map_err(AdapterError::from)?;
    let bytes = fs::read(mks_path).map_err(|source| {
        AdapterError::external_io(
            "read extracted VobSub subtitle",
            Some(mks_path.to_path_buf()),
            source,
        )
    })?;
    let mut parser = VobSubParser::new();
    parser.load_from_mks(&bytes).map_err(|message| {
        AdapterError::invalid_input(format!("invalid VobSub track: {message}"))
    })?;
    if parser.count() == 0 {
        return Err(AdapterError::invalid_input(format!(
            "VobSub subtitle contains no decodable packets: {}",
            mks_path.display()
        )));
    }

    convert_decoder_to_srt(
        &mut parser,
        srt_path,
        source_language,
        stream_language,
        tesseract,
        cancellation,
        progress,
    )
}

pub(crate) fn convert_dvb_pes_to_srt(
    pes_path: &Path,
    srt_path: &Path,
    source_language: &str,
    stream_language: Option<&str>,
    tesseract: &Path,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
) -> AdapterResult<BitmapOcrOutcome> {
    cancellation.check().map_err(AdapterError::from)?;
    let bytes = fs::read(pes_path).map_err(|source| {
        AdapterError::external_io(
            "read extracted DVB subtitle",
            Some(pes_path.to_path_buf()),
            source,
        )
    })?;
    let mut parser = DvbParser::new();
    if parser.parse(&bytes) == 0 {
        return Err(AdapterError::invalid_input(format!(
            "DVB subtitle contains no decodable timed display sets: {}",
            pes_path.display()
        )));
    }

    convert_decoder_to_srt(
        &mut parser,
        srt_path,
        source_language,
        stream_language,
        tesseract,
        cancellation,
        progress,
    )
}

fn convert_decoder_to_srt(
    decoder: &mut impl BitmapCueDecoder,
    srt_path: &Path,
    source_language: &str,
    stream_language: Option<&str>,
    tesseract: &Path,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
) -> AdapterResult<BitmapOcrOutcome> {
    let language = tesseract_language(source_language, stream_language)?;
    ensure_tesseract_language(tesseract, &language, cancellation)?;

    let images_dir = srt_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("bitmap-ocr-images");
    fs::create_dir_all(&images_dir).map_err(|source| {
        AdapterError::external_io(
            "create bitmap OCR image directory",
            Some(images_dir.clone()),
            source,
        )
    })?;
    let cues = render_cues(decoder, &images_dir, cancellation)?;
    if cues.is_empty() {
        return Err(AdapterError::invalid_input(
            "bitmap subtitle contains no visible subtitle images",
        ));
    }

    let cache = recognize_unique_images(tesseract, &cues, &language, cancellation, progress)?;
    let total = u64::try_from(cues.len()).unwrap_or(u64::MAX);
    let mut recognized = Vec::with_capacity(cues.len());
    let mut metadata = Vec::with_capacity(cues.len());
    let mut low_confidence_cues = 0usize;
    let mut empty_cues = Vec::new();
    for cue in &cues {
        cancellation.check().map_err(AdapterError::from)?;
        let text = cache.get(&cue.fingerprint).ok_or_else(|| {
            AdapterError::invalid_input("bitmap OCR result is missing for a rendered cue")
        })?;
        if text.text.trim().is_empty() {
            empty_cues.push(cue.start_ms);
        } else {
            if text
                .words
                .iter()
                .any(|word| word.confidence.is_some_and(|value| value < 70))
                || text.confidence.is_some_and(|value| value < 55)
            {
                low_confidence_cues += 1;
            }
            recognized.push((cue.start_ms, cue.end_ms, text.text.clone()));
            metadata.push(OcrCueMetadata {
                id: recognized.len().to_string(),
                words: text.words.clone(),
            });
        }
    }

    if !empty_cues.is_empty() {
        let preview = empty_cues
            .iter()
            .take(5)
            .map(|value| format_srt_time(*value))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(AdapterError::invalid_input(format!(
            "bitmap subtitle OCR produced no text for {} cue(s), including {preview}; choose the correct source language or install a matching Tesseract language model",
            empty_cues.len()
        )));
    }

    let srt = render_srt(&recognized);
    fs::write(srt_path, srt).map_err(|source| {
        AdapterError::external_io("write OCR subtitle", Some(srt_path.to_path_buf()), source)
    })?;
    let mut completed = ProgressEvent::running(
        TaskKind::Pipeline,
        "OCR_BITMAP_SUBTITLE",
        total,
        Some(total),
        ProgressUnit::Lines,
    );
    completed.state = TaskState::Completed;
    if low_confidence_cues > 0 {
        completed.message = Some(format!(
            "{low_confidence_cues} OCR cue(s) contain a word below confidence 70 and may need review"
        ));
    }
    progress.emit(completed);
    Ok(BitmapOcrOutcome {
        cue_count: recognized.len(),
        low_confidence_cues,
        source_language: language,
        cues: metadata,
    })
}

trait BitmapCueDecoder {
    fn count(&self) -> usize;
    fn cue_start_time(&self, index: usize) -> f64;
    fn cue_end_time(&mut self, index: usize) -> f64;
    fn render_at_index(&mut self, index: usize) -> Option<SubtitleFrame>;
    fn frame_persists_until_next_update(&self) -> bool;
}

impl BitmapCueDecoder for PgsParser {
    fn count(&self) -> usize {
        PgsParser::count(self)
    }

    fn cue_start_time(&self, index: usize) -> f64 {
        self.get_cue_start_time(index)
    }

    fn cue_end_time(&mut self, index: usize) -> f64 {
        self.get_cue_end_time(index)
    }

    fn render_at_index(&mut self, index: usize) -> Option<SubtitleFrame> {
        PgsParser::render_at_index(self, index)
    }

    fn frame_persists_until_next_update(&self) -> bool {
        true
    }
}

impl BitmapCueDecoder for VobSubParser {
    fn count(&self) -> usize {
        VobSubParser::count(self)
    }

    fn cue_start_time(&self, index: usize) -> f64 {
        self.get_cue_start_time(index)
    }

    fn cue_end_time(&mut self, index: usize) -> f64 {
        self.get_cue_end_time(index)
    }

    fn render_at_index(&mut self, index: usize) -> Option<SubtitleFrame> {
        let frame = VobSubParser::render_at_index(self, index)?;
        Some(SubtitleFrame {
            width: frame.screen_width,
            height: frame.screen_height,
            compositions: vec![SubtitleComposition {
                x: frame.x,
                y: frame.y,
                width: frame.width,
                height: frame.height,
                rgba: frame.rgba,
            }],
        })
    }

    fn frame_persists_until_next_update(&self) -> bool {
        false
    }
}

impl BitmapCueDecoder for DvbParser {
    fn count(&self) -> usize {
        DvbParser::count(self)
    }

    fn cue_start_time(&self, index: usize) -> f64 {
        self.get_cue_start_time(index)
    }

    fn cue_end_time(&mut self, index: usize) -> f64 {
        self.get_cue_end_time(index)
    }

    fn render_at_index(&mut self, index: usize) -> Option<SubtitleFrame> {
        let frame = DvbParser::render_at_index(self, index)?;
        Some(SubtitleFrame {
            width: frame.width,
            height: frame.height,
            compositions: frame
                .compositions
                .into_iter()
                .map(|composition| SubtitleComposition {
                    x: composition.x,
                    y: composition.y,
                    width: composition.width,
                    height: composition.height,
                    rgba: composition.rgba,
                })
                .collect(),
        })
    }

    fn frame_persists_until_next_update(&self) -> bool {
        false
    }
}

fn render_cues(
    decoder: &mut impl BitmapCueDecoder,
    images_dir: &Path,
    cancellation: &CancellationGuard,
) -> AdapterResult<Vec<RenderedCue>> {
    let frame_persists = decoder.frame_persists_until_next_update();
    let mut cues = Vec::new();
    let mut active: Option<ActiveCue> = None;
    let mut image_paths = HashMap::<[u8; 32], PathBuf>::new();

    for index in 0..decoder.count() {
        cancellation.check().map_err(AdapterError::from)?;
        let start_ms = timestamp_ms(decoder.cue_start_time(index))?;
        let next_ms = timestamp_ms(decoder.cue_end_time(index))?;
        let rendered = decoder
            .render_at_index(index)
            .and_then(|frame| frame_to_ocr_image(&frame));

        let Some((width, height, pixels)) = rendered else {
            if let Some(previous) = active.take() {
                let boundary = if frame_persists {
                    start_ms
                } else {
                    previous.end_ms
                };
                cues.push(finish_active(previous, boundary));
            }
            continue;
        };
        let mut hasher = Sha256::new();
        hasher.update(width.to_le_bytes());
        hasher.update(height.to_le_bytes());
        hasher.update(&pixels);
        let fingerprint: [u8; 32] = hasher.finalize().into();
        if let Some(previous) = active.as_mut()
            && previous.fingerprint == fingerprint
            && (frame_persists || previous.end_ms >= start_ms)
        {
            previous.end_ms = next_ms.max(previous.end_ms);
            continue;
        }
        if let Some(previous) = active.take() {
            let boundary = if frame_persists {
                start_ms
            } else {
                previous.end_ms
            };
            cues.push(finish_active(previous, boundary));
        }
        let image_path = if let Some(path) = image_paths.get(&fingerprint) {
            path.clone()
        } else {
            let name = fingerprint
                .iter()
                .take(12)
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let path = images_dir.join(format!("{name}.png"));
            write_grayscale_png(&path, width, height, &pixels)?;
            image_paths.insert(fingerprint, path.clone());
            path
        };
        active = Some(ActiveCue {
            start_ms,
            end_ms: next_ms,
            fingerprint,
            image_path,
        });
    }
    if let Some(previous) = active {
        let end_ms = previous.end_ms;
        cues.push(finish_active(previous, end_ms));
    }
    Ok(cues)
}

fn finish_active(active: ActiveCue, boundary_ms: u64) -> RenderedCue {
    RenderedCue {
        start_ms: active.start_ms,
        end_ms: boundary_ms.max(active.start_ms.saturating_add(1)),
        fingerprint: active.fingerprint,
        image_path: active.image_path,
    }
}

fn frame_to_ocr_image(frame: &SubtitleFrame) -> Option<(u32, u32, Vec<u8>)> {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut visible = false;
    for composition in &frame.compositions {
        let width = usize::from(composition.width);
        for (pixel_index, pixel) in composition.rgba.as_chunks::<4>().0.iter().enumerate() {
            if pixel[3] <= MIN_VISIBLE_ALPHA {
                continue;
            }
            visible = true;
            let x =
                u32::from(composition.x).saturating_add(u32::try_from(pixel_index % width).ok()?);
            let y =
                u32::from(composition.y).saturating_add(u32::try_from(pixel_index / width).ok()?);
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !visible {
        return None;
    }

    let content_width = max_x.checked_sub(min_x)?.checked_add(1)?;
    let content_height = max_y.checked_sub(min_y)?.checked_add(1)?;
    let padded_width = content_width.checked_add(OCR_PADDING.checked_mul(2)?)?;
    let padded_height = content_height.checked_add(OCR_PADDING.checked_mul(2)?)?;
    let width = padded_width.checked_mul(OCR_SCALE)?;
    let height = padded_height.checked_mul(OCR_SCALE)?;
    let len = usize::try_from(width.checked_mul(height)?).ok()?;
    let mut output = vec![255u8; len];

    for composition in &frame.compositions {
        let composition_width = usize::from(composition.width);
        for (pixel_index, pixel) in composition.rgba.as_chunks::<4>().0.iter().enumerate() {
            if pixel[3] <= MIN_VISIBLE_ALPHA {
                continue;
            }
            let source_x = u32::try_from(pixel_index % composition_width).ok()?;
            let source_y = u32::try_from(pixel_index / composition_width).ok()?;
            let x = u32::from(composition.x)
                .checked_add(source_x)?
                .checked_sub(min_x)?
                .checked_add(OCR_PADDING)?
                .checked_mul(OCR_SCALE)?;
            let y = u32::from(composition.y)
                .checked_add(source_y)?
                .checked_sub(min_y)?
                .checked_add(OCR_PADDING)?
                .checked_mul(OCR_SCALE)?;
            let foreground = 255u8.saturating_sub(pixel[3]);
            for offset_y in 0..OCR_SCALE {
                for offset_x in 0..OCR_SCALE {
                    let target = y
                        .checked_add(offset_y)?
                        .checked_mul(width)?
                        .checked_add(x.checked_add(offset_x)?)?;
                    let target = usize::try_from(target).ok()?;
                    if let Some(value) = output.get_mut(target) {
                        *value = (*value).min(foreground);
                    }
                }
            }
        }
    }
    Some((width, height, output))
}

fn write_grayscale_png(path: &Path, width: u32, height: u32, pixels: &[u8]) -> AdapterResult<()> {
    let file = fs::File::create(path).map_err(|source| {
        AdapterError::external_io("create bitmap OCR image", Some(path.to_path_buf()), source)
    })?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|source| png_io_error("write bitmap OCR PNG header", source))?;
    writer
        .write_image_data(pixels)
        .map_err(|source| png_io_error("write bitmap OCR PNG pixels", source))?;
    Ok(())
}

fn png_io_error(operation: &'static str, source: png::EncodingError) -> AdapterError {
    AdapterError::external_io(operation, None, io::Error::other(source))
}

fn ensure_tesseract_language(
    tesseract: &Path,
    language: &str,
    cancellation: &CancellationGuard,
) -> AdapterResult<()> {
    let output = ProcessSupervisor::run(
        Command::new(tesseract).arg(OsStr::new("--list-langs")),
        cancellation,
        "list Tesseract OCR languages",
    )
    .map_err(|error| {
        if error.is_not_found() {
            AdapterError::invalid_input(tesseract_missing_message(language))
        } else {
            error
        }
    })?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(AdapterError::invalid_input(format!(
            "Tesseract could not load its language data. {}{}",
            tesseract_language_missing_message(language),
            if detail.is_empty() {
                String::new()
            } else {
                format!(" Tesseract reported: {detail}")
            }
        )));
    }
    let installed = String::from_utf8_lossy(&output.stdout);
    if !installed.lines().any(|value| value.trim() == language) {
        return Err(AdapterError::invalid_input(format!(
            "Tesseract OCR language `{language}` is not installed. {}",
            tesseract_language_missing_message(language)
        )));
    }
    Ok(())
}

fn tesseract_missing_message(language: &str) -> String {
    format!(
        "Required OCR dependency `tesseract` is missing or not on PATH, and `{language}` source-language data is required. Install them and verify with `tesseract --list-langs`, or explicitly use audio transcription as a substitute source; audio transcription regenerates dialogue text and does not translate the existing bitmap subtitle."
    )
}

fn tesseract_language_missing_message(language: &str) -> String {
    format!(
        "Required Tesseract source-language data `{language}` is missing. Install that language data and verify with `tesseract --list-langs`, or explicitly use audio transcription as a substitute source; it does not translate the existing bitmap subtitle."
    )
}

fn recognize_unique_images(
    tesseract: &Path,
    cues: &[RenderedCue],
    language: &str,
    cancellation: &CancellationGuard,
    progress: &SharedProgress,
) -> AdapterResult<HashMap<[u8; 32], OcrText>> {
    let mut unique = Vec::<([u8; 32], PathBuf)>::new();
    let mut indexes = HashMap::<[u8; 32], usize>::new();
    for cue in cues {
        if indexes.contains_key(&cue.fingerprint) {
            continue;
        }
        indexes.insert(cue.fingerprint, unique.len());
        unique.push((cue.fingerprint, cue.image_path.clone()));
    }
    let total = unique.len();
    let worker_count = std::thread::available_parallelism()
        .map_or(2, usize::from)
        .clamp(1, 4)
        .min(total.max(1));
    let next = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let progress_serial = Mutex::new(());
    let results = Mutex::new(
        std::iter::repeat_with(|| None)
            .take(total)
            .collect::<Vec<Option<OcrText>>>(),
    );

    std::thread::scope(|scope| -> AdapterResult<()> {
        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            workers.push(scope.spawn(|| -> AdapterResult<()> {
                loop {
                    if failed.load(Ordering::Acquire) {
                        return Ok(());
                    }
                    if let Err(error) = cancellation.check() {
                        failed.store(true, Ordering::Release);
                        return Err(AdapterError::from(error));
                    }
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some((_, image_path)) = unique.get(index) else {
                        return Ok(());
                    };
                    let text = match recognize_image(tesseract, image_path, language, cancellation)
                    {
                        Ok(text) => text,
                        Err(error) => {
                            failed.store(true, Ordering::Release);
                            return Err(error);
                        }
                    };
                    let mut guard = results.lock().map_err(|_| {
                        AdapterError::invalid_input("bitmap OCR result lock poisoned")
                    })?;
                    if let Some(slot) = guard.get_mut(index) {
                        *slot = Some(text);
                    }
                    drop(guard);
                    let progress_guard = progress_serial.lock().map_err(|_| {
                        AdapterError::invalid_input("bitmap OCR progress lock poisoned")
                    })?;
                    let current = completed.fetch_add(1, Ordering::Relaxed).saturating_add(1);
                    progress.emit(ProgressEvent::running(
                        TaskKind::Pipeline,
                        "OCR_BITMAP_SUBTITLE",
                        u64::try_from(current).unwrap_or(u64::MAX),
                        Some(u64::try_from(total).unwrap_or(u64::MAX)),
                        ProgressUnit::Lines,
                    ));
                    drop(progress_guard);
                }
            }));
        }
        for worker in workers {
            worker
                .join()
                .map_err(|_| AdapterError::invalid_input("bitmap OCR worker panicked"))??;
        }
        Ok(())
    })?;

    let mut results = results
        .into_inner()
        .map_err(|_| AdapterError::invalid_input("bitmap OCR result lock poisoned"))?;
    let mut recognized = HashMap::with_capacity(total);
    for (index, (fingerprint, _)) in unique.into_iter().enumerate() {
        let text = results
            .get_mut(index)
            .and_then(Option::take)
            .ok_or_else(|| AdapterError::invalid_input("bitmap OCR worker returned no result"))?;
        recognized.insert(fingerprint, text);
    }
    Ok(recognized)
}

fn recognize_image(
    tesseract: &Path,
    image_path: &Path,
    language: &str,
    cancellation: &CancellationGuard,
) -> AdapterResult<OcrText> {
    let mut last = OcrText {
        text: String::new(),
        confidence: None,
        words: Vec::new(),
    };
    for page_segmentation_mode in [6, 7, 11, 13] {
        let mode = page_segmentation_mode.to_string();
        let output = ProcessSupervisor::run(
            Command::new(tesseract).args([
                image_path.as_os_str(),
                OsStr::new("stdout"),
                OsStr::new("-l"),
                OsStr::new(language),
                OsStr::new("--psm"),
                OsStr::new(&mode),
                OsStr::new("tsv"),
            ]),
            cancellation,
            "recognize bitmap subtitle image",
        )?;
        if !output.status.success() {
            return Err(ocr_process_error(
                &output,
                "Tesseract failed to recognize a bitmap subtitle cue",
            ));
        }
        last = parse_tsv(&String::from_utf8_lossy(&output.stdout))?;
        if !last.text.trim().is_empty() {
            return Ok(last);
        }
    }
    Ok(last)
}

fn ocr_process_error(output: &Output, fallback: &str) -> AdapterError {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    AdapterError::ChildProcess {
        program: "tesseract",
        status: output.status.code(),
        message: if stderr.is_empty() {
            fallback.to_owned()
        } else {
            stderr
        },
    }
}

fn parse_tsv(tsv: &str) -> AdapterResult<OcrText> {
    let mut lines = Vec::<((u32, u32, u32), Vec<String>)>::new();
    let mut word_confidences = Vec::new();
    let mut confidence_sum = 0u64;
    let mut confidence_count = 0u64;
    for row in tsv.lines().skip(1) {
        let fields = row.splitn(12, '\t').collect::<Vec<_>>();
        if fields.len() != 12 || fields[0] != "5" || fields[11].trim().is_empty() {
            continue;
        }
        let key = (
            parse_tsv_number(fields[2])?,
            parse_tsv_number(fields[3])?,
            parse_tsv_number(fields[4])?,
        );
        if lines.last().is_none_or(|(previous, _)| *previous != key) {
            lines.push((key, Vec::new()));
        }
        if let Some((_, words)) = lines.last_mut() {
            words.push(fields[11].trim().to_owned());
        }
        let word_confidence = fields[10]
            .parse::<f64>()
            .ok()
            .filter(|value| *value >= 0.0)
            .map(|value| u8::try_from((value.round() as u64).min(100)).unwrap_or(100));
        if let Some(value) = word_confidence {
            confidence_sum = confidence_sum.saturating_add(u64::from(value));
            confidence_count = confidence_count.saturating_add(1);
        }
        word_confidences.push(OcrWordConfidence {
            text: fields[11].trim().to_owned(),
            confidence: word_confidence,
        });
    }
    let text = lines
        .into_iter()
        .map(|(_, words)| words.join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let confidence = (confidence_count > 0)
        .then(|| u8::try_from((confidence_sum / confidence_count).min(100)).unwrap_or(100));
    Ok(OcrText {
        text,
        confidence,
        words: word_confidences,
    })
}

fn parse_tsv_number(value: &str) -> AdapterResult<u32> {
    value.parse::<u32>().map_err(|source| {
        AdapterError::invalid_input(format!("invalid Tesseract TSV number `{value}`: {source}"))
    })
}

fn tesseract_language(
    source_language: &str,
    stream_language: Option<&str>,
) -> AdapterResult<String> {
    let requested = normalized_language(source_language);
    let source = if requested.as_deref().is_none_or(|value| value == "auto") {
        stream_language.and_then(normalized_language).ok_or_else(|| {
            AdapterError::invalid_input(
                "bitmap subtitle OCR needs a source language because the selected stream has no language tag",
            )
        })?
    } else {
        requested.unwrap_or_default()
    };
    let code = match source.as_str() {
        "en" | "eng" => "eng",
        "zh" | "zho" | "chi" | "zh-hans" => "chi_sim",
        "zh-hant" => "chi_tra",
        "ja" | "jpn" => "jpn",
        "ko" | "kor" => "kor",
        "fr" | "fra" | "fre" => "fra",
        "es" | "spa" => "spa",
        "de" | "deu" | "ger" => "deu",
        "pt" | "por" => "por",
        "ru" | "rus" => "rus",
        "it" | "ita" => "ita",
        "ar" | "ara" => "ara",
        "hi" | "hin" => "hin",
        "nl" | "nld" | "dut" => "nld",
        "pl" | "pol" => "pol",
        "tr" | "tur" => "tur",
        "uk" | "ukr" => "ukr",
        "vi" | "vie" => "vie",
        "th" | "tha" => "tha",
        "id" | "ind" => "ind",
        other => other,
    };
    Ok(code.to_owned())
}

fn normalized_language(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("und") {
        return None;
    }
    let normalized = subbake_core::languages::normalize_language_name(value, true);
    (normalized != "und").then(|| normalized.to_ascii_lowercase())
}

fn timestamp_ms(value: f64) -> AdapterResult<u64> {
    if !value.is_finite() || value < 0.0 || value > u64::MAX as f64 {
        return Err(AdapterError::invalid_input(
            "bitmap subtitle has an invalid timestamp",
        ));
    }
    Ok(value.round() as u64)
}

fn render_srt(cues: &[(u64, u64, String)]) -> String {
    let mut output = String::new();
    for (index, (start_ms, end_ms, text)) in cues.iter().enumerate() {
        output.push_str(&(index + 1).to_string());
        output.push('\n');
        output.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(*start_ms),
            format_srt_time(*end_ms)
        ));
        output.push_str(text.trim());
        output.push_str("\n\n");
    }
    output
}

fn format_srt_time(milliseconds: u64) -> String {
    let hours = milliseconds / 3_600_000;
    let minutes = (milliseconds / 60_000) % 60;
    let seconds = (milliseconds / 1_000) % 60;
    let millis = milliseconds % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02},{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use libbitsub_core::pgs::SubtitleComposition;

    struct FakeDecoder {
        starts: Vec<f64>,
        ends: Vec<f64>,
        persists: bool,
    }

    impl BitmapCueDecoder for FakeDecoder {
        fn count(&self) -> usize {
            self.starts.len()
        }

        fn cue_start_time(&self, index: usize) -> f64 {
            self.starts[index]
        }

        fn cue_end_time(&mut self, index: usize) -> f64 {
            self.ends[index]
        }

        fn render_at_index(&mut self, _index: usize) -> Option<SubtitleFrame> {
            Some(SubtitleFrame {
                width: 1,
                height: 1,
                compositions: vec![SubtitleComposition {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                    rgba: vec![255, 255, 255, 255],
                }],
            })
        }

        fn frame_persists_until_next_update(&self) -> bool {
            self.persists
        }
    }

    #[test]
    fn tsv_reconstructs_lines_and_average_confidence() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90.1\tHello\n\
5\t1\t1\t1\t1\t2\t12\t0\t10\t10\t80.0\tworld\n\
5\t1\t1\t1\t2\t1\t0\t12\t10\t10\t70.0\tAgain\n";

        let parsed = parse_tsv(tsv).expect("parse TSV");

        assert_eq!(parsed.text, "Hello world\nAgain");
        assert_eq!(parsed.confidence, Some(80));
    }

    #[test]
    fn transparent_frame_is_cropped_padded_and_scaled() {
        let mut rgba = vec![0u8; 4 * 3 * 4];
        rgba[(5 * 4) + 3] = 255;
        let frame = SubtitleFrame {
            width: 1920,
            height: 1080,
            compositions: vec![SubtitleComposition {
                x: 100,
                y: 200,
                width: 4,
                height: 3,
                rgba,
            }],
        };

        let (width, height, pixels) = frame_to_ocr_image(&frame).expect("visible frame");

        assert_eq!(width, (1 + OCR_PADDING * 2) * OCR_SCALE);
        assert_eq!(height, (1 + OCR_PADDING * 2) * OCR_SCALE);
        assert_eq!(
            pixels.len(),
            usize::try_from(width * height).expect("image size")
        );
        assert!(pixels.contains(&0));
    }

    #[test]
    fn independent_bitmap_cues_do_not_merge_across_a_gap() {
        let mut decoder = FakeDecoder {
            starts: vec![0.0, 2_000.0],
            ends: vec![1_000.0, 3_000.0],
            persists: false,
        };
        let temporary = tempfile::tempdir().expect("temporary directory");

        let cues = render_cues(&mut decoder, temporary.path(), &CancellationGuard::never())
            .expect("render cues");

        assert_eq!(cues.len(), 2);
        assert_eq!((cues[0].start_ms, cues[0].end_ms), (0, 1_000));
        assert_eq!((cues[1].start_ms, cues[1].end_ms), (2_000, 3_000));
    }

    #[test]
    fn language_uses_stream_tag_when_translation_source_is_auto() {
        assert_eq!(
            tesseract_language("Auto", Some("eng")).expect("language"),
            "eng"
        );
        assert_eq!(
            tesseract_language("Traditional Chinese", None).expect("language"),
            "chi_tra"
        );
        assert!(tesseract_language("Auto", None).is_err());
    }

    #[test]
    fn missing_tesseract_reports_dependency_and_source_substitution_boundary() {
        let error = ensure_tesseract_language(
            Path::new("subbake-test-tesseract-that-does-not-exist"),
            "chi_sim",
            &CancellationGuard::never(),
        )
        .expect_err("missing Tesseract should fail");
        let message = error.to_string();

        assert!(message.contains("Required OCR dependency `tesseract` is missing"));
        assert!(message.contains("`chi_sim` source-language data is required"));
        assert!(message.contains("tesseract --list-langs"));
        assert!(message.contains("audio transcription as a substitute source"));
        assert!(!message.contains("sudo "));
    }

    #[test]
    fn missing_language_message_names_data_without_hard_coding_a_package_manager() {
        let message = tesseract_language_missing_message("chi_sim");

        assert!(message.contains("source-language data `chi_sim` is missing"));
        assert!(message.contains("audio transcription as a substitute source"));
        assert!(!message.contains("sudo "));
    }

    #[test]
    fn srt_keeps_pgs_timestamps_and_multiline_text() {
        let rendered = render_srt(&[(1_234, 65_006, "First\nSecond".to_owned())]);
        assert_eq!(
            rendered,
            "1\n00:00:01,234 --> 00:01:05,006\nFirst\nSecond\n\n"
        );
    }
}
