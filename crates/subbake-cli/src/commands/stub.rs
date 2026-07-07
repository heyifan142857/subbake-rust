use std::io;

pub fn run_transcribe(args: &[String]) -> io::Result<()> {
    let media = args
        .first()
        .ok_or_else(|| io::Error::other("transcribe requires a media path"))?;
    println!("Transcription adapter is pending migration: {media}");
    Ok(())
}

pub fn run_pipeline(args: &[String]) -> io::Result<()> {
    let input = args
        .first()
        .ok_or_else(|| io::Error::other("pipeline requires an input path"))?;
    println!(
        "Pipeline adapter is pending migration for {input}. Use `translate` for subtitle files."
    );
    Ok(())
}

pub fn run_whisper(args: &[String]) -> io::Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("status");
    println!("whisper command `{command}` is pending adapter migration.");
    Ok(())
}
