use std::io;

pub fn run_whisper(args: &[String]) -> io::Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("status");
    println!("whisper command `{command}` is pending adapter migration.");
    Ok(())
}
