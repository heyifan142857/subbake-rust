use std::io;

pub fn run(args: &[String]) -> io::Result<()> {
    if args.first().is_some_and(|value| value == "resume") {
        println!(
            "{}",
            subbake_agent::resume_agent(args.get(1).map(String::as_str))
        );
    } else if args.is_empty() {
        println!("{}", subbake_agent::start_agent());
    } else {
        return Err(io::Error::other(
            "unsupported agent command; use `agent resume [SESSION_ID]`",
        ));
    }
    Ok(())
}
