use std::env;

fn main() {
    if let Err(error) = subbake_cli::run(env::args().skip(1).collect()) {
        eprintln!("Error: {error}");
        std::process::exit(error.exit_code());
    }
}
