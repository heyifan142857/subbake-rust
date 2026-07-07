use std::io;

use subbake_adapters::{BackendConfig, build_backend};

use crate::args::required_value;

pub fn run(args: &[String]) -> io::Result<()> {
    if args.first().is_none_or(|value| value != "check") {
        return Err(io::Error::other("provider requires `check`"));
    }
    let mut provider = "mock".to_owned();
    let mut model = "mock-zh".to_owned();
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--provider" => provider = required_value(args, &mut index, "--provider")?,
            "--model" => model = required_value(args, &mut index, "--model")?,
            other => return Err(io::Error::other(format!("unknown provider option `{other}`"))),
        }
        index += 1;
    }

    let backend = build_backend(&BackendConfig::new(provider, model)).map_err(io::Error::other)?;
    let (ok, message) = backend.check_credentials().map_err(io::Error::other)?;
    if ok {
        println!("Provider check passed.");
        println!("{message}");
        Ok(())
    } else {
        Err(io::Error::other(message))
    }
}
