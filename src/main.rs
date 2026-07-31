//! Thin binary shim over [`laughing_man::run`]. All logic lives in the library so it is
//! unit-testable; this binary only parses arguments and forwards them.

use clap::Parser;

use laughing_man::cli::Cli;

fn main() -> anyhow::Result<()> {
    laughing_man::run(Cli::parse())
}
