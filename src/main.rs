use clap::Parser;

use laughing_man::cli::Cli;

fn main() -> laughing_man::Result<()> {
    laughing_man::run(Cli::parse())
}
