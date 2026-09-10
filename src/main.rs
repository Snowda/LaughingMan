use clap::Parser;

use laughing_man::cli::Cli;

fn main() -> anyhow::Result<()> {
    laughing_man::run(Cli::parse())
}
