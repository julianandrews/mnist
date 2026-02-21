use std::path::Path;

use anyhow::bail;
use clap::Parser;

use mnist::args::{Args, Command};
use mnist::mnist_sample::MnistSample;
use mnist::neural_net::NeuralNet;

fn main() -> Result<(), anyhow::Error> {
    let parsed_args = Args::parse();
    match parsed_args.command {
        Command::Train {
            file,
            layers,
            activation,
            initialization,
        } => {
            let samples = read_samples(&file)?;
            let sample_len = samples.first().map(|s| s.size()).unwrap_or(0);
            if samples.iter().any(|s| s.size() != sample_len) {
                bail!("Training failed: uneven sample sizes");
            }
            let layers: Vec<usize> = std::iter::once(sample_len)
                .chain(layers)
                .chain(std::iter::once(10))
                .collect();
            let mut net = NeuralNet::new(&layers, activation.into(), initialization.into());
            println!("{net:?}");
        }
        Command::Test => todo!(),
    }
    Ok(())
}

fn read_input<P: AsRef<Path>>(infile: &Option<P>) -> Result<String, Box<dyn std::error::Error>> {
    let mut reader: Box<dyn std::io::Read> = match infile {
        Some(filename) => Box::new(std::io::BufReader::new(std::fs::File::open(filename)?)),
        None => Box::new(std::io::stdin()),
    };
    let mut input = String::new();
    reader.read_to_string(&mut input)?;
    Ok(input)
}

fn read_samples<P: AsRef<Path>>(infile: &Option<P>) -> Result<Vec<MnistSample>, anyhow::Error> {
    let input = match read_input(infile) {
        Ok(input) => input,
        Err(e) => bail!("Failed to read input: {e}"),
    };
    Ok(input
        .lines()
        .map(|line| line.parse())
        .collect::<Result<_, _>>()?)
}
