use clap::Parser;

use mnist::args::{Args, Command};
use mnist::data::CsvDataset;
use mnist::neural_net::NeuralNet;

fn main() -> Result<(), anyhow::Error> {
    let parsed_args = Args::parse();
    match parsed_args.command {
        Command::Train {
            sample_file,
            hidden_layers,
            batch_size,
            activation,
            init_method,
        } => {
            let mut reader: Box<dyn std::io::Read> = match sample_file {
                Some(filename) => Box::new(std::io::BufReader::new(std::fs::File::open(filename)?)),
                None => Box::new(std::io::stdin()),
            };
            let dataset = CsvDataset::new(&mut reader)?;

            // Add the input layer (the same size as the samples), and the output layer (size 10).
            let layers: Vec<usize> = std::iter::once(784)
                .chain(hidden_layers)
                .chain(std::iter::once(10))
                .collect();

            let mut net = NeuralNet::new(&layers, activation.into(), init_method.into());
            // TODO: verbose flag?
            // let optimizer = todo!();
            // optimizer.train(&mut net, data_loader);
            // TODO: serialize net
        }
        Command::Test => todo!(),
    }
    Ok(())
}
