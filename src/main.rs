use clap::Parser;

use mnist::args::{Args, Command};
use mnist::data::csv_loader_from_file;
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
            // TODO: alternate filetype loading
            // TODO: shuffle bool from command line flags
            // TODO: RNG seed from command line flags
            let loader = csv_loader_from_file(&sample_file, batch_size, true, None)?;

            // Add the input layer and the output layer to the hidden layers
            let layers: Vec<usize> = std::iter::once(loader.input_size())
                .chain(hidden_layers)
                .chain(std::iter::once(loader.output_size()))
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
