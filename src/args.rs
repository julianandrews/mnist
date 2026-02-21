use clap::builder::styling::{AnsiColor, Styles};
use clap::{Parser, Subcommand, ValueEnum};

use crate::neural_net::{ActivationFunction, InitMethod};

// clap v3 styling
const CLAP_STYLES: Styles = Styles::styled()
    .header(AnsiColor::Yellow.on_default())
    .usage(AnsiColor::Green.on_default())
    .literal(AnsiColor::Green.on_default())
    .placeholder(AnsiColor::Green.on_default());

#[derive(Debug, Parser)]
#[command(version, about, styles=CLAP_STYLES)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    Train {
        /// File containing training samples [default: stdin]
        file: Option<String>,

        /// Hidden layer sizes. Can be specified either as a comma-separated list: --layers
        /// 64,32, or with repeated flags: --layer 64 --layer 32
        #[arg(
            long,
            value_delimiter = ',',
            num_args = 1..,
            default_value = "16,16",
            value_name = "SIZE"
        )]
        layers: Vec<usize>,

        /// Activation function to use
        #[arg(long, short, value_enum, default_value_t = CliActivationFunction::Sigmoid)]
        activation: CliActivationFunction,

        /// Initialization method for weights
        #[arg(long, short, value_enum, default_value_t = CliInitMethod::LeCunn)]
        initialization: CliInitMethod,
    },
    Test,
}

#[derive(Debug, Clone, ValueEnum)]
#[clap(rename_all = "lower")]
pub enum CliActivationFunction {
    Sigmoid,
    Tanh,
    ReLU,
}

impl From<CliActivationFunction> for ActivationFunction {
    fn from(cli: CliActivationFunction) -> Self {
        match cli {
            CliActivationFunction::Sigmoid => ActivationFunction::Sigmoid,
            CliActivationFunction::Tanh => ActivationFunction::Tanh,
            CliActivationFunction::ReLU => ActivationFunction::ReLU,
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
#[clap(rename_all = "lower")]
pub enum CliInitMethod {
    LeCunn,
    Glorot,
    He,
}

impl From<CliInitMethod> for InitMethod {
    fn from(cli: CliInitMethod) -> Self {
        match cli {
            CliInitMethod::LeCunn => InitMethod::LeCunn,
            CliInitMethod::Glorot => InitMethod::Glorot,
            CliInitMethod::He => InitMethod::He,
        }
    }
}
