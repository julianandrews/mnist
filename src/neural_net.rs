use ndarray::{Array1, Array2};
use rand::distr::{Distribution, Uniform};

#[derive(Debug, Clone)]
pub struct NeuralNet {
    activations: Vec<Array1<f64>>,
    biases: Vec<Array1<f64>>,
    weights: Vec<Array2<f64>>,
    activation_function: ActivationFunction,
}

impl NeuralNet {
    pub fn new(
        layers: &[usize],
        activation_function: ActivationFunction,
        initialization_method: InitMethod,
    ) -> Self {
        let activations = layers.iter().skip(1).map(|&l| Array1::zeros(l)).collect();
        let biases = layers.iter().skip(1).map(|&l| Array1::zeros(l)).collect();
        let weights = layers
            .windows(2)
            .map(|pair| initialization_method.initialize_weights(pair[0], pair[1]))
            .collect();

        NeuralNet {
            activations,
            biases,
            weights,
            activation_function,
        }
    }
}

#[derive(Debug, Clone)]
pub enum InitMethod {
    LeCunn,
    Glorot,
    He,
}

impl InitMethod {
    fn initialize_weights(&self, n_in: usize, n_out: usize) -> Array2<f64> {
        let limit = match self {
            InitMethod::LeCunn => (1.0 / n_in as f64).sqrt(),
            InitMethod::Glorot => (6.0 / (n_in + n_out) as f64).sqrt(),
            InitMethod::He => (6.0 / n_in as f64).sqrt(),
        };
        let mut rng = rand::rng();
        let dist = Uniform::new(-limit, limit).unwrap();
        Array2::from_shape_fn((n_in, n_out), |_| dist.sample(&mut rng))
    }
}

#[derive(Debug, Clone)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    ReLU,
}
