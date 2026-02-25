use ndarray::{Array1, Array2};
use rand::distr::{Distribution, Uniform};

// TODO: per layer activation functions?
#[derive(Debug, Clone)]
pub struct NeuralNet {
    biases: Vec<Array1<f32>>,
    weights: Vec<Array2<f32>>,
    activation_function: ActivationFunction,
}

impl NeuralNet {
    pub fn new(
        layers: &[usize],
        activation_function: ActivationFunction,
        init_method: InitMethod,
    ) -> Self {
        // Initialize biasses for all layers but the input layer.
        // Initializing to zero is a good default. There are use cases for non-zero biases though.
        let biases = layers.iter().skip(1).map(|&l| Array1::zeros(l)).collect();

        // Initialize weights for each adjacent pair of layers using the requested init method.
        let weights = layers
            .windows(2)
            .map(|pair| init_method.init_weights(pair[0], pair[1]))
            .collect();

        NeuralNet {
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
    fn init_weights(&self, n_in: usize, n_out: usize) -> Array2<f32> {
        // LeCunn, Glorot, and He initialization all just specify the variance to aim for as a
        // function of the number of input and output neurons.
        //
        // The specific distribution doesn't seem to be very important, and a uniform distribution
        // clamped to achieve the desired variance seems to be standard practice.
        let limit = match self {
            InitMethod::LeCunn => (1.0 / n_in as f32).sqrt(),
            InitMethod::Glorot => (6.0 / (n_in + n_out) as f32).sqrt(),
            InitMethod::He => (6.0 / n_in as f32).sqrt(),
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
