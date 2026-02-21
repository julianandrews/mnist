use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct Sample {
    pub inputs: Array1<f64>,
    pub expected: Array1<f64>,
}

impl Sample {
    pub fn input_len(&self) -> usize {
        self.inputs.len()
    }

    pub fn output_len(&self) -> usize {
        self.expected.len()
    }
}
