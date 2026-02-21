use std::{borrow::Cow, io::Read};

use ndarray::Array1;

use crate::data::{Dataset, Sample};

#[derive(Debug)]
pub struct CsvDataset {
    samples: Vec<Sample>,
}

impl CsvDataset {
    // let samples = read_samples(&sample_file)?;
    // let sample_len = samples.first().map(|s| s.size()).unwrap_or(0);
    // if samples.iter().any(|s| s.size() != sample_len) {
    //     bail!("Training failed: uneven sample sizes");
    // }
    // TODO: pull the possible outputs from the samples instead of assuming 10 neurons in
    // the output layer.
    pub fn new(mut reader: impl Read) -> Result<Self, CsvLoadError> {
        fn parse_line(line: &str) -> Result<Sample, CsvLoadError> {
            let (expected_part, features_part) =
                line.split_once(',').ok_or(CsvLoadError::FailedSplit)?;
            let inputs: Array1<f64> = features_part
                .split(',')
                .map(|w| w.parse::<u8>().map(|w| (w as f64 - 127.5) / 127.5))
                .collect::<Result<_, _>>()
                .map_err(|_| CsvLoadError::FeatureParseError)?;
            let expected_value: usize = expected_part
                .parse()
                .map_err(|_| CsvLoadError::ExpectedParseError)?;
            // TODO: infer output shape from inputs?
            let mut expected = Array1::zeros(10);
            expected[expected_value] = 1.0;
            Ok(Sample { inputs, expected })
        }

        // TODO: streaming reads for efficiency
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        let samples = input.lines().map(parse_line).collect::<Result<_, _>>()?;
        // TODO: validate samples have same inputs
        Ok(CsvDataset { samples })
    }
}

impl Dataset for CsvDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<Cow<'_, Sample>> {
        self.samples.get(index).map(Cow::Borrowed)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CsvLoadError {
    #[error("failed to read input")]
    ReadError(#[from] std::io::Error),
    #[error("failed to split inputs from expected")]
    FailedSplit,
    #[error("failed to parse expected")]
    ExpectedParseError,
    #[error("failed to parse feature")]
    FeatureParseError,
}
