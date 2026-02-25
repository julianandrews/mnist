use std::{borrow::Cow, io::Read};

use ndarray::Array1;

use crate::data::{Dataset, Sample};

#[derive(Debug)]
pub struct CsvDataset {
    samples: Vec<Sample>,
}

impl CsvDataset {
    pub fn new(mut reader: impl Read) -> Result<Self, CsvLoadError> {
        // TODO: use csv library
        // TODO: streaming reads for efficiency
        // TODO: get output range from the samples instead of assuming 10
        fn parse_line(line: &str) -> Result<Sample, CsvLoadError> {
            let (expected_part, features_part) =
                line.split_once(',').ok_or(CsvLoadError::FailedSplit)?;
            let inputs: Array1<f32> = features_part
                .split(',')
                .map(|w| w.parse::<u8>().map(|w| (w as f32 - 127.5) / 127.5))
                .collect::<Result<_, _>>()
                .map_err(|_| CsvLoadError::FeatureParseError)?;
            let expected_value: usize = expected_part
                .parse()
                .map_err(|_| CsvLoadError::ExpectedParseError)?;
            let mut expected = Array1::zeros(10);
            expected[expected_value] = 1.0;
            Ok(Sample { inputs, expected })
        }

        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        let samples: Vec<Sample> = input.lines().map(parse_line).collect::<Result<_, _>>()?;
        let input_size = samples.first().map(|s| s.input_size()).unwrap_or(0);
        if samples.iter().any(|s| s.input_size() != input_size) {
            return Err(CsvLoadError::InconsistentInputSizes);
        }
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
    #[error("inconsistent input sizes in data")]
    InconsistentInputSizes,
}
