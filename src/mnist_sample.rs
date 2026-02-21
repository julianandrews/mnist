use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct MnistSample {
    pub features: Array1<f64>,
    pub expected: Array1<f64>,
}

impl MnistSample {
    pub fn size(&self) -> usize {
        self.features.len()
    }
}

impl std::str::FromStr for MnistSample {
    type Err = SampleParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (expected_part, features_part) =
            s.split_once(',').ok_or(SampleParseError::FailedSplit)?;
        let features: Array1<f64> = features_part
            .split(',')
            .map(|w| w.parse::<u8>().map(|w| (w as f64 - 127.5) / 127.5))
            .collect::<Result<_, _>>()
            .map_err(|_| SampleParseError::FeatureParseError)?;
        let expected_value: usize = expected_part
            .parse()
            .map_err(|_| SampleParseError::ExpectedParseError)?;
        let mut expected = Array1::zeros(10);
        expected[expected_value] = 1.0;
        Ok(MnistSample { features, expected })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SampleParseError {
    #[error("failed to split features from expected")]
    FailedSplit,
    #[error("failed to parse expected")]
    ExpectedParseError,
    #[error("failed to parse feature")]
    FeatureParseError,
}
