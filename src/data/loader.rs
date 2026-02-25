use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::data::{CsvDataset, Dataset, csv_dataset::CsvLoadError};

#[derive(Debug)]
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    seed: Option<u64>,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize, shuffle: bool, seed: Option<u64>) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            seed,
        }
    }

    pub fn input_size(&self) -> usize {
        self.dataset.get(0).map(|s| s.input_size()).unwrap_or(0)
    }

    pub fn output_size(&self) -> usize {
        self.dataset.get(0).map(|s| s.output_size()).unwrap_or(0)
    }

    /// Returns an iterator over batches of samples from the dataset.
    ///
    /// Each batch is a tuple `(inputs, targets)` where:
    /// - `inputs` is an `Array2<f32>` of shape `(batch_size, input_dim)`
    /// - `targets` is an `Array2<f32>` of shape `(batch_size, num_classes)`
    ///
    /// The last batch may be smaller than `batch_size` if the dataset size is
    /// not a multiple of the batch size.
    ///
    /// # Performance
    ///
    /// Each call to `next()` allocates new `Array2` buffers for the batch and
    /// copies the sample data into them. This copy is necessary to produce
    /// contiguous matrices suitable for efficient vectorized operations.
    ///
    /// # Example
    /// ```
    /// let loader = DataLoader::new(dataset, 32, true)?;
    /// // TODO: This is nonsense, but an example's a good idea.
    /// for (inputs, targets) in loader.batches() {
    ///     // inputs: (32, 784), targets: (32, 10)
    ///     let predictions = model.forward(&inputs);
    ///     let loss = compute_loss(&predictions, &targets);
    ///     // ...
    /// }
    /// ```
    pub fn batches(&self) -> impl Iterator<Item = (Array2<f32>, Array2<f32>)> + '_ {
        BatchIter::new(self)
    }
}

pub struct BatchIter<'a, D: Dataset> {
    dataset: &'a D,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    indices: Vec<usize>,
    pos: usize,
}

impl<'a, D: Dataset> BatchIter<'a, D> {
    // TODO: Implement a shuffle buffer for handling larger datasets
    fn new(data_loader: &'a DataLoader<D>) -> Self {
        let mut indices: Vec<usize> = (0..data_loader.dataset.len()).collect();
        if data_loader.shuffle {
            match data_loader.seed {
                Some(seed) => {
                    let mut rng = StdRng::seed_from_u64(seed);
                    indices.shuffle(&mut rng);
                }
                None => {
                    let mut rng = rand::rng();
                    indices.shuffle(&mut rng);
                }
            }
        }
        BatchIter {
            dataset: &data_loader.dataset,
            batch_size: data_loader.batch_size,
            input_size: data_loader.input_size(),
            output_size: data_loader.output_size(),
            indices,
            pos: 0,
        }
    }
}

impl<'a, D: Dataset> Iterator for BatchIter<'a, D> {
    type Item = (Array2<f32>, Array2<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.indices.len() {
            return None;
        }

        let end = (self.pos + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.pos..end];
        self.pos = end;
        let batch_size = batch_indices.len();

        let mut inputs = Array2::zeros((batch_size, self.input_size));
        let mut targets = Array2::zeros((batch_size, self.output_size));

        for (i, &idx) in batch_indices.iter().enumerate() {
            let sample = self.dataset.get(idx).expect("missing sample");
            inputs.row_mut(i).assign(&sample.inputs);
            targets.row_mut(i).assign(&sample.expected);
        }

        Some((inputs, targets))
    }
}

pub fn csv_loader_from_file(
    filename: &str,
    batch_size: usize,
    shuffle: bool,
    seed: Option<u64>,
) -> Result<DataLoader<CsvDataset>, CsvLoadError> {
    let mut reader = std::io::BufReader::new(std::fs::File::open(filename)?);
    let dataset = CsvDataset::new(&mut reader)?;
    Ok(DataLoader::new(dataset, batch_size, shuffle, seed))
}
