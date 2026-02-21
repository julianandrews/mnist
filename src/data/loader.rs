use crate::data::{CsvDataset, Dataset, csv_dataset::CsvLoadError};

#[derive(Debug)]
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
        }
    }

    pub fn input_size(&self) -> usize {
        self.dataset.get(0).map(|s| s.input_size()).unwrap_or(0)
    }

    pub fn output_size(&self) -> usize {
        self.dataset.get(0).map(|s| s.output_size()).unwrap_or(0)
    }
}

pub fn csv_loader_from_file(
    filename: &str,
    batch_size: usize,
) -> Result<DataLoader<CsvDataset>, CsvLoadError> {
    let mut reader = std::io::BufReader::new(std::fs::File::open(filename)?);
    let dataset = CsvDataset::new(&mut reader)?;
    Ok(DataLoader::new(dataset, batch_size))
}
