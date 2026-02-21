use crate::data::Dataset;

#[derive(Debug)]
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
}
