use std::borrow::Cow;

use crate::data::Sample;

/// A source of samples for training or evaluation.
///
/// The `Dataset` trait abstracts over different data sources (e.g., CSV files, in‑memory arrays,
/// binary formats) and provides random access to samples. This lets data loaders iterate, shuffle,
/// and batch samples without knowing the underlying storage details.
pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> Option<Cow<'_, Sample>>;
}
