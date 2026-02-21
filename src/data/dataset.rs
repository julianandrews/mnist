use std::borrow::Cow;

use crate::data::Sample;

pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> Option<Cow<'_, Sample>>;
}
