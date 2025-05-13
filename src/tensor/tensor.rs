use ndarray::ArrayD;
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum TensorErrors {
    #[error("Invalid tensor dimensions.")]
    InvalidShape,
    #[error("Index out of bound.")]
    OutOfBound,
    #[error("Invalid slice range.")]
    InvalidRange,
    #[error("Trying to adding to miss-matched shapes tensors.")]
    MissMatchedShapes,
}

#[derive(Clone)]
pub struct Tensor {
    pub data: ArrayD<f32>,
    pub requires_grad: bool,
    pub grad: Option<ArrayD<f32>>
}

impl Tensor {
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn strides(&self) -> Vec<isize> {
        self.data.strides().to_vec()
    }
}
