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
    pub grad: Option<ArrayD<f32>>,
}

impl Tensor {
    /// Returns the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A vector of `usize` values specifying the shape of the tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Returns the number of dimensions of the tensor.
    ///
    /// # Returns
    ///
    /// A usize value representing the number of dimensions.

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// # Returns
    ///
    /// A usize value representing the total number of elements in the tensor.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the strides of the tensor.
    ///
    /// # Returns
    ///
    /// A vector of isize values representing the strides of the tensor. The
    /// strides are the number of elements to jump when moving to the next element
    /// in each dimension. The strides are ordered from slowest changing dimension
    /// to fastest changing dimension.
    pub fn strides(&self) -> Vec<isize> {
        self.data.strides().to_vec()
    }
}
