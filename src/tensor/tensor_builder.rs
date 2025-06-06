use crate::tensor::tensor::{Tensor, TensorErrors};
use ndarray::{ArrayD, IxDyn};
use std::iter::repeat;

pub enum InitMethod {
    FromData(Vec<f32>),
    Ones,
    Zeros,
}

pub struct TensorBuilder {
    shape: Vec<usize>,
    requires_grad: bool,
    init_method: InitMethod,
}

impl TensorBuilder {
    /// Returns a new `TensorBuilder` with default values set.
    ///
    /// The default values are:
    ///
    /// * `shape`: an empty vector
    /// * `requires_grad`: `false`
    /// * `init_method`: `InitMethod::Zeros`
    ///
    pub fn new() -> Self {
        Self {
            shape: vec![],
            requires_grad: false,
            init_method: InitMethod::Zeros,
        }
    }

    /// Sets the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - A slice of `usize` values specifying the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new `TensorBuilder` with the specified shape.
    ///
    pub fn shape(mut self, shape: &[usize]) -> Self {
        self.shape = shape.to_vec();
        self
    }

    /// Sets whether the tensor requires gradients.
    ///
    /// If `true`, the tensor is marked as requiring gradients and any
    /// operations performed on it will be tracked. If `false`, the tensor
    /// is marked as not requiring gradients and operations will not be tracked.
    ///
    /// # Arguments
    ///
    /// * `flag` - A boolean indicating whether the tensor requires gradients.
    ///
    /// # Returns
    ///
    /// A new `TensorBuilder` with the specified `requires_grad` flag.
    pub fn requires_grad(mut self, flag: bool) -> Self {
        self.requires_grad = flag;
        self
    }

    /// Sets the method to use for initializing the tensor.
    ///
    /// # Arguments
    ///
    /// * `method` - An `InitMethod` value specifying the method to use for initializing
    ///              the tensor.
    ///
    /// # Returns
    ///
    /// A new `TensorBuilder` with the specified initialization method.
    ///
    pub fn init(mut self, method: InitMethod) -> Self {
        self.init_method = method;
        self
    }

    /// Builds and returns a `Tensor` based on the current configuration of the `TensorBuilder`.
    ///
    /// This function finalizes the construction of a `Tensor` using the shape, initialization
    /// method, and other properties specified in the `TensorBuilder`. The `shape` must be
    /// non-empty, and the total size of the tensor must match the number of elements specified
    /// by the initialization method.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor)` - If the tensor is successfully created with the specified configuration.
    /// * `Err(TensorErrors::InvalidShape)` - If the shape is empty or if there is a mismatch
    ///   between the size of the shape and the data provided for initialization.

    pub fn build(self) -> Result<Tensor, TensorErrors> {
        if self.shape.is_empty() {
            return Err(TensorErrors::InvalidShape);
        }

        let size: usize = self.shape.iter().product();
        let shape_dyn = IxDyn(&self.shape);

        let array = match self.init_method {
            InitMethod::Zeros => {
                ArrayD::from_shape_vec(shape_dyn, repeat(0.0).take(size).collect())
                    .map_err(|_| TensorErrors::InvalidShape)?
            }
            InitMethod::Ones => ArrayD::from_shape_vec(shape_dyn, repeat(1.0).take(size).collect())
                .map_err(|_| TensorErrors::InvalidShape)?,
            InitMethod::FromData(data) => {
                if data.len() != size {
                    return Err(TensorErrors::InvalidShape);
                }
                ArrayD::from_shape_vec(shape_dyn, data).map_err(|_| TensorErrors::InvalidShape)?
            }
        };

        Ok(Tensor {
            data: array,
            requires_grad: self.requires_grad,
            grad: None,
        })
    }
}
