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
    pub fn new() -> Self {
        Self {
            shape: vec![],
            requires_grad: false,
            init_method: InitMethod::Zeros,
        }
    }

    pub fn shape(mut self, shape: &[usize]) -> Self {
        self.shape = shape.to_vec();
        self
    }

    pub fn requires_grad(mut self, flag: bool) -> Self {
        self.requires_grad = flag;
        self
    }

    pub fn init(mut self, method: InitMethod) -> Self {
        self.init_method = method;
        self
    }

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
            InitMethod::Ones => {
                ArrayD::from_shape_vec(shape_dyn, repeat(1.0).take(size).collect())
                    .map_err(|_| TensorErrors::InvalidShape)?
            }
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
