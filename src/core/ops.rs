use crate::core::tensor::Tensor;
use ndarray::{ArrayD, IxDyn, Ix2};
use std::ops::{Add, Sub, Mul};
use rayon::prelude::*;

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.shape == rhs.shape);
        let data = self.data + rhs.data;
        Tensor::new(data, self.shape.clone(), self.requires_grad)
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        let data = self.data - rhs.data;
        Tensor::new(data, self.shape.clone(), self.requires_grad)
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f64) -> Self::Output {
        let data = self.data * scalar;
        Tensor::new(data, self.shape.clone(), self.requires_grad)
    }
}

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // Ensure tensors are at least 2D
        assert!(self.rank >= 2 && other.rank >= 2, "Tensors must be at least 2D for matmul");

        // Extract dimensions for matrix multiplication
        let (m, k1) = (self.shape[self.rank - 2], self.shape[self.rank - 1]);
        let (k2, n) = (other.shape[other.rank - 2], other.shape[other.rank - 1]);
        assert!(k1 == k2, "Shape mismatch: {:?} @ {:?} is invalid", self.shape, other.shape);

        // Convert to 2D arrays for dot product
        let self_matrix = self.data.clone().into_dimensionality::<Ix2>().unwrap();
        let other_matrix = other.data.clone().into_dimensionality::<Ix2>().unwrap();

        // Perform matrix multiplication
        let result_data = self_matrix.dot(&other_matrix);

        // Convert back to dynamic dimensions
        let new_shape = vec![m, n];
        let result_data = result_data.into_dyn();

        // Initialize gradient if needed
        let mut result_grad = None;
        if self.requires_grad || other.requires_grad {
            result_grad = Some(ArrayD::<f64>::zeros(IxDyn(&new_shape)));
        }

        Tensor {
            size: (m * n) as i32,
            rank: new_shape.len(),
            shape: new_shape,
            data: result_data,
            offset: 0,
            requires_grad: self.requires_grad || other.requires_grad,
            grad: result_grad,
        }
    }
}
