use crate::core::tensor::Tensor;
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
    fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(self.shape.len() >= 2 && other.shape.len() >= 2, "Tensors must have at least 2 dimensions for matrix multiplication.");

        let (m, k1) = (self.shape[self.shape.len() - 2], self.shape[self.shape.len() - 1]);
        let (k2, n) = (other.shape[other.shape.len() - 2], other.shape[other.shape.len() - 1]);

        assert!(k1 == k2, "Inner dimensions must match for matrix multiplication.");

        let mut result = vec![0.0; m*n];
        result.par_chunk_mut(n).enumerate().for_each(|(i, chunk)| {
            for j in 0..n {
                for k in 0..k1 {
                    chunk[j] += self.data[[i, k]] * other.data[[k, j]];
                }
            }
        });
        Tensor::ones(&self.shape, true)
    }
}