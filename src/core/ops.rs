use crate::core::tensor::Tensor;
use std::ops::{Add, Mul, MulAssign};

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.shape == rhs.shape);
        let data = self.data + rhs.data;
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
