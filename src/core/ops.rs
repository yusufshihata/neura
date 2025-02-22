use crate::core::tensor::Tensor;
use std::ops::{Add, Mul};

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.shape == rhs.shape);
        let data = self.data + rhs.data;
        Tensor::new(data, self.shape.clone(), self.requires_grad)
    }
}
