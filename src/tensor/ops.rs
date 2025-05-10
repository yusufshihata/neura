use neura::tensor::tensor::{ Tensor, TensorErrors };
use std::ops;

impl ops::Add<Tensor> for Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorErrors> {
        if &self.shape() != other.shape() {
            return Err(TensorErrors::MissMatchedShapes);
        }

        let mut new_tensor: Tensor = self.clone();
        for i in 0..&self.size() {
            new_tensor.data[i] += *other.data[i];
        }

        Ok(new_tensor)
    }
}

impl ops::Sub<Tensor> for Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorErrors> {
        if &self.shape() != other.shape() {
            return Err(TensorErrors::MissMatchedShapes);
        }

        let mut new_tensor: Tensor = self.clone();
        for i in 0..&self.size() {
            new_tensor.data[i] -= *other.data[i];
        }
    }
}

