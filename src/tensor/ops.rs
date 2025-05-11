use std::ops::{Add, Sub, AddAssign, SubAssign, Mul, MulAssign};
use crate::tensor::tensor::{Tensor, TensorErrors};

impl Add for Tensor {
    type Output = Result<Tensor, TensorErrors>;

    fn add(self, other: Tensor) -> Self::Output {
        if self.shape != other.shape {
            return Err(TensorErrors::MissMatchedShapes);
        }
        let data = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a + b).collect();
        let requires_grad = self.requires_grad || other.requires_grad;
        Ok(Tensor {
            dims: self.dims,
            shape: self.shape.clone(),
            size: self.size,
            data,
            strides: self.strides.clone(),
            requires_grad,
        })
    }
}

impl Sub for Tensor {
    type Output = Result<Tensor, TensorErrors>;

    fn sub(self, other: Tensor) -> Self::Output {
        if self.shape != other.shape {
            return Err(TensorErrors::MissMatchedShapes);
        }
        let data = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a - b).collect();
        let requires_grad = self.requires_grad || other.requires_grad;
        Ok(Tensor {
            dims: self.dims,
            shape: self.shape.clone(),
            size: self.size,
            data,
            strides: self.strides.clone(),
            requires_grad,
        })
    }
}

impl AddAssign for Tensor {
    fn add_assign(&mut self, other: Tensor) {
        if self.shape != other.shape {
            panic!("MissMatchedShapes: Cannot add_assign tensors with different shapes");
        }

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
        self.requires_grad |= other.requires_grad;
    }
}

impl SubAssign for Tensor {
    fn sub_assign(&mut self, other: Tensor) {
        if self.shape != other.shape {
            panic!("MissMatchedShapes: Cannot sub_assign tensors with different shapes");
        }

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= b;
        }
        self.requires_grad |= other.requires_grad;
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Self::Output {
        let new_data = self.data.iter().map(|&x| x * scalar).collect();

        Tensor::new(new_data, self.shape.clone(), Some(self.requires_grad)).expect("No requires grad for the tensor.")
    }
}

impl MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, scalar: f32) {
        for element in &mut self.data {
            *element *= scalar;
        }
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Result<Tensor, TensorErrors>;

    fn mul(self, other: Tensor) -> Self::Output {
        if self.shape != other.shape {
            return Err(TensorErrors::MissMatchedShapes);
        }
        let mut new_data = self.data;

        for i in 0..self.size {
            new_data[i] *= other.data[i];
        }

        Ok(Tensor::new(new_data, self.shape.clone(), Some(self.requires_grad)).expect("No requires grad for tensor."))
    }
}

impl MulAssign<Tensor> for Tensor {
    fn mul_assign(&mut self, other: Tensor) {
        if self.shape != other.shape {
            panic!("Miss matched shapes.");
        }
        for i in 0..self.size {
            self.data[i] *= other.data[i];
        }
    }
}

