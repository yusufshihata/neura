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

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorErrors> {
        if self.shape.is_empty() || other.shape.is_empty() {
            return Err(TensorErrors::InvalidShape);
        }

        let a_is_1d = self.shape.len() == 1;
        let b_is_1d = other.shape.len() == 1;

        let a_rows = if a_is_1d { 1 } else { self.shape[self.shape.len() - 2] }; // m
        let a_cols = self.shape[self.shape.len() - 1]; // n
        let b_rows = if b_is_1d { other.shape[0] } else { other.shape[other.shape.len() - 2] }; // n
        let b_cols = if b_is_1d { 1 } else { other.shape[other.shape.len() - 1] }; // p

        if a_cols != b_rows {
            return Err(TensorErrors::MissMatchedShapes);
        }

        let a_batch: Vec<usize> = if a_is_1d {
            vec![]
        } else {
            self.shape[..self.shape.len().saturating_sub(2)].to_vec()
        };
        
        let b_batch: Vec<usize> = if b_is_1d {
            vec![]
        } else {
            other.shape[..other.shape.len().saturating_sub(2)].to_vec()
        };

        let batch_shape = broadcast_shapes(&a_batch, &b_batch)?;
        let mut output_shape = [batch_shape.as_slice(), &[a_rows, b_cols]].concat();

        if a_is_1d && output_shape.len() > 1 && output_shape[output_shape.len() - 2] == 1 {
            output_shape.remove(output_shape.len() - 2);
        }
        if b_is_1d && output_shape.len() > 1 && output_shape[output_shape.len() - 1] == 1 {
            output_shape.pop();
        }

        let batch_size = batch_shape.iter().product::<usize>();
        let output_size = batch_size * a_rows * b_cols;

        let mut result_data = vec![0.0; output_size];

        for batch_idx in 0..batch_size {
            let a_batch_offset = compute_batch_offset(&a_batch, batch_idx, &batch_shape)
                * a_rows
                * a_cols;
            let b_batch_offset = compute_batch_offset(&b_batch, batch_idx, &batch_shape)
                * b_rows
                * b_cols;

            for i in 0..a_rows {
                for j in 0..b_cols {
                    let mut sum = 0.0;
                    for k in 0..a_cols {
                        let a_idx = a_batch_offset
                            + (if a_is_1d { 0 } else { i * a_cols }) + k;
                        let b_idx = b_batch_offset
                            + k * (if b_is_1d { 1 } else { b_cols }) + (if b_is_1d { 0 } else { j });
                        sum += self.data[a_idx] * other.data[b_idx];
                    }
                    let result_idx = batch_idx * a_rows * b_cols + i * b_cols + j;
                    result_data[result_idx] = sum;
                }
            }
        }

        Tensor::new(
            result_data,
            output_shape,
            Some(self.requires_grad || other.requires_grad),
        )
    }
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, TensorErrors> {
    let max_len = a.len().max(b.len());
    let mut result = vec![1; max_len];

    let a_padded = pad_shape(a, max_len);
    let b_padded = pad_shape(b, max_len);

    for i in 0..max_len {
        if a_padded[i] != b_padded[i] && a_padded[i] != 1 && b_padded[i] != 1 {
            return Err(TensorErrors::MissMatchedShapes);
        }
        result[i] = a_padded[i].max(b_padded[i]);
    }

    Ok(result)
}

fn pad_shape(shape: &[usize], len: usize) -> Vec<usize> {
    let mut padded = vec![1; len - shape.len()];
    padded.extend_from_slice(shape);
    padded
}

fn compute_batch_offset(batch_shape: &[usize], batch_idx: usize, broadcast_shape: &[usize]) -> usize {
    let mut offset = 0;
    let mut idx = batch_idx;
    for (dim, &bcast_dim) in batch_shape.iter().zip(broadcast_shape.iter()).rev() {
        let stride = idx % bcast_dim;
        offset = offset * bcast_dim + if *dim == 1 { 0 } else { stride };
        idx /= bcast_dim;
    }
    offset
}

