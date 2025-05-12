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
    pub grad: Option<ArrayD<f32>>
}

impl Tensor {
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn strides(&self) -> Vec<isize> {
        self.data.strides().to_vec()
    }
}

/*
impl Tensor {
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn strides(&self) -> &Vec<usize> {
        &self.strides
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn get(&self, index: &[usize]) -> Result<f32, TensorErrors> {
        if index.len() != self.dims {
            return Err(TensorErrors::InvalidShape);
        }
        let mut idx = 0;
        for (i, &dim_idx) in index.iter().enumerate() {
            if dim_idx >= self.shape[i] {
                return Err(TensorErrors::OutOfBound);
            }
            idx += dim_idx * self.strides[i];
        }
        Ok(self.data[idx])
    }


    pub fn compute_strides(self, shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; self.shape.len()];

        for i in (0..self.shape.len() - 1).rev() {
            strides[i] = strides[i+1] * self.shape[i+1];
        }

        strides
    }
    
    pub fn slice(&self, ranges: Vec<std::ops::Range<usize>>) -> Result<Tensor, TensorErrors> {

        if ranges.len() != self.dims {
                return Err(TensorErrors::InvalidShape);
            }

        let mut new_shape = Vec::with_capacity(self.dims);
        for (i, range) in ranges.iter().enumerate() {
            // Corrected the range check here: changed >= to >
            if range.start > self.shape[i] || range.end > self.shape[i] || range.start > range.end {
                return Err(TensorErrors::InvalidRange);
            }
            new_shape.push(range.end - range.start);
        }
        
        let new_size = new_shape.iter().product();
        let new_strides = Self::compute_strides(&new_shape);

        if new_size == 0 {
            return Ok(Tensor {
                dims: self.dims,
                shape: new_shape,
                size: 0,
                data: Vec::new(),
                strides: new_strides, requires_grad: self.requires_grad,
            });
        }

        let mut new_data = vec![0.0; new_size];
        let mut indices = vec![0; self.dims];
        let mut pos = 0;
        loop {
            let orig_index: Vec<usize> = indices
                .iter()
                .zip(ranges.iter())
                .map(|(&i, r)| r.start + i)
                .collect();
            new_data[pos] = self.get(&orig_index)?;
            pos += 1;

            let mut dim = self.dims - 1;
            loop {
                indices[dim] += 1;
                if indices[dim] < new_shape[dim] {
                    break;
                }
                indices[dim] = 0;
                if dim == 0 {
                    return Ok(Tensor {
                        dims: self.dims,
                        shape: new_shape,
                        size: new_size,
                        data: new_data,
                        strides: new_strides,
                        requires_grad: self.requires_grad,
                    });
                }
                dim -= 1;
            }
        }
    }
    
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor, TensorErrors> {
        let new_size = new_shape.iter().product::<usize>();
        if new_size != self.size {
            return Err(TensorErrors::InvalidShape);
        }
        Tensor::new(self.data.clone(), new_shape, Some(self.requires_grad))
    }

    pub fn apply<F>(&self, f: F) -> Result<Tensor, TensorErrors>
    where
        F: Fn(f32) -> f32,
    {
        let new_data = self.data.iter().map(|&x| f(x)).collect();
        Tensor::new(new_data, self.shape.clone(), Some(self.requires_grad))
    }
}

*/