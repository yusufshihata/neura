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
    pub dims: usize,
    pub shape: Vec<usize>,
    pub size: usize,
    pub data: Vec<f32>,
    pub strides: Vec<usize>,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: Option<bool>) -> Result<Self, TensorErrors> {
        let size = shape.iter().product::<usize>();
        if size != data.len() {
            return Err(TensorErrors::InvalidShape);
        }
        let dims = shape.len();
        let strides = Self::compute_strides(&shape);
        let grad = requires_grad.unwrap_or(false);
        Ok(Tensor {
            dims,
            shape,
            size,
            data,
            strides,
            requires_grad: grad,
        })
    }
    
    pub fn zeros(shape: Vec<usize>, requires_grad: Option<bool>) -> Self {
        let grad = requires_grad.unwrap_or(false);
        let dims = shape.len();
        let size = shape.iter().product();
        let strides = Self::compute_strides(&shape);
        Tensor {
            dims,
            shape,
            size,
            data: vec![0.0; size],
            strides,
            requires_grad: grad,
        }
    }

    pub fn ones(shape: Vec<usize>, requires_grad: Option<bool>) -> Self {
        let grad = requires_grad.unwrap_or(false);
        let dims = shape.len();
        let size = shape.iter().product();
        let strides = Self::compute_strides(&shape);
        Tensor {
            dims,
            shape,
            size,
            data: vec![1.0; size],
            strides,
            requires_grad: grad,
        }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        if !shape.is_empty() {
            strides[shape.len() - 1] = 1;
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        strides
    }

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
                strides: new_strides,
                requires_grad: self.requires_grad,
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
}
