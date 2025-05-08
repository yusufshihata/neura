use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum TensorErrors {
    #[error("Invalid tensor dimensions.")]
    InvalidShape,
    #[error("Index out of bound.")]
    OutOfBound,
}

pub struct Tensor {
    dims: usize,
    shape: Vec<usize>,
    size: usize,
    data: Vec<f32>,
    strides: Vec<usize>,
    requires_grad: bool,
}

impl<'a> Tensor {
    pub fn zeros(shape: Vec<usize>, requires_grad: Option<bool>) -> Self {
        let grad = match requires_grad {
            Some(requires_grad) => requires_grad,
            None => false,
        };

        let dims = shape.len();

        let mut size = 1;

        for dim in &shape {
            size *= dim;
        }

        let mut strides = vec![];

        for i in 0..dims {
            let mut stride = 1;
            
            for j in i+1..dims {
                stride *= shape[j];
            }
            
            strides.push(stride);
        }

        Tensor {
            dims: dims,
            shape: shape,
            size: size,
            data: vec![0.0; size],
            strides: strides,
            requires_grad: grad,
        }
    }

    
    pub fn ones(shape: Vec<usize>, requires_grad: Option<bool>) -> Self {
        let grad = match requires_grad {
            Some(requires_grad) => requires_grad,
            None => false,
        };

        let dims = shape.len();

        let mut size = 1;

        for dim in &shape {
            size *= dim;
        }

        let mut strides = vec![];

        for i in 0..dims {
            let mut stride = 1;
            
            for j in i+1..dims {
                stride *= shape[j];
            }
            
            strides.push(stride);
        }

        Tensor {
            dims: dims,
            shape: shape,
            size: size,
            data: vec![1.0; size],
            strides: strides,
            requires_grad: grad,
        }
    }

    pub fn size(&'a self) -> &'a usize {
        &self.size
    }

    pub fn view(&'a self) -> &'a Vec<usize> {
        &self.shape
    }

    pub fn strides(&'a self) -> &'a Vec<usize> {
        &self.strides
    }

    pub fn get(&'a self, index: Vec<usize>) -> Result<f32, TensorErrors> {
        let n = &index.len();
        if *n != self.dims {
            return Err(TensorErrors::InvalidShape);
        }

        let mut idx = 0;

        for i in 0..*n {
            if index[i] >= self.shape[i] {
                return Err(TensorErrors::OutOfBound);
            }

            idx += index[i] * self.strides[i];
        }
        
        Ok(self.data[idx])
    }
    pub fn data(&'a self) {
        for i in 0..(*self.size() as i32) {
            println!("{} ", self.data[i as usize]);
        }
    }

    pub fn requires_grad(&'a self) -> &'a bool {
        &self.requires_grad
    }
}

