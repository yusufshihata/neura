use crate::tensor::tensor::{ Tensor, TensorErrors };

pub enum InitMethod {
    FromData(Vec<f32>),
    Ones,
    Zeros,
}

pub struct TensorBuilder {
    shape: Vec<usize>,
    requires_grad: bool,
    init_method: InitMethod,
}

impl TensorBuilder {
    pub fn shape(mut self, shape: &[usize]) -> Self {
        self.shape = shape.to_vec();

        self
    }

    pub fn requires_grad(mut self, flag: bool) -> Self {
        self.requires_grad = flag;

        self
    }


    pub fn init(mut self, method: InitMethod) -> Self {
        self.init_method = method;

        self
    }

    pub fn compute_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.shape.len()];

        for i in (0..self.shape.len() - 1).rev() {
            strides[i] = strides[i+1] * self.shape[i+1];
        }

        strides
    }

    pub fn build(self) -> Result<Tensor, TensorErrors> {
        let size: usize = self.shape.iter().product();

        if self.shape == vec![] {
            return Err(TensorErrors::InvalidShape);
        }

        let strides = self.compute_strides();

        let data = match &self.init_method {
            InitMethod::Zeros => vec![0.0; size],
            InitMethod::Ones => vec![1.0; size],
            InitMethod::FromData(d) => {
                if d.len() != size {
                    return Err(TensorErrors::InvalidShape);
                }
                d.clone()
            },
        };

        Ok(Tensor {
            size,
            dims: self.shape.len(),
            shape: self.shape,
            data,
            strides,
            requires_grad: self.requires_grad,
        })
    }

    pub fn new() -> Self {
        Self {
            shape: vec![],
            requires_grad: false,
            init_method: InitMethod::Zeros,
        }
    }
}