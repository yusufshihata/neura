use ndarray::{ArrayD, IxDyn};

#[derive(Debug)]
pub struct Tensor {
    pub size: i32,
    pub rank: usize,
    pub shape: Vec<usize>,
    pub data: ArrayD<f64>,
    pub offset: i32,
    pub requires_grad: bool,
    pub grad: Option<ArrayD<f64>>,
}

impl Tensor {
    pub fn new(data: ArrayD<f64>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let size: i32 = shape.iter().product::<usize>() as i32;
        let rank: usize = shape.len();

        let grad = if requires_grad {
            Some(ArrayD::<f64>::default(IxDyn(&shape)))
        } else {
            None
        };

        Tensor {
            size,
            rank,
            shape,
            data,
            offset: 0,
            requires_grad: false,
            grad: grad,
        }
    }

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        let size: i32 = shape.iter().product::<usize>() as i32;
        let rank: usize = shape.len();
        let data: ArrayD<f64> = ArrayD::<f64>::zeros(IxDyn(shape));

        let grad = if requires_grad {
            Some(ArrayD::<f64>::zeros(IxDyn(shape)))
        } else {
            None
        };

        Tensor {
            size,
            rank,
            shape: shape.to_vec(),
            data,
            offset: 0,
            requires_grad,
            grad,
        }
    }

    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        let size: i32 = shape.iter().product::<usize>() as i32;
        let rank: usize = shape.len();
        let data: ArrayD<f64> = ArrayD::<f64>::ones(IxDyn(shape));

        let grad = if requires_grad {
            Some(ArrayD::<f64>::ones(IxDyn(shape)))
        } else {
            None
        };

        Tensor {
            size,
            rank,
            shape: shape.to_vec(),
            data,
            offset: 0,
            requires_grad,
            grad,
        }
    }
}
