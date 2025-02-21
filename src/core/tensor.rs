use ndarray::{ArrayD, IxDyn};

#[derive(Debug)]
pub struct Tensor<T> {
    pub size: i32,
    pub ndim: usize,
    pub shape: Vec<usize>,
    pub data: ArrayD<T>,
    pub offset: i32,
    pub requires_grad: bool,
    pub grad: Option<ArrayD<T>>,
}

impl<T: Default + Clone> Tensor<T> {
    pub fn new(shape: &[usize], requires_grad: bool) -> Self {
        let size: i32 = shape.iter().product::<usize>() as i32;
        let ndim: usize = shape.len();
        let data: ArrayD<T> = ArrayD::<T>::default(IxDyn(shape));

        let grad = if requires_grad {
            Some(ArrayD::<T>::default(IxDyn(shape)))
        } else {
            None
        };

        Tensor {
            size,
            ndim,
            shape: shape.to_vec(),
            data,
            offset: 0,
            requires_grad,
            grad,
        }
    }
}
