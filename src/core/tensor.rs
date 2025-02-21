// Create a tensor sturct
#[derive(Debug)]
pub struct Tensor {
    size: i32,
    ndim: i32,
    shape: Vec<i32>,
    data: Vec<f32>,
    offset: i32,
}