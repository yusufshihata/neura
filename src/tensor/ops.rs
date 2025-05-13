use std::ops::{ Index, IndexMut, Add, AddAssign, Sub, SubAssign, Mul, MulAssign };
use ndarray::{ArrayD, s};
use crate::tensor::tensor::{ Tensor, TensorErrors };


impl Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        self.data.as_slice_memory_order().unwrap().index(index)
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.data.as_slice_memory_order_mut().unwrap().index_mut(index)
    }
}

impl Index<std::ops::Range<usize>> for Tensor {
    type Output = [f32];

    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        self.data.as_slice_memory_order().unwrap().index(index)
    }
}

impl IndexMut<std::ops::Range<usize>> for Tensor {
    fn index_mut(&mut self, index: std::ops::Range<usize>) -> &mut Self::Output {
        self.data.as_slice_memory_order_mut().unwrap().index_mut(index)
    }
}

impl Tensor {
    pub fn slice(&self, ranges: Vec<std::ops::Range<usize>>) -> Result<Tensor, TensorErrors> {
        if ranges.len() != self.ndim() {
            return Err(TensorErrors::InvalidShape);
        }

        let slice_spec: Vec<_> = ranges.into_iter()
            .map(|r| r.start as isize..r.end as isize)
            .collect();
        
        let sliced = match slice_spec.len() {
            1 => self.data.slice(s![slice_spec[0].clone()]).into_dyn(),
            2 => self.data.slice(s![
                slice_spec[0].clone(),
                slice_spec[1].clone()
            ]).into_dyn(),
            3 => self.data.slice(s![
                slice_spec[0].clone(),
                slice_spec[1].clone(),
                slice_spec[2].clone()
            ]).into_dyn(),
            4 => self.data.slice(s![
                slice_spec[0].clone(),
                slice_spec[1].clone(),
                slice_spec[2].clone(),
                slice_spec[3].clone()
            ]).into_dyn(),
            5 => self.data.slice(s![
                slice_spec[0].clone(),
                slice_spec[1].clone(),
                slice_spec[2].clone(),
                slice_spec[3].clone(),
                slice_spec[4].clone()
            ]).into_dyn(),
            _ => return Err(TensorErrors::InvalidShape),
        };
        
        Ok(Tensor {
            data: sliced.into_owned(),
            requires_grad: self.requires_grad,
            grad: None,
        })
    }
}

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, other: &'b Tensor) -> Self::Output {
        let data = &self.data + &other.data;

        Tensor {
            data,
            requires_grad: self.requires_grad || other.requires_grad,
            grad: None,
        }
    }
}

impl<'b> AddAssign<&'b Tensor> for Tensor {
    fn add_assign(&mut self, other: &'b Tensor) {
        self.data += &other.data;
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    
    fn sub(self, other: &'b Tensor) -> Self::Output {
        let data = &self.data - &other.data;

        Tensor {
            data,
            requires_grad: self.requires_grad || other.requires_grad,
            grad: None
        }
    }
}

impl<'b> SubAssign<&'b Tensor> for Tensor {
    fn sub_assign(&mut self, other: &'b Tensor) {
        self.data -= &other.data;
    }
}


impl<'a> Mul<f32> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Self::Output {
        let data = &self.data * scalar;

        Tensor {
            data,
            requires_grad: self.requires_grad,
            grad: None
        }
    }
}

impl MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, scalar: f32) {
        self.data *= scalar;
    }
}

