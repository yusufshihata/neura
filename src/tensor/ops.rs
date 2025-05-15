use crate::tensor::tensor::{Tensor, TensorErrors};
use ndarray::{ArrayD, Axis, IxDyn, ShapeError, Zip, linalg::general_mat_mul, s};
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

impl Index<usize> for Tensor {
    type Output = f32;

    /// Returns a reference to the element of the tensor at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - A usize index specifying the position of the element within
    ///             the flattened tensor data.
    ///
    /// # Panics
    ///
    /// This function will panic if the index is out of bounds.

    fn index(&self, index: usize) -> &Self::Output {
        self.data.as_slice_memory_order().unwrap().index(index)
    }
}

impl IndexMut<usize> for Tensor {
    /// Returns a mutable reference to the element of the tensor at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - An usize index specifying the position of the element within
    ///             the flattened tensor data.
    ///
    /// # Panics
    ///
    /// This function will panic if the index is out of bounds.

    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.data
            .as_slice_memory_order_mut()
            .unwrap()
            .index_mut(index)
    }
}

impl Index<std::ops::Range<usize>> for Tensor {
    type Output = [f32];

    /// Returns a slice of elements from the tensor.
    ///
    /// # Arguments
    ///
    /// * `index` - A range of usize indices specifying the start and end of the
    ///             slice.
    ///
    /// # Panics
    ///
    /// This function will panic if the range is out of bounds.
    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        self.data.as_slice_memory_order().unwrap().index(index)
    }
}

impl IndexMut<std::ops::Range<usize>> for Tensor {
    /// Returns a mutable slice of elements from the tensor.
    ///
    /// # Arguments
    ///
    /// * `index` - A range of usize indices specifying the start and end of the
    ///             slice.
    ///
    /// # Panics
    ///
    /// This function will panic if the range is out of bounds.
    fn index_mut(&mut self, index: std::ops::Range<usize>) -> &mut Self::Output {
        self.data
            .as_slice_memory_order_mut()
            .unwrap()
            .index_mut(index)
    }
}

impl Tensor {
    /// Returns a new tensor which is a slice of the original tensor. The slice
    /// is specified by a vector of ranges, where each range corresponds to a
    /// dimension of the tensor. The range is specified as a start and end index.
    ///
    /// # Arguments
    ///
    /// * `ranges` - A vector of ranges, where each range is a start and end
    ///              index.
    ///
    /// # Errors
    ///
    /// This function will return an `Err` if the number of ranges does not match
    /// the number of dimensions of the tensor, or if any of the ranges are out
    /// of bounds.
    pub fn slice(&self, ranges: Vec<std::ops::Range<usize>>) -> Result<Tensor, TensorErrors> {
        if ranges.len() != self.ndim() {
            return Err(TensorErrors::InvalidShape);
        }

        let slice_spec: Vec<_> = ranges
            .into_iter()
            .map(|r| r.start as isize..r.end as isize)
            .collect();

        let sliced = match slice_spec.len() {
            1 => self.data.slice(s![slice_spec[0].clone()]).into_dyn(),
            2 => self
                .data
                .slice(s![slice_spec[0].clone(), slice_spec[1].clone()])
                .into_dyn(),
            3 => self
                .data
                .slice(s![
                    slice_spec[0].clone(),
                    slice_spec[1].clone(),
                    slice_spec[2].clone()
                ])
                .into_dyn(),
            4 => self
                .data
                .slice(s![
                    slice_spec[0].clone(),
                    slice_spec[1].clone(),
                    slice_spec[2].clone(),
                    slice_spec[3].clone()
                ])
                .into_dyn(),
            5 => self
                .data
                .slice(s![
                    slice_spec[0].clone(),
                    slice_spec[1].clone(),
                    slice_spec[2].clone(),
                    slice_spec[3].clone(),
                    slice_spec[4].clone()
                ])
                .into_dyn(),
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

    /// Adds two tensors element-wise and returns the resulting tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the first tensor.
    /// * `other` - A reference to the second tensor.
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise sum of the two input tensors. The
    /// `requires_grad` flag of the resulting tensor is set to `true` if either of
    /// the input tensors has `requires_grad` set to `true`.
    ///
    /// # Panics
    ///
    /// This function will panic if the shapes of the two tensors do not match.

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
    /// Element-wise adds the other tensor to the current tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to the tensor to add to the current tensor.
    ///
    /// # Returns
    ///
    /// The mutated current tensor.
    ///
    /// # Panics
    ///
    /// This function will panic if the shapes of the two tensors do not match.
    fn add_assign(&mut self, other: &'b Tensor) {
        self.data += &other.data;
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    /// Subtracts the second tensor from the first tensor element-wise and returns the resulting tensor.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the first tensor.
    /// * `other` - A reference to the second tensor.
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise difference of the two input tensors. The
    /// `requires_grad` flag of the resulting tensor is set to `true` if either of
    /// the input tensors has `requires_grad` set to `true`.
    ///
    /// # Panics
    ///
    /// This function will panic if the shapes of the two tensors do not match.

    fn sub(self, other: &'b Tensor) -> Self::Output {
        let data = &self.data - &other.data;

        Tensor {
            data,
            requires_grad: self.requires_grad || other.requires_grad,
            grad: None,
        }
    }
}

impl<'b> SubAssign<&'b Tensor> for Tensor {
    /// Subtracts the second tensor from the first tensor element-wise in-place.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to the second tensor.
    ///
    /// # Panics
    ///
    /// This function will panic if the shapes of the two tensors do not match.
    fn sub_assign(&mut self, other: &'b Tensor) {
        self.data -= &other.data;
    }
}

impl<'a> Mul<f32> for &'a Tensor {
    type Output = Tensor;

    /// Scales the tensor by the given scalar.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to scale the tensor with.
    ///
    /// # Returns
    ///
    /// A new tensor containing the scaled data of the current tensor. The
    /// `requires_grad` flag of the resulting tensor is set to the `requires_grad`
    /// flag of the current tensor.
    fn mul(self, scalar: f32) -> Self::Output {
        let data = &self.data * scalar;

        Tensor {
            data,
            requires_grad: self.requires_grad,
            grad: None,
        }
    }
}

impl MulAssign<f32> for Tensor {
    /// Multiplies each element of the tensor by the given scalar in-place.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value by which to multiply each element of the tensor.
    ///
    /// # Panics
    ///
    /// This function will panic if the tensor's data cannot be accessed as a mutable slice.

    fn mul_assign(&mut self, scalar: f32) {
        self.data *= scalar;
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'b Tensor {
    type Output = Result<Tensor, ShapeError>;
    /// Element-wise multiplies the two tensors and returns a new tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to the second tensor.
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise product of the two input tensors. The
    /// `requires_grad` flag of the resulting tensor is set to `true` if either of
    /// the input tensors has `requires_grad` set to `true`.
    ///
    /// # Errors
    ///
    /// This function will return an error if the two tensors do not have the same shape.
    fn mul(self, other: &'b Tensor) -> Self::Output {
        if self.shape() != other.shape() {
            return Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape));
        }

        let data = &self.data * &other.data;
        Ok(Tensor {
            data,
            requires_grad: self.requires_grad || other.requires_grad,
            grad: None,
        })
    }
}

impl<'b> MulAssign<&'b Tensor> for Tensor {
    /// Element-wise multiplies the second tensor with the first tensor in-place.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to the second tensor.
    ///
    /// # Panics
    ///
    /// This function will panic if the two tensors do not have the same shape.
    fn mul_assign(&mut self, other: &'b Tensor) {
        if self.shape() != other.shape() {
            panic!("Invalid Shapes.");
        }
        self.data *= &other.data;
    }
}
