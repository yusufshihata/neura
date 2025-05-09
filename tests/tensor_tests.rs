#[cfg(test)]
#[warn(unused_imports)]
mod tests {
    use super::*;
    use neura::tensor::tensor::{Tensor, TensorErrors};

    fn create_sequential_tensor(shape: Vec<usize>) -> Tensor {
        let mut tensor = Tensor::zeros(shape, None);
        let size = tensor.size();
        for i in 0..size {
            tensor.data[i] = (i + 1) as f32;
        }
        tensor
    }

    #[test]
    fn test_zeros() {
        let tensor = Tensor::zeros(vec![2, 3], Some(true));
        assert_eq!(tensor.size(), 6);
        assert_eq!(tensor.shape(), &vec![2, 3]);
        assert_eq!(tensor.strides(), &vec![3, 1]);
        assert!(tensor.requires_grad());
        assert_eq!(tensor.data, vec![0.0; 6]);
    }

    #[test]
    fn test_ones() {
        let tensor = Tensor::ones(vec![2, 3], Some(false));
        assert_eq!(tensor.size(), 6);
        assert_eq!(tensor.shape(), &vec![2, 3]);
        assert_eq!(tensor.strides(), &vec![3, 1]);
        assert!(!tensor.requires_grad());
        assert_eq!(tensor.data, vec![1.0; 6]);
    }

    #[test]
    fn test_get_miss_matched_shape() {
        let tensor = Tensor::zeros(vec![3, 4, 2], None);
        assert!(matches!(
            tensor.get(&vec![9, 4, 5, 6]),
            Err(TensorErrors::InvalidShape)
        ));
    }

    #[test]
    fn test_get_out_of_bound() {
        let tensor = Tensor::zeros(vec![3, 4, 5], None);
        assert!(matches!(
            tensor.get(&vec![0, 4, 0]),
            Err(TensorErrors::OutOfBound)
        ));
    }

    #[test]
    fn test_get_success() {
        let tensor = create_sequential_tensor(vec![3, 4, 5]);
        assert_eq!(tensor.get(&vec![0, 0, 0]), Ok(1.0));
        assert_eq!(tensor.get(&vec![2, 3, 4]), Ok(60.0));
    }

    #[test]
    fn test_get_1d() {
        let tensor = create_sequential_tensor(vec![5]);
        assert_eq!(tensor.get(&vec![0]), Ok(1.0));
        assert_eq!(tensor.get(&vec![4]), Ok(5.0));
        assert!(matches!(
            tensor.get(&vec![5]),
            Err(TensorErrors::OutOfBound)
        ));
    }

    #[test]
    fn test_slice_1d() {
        let tensor = create_sequential_tensor(vec![10]);
        let sliced = tensor.slice(vec![1..8]).unwrap();
        assert_eq!(sliced.shape(), &vec![7]);
        assert_eq!(sliced.data, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(sliced.strides(), &vec![1]);
    }

    #[test]
    fn test_slice_2d() {
        let tensor = create_sequential_tensor(vec![10, 5]);
        let sliced = tensor.slice(vec![1..8, 0..2]).unwrap();
        assert_eq!(sliced.shape(), &vec![7, 2]);
        assert_eq!(
            sliced.data,
            vec![6.0, 7.0, 11.0, 12.0, 16.0, 17.0, 21.0, 22.0, 26.0, 27.0, 31.0, 32.0, 36.0, 37.0]
        );
        assert_eq!(sliced.strides(), &vec![2, 1]);
    }

    #[test]
    fn test_slice_3d() {
        let tensor = create_sequential_tensor(vec![4, 3, 2]);
        let sliced = tensor.slice(vec![1..3, 0..2, 0..1]).unwrap();
        assert_eq!(sliced.shape(), &vec![2, 2, 1]);
        assert_eq!(sliced.data, vec![7.0, 9.0, 13.0, 15.0]);
        assert_eq!(sliced.strides(), &vec![2, 1, 1]);
    }

    #[test]
    fn test_slice_invalid_dims() {
        let tensor = Tensor::zeros(vec![10, 5], None);
        let result = tensor.slice(vec![1..8]);
        assert!(matches!(result, Err(TensorErrors::InvalidShape)));
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let tensor = Tensor::zeros(vec![10, 5], None);
        let result = tensor.slice(vec![1..11, 0..2]);
        assert!(matches!(result, Err(TensorErrors::InvalidRange)));
    }

    #[test]
    fn test_slice_invalid_range() {
        let tensor = Tensor::zeros(vec![10, 5], None);
        let result = tensor.slice(vec![8..1, 0..2]);
        assert!(matches!(result, Err(TensorErrors::InvalidRange)));
    }

    #[test]
    fn test_slice_empty() {
        let tensor = Tensor::zeros(vec![10, 5], None);
        let sliced = tensor.slice(vec![1..1, 0..0]).unwrap();
        assert_eq!(sliced.shape(), &vec![0, 0]);
        assert_eq!(sliced.data, vec![]);
        assert_eq!(sliced.strides(), &vec![0, 1]);
    }

    #[test]
    fn test_slice_full() {
        let tensor = create_sequential_tensor(vec![3, 2]);
        let sliced = tensor.slice(vec![0..3, 0..2]).unwrap();
        assert_eq!(sliced.shape(), &vec![3, 2]);
        assert_eq!(sliced.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(sliced.strides(), &vec![2, 1]);
    }

    #[test]
    fn test_slice_2d_partial() {
        let tensor = create_sequential_tensor(vec![3, 4]);
        let sliced = tensor.slice(vec![0..2, 1..3]).unwrap();
        assert_eq!(sliced.shape(), &vec![2, 2]);
        assert_eq!(sliced.data, vec![2.0, 3.0, 6.0, 7.0]);
        assert_eq!(sliced.strides(), &vec![2, 1]);
    }

    #[test]
    fn test_slice_preserves_requires_grad() {
        let tensor = Tensor::ones(vec![2, 3], Some(true));
        let sliced = tensor.slice(vec![0..1, 1..2]).unwrap();
        assert!(sliced.requires_grad());
    }

    #[test]
    fn test_slice_on_zero_size_tensor() {
        let tensor = Tensor::zeros(vec![0, 3], None);
        let sliced = tensor.slice(vec![0..0, 1..2]).unwrap();
        assert_eq!(sliced.shape(), &vec![0, 1]);
        assert_eq!(sliced.data, vec![]);
        assert_eq!(sliced.strides(), &vec![1, 1]);
    }
}
