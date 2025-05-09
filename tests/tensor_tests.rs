#[cfg(test)]
#[warn(unused_imports)]
mod tests {
    use neura::tensor::tensor::{Tensor,TensorErrors};
    use super::*;

    #[test]
    fn test_zeros() {
        let tensor = Tensor::zeros(vec![2,3], Some(true));

        
        assert_eq!(*tensor.size(), 6 as usize);
        assert_eq!(*tensor.view(), vec![2 as usize, 3 as usize]);
        assert_eq!(*tensor.strides(), vec![3 as usize, 1 as usize]);
    }

    #[test]
    fn test_ones() {
        let tensor = Tensor::ones(vec![2,3], Some(false));

        assert_eq!(*tensor.size(), 6 as usize);
        assert_eq!(*tensor.view(), vec![2 as usize, 3 as usize]);
        assert_eq!(*tensor.strides(), vec![3 as usize, 1 as usize]);
    }

   #[test]
    fn test_get_miss_matched_shape() {
        let tensor = Tensor::zeros(vec![3, 4, 2], None);
        assert!(matches!(
            tensor.get(vec![9, 4, 5, 6]),
            Err(TensorErrors::InvalidShape)
        ));
    }

    #[test]
    fn test_get_out_of_bound() {
        let tensor = Tensor::zeros(vec![3, 4, 5], None);
        assert!(matches!(
            tensor.get(vec![0, 4, 0]),
            Err(TensorErrors::OutOfBound)
        ));
    }

    #[test]
    fn test_get_success() {
        let tensor = Tensor::zeros(vec![3, 4, 5], None);
        assert_eq!(tensor.get(vec![0, 0, 0]), Ok(0.0));
    }

    #[test]
    fn test_slice_1d() {
        let mut tensor = Tensor::zeros(vec![10], None);
        // Populate with 1.0, 2.0, ..., 10.0
        for i in 0..10 {
            tensor.data[i] = (i + 1) as f32;
        }

        let sliced = tensor.slice(vec![1..8]).unwrap();
        assert_eq!(sliced.shape, vec![7]);
        assert_eq!(sliced.data, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(sliced.strides, vec![1]);
    }

    #[test]
    fn test_slice_2d() {
        let mut tensor = Tensor::zeros(vec![10, 5], None);
        // Populate with 1.0, 2.0, ..., 50.0
        for i in 0..10 {
            for j in 0..5 {
                tensor.data[i * 5 + j] = (i * 5 + j + 1) as f32;
            }
        }

        let sliced = tensor.slice(vec![1..8, 0..2]).unwrap();
        assert_eq!(sliced.shape, vec![7, 2]);
        assert_eq!(
            sliced.data,
            vec![6.0, 7.0, 11.0, 12.0, 16.0, 17.0, 21.0, 22.0, 26.0, 27.0, 31.0, 32.0, 36.0, 37.0]
        );
        assert_eq!(sliced.strides, vec![2, 1]);
    }

    #[test]
    fn test_slice_3d() {
        let mut tensor = Tensor::zeros(vec![4, 3, 2], None);
        // Populate with 1.0, 2.0, ..., 24.0
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..2 {
                    tensor.data[i * 6 + j * 2 + k] = (i * 6 + j * 2 + k + 1) as f32;
                }
            }
        }

        let sliced = tensor.slice(vec![1..3, 0..2, 0..1]).unwrap();
        assert_eq!(sliced.shape, vec![2, 2, 1]);
        assert_eq!(sliced.data, vec![7.0, 9.0, 13.0, 15.0]);
        assert_eq!(sliced.strides, vec![2, 1, 1]);
    }

    #[test]
    fn test_slice_invalid_dims() {
        let tensor = Tensor::zeros(vec![10, 5], None);
        let result = tensor.slice(vec![1..8]); // Wrong number of ranges
        assert!(matches!(result, Err(TensorErrors::InvalidShape)));
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let tensor = Tensor::zeros(vec![10, 5], None);
        let result = tensor.slice(vec![1..11, 0..2]); // Row range exceeds shape
        assert!(matches!(result, Err(TensorErrors::InvalidRange)));
    }

    #[test]
    fn test_slice_invalid_range() {
        let tensor = Tensor::zeros(vec![10, 5], None);
        let result = tensor.slice(vec![8..1, 0..2]); // Start > end
        assert!(matches!(result, Err(TensorErrors::InvalidRange)));
    }

    #[test]
    fn test_slice_empty() {
        let tensor = Tensor::zeros(vec![10, 5], None);
        let sliced = tensor.slice(vec![1..1, 0..0]).unwrap();
        assert_eq!(sliced.shape, vec![0, 0]);
        assert_eq!(sliced.data, vec![]);
        assert_eq!(sliced.strides, vec![0, 1]);
    }

    #[test]
    fn test_slice_full() {
        let mut tensor = Tensor::zeros(vec![3, 2], None);
        // Populate with 1.0, 2.0, ..., 6.0
        for i in 0..3 {
            for j in 0..2 {
                tensor.data[i * 2 + j] = (i * 2 + j + 1) as f32;
            }
        }

        let sliced = tensor.slice(vec![0..3, 0..2]).unwrap();
        assert_eq!(sliced.shape, vec![3, 2]);
        assert_eq!(sliced.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(sliced.strides, vec![2, 1]);
    }
}
