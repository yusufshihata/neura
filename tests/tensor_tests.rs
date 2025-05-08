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
}
