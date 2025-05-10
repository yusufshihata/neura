#[cfg(test)]
mod tests {
    use super::*;
    use neura::tensor::tensor::{Tensor, TensorErrors};
    use std::ops::{Add, Sub};

    #[test]
    fn test_tensor_add_same_shape() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = t1 + t2;
        let expected = Tensor::new(vec![6.0, 8.0, 10.0, 12.0], vec![2, 2]);
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_sub_same_shape() {
        let t1 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = t1 - t2;
        let expected = Tensor::new(vec![4.0, 4.0, 4.0, 4.0], vec![2, 2]);
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_add_different_shapes_panic() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let t2 = Tensor::new(vec![4.0, 5.0], vec![2]);
        let result = t1 + t2;
        assert!(matches!(result, Err(TensorErrors::MissMatchedShapes)));
    }

    #[test]
    fn test_tensor_sub_different_shapes_panic() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let t2 = Tensor::new(vec![4.0, 5.0], vec![2]);
        let result = t1 - t2;
        assert!(matches!(result, Err(TensorErrors::MissMatchedShapes)));
    }

    #[test]
    fn test_tensor_add_scalar() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = t + 2.0;
        let expected = Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_sub_scalar() {
        let t = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = t - 2.0;
        let expected = Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_add_empty() {
        let t1 = Tensor::new(vec![], vec![]);
        let t2 = Tensor::new(vec![], vec![]);
        let result = t1 + t2;
        let expected = Tensor::new(vec![], vec![]);
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_sub_empty() {
        let t1 = Tensor::new(vec![], vec![]);
        let t2 = Tensor::new(vec![], vec![]);
        let result = t1 - t2;
        let expected = Tensor::new(vec![], vec![]);
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_add_assign_same_shape() {
        let mut t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        t1 += t2;
        let expected = Tensor::new(vec![6.0, 8.0, 10.0, 12.0], vec![2, 2]);
        assert_eq!(t1.data, expected.data);
        assert_eq!(t1.shape, expected.shape);
    }

    #[test]
    fn test_tensor_sub_assign_same_shape() {
        let mut t1 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        t1 -= t2;
        let expected = Tensor::new(vec![4.0, 4.0, 4.0, 4.0], vec![2, 2]);
        assert_eq!(t1.data, expected.data);
        assert_eq!(t1.shape, expected.shape);
    }

    #[test]
    fn test_tensor_add_assign_different_shapes_panic() {
        let mut t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let t2 = Tensor::new(vec![4.0, 5.0], vec![2]);
        let t1 += t2;
        assert!(matches!(result, Err(TensorErrors::MissMatchedShapes)));
    }

    #[test]
    fn test_tensor_sub_assign_different_shapes_panic() {
        let mut t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let t2 = Tensor::new(vec![4.0, 5.0], vec![2]);
        let t1 -= t2;
        assert!(matches!(result, Err(TensorErrors::MissMatchedShapes)));
    }
}
