#[cfg(test)]
mod tests {
    use neura::tensor::tensor::{Tensor, TensorErrors};

    #[test]
    fn test_tensor_add_same_shape() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], None).unwrap();
        let t2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], None).unwrap();
        let result = (t1 + t2).unwrap();
        let expected = Tensor::new(vec![6.0, 8.0, 10.0, 12.0], vec![2, 2], None).unwrap();
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_sub_same_shape() {
        let t1 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], None).unwrap();
        let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], None).unwrap();
        let result = (t1 - t2).unwrap();
        let expected = Tensor::new(vec![4.0, 4.0, 4.0, 4.0], vec![2, 2], None).unwrap();
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_add_different_shapes() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], None).unwrap();
        let t2 = Tensor::new(vec![4.0, 5.0], vec![2], None).unwrap();
        let result = t1 + t2;
        assert!(matches!(result, Err(TensorErrors::MissMatchedShapes)));
    }

    #[test]
    fn test_tensor_sub_different_shapes() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], None).unwrap();
        let t2 = Tensor::new(vec![4.0, 5.0], vec![2], None).unwrap();
        let result = t1 - t2;
        assert!(matches!(result, Err(TensorErrors::MissMatchedShapes)));
    }


    #[test]
    fn test_tensor_add_assign_same_shape() {
        let mut t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], None).unwrap();
        let t2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], None).unwrap();
        t1 += t2;
        let expected = Tensor::new(vec![6.0, 8.0, 10.0, 12.0], vec![2, 2], None).unwrap();
        assert_eq!(t1.data, expected.data);
        assert_eq!(t1.shape, expected.shape);
    }

    #[test]
    fn test_tensor_sub_assign_same_shape() {
        let mut t1 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], None).unwrap();
        let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], None).unwrap();
        t1 -= t2;
        let expected = Tensor::new(vec![4.0, 4.0, 4.0, 4.0], vec![2, 2], None).unwrap();
        assert_eq!(t1.data, expected.data);
        assert_eq!(t1.shape, expected.shape);
    }

    #[test]
    #[should_panic]
    fn test_tensor_add_assign_different_shapes() {
        let mut t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], None).unwrap();
        let t2 = Tensor::new(vec![4.0, 5.0], vec![2], None).unwrap();
        t1 += t2;
    }

    #[test]
    #[should_panic]
    fn test_tensor_sub_assign_different_shapes() {
        let mut t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], None).unwrap();
        let t2 = Tensor::new(vec![4.0, 5.0], vec![2], None).unwrap();
        t1 -= t2;
    }

    #[test]
    fn test_tensor_new_invalid_shape() {
        let result = Tensor::new(vec![1.0, 2.0], vec![3], None);
        assert!(matches!(result, Err(TensorErrors::InvalidShape)));
    }

    #[test]
    fn test_tensor_scalar_mul() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], None).unwrap();
        let scalar = 2.0;
        let result = (t * scalar);
        let expected = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2], None).unwrap();
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_scalar_mul_zero() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], None).unwrap();
        let scalar = 0.0;
        let result = (t * scalar);
        let expected = Tensor::new(vec![0.0, 0.0, 0.0], vec![3], None).unwrap();
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_scalar_mul_negative() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], None).unwrap();
        let scalar = -1.0;
        let result = (t * scalar);
        let expected = Tensor::new(vec![-1.0, -2.0, -3.0, -4.0], vec![2, 2], None).unwrap();
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_scalar_mul_assign() {
        let mut t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], None).unwrap();
        let scalar = 3.0;
        t *= scalar;
        let expected = Tensor::new(vec![3.0, 6.0, 9.0, 12.0], vec![2, 2], None).unwrap();
        assert_eq!(t.data, expected.data);
        assert_eq!(t.shape, expected.shape);
    }
    
    #[test]
    fn test_tensor_element_wise_mul_same_shape() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], None).unwrap();
        let t2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], None).unwrap();
        let result = (t1 * t2).unwrap();
        let expected = Tensor::new(vec![5.0, 12.0, 21.0, 32.0], vec![2, 2], None).unwrap();
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_element_wise_mul_different_shapes() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], None).unwrap();
        let t2 = Tensor::new(vec![4.0, 5.0], vec![2], None).unwrap();
        let result = t1 * t2;
        assert!(matches!(result, Err(TensorErrors::MissMatchedShapes)));
    }

    #[test]
    fn test_tensor_element_wise_mul_empty_tensors() {
        let t1 = Tensor::new(vec![], vec![0], None).unwrap();
        let t2 = Tensor::new(vec![], vec![0], None).unwrap();
        let result = (t1 * t2).unwrap();
        let expected = Tensor::new(vec![], vec![0], None).unwrap();
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }

    #[test]
    fn test_tensor_element_wise_mul_1d_tensors() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], None).unwrap();
        let t2 = Tensor::new(vec![4.0, 5.0, 6.0], vec![3], None).unwrap();
        let result = (t1 * t2).unwrap();
        let expected = Tensor::new(vec![4.0, 10.0, 18.0], vec![3], None).unwrap();
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }
}
