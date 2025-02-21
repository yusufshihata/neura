#[cfg(test)]
mod tests {
    use neura::core::tensor::Tensor;
    use ndarray::{IxDyn, Dimension};

    #[test]
    fn test_tensor_creation() {
        let shape = vec![3, 4, 5]; 
        let tensor: Tensor<f32> = Tensor::new(&shape, false);

        // Check basic properties
        assert_eq!(tensor.ndim, 3);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.size, (3 * 4 * 5) as i32);
        assert_eq!(tensor.requires_grad, false);
        assert!(tensor.grad.is_none());
    }

    #[test]
    fn test_tensor_requires_grad() {
        let shape = vec![2, 2];
        let tensor: Tensor<f32> = Tensor::new(&shape, true);

        assert!(tensor.requires_grad);
        assert!(tensor.grad.is_some());
    }

    #[test]
    fn test_tensor_data_shape() {
        let shape = vec![2, 3, 4];
        let tensor: Tensor<f32> = Tensor::new(&shape, false);

        assert_eq!(tensor.data.shape(), IxDyn(&shape).as_array_view().shape());
    }
}
