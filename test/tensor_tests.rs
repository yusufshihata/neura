#[cfg(test)]
mod tests {
    use neura::core::tensor::Tensor;
    use ndarray::{s, ArrayD, IxDyn};

    #[test]
    fn test_tensor_creation() {
        let shape = vec![3, 4, 5];
        let data = ArrayD::<f64>::ones(IxDyn(&shape));
        let tensor: Tensor = Tensor::new(data, shape, false);

        // Check basic properties
        assert_eq!(tensor.rank, 3);
        assert_eq!(tensor.size, (3 * 4 * 5) as i32);
        assert_eq!(tensor.requires_grad, false);
        assert!(tensor.grad.is_none());
    }

    #[test]
    fn test_tensor_requires_grad() {
        let shape = vec![2, 2];
        let data = ArrayD::<f64>::ones(IxDyn(&shape));
        let tensor: Tensor = Tensor::new(data, shape, true);

        assert!(tensor.grad.is_some());
    }

    #[test]
    fn test_tensor_zeros() {
        let shape = vec![2, 3, 4];
        let tensor: Tensor = Tensor::zeros(&shape, false);

        println!("{:?}", tensor.data);

        assert_eq!(tensor.data.shape(), shape);
    }

    #[test]
    fn test_tensor_ones() {
        let shape = vec![2, 3, 4];
        let tensor: Tensor = Tensor::ones(&shape, false);

        println!("{:?}", tensor.data);

        assert_eq!(tensor.data.shape(), shape);
    }

    #[test]
    fn test_tensor_get_item() {
        let shape = vec![1, 2, 2];
        let tensor: Tensor = Tensor::ones(&shape, false);

        assert_eq!(tensor.data[[0,1,1]], 1.0);
    }

    #[test]
    fn test_tensor_set_item() {
        let shape = vec![1, 2, 2];
        let mut tensor: Tensor = Tensor::ones(&shape, false);
        tensor.data[[0, 1, 1]] = 10.0;

        assert_eq!(tensor.data[[0, 1, 1]], 10.0);
    }

    #[test]
    fn test_tensor_get_slice() {
        let shape = vec![1,2,3];
        let tensor: Tensor = Tensor::ones(&shape, false);

        println!("{:?}", tensor.data.slice(s![0, 0, 0..2]));
    }
}
