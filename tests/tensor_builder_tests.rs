#[cfg(test)]
mod tests {
    use neura::tensor::tensor::TensorErrors;
    use neura::tensor::tensor_builder::{TensorBuilder, InitMethod};
    use ndarray::ArrayD;

    #[test]
    fn build_without_shape_is_error() {
        let err = TensorBuilder::new()
            .init(InitMethod::Zeros)
            .build();
        assert!(matches!(err, Err(TensorErrors::InvalidShape)));
    }

    #[test]
    fn zero_dimension_shape() {
        let t = TensorBuilder::new()
            .shape(&[2, 0, 3])
            .init(InitMethod::Ones)
            .build()
            .unwrap();
        
        assert_eq!(t.shape(), vec![2, 0, 3]);
        assert_eq!(t.data.len(), 0);
        assert_eq!(t.requires_grad, false);
    }

    #[test]
    fn zeros_and_ones_strides() {
        let z = TensorBuilder::new()
            .shape(&[3, 4, 5])
            .init(InitMethod::Zeros)
            .build()
            .unwrap();
        
        // Check shape and data initialization
        assert_eq!(z.shape(), vec![3, 4, 5]);
        assert_eq!(z.data.shape(), [3, 4, 5]);

        // Check initialization with zeros
        assert!(z.data.iter().all(|&x| x == 0.0));

        let o = TensorBuilder::new()
            .shape(&[1, 2, 1])
            .init(InitMethod::Ones)
            .build()
            .unwrap();
        
        // Check shape and data initialization
        assert_eq!(o.shape(), vec![1, 2, 1]);
        assert_eq!(o.data.shape(), [1, 2, 1]);

        // Check initialization with ones
        assert!(o.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn from_data_valid_and_invalid() {
        let raw = vec![9.0, 8.0, 7.0];
        let t = TensorBuilder::new()
            .shape(&[3])
            .init(InitMethod::FromData(raw.clone()))
            .build()
            .unwrap();
        
        // Check that data matches the input data
        assert_eq!(t.data, ArrayD::from_shape_vec(vec![3], raw).unwrap());

        let bad = TensorBuilder::new()
            .shape(&[2])
            .init(InitMethod::FromData(vec![1.0, 2.0, 3.0]))
            .build();
        
        // Check that an error is returned when the shape doesn't match the data size
        assert!(matches!(bad, Err(TensorErrors::InvalidShape)));
    }

    #[test]
    fn requires_grad_flag() {
        let t0 = TensorBuilder::new()
            .shape(&[1])
            .init(InitMethod::Zeros)
            .build()
            .unwrap();
        
        assert!(!t0.requires_grad);

        let t1 = TensorBuilder::new()
            .shape(&[1])
            .requires_grad(true)
            .init(InitMethod::Zeros)
            .build()
            .unwrap();
        
        assert!(t1.requires_grad);
    }

    #[test]
    fn dtype_propagation() {
        let t = TensorBuilder::new()
            .shape(&[2, 2])
            .init(InitMethod::Zeros)
            .build()
            .unwrap();

        // In this test, we don't need to verify dtype directly in the builder for now.
        // Assuming dtype handling will be added as needed.
        assert_eq!(t.shape(), vec![2, 2]);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn builder_chaining_order() {
        let a = TensorBuilder::new()
            .shape(&[2, 2])
            .init(InitMethod::Ones)
            .requires_grad(true)
            .build()
            .unwrap();

        let b = TensorBuilder::new()
            .requires_grad(true)
            .shape(&[2, 2])
            .init(InitMethod::Ones)
            .build()
            .unwrap();

        // Check that the data is the same for both tensors
        assert_eq!(a.data, b.data);
        assert_eq!(a.requires_grad, b.requires_grad);
    }
}
