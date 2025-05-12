#[cfg(test)]
mod tests {
    use neura::tensor::tensor::{Tensor, TensorErrors};
    use neura::tensor::tensor_builder::{TensorBuilder, InitMethod};

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
        assert_eq!(t.shape, vec![2, 0, 3]);
        assert_eq!(t.size, 0);
        assert_eq!(t.data.len(), 0);
        assert_eq!(t.strides, vec![0, 3, 1]);
    }

    #[test]
    fn zeros_and_ones_strides() {
        let z = TensorBuilder::new()
            .shape(&[3, 4, 5])
            .init(InitMethod::Zeros)
            .build()
            .unwrap();
        assert_eq!(z.size, 60);
        assert_eq!(z.strides, vec![20, 5, 1]);
        assert!(z.data.iter().all(|&x| x == 0.0));

        let o = TensorBuilder::new()
            .shape(&[1, 2, 1])
            .init(InitMethod::Ones)
            .build()
            .unwrap();
        assert_eq!(o.size, 2);
        assert!(o.data.iter().all(|&x| x == 1.0));
        assert_eq!(o.strides, vec![2, 1, 1]);
    }

    #[test]
    fn from_data_valid_and_invalid() {
        let raw = vec![9.0, 8.0, 7.0];
        let t = TensorBuilder::new()
            .shape(&[3])
            .init(InitMethod::FromData(raw.clone()))
            .build()
            .unwrap();
        assert_eq!(t.data, raw);

        let bad = TensorBuilder::new()
            .shape(&[2])
            .init(InitMethod::FromData(vec![1.0, 2.0, 3.0]))
            .build();
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

        assert_eq!(a.data, b.data);
        assert_eq!(a.requires_grad, b.requires_grad);
    }
}
