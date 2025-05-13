#[cfg(test)]
mod tests {
    use neura::tensor::tensor::{Tensor, TensorErrors};
    use neura::tensor::tensor_builder::{TensorBuilder, InitMethod};

    #[test]
    fn test_tensor_is_contiguous() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        assert!(tensor.data.is_standard_layout());
    }

    #[test]
    fn test_index_usize_valid() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        assert_eq!(tensor[0], 0.0);
        assert_eq!(tensor[3], 3.0);
        assert_eq!(tensor[5], 5.0);
    }

    #[test]
    #[should_panic]
    fn test_index_usize_invalid() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        let _ = tensor[6];
    }

    #[test]
    fn test_index_mut_usize_valid() {
        let mut tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        tensor[1] = 10.0;
        assert_eq!(tensor[1], 10.0);
        tensor[4] = 20.0;
        assert_eq!(tensor[4], 20.0);
    }

    #[test]
    #[should_panic]
    fn test_index_mut_usize_invalid() {
        let mut tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        tensor[6] = 100.0;
    }

    #[test]
    fn test_index_range_valid() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        assert_eq!(&tensor[1..4], &[1.0, 2.0, 3.0]);
        assert_eq!(&tensor[0..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(&tensor[3..3], &[]);
    }

    #[test]
    #[should_panic]
    fn test_index_range_partially_out_of_bounds() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        let _ = &tensor[4..7];
    }

    #[test]
    #[should_panic]
    fn test_index_range_completely_out_of_bounds() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        let _ = &tensor[6..8];
    }

    #[test]
    fn test_index_mut_range_valid() {
        let mut tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        tensor[2..5].copy_from_slice(&[100.0, 200.0, 300.0]);
        assert_eq!(tensor[2], 100.0);
        assert_eq!(tensor[3], 200.0);
        assert_eq!(tensor[4], 300.0);
    }

    #[test]
    #[should_panic]
    fn test_index_mut_range_partially_out_of_bounds() {
        let mut tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        tensor[4..7].copy_from_slice(&[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic]
    fn test_index_mut_range_completely_out_of_bounds() {
        let mut tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        tensor[6..8].copy_from_slice(&[1.0, 2.0]);
    }

    #[test]
    fn test_zero_element_tensor_empty_range() {
        let tensor = TensorBuilder::new()
            .shape(&[0])
            .init(InitMethod::FromData(vec![]))
            .build()
            .unwrap();
        assert_eq!(&tensor[0..0], &[]);
    }

    #[test]
    #[should_panic]
    fn test_zero_element_tensor_panic() {
        let tensor = TensorBuilder::new()
            .shape(&[0])
            .init(InitMethod::FromData(vec![]))
            .build()
            .unwrap();
        let _ = tensor[0];
    }

    #[test]
    fn test_one_element_tensor() {
        let mut tensor = TensorBuilder::new()
            .shape(&[1])
            .init(InitMethod::FromData(vec![42.0]))
            .build()
            .unwrap();
        assert_eq!(tensor[0], 42.0);
        tensor[0] = 100.0;
        assert_eq!(tensor[0], 100.0);
        assert_eq!(&tensor[0..1], &[100.0]);
        tensor[0..1].copy_from_slice(&[200.0]);
        assert_eq!(tensor[0], 200.0);
    }

    #[test]
    fn test_empty_range() {
        let mut tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        assert_eq!(&tensor[2..2], &[]);
        tensor[2..2].copy_from_slice(&[]); // No-op
        assert_eq!(&tensor[0..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic]
    fn test_reverse_range() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        let _ = &tensor[5..2];
    }

    #[test]
    fn test_different_shape() {
        let tensor = TensorBuilder::new()
            .shape(&[3, 2])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        assert_eq!(tensor[0], 0.0);
        assert_eq!(tensor[1], 1.0);
        assert_eq!(tensor[2], 2.0);
        assert_eq!(tensor[3], 3.0);
        assert_eq!(tensor[4], 4.0);
        assert_eq!(tensor[5], 5.0);
        assert_eq!(&tensor[1..4], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_flat_indexing_1d() {
        let tensor = TensorBuilder::new()
            .shape(&[5])
            .init(InitMethod::FromData(vec![10.0, 20.0, 30.0, 40.0, 50.0]))
            .build()
            .unwrap();
        assert_eq!(tensor[0], 10.0);
        assert_eq!(tensor[2], 30.0);
        assert_eq!(tensor[4], 50.0);
    }

    #[test]
    fn test_flat_indexing_2d() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        // Row-major: [0.0, 1.0, 2.0] (row 0), [3.0, 4.0, 5.0] (row 1)
        assert_eq!(tensor[0], 0.0); // (0, 0)
        assert_eq!(tensor[1], 1.0); // (0, 1)
        assert_eq!(tensor[2], 2.0); // (0, 2)
        assert_eq!(tensor[3], 3.0); // (1, 0)
        assert_eq!(tensor[4], 4.0); // (1, 1)
        assert_eq!(tensor[5], 5.0); // (1, 2)
    }

    #[test]
    fn test_flat_indexing_3d() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 2, 2])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
            .build()
            .unwrap();
        // Row-major: [0.0, 1.0] (0,0,:), [2.0, 3.0] (0,1,:), [4.0, 5.0] (1,0,:), [6.0, 7.0] (1,1,:)
        assert_eq!(tensor[0], 0.0); // (0, 0, 0)
        assert_eq!(tensor[1], 1.0); // (0, 0, 1)
        assert_eq!(tensor[2], 2.0); // (0, 1, 0)
        assert_eq!(tensor[3], 3.0); // (0, 1, 1)
        assert_eq!(tensor[4], 4.0); // (1, 0, 0)
        assert_eq!(tensor[5], 5.0); // (1, 0, 1)
        assert_eq!(tensor[6], 6.0); // (1, 1, 0)
        assert_eq!(tensor[7], 7.0); // (1, 1, 1)
    }

    #[test]
    fn test_range_slicing_2d() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        // Flat indices: 0..3 is row 0, 3..6 is row 1
        assert_eq!(&tensor[0..3], &[0.0, 1.0, 2.0]); // First row
        assert_eq!(&tensor[3..6], &[3.0, 4.0, 5.0]); // Second row
        assert_eq!(&tensor[1..4], &[1.0, 2.0, 3.0]); // Across rows
    }

    #[test]
    fn test_range_slicing_3d() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 2, 2])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
            .build()
            .unwrap();
        assert_eq!(&tensor[0..4], &[0.0, 1.0, 2.0, 3.0]); // First "plane"
        assert_eq!(&tensor[4..8], &[4.0, 5.0, 6.0, 7.0]); // Second "plane"
        assert_eq!(&tensor[2..6], &[2.0, 3.0, 4.0, 5.0]); // Across planes
    }

    #[test]
    fn test_range_slicing_edge_cases_2d() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        assert_eq!(&tensor[0..1], &[0.0]); // Start
        assert_eq!(&tensor[5..6], &[5.0]); // End
        assert_eq!(&tensor[2..4], &[2.0, 3.0]); // Middle
        assert_eq!(&tensor[0..0], &[]); // Empty at start
        assert_eq!(&tensor[6..6], &[]); // Empty at end
    }

    #[test]
    fn test_mutation_through_range_slicing_2d() {
        let mut tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        tensor[0..3].copy_from_slice(&[10.0, 20.0, 30.0]); // Mutate first row
        assert_eq!(&tensor[0..6], &[10.0, 20.0, 30.0, 3.0, 4.0, 5.0]);
        tensor[3..6].copy_from_slice(&[40.0, 50.0, 60.0]); // Mutate second row
        assert_eq!(&tensor[0..6], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn test_mutation_through_range_slicing_3d() {
        let mut tensor = TensorBuilder::new()
            .shape(&[2, 2, 2])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
            .build()
            .unwrap();
        tensor[0..4].copy_from_slice(&[10.0, 20.0, 30.0, 40.0]); // Mutate first plane
        assert_eq!(&tensor[0..8], &[10.0, 20.0, 30.0, 40.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_slice_2d_tensor() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();

        let sliced = tensor.slice(vec![0..1, 0..3]).unwrap();
        assert_eq!(sliced.shape(), &[1, 3]);
        assert_eq!(sliced.data.as_slice().unwrap(), &[0.0, 1.0, 2.0]);

        let sliced = tensor.slice(vec![0..2, 1..2]).unwrap();
        assert_eq!(sliced.shape(), &[2, 1]);
        assert_eq!(sliced.data.as_slice().unwrap(), &[1.0, 4.0]);

        let sliced = tensor.slice(vec![0..2, 1..3]).unwrap();
        assert_eq!(sliced.shape(), &[2, 2]);
        assert_eq!(sliced.data.as_slice().unwrap(), &[1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn test_slice_3d_tensor() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 2, 2])
            .init(InitMethod::FromData((0..8).map(|x| x as f32).collect()))
            .build()
            .unwrap();

        let sliced = tensor.slice(vec![0..1, 0..2, 0..2]).unwrap();
        assert_eq!(sliced.shape(), &[1, 2, 2]);
        assert_eq!(sliced.data.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0]);

        let sliced = tensor.slice(vec![0..2, 1..2, 1..2]).unwrap();
        assert_eq!(sliced.shape(), &[2, 1, 1]);
        assert_eq!(sliced.data.as_slice().unwrap(), &[3.0, 7.0]);
    }

    #[test]
    fn test_slice_invalid_dimensions() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();

        let result = tensor.slice(vec![0..1]);
        assert!(matches!(result, Err(TensorErrors::InvalidShape)));

        let result = tensor.slice(vec![0..1, 0..1, 0..1]);
        assert!(matches!(result, Err(TensorErrors::InvalidShape)));
    }

    #[test]
    fn test_slice_empty_ranges() {
        let tensor = TensorBuilder::new()
            .shape(&[2, 3])
            .init(InitMethod::FromData(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();

        let sliced = tensor.slice(vec![0..0, 0..3]).unwrap();
        assert_eq!(sliced.shape(), &[0, 3]);
        assert_eq!(sliced.data.as_slice().unwrap(), &[]);

        let sliced = tensor.slice(vec![0..2, 1..1]).unwrap();
        assert_eq!(sliced.shape(), &[2, 0]);
        assert_eq!(sliced.data.as_slice().unwrap(), &[]);
    }
}