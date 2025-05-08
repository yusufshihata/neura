use neura::tensor::tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        // TODO: Here comes zeros functionality test
        let tensor = Tensor::zeros(vec![2,3], Some(true));

        
        assert_eq!(*tensor.size(), 6 as usize);
        assert_eq!(*tensor.view(), vec![2 as usize, 3 as usize]);
        assert_eq!(*tensor.strides(), vec![3 as usize, 1 as usize]);
    }
}
