#[cfg(test)]

mod tests {
    use neura::core::tensor::Tensor;

    #[test]
    fn add_tensors_test() {
        let t1 = Tensor::ones(&vec![2, 2], false);
        let t2 = Tensor::ones(&vec![2, 2], false);
        let t3 = t1 + t2;

        assert_eq!(t3.data[[0, 0]], 2.0);
    }

    #[test]
    fn scalar_mul_test() {
        let t1 = Tensor::ones(&vec![2,2], false);

        let t2 = t1 * 2.0;

        assert_eq!(t2.data[[0, 0]], 2.0);
    }
}