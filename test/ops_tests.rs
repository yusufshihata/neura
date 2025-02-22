#[cfg(test)]

mod tests {
    use neura::core::tensor::Tensor;

    #[test]
    fn add_tensors() {
        let t1 = Tensor::ones(&vec![2, 2], false);
        let t2 = Tensor::ones(&vec![2, 2], false);
        let t3 = t1 + t2;

        assert_eq!(t3.data[[0, 0]], 2.0);
    }
}