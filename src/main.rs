use neura::tensor::tensor::Tensor;

fn main() {
    let tensor = Tensor::zeros(vec![2,3], Some(true));

    tensor.data();
}
