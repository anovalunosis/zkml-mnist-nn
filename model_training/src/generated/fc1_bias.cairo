use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::signed_integer::i32::i32;

fn fc1_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 9581, sign: false });
    data.append(i32 { mag: 267, sign: false });
    data.append(i32 { mag: 5255, sign: true });
    data.append(i32 { mag: 8274, sign: false });
    data.append(i32 { mag: 2762, sign: false });
    data.append(i32 { mag: 7924, sign: false });
    data.append(i32 { mag: 10408, sign: false });
    data.append(i32 { mag: 1243, sign: true });
    data.append(i32 { mag: 6851, sign: false });
    data.append(i32 { mag: 1216, sign: true });
let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) }; 
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}
