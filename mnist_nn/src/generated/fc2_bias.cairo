use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::signed_integer::i32::i32;

fn fc2_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 210, sign: true });
    data.append(i32 { mag: 252, sign: false });
    data.append(i32 { mag: 381, sign: false });
    data.append(i32 { mag: 233, sign: true });
    data.append(i32 { mag: 298, sign: true });
    data.append(i32 { mag: 767, sign: false });
    data.append(i32 { mag: 277, sign: true });
    data.append(i32 { mag: 226, sign: false });
    data.append(i32 { mag: 1193, sign: true });
    data.append(i32 { mag: 383, sign: false });
let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) }; 
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}
