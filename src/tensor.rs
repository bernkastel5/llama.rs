use half::f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    Int8,
    Int4,
}

#[derive(Debug, Clone)]
pub struct QuantTensor {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub zero_points: Option<Vec<f32>>,
    pub shape: Vec<usize>,
    pub quant_type: QuantType,
}

impl QuantTensor {
    pub fn new(data: Vec<u8>, scales: Vec<f32>, zero_points: Option<Vec<f32>>, shape: Vec<usize>, quant_type: QuantType) -> Self {
        Self { data, scales, zero_points, shape, quant_type }
    }
}

/// Basic min/max linear quantization from FP16/FP32 to INT8.
/// TODO: Plug SIMD kernel here for production performance.
pub fn quantize_fp16_to_int8(src: &[f16]) -> QuantTensor {
    if src.is_empty() {
        return QuantTensor::new(vec![], vec![], None, vec![], QuantType::Int8);
    }

    let min_val = src.iter().map(|&x| x.to_f32()).fold(f32::INFINITY, f32::min);
    let max_val = src.iter().map(|&x| x.to_f32()).fold(f32::NEG_INFINITY, f32::max);

    let scale = if max_val != min_val { (max_val - min_val) / 255.0 } else { 1.0 };
    let zero = if scale != 0.0 { -min_val / scale } else { 0.0 };

    let mut data = Vec::with_capacity(src.len());
    let mut scales = vec![scale];
    let zero_points = Some(vec![zero]);

    for &val in src {
        let q = ((val.to_f32() - min_val) / scale).round().clamp(0.0, 255.0) as u8;
        data.push(q);
    }

    QuantTensor::new(data, scales, zero_points, vec![src.len()], QuantType::Int8)
}