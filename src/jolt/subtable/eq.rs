use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::split_bits;

#[derive(Default, Debug)]
pub struct EQSubtable<F: PrimeField> {
  _field: PhantomData<F>,
}

impl<F: PrimeField> EQSubtable<F> {
  pub fn new() -> Self {
    Self {
      _field: PhantomData,
    }
  }
}

impl<F: PrimeField> LassoSubtable<F> for EQSubtable<F> {
  fn materialize(&self, M: usize) -> Vec<F> {
    let mut entries: Vec<F> = Vec::with_capacity(M);
    let bits_per_operand = (log2(M) / 2) as usize;

    // Materialize table entries in order where (x | y) ranges 0..M
    for idx in 0..M {
      let (x, y) = split_bits(idx, bits_per_operand);
      let row = if x == y { F::one() } else { F::zero() };
      entries.push(row);
    }
    entries
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    // \prod_i x_i * y_i + (1 - x_i) * (1 - y_i)
    debug_assert!(point.len() % 2 == 0);
    let b = point.len() / 2;
    let (x, y) = point.split_at(b);

    let mut result = F::one();
    for i in 0..b {
      result *= x[i] * y[i] + (F::one() - x[i]) * (F::one() - y[i])
    }
    result
  }
}
