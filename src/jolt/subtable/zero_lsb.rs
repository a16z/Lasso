use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default)]
pub struct ZeroLSBSubtable<F: PrimeField> {
  _field: PhantomData<F>,
}

impl<F: PrimeField> ZeroLSBSubtable<F> {
  pub fn new() -> Self {
    Self {
      _field: PhantomData,
    }
  }
}

impl<F: PrimeField> LassoSubtable<F> for ZeroLSBSubtable<F> {
  fn materialize(&self, M: usize) -> Vec<F> {
    // always set LSB to 0
    (0..M).map(|i| F::from((i-(i%2)) as u64)).collect()
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    let mut result = F::zero();
    // skip LSB
    for i in 1..point.len() {
      result += F::from(1u64 << i) * point[point.len() - 1 - i];
    }
    result
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;

  use crate::{
    jolt::subtable::{zero_lsb::ZeroLSBSubtable, LassoSubtable},
    subtable_materialize_mle_parity_test,
  };

  subtable_materialize_mle_parity_test!(zero_lsb_materialize_mle_parity, ZeroLSBSubtable<Fr>, Fr, 256);
}
