use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use crate::utils::split_bits;

use super::LassoSubtable;

/// Example usage in ADD:
/// Input z is of 65 bits, which is split into 20-bit chunks.
/// This subtable is used to remove the overflow bit from the 4th chunk.
#[derive(Default)]
pub struct TruncateOverflowSubtable<F: PrimeField, const WORD_SIZE: usize> {
    _field: PhantomData<F>,
}

impl<F: PrimeField, const WORD_SIZE: usize> TruncateOverflowSubtable<F, WORD_SIZE> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: PrimeField, const WORD_SIZE: usize> LassoSubtable<F>
    for TruncateOverflowSubtable<F, WORD_SIZE>
{
    fn materialize(&self, M: usize) -> Vec<F> {
        let cutoff = WORD_SIZE % log2(M) as usize;

        let mut entries: Vec<F> = Vec::with_capacity(M);
        for idx in 0..M {
            let (_, lower_bits) = split_bits(idx, cutoff);
            let row = F::from_u64(lower_bits as u64).unwrap();
            entries.push(row);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        let log_M = point.len();
        let cutoff = WORD_SIZE % log_M as usize;

        let mut result = F::zero();
        for i in 0..cutoff {
            result += F::from_u64(1u64 << i).unwrap() * point[point.len() - 1 - i];
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        jolt::subtable::{truncate_overflow::TruncateOverflowSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
      truncate_overflow_materialize_mle_parity,
      TruncateOverflowSubtable<Fr, 32>,
      Fr,
      256
    );
}
