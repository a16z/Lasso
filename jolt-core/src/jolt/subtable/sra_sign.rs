use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::math::Math;
use crate::utils::split_bits;

#[derive(Default)]
pub struct SraSignSubtable<F: PrimeField, const WORD_SIZE: usize> {
    _field: PhantomData<F>,
}

impl<F: PrimeField, const WORD_SIZE: usize> SraSignSubtable<F, WORD_SIZE> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: PrimeField, const WORD_SIZE: usize> LassoSubtable<F> for SraSignSubtable<F, WORD_SIZE> {
    fn materialize(&self, M: usize) -> Vec<F> {
        let mut entries: Vec<F> = Vec::with_capacity(M);

        let operand_chunk_width: usize = (log2(M) / 2) as usize;

        // find position of sign bit in the chunk
        let sign_bit_index = (WORD_SIZE - 1) % operand_chunk_width;

        for idx in 0..M {
            let (x, y) = split_bits(idx, operand_chunk_width);

            let x_sign = F::from_u64(((x >> sign_bit_index) & 1) as u64).unwrap();

            let row = (0..(y % WORD_SIZE) as u32)
                .into_iter()
                .fold(F::zero(), |acc, i: u32| {
                    acc + F::from_u64(1_u64 << (WORD_SIZE as u32 - 1 - i)).unwrap() * x_sign
                });

            entries.push(row);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // first half is chunk X_i
        // and second half is always chunk Y_0
        debug_assert!(point.len() % 2 == 0);

        let log_WORD_SIZE = log2(WORD_SIZE) as usize;

        let b = point.len() / 2;
        let (x, y) = point.split_at(b);

        let mut result = F::zero();

        let sign_index = (WORD_SIZE - 1) % b;
        let x_sign = x[b - 1 - sign_index];

        // min with 1 << b is included for test cases with subtables of bit-length smaller than 6
        for k in 0..std::cmp::min(WORD_SIZE, 1 << b) {
            let k_bits = (k as usize)
                .get_bits(log_WORD_SIZE)
                .iter()
                .map(|bit| if *bit { F::one() } else { F::zero() })
                .collect::<Vec<F>>(); // big-endian

            let mut eq_term = F::one();
            // again, min with b is included when subtables of bit-length less than 6 are used
            for i in 0..std::cmp::min(log_WORD_SIZE, b) {
                eq_term *= k_bits[log_WORD_SIZE - 1 - i] * y[b - 1 - i]
                    + (F::one() - k_bits[log_WORD_SIZE - 1 - i]) * (F::one() - y[b - 1 - i]);
            }

            let x_sign_upper = (0..k).into_iter().fold(F::zero(), |acc, i| {
                acc + F::from_u64(1_u64 << (WORD_SIZE - 1 - i)).unwrap() * x_sign
            });

            result += eq_term * x_sign_upper;
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        jolt::subtable::{sra_sign::SraSignSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
      sra_sign_materialize_mle_parity,
      SraSignSubtable<Fr, 32>,
      Fr,
      256
    );
}
