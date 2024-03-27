use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default)]
pub struct EqSubtable<F: PrimeField> {
    _field: PhantomData<F>,
}

impl<F: PrimeField> EqSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: PrimeField> LassoSubtable<F> for EqSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        let mut entries: Vec<F> = vec![F::zero(); M];
        let bits_per_operand = (log2(M) / 2) as usize;

        // Materialize table entries in order where (x | y) ranges 0..M
        // Below is the optimized loop for the condition:
        // table[x | y] = x == y
        for idx in 0..(1 << bits_per_operand) {
            let concat_idx = idx | (idx << bits_per_operand);
            entries[concat_idx] = F::one();
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
            result *= x[i] * y[i] + (F::one() - x[i]) * (F::one() - y[i]);
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        jolt::subtable::{eq::EqSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(eq_materialize_mle_parity, EqSubtable<Fr>, Fr, 256);
}
