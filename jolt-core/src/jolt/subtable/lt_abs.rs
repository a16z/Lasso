use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::split_bits;

#[derive(Default)]
pub struct LtAbsSubtable<F: PrimeField> {
    _field: PhantomData<F>,
}

impl<F: PrimeField> LtAbsSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: PrimeField> LassoSubtable<F> for LtAbsSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        let mut entries: Vec<F> = Vec::with_capacity(M);
        let bits_per_operand = (log2(M) / 2) as usize;
        // 0b01111...11
        let lower_bits_mask = (1 << (bits_per_operand - 1)) - 1;

        // Materialize table entries in order where (x | y) ranges 0..M
        for idx in 0..M {
            let (x, y) = split_bits(idx, bits_per_operand);
            let row = if (x & lower_bits_mask) < (y & lower_bits_mask) {
                F::one()
            } else {
                F::zero()
            };
            entries.push(row);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \prod_i x_i * y_i + (1 - x_i) * (1 - y_i)
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (x, y) = point.split_at(b);

        let mut result = F::zero();
        let mut eq_term = F::one();
        // Skip i=0
        for i in 1..b {
            result += (F::one() - x[i]) * y[i] * eq_term;
            eq_term *= F::one() - x[i] - y[i] + F::from_u64(2u64).unwrap() * x[i] * y[i];
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        jolt::subtable::{lt_abs::LtAbsSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        lt_abs_materialize_mle_parity,
        LtAbsSubtable<Fr>,
        Fr,
        256
    );
}
