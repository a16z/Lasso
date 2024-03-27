use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::split_bits;

#[derive(Default)]
pub struct GtMSBSubtable<F: PrimeField> {
    _field: PhantomData<F>,
}

impl<F: PrimeField> GtMSBSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: PrimeField> LassoSubtable<F> for GtMSBSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        let mut entries: Vec<F> = Vec::with_capacity(M);
        let bits_per_operand = (log2(M) / 2) as usize;
        let high_bit = 1usize << (bits_per_operand - 1);

        // Materialize table entries in order from 0..M
        for idx in 0..M {
            let (x, y) = split_bits(idx, bits_per_operand);
            let row = (x & high_bit) > (y & high_bit);
            entries.push(if row { F::one() } else { F::zero() });
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (x, y) = point.split_at(b);
        // x_0 * (1 - y_0)
        x[0] * (F::one() - y[0])
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        jolt::subtable::{gt_msb::GtMSBSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        gt_msb_materialize_mle_parity,
        GtMSBSubtable<Fr>,
        Fr,
        256
    );
}
