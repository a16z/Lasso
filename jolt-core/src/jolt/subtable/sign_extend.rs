use ark_ff::PrimeField;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default)]
pub struct SignExtendSubtable<F: PrimeField, const WIDTH: usize> {
    _field: PhantomData<F>,
}

impl<F: PrimeField, const WIDTH: usize> SignExtendSubtable<F, WIDTH> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: PrimeField, const WIDTH: usize> LassoSubtable<F> for SignExtendSubtable<F, WIDTH> {
    fn materialize(&self, M: usize) -> Vec<F> {
        // TODO(moodlezoup): This subtable currently only works for M = 2^16
        assert_eq!(M, 1 << 16);
        let mut entries: Vec<F> = Vec::with_capacity(M);

        // The sign-extension will be the same width as the value being extended
        // –– this is convenient for how this subtable is used in LB and LH.
        let ones: u64 = (1 << WIDTH) - 1;

        for idx in 0..M {
            let sign_bit = ((idx >> (WIDTH - 1)) & 1) as u64;
            let sign_extension = sign_bit * ones;
            let row = F::from_u64(sign_extension).unwrap();
            entries.push(row);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        assert_eq!(point.len(), 16);

        let sign_bit = point[point.len() - WIDTH];
        let ones: u64 = (1 << WIDTH) - 1;
        sign_bit * F::from_u64(ones).unwrap()
    }
}

#[cfg(test)]
mod test {
    use ark_curve25519::Fr;

    use crate::{
        jolt::subtable::{sign_extend::SignExtendSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        sign_extend_8_materialize_mle_parity,
        SignExtendSubtable<Fr, 8>,
        Fr,
        1 << 16
    );

    subtable_materialize_mle_parity_test!(
        sign_extend_16_materialize_mle_parity,
        SignExtendSubtable<Fr, 16>,
        Fr,
        1 << 16
    );
}
