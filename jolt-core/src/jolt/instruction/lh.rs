use ark_ff::PrimeField;
use rand::prelude::StdRng;

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{
    identity::IdentitySubtable, sign_extend::SignExtendSubtable, LassoSubtable,
};
use crate::utils::instruction_utils::chunk_operand_usize;

#[derive(Copy, Clone, Default, Debug)]
pub struct LHInstruction(pub u64);

impl JoltInstruction for LHInstruction {
    fn operands(&self) -> [u64; 2] {
        [0, self.0]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        assert!(vals.len() == 2);

        let half = vals[0];
        let sign_extension = vals[1];

        half + F::from_u64(1 << 16).unwrap() * sign_extension
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: PrimeField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // This assertion ensures that we only need one TruncateOverflowSubtable
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        vec![
            (
                Box::new(IdentitySubtable::<F>::new()),
                SubtableIndices::from(C - 1),
            ),
            (
                // Sign extend the lowest 16 bits of the loaded value,
                // Which will be in the second-to-last chunk.
                Box::new(SignExtendSubtable::<F, 16>::new()),
                SubtableIndices::from(C - 1),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // Sign-extend lower 16 bits of the loaded value
        (self.0 & 0xffff) as i16 as i32 as u32 as u64
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        use rand_core::RngCore;
        Self(rng.next_u32() as u64)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::LHInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn lh_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = LHInstruction(x);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            LHInstruction(0),
            LHInstruction(1),
            LHInstruction(100),
            LHInstruction(u32_max),
            LHInstruction(1 << 8),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
