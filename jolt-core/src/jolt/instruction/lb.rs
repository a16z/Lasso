use ark_ff::PrimeField;
use rand::prelude::StdRng;

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{
    sign_extend::SignExtendSubtable, truncate_overflow::TruncateOverflowSubtable, LassoSubtable,
};
use crate::utils::instruction_utils::chunk_operand_usize;

#[derive(Copy, Clone, Default, Debug)]
pub struct LBInstruction(pub u64);

impl JoltInstruction for LBInstruction {
    fn operands(&self) -> [u64; 2] {
        [0, self.0]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(M >= 1 << 8);
        assert!(vals.len() == 2);

        let byte = vals[0];
        let sign_extension = vals[1];

        let mut result = byte;
        for i in 1..C {
            result += F::from_u64(1 << (8 * i)).unwrap() * sign_extension;
        }
        result
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
        assert!(M >= 1 << 8);
        vec![
            (
                // Truncate all but the lowest eight bits of the last chunk,
                // which contains the lower 8 bits of the loaded value.
                Box::new(TruncateOverflowSubtable::<F, 8>::new()),
                SubtableIndices::from(C - 1),
            ),
            (
                // Sign extend the lowest eight bits of the last chunk,
                // which contains the lower 8 bits of the loaded value.
                Box::new(SignExtendSubtable::<F, 8>::new()),
                SubtableIndices::from(C - 1),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // Sign-extend lower 8 bits of the loaded value
        (self.0 & 0xff) as i8 as i32 as u32 as u64
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        use rand_core::RngCore;
        Self(rng.next_u32() as u64)
    }
}

#[cfg(test)]
mod test {
    use ark_curve25519::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::LBInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn lb_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = LBInstruction(x);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            LBInstruction(0),
            LBInstruction(1),
            LBInstruction(100),
            LBInstruction(u32_max),
            LBInstruction(1 << 8),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
