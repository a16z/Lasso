use ark_ff::PrimeField;
use ark_std::log2;
use rand::prelude::StdRng;

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use crate::utils::instruction_utils::{chunk_operand_usize, concatenate_lookups};

#[derive(Copy, Clone, Default, Debug)]
pub struct SWInstruction(pub u64);

impl JoltInstruction for SWInstruction {
    fn operands(&self) -> [u64; 2] {
        [0, self.0]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], _: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        assert!(vals.len() == 2);
        concatenate_lookups(vals, 2, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: PrimeField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // This assertion ensures that we only need two IdentitySubtables
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        vec![(
            Box::new(IdentitySubtable::<F>::new()),
            SubtableIndices::from(C - 2..C),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // Lower 32 bits of the rs2 value
        self.0 & 0xffffffff
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

    use super::SWInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn sw_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = SWInstruction(x);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SWInstruction(0),
            SWInstruction(1),
            SWInstruction(100),
            SWInstruction(u32_max),
            SWInstruction(1 << 8),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
