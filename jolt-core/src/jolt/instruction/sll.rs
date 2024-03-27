use ark_ff::PrimeField;
use ark_std::log2;
use rand::prelude::StdRng;

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{sll::SllSubtable, LassoSubtable};
use crate::utils::instruction_utils::{
    assert_valid_parameters, chunk_and_concatenate_for_shift, concatenate_lookups,
};

#[derive(Copy, Clone, Default, Debug)]
pub struct SLLInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SLLInstruction<WORD_SIZE> {
    fn operands(&self) -> [u64; 2] {
        [self.0, self.1]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(C <= 10);
        concatenate_lookups(vals, C, (log2(M) / 2) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: PrimeField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        let mut subtables: Vec<Box<dyn LassoSubtable<F>>> = vec![
            Box::new(SllSubtable::<F, 0, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 1, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 2, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 3, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 4, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 5, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 6, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 7, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 8, WORD_SIZE>::new()),
            Box::new(SllSubtable::<F, 9, WORD_SIZE>::new()),
        ];
        subtables.truncate(C);
        subtables.reverse();

        let indices = (0..C).into_iter().map(|i| SubtableIndices::from(i));
        subtables.into_iter().zip(indices).collect()
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        chunk_and_concatenate_for_shift(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // SLL is specified to ignore all but the last 5 bits of y: https://jemu.oscc.cc/SLL
        (self.0 as u32)
            .checked_shl(self.1 as u32 % WORD_SIZE as u32)
            .unwrap_or(0)
            .into()
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        use rand_core::RngCore;
        Self(rng.next_u32() as u64, rng.next_u32() as u64)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::SLLInstruction;

    #[test]
    fn sll_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32(), rng.next_u32());
            let instruction = SLLInstruction::<WORD_SIZE>(x as u64, y as u64);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SLLInstruction::<32>(100, 0),
            SLLInstruction::<32>(0, 100),
            SLLInstruction::<32>(1, 0),
            SLLInstruction::<32>(0, u32_max),
            SLLInstruction::<32>(u32_max, 0),
            SLLInstruction::<32>(u32_max, u32_max),
            SLLInstruction::<32>(u32_max, 1 << 8),
            SLLInstruction::<32>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
