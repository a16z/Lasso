use ark_ff::PrimeField;
use rand::prelude::StdRng;

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{sra_sign::SraSignSubtable, srl::SrlSubtable, LassoSubtable};
use crate::utils::instruction_utils::{assert_valid_parameters, chunk_and_concatenate_for_shift};

#[derive(Copy, Clone, Default, Debug)]
pub struct SRAInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SRAInstruction<WORD_SIZE> {
    fn operands(&self) -> [u64; 2] {
        [self.0, self.1]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, _: usize) -> F {
        assert!(C <= 10);
        assert_eq!(vals.len(), C + 1);
        vals.iter().sum()
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
            Box::new(SrlSubtable::<F, 0, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 1, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 2, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 3, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 4, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 5, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 6, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 7, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 8, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 9, WORD_SIZE>::new()),
        ];
        subtables.truncate(C);
        subtables.reverse();
        let indices = (0..C).into_iter().map(|i| SubtableIndices::from(i));
        let mut subtables_and_indices: Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> =
            subtables.into_iter().zip(indices).collect();

        subtables_and_indices.push((
            Box::new(SraSignSubtable::<F, WORD_SIZE>::new()),
            SubtableIndices::from(0),
        ));

        subtables_and_indices
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        chunk_and_concatenate_for_shift(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        let x = self.0 as i32;
        let y = self.1 as u32 % (WORD_SIZE as u32);
        (x.checked_shr(y).unwrap_or(0) as u32).into()
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

    use super::SRAInstruction;

    #[test]
    fn sra_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32(), rng.next_u32());
            let instruction = SRAInstruction::<WORD_SIZE>(x as u64, y as u64);
            jolt_instruction_test!(instruction);
        }
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SRAInstruction::<32>(100, 0),
            SRAInstruction::<32>(0, 2),
            SRAInstruction::<32>(1, 2),
            SRAInstruction::<32>(0, 32),
            SRAInstruction::<32>(u32_max, 0),
            SRAInstruction::<32>(u32_max, 31),
            SRAInstruction::<32>(u32_max, 1 << 8),
            SRAInstruction::<32>(1 << 8, 1 << 16),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
