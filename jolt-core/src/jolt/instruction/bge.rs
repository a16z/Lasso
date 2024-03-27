use ark_ff::PrimeField;
use rand::prelude::StdRng;

use super::{slt::SLTInstruction, JoltInstruction, SubtableIndices};
use crate::{
    jolt::subtable::{
        eq::EqSubtable, eq_abs::EqAbsSubtable, eq_msb::EqMSBSubtable, gt_msb::GtMSBSubtable,
        lt_abs::LtAbsSubtable, ltu::LtuSubtable, LassoSubtable,
    },
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Copy, Clone, Default, Debug)]
pub struct BGEInstruction(pub u64, pub u64);

impl JoltInstruction for BGEInstruction {
    fn operands(&self) -> [u64; 2] {
        [self.0, self.1]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
        // 1 - LTS(x, y) =
        F::one() - SLTInstruction(self.0, self.1).combine_lookups(vals, C, M)
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C + 1
    }

    fn subtables<F: PrimeField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(GtMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(EqMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(LtuSubtable::new()), SubtableIndices::from(1..C)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(1..C)),
            (Box::new(LtAbsSubtable::new()), SubtableIndices::from(0)),
            (Box::new(EqAbsSubtable::new()), SubtableIndices::from(0)),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        ((self.0 as i32) >= (self.1 as i32)).into()
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

    use super::BGEInstruction;

    #[test]
    fn bge_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        // Random
        for _ in 0..256 {
            let x = rng.next_u32();
            let y = rng.next_u32();

            let instruction = BGEInstruction(x as u64, y as u64);

            jolt_instruction_test!(instruction);
        }

        // Ones
        for _ in 0..256 {
            let x = rng.next_u32();
            jolt_instruction_test!(BGEInstruction(x as u64, x as u64));
        }

        // Edge-cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            BGEInstruction(100, 0),
            BGEInstruction(0, 100),
            BGEInstruction(1, 0),
            BGEInstruction(0, u32_max),
            BGEInstruction(u32_max, 0),
            BGEInstruction(u32_max, u32_max),
            BGEInstruction(u32_max, 1 << 8),
            BGEInstruction(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
