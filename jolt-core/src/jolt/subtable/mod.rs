use ark_ff::PrimeField;
use enum_dispatch::enum_dispatch;
use std::any::TypeId;
use std::marker::Sync;
use strum::{EnumCount, IntoEnumIterator};

#[enum_dispatch]
pub trait LassoSubtable<F: PrimeField>: 'static + Sync {
    /// Returns the TypeId of this subtable.
    /// The `Jolt` trait has associated enum types `InstructionSet` and `Subtables`.
    /// This function is used to resolve the many-to-many mapping between `InstructionSet` variants
    /// and `Subtables` variants,
    fn subtable_id(&self) -> SubtableId {
        TypeId::of::<Self>()
    }
    /// Fully materializes a subtable of size `M`, reprensented as a Vec of length `M`.
    fn materialize(&self, M: usize) -> Vec<F>;
    /// Evaluates the multilinear extension polynomial for this subtable at the given `point`,
    /// interpreted to be of size log_2(M), where M is the size of the subtable.
    fn evaluate_mle(&self, point: &[F]) -> F;
}

pub type SubtableId = TypeId;
pub trait JoltSubtableSet<F: PrimeField>:
    LassoSubtable<F> + IntoEnumIterator + EnumCount + From<SubtableId> + Into<usize> + Send + Sync
{
    fn enum_index(subtable: Box<dyn LassoSubtable<F>>) -> usize {
        Self::from(subtable.subtable_id()).into()
    }
}

pub mod and;
pub mod eq;
pub mod eq_abs;
pub mod eq_msb;
pub mod gt_msb;
pub mod identity;
pub mod lt_abs;
pub mod ltu;
pub mod or;
pub mod sign_extend;
pub mod sll;
pub mod sra_sign;
pub mod srl;
pub mod truncate_overflow;
pub mod xor;
pub mod zero_lsb;

#[cfg(test)]
pub mod test;
