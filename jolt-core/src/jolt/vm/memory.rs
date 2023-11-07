use ark_ff::PrimeField;

use crate::{
  poly::dense_mlpoly::DensePolynomial,
  subprotocols::grand_product::{BGPCInterpretable, BatchedGrandProductCircuit, GPEvals},
};

pub enum MemoryOp {
  Read(u64, u64),       // (address, value)
  Write(u64, u64, u64), // (address, old_value, new_value)
}

pub struct Memory<F: PrimeField> {
  a: DensePolynomial<F>,
  v: DensePolynomial<F>,

  read_t: DensePolynomial<F>,
}

impl<F: PrimeField> Memory<F> {
  fn new(read_set: Vec<(F, F, F)>, write_set: Vec<(F, F, F)>, final_set: Vec<(F, F, F)>) -> Self {
    todo!("construct")
  }
}

impl<F: PrimeField> BGPCInterpretable<F> for Memory<F> {
  fn compute_leaves(
    &self,
    memory_index: usize,
    r_hash: (&F, &F),
  ) -> (
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
  ) {
    todo!()
  }

  fn construct_batches(
    &self,
    r_hash: (&F, &F),
  ) -> (
    BatchedGrandProductCircuit<F>,
    BatchedGrandProductCircuit<F>,
    Vec<GPEvals<F>>,
  ) {
    todo!()
  }
}

// TODO(sragss): FingerprintStrategy

#[cfg(test)]
mod tests {
  #[test]
  fn prod_layer_proof() {
    todo!()
  }

  #[test]
  fn e2e_mem_checking() {
    todo!()
  }
}