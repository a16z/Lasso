use ark_curve25519::{EdwardsProjective as G1Projective, Fr};
use merlin::Transcript;

use crate::{
  lasso::{
    densified::DensifiedRepresentation,
    surge::{SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
  }, subprotocols::{traits::PolynomialCommitmentScheme, hyrax::Hyrax}, subtables::{
    and::AndSubtableStrategy, lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy,
    SubtableStrategy,
  }, utils::math::Math, utils::random::RandomTape
};

macro_rules! e2e_test {
  ($test_name:ident, $Strategy:ty, $G:ty, $F:ty, $C:expr, $M:expr, $sparsity:expr) => {
    #[test]
    fn $test_name() {
      use crate::utils::test::{gen_indices, gen_random_point};
      use ark_std::log2;

      const C: usize = $C;
      const M: usize = $M;

      // parameters
      const NUM_MEMORIES: usize = <$Strategy as SubtableStrategy<$F, C, M>>::NUM_MEMORIES;
      let log_M: usize = M.log_2();
      let log_s: usize = log2($sparsity) as usize;

      // generate sparse polynomial
      let nz: Vec<[usize; C]> = gen_indices($sparsity, M);

      let mut dense: DensifiedRepresentation<$F, C> =
        DensifiedRepresentation::from_lookup_indices(&nz, log_M);
      let gens =
        SparsePolyCommitmentGens::<$G>::new(b"gens_sparse_poly", C, $sparsity, NUM_MEMORIES, log_M);
      let commitment = dense.commit(&gens);

      let r: Vec<$F> = gen_random_point(log_s);

      let mut random_tape = RandomTape::new(b"proof");
      let mut prover_transcript = Transcript::new(b"example");
      let proof = SparsePolynomialEvaluationProof::<$G, C, $M, $Strategy>::prove(
        &mut dense,
        &r,
        &gens,
        &mut prover_transcript,
        &mut random_tape,
      );

      let mut verifier_transcript = Transcript::new(b"example");
      assert!(
        proof
          .verify(&commitment, &r, &gens, &mut verifier_transcript)
          .is_ok(),
        "Failed to verify proof."
      );
    }
  };
}

/// Hyrax
e2e_test!(
  prove_4d_lt,
  LTSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 4,
  /* M= */ 16,
  /* sparsity= */ 16
);
e2e_test!(
  prove_4d_lt_big_s,
  LTSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 4,
  /* M= */ 16,
  /* sparsity= */ 128
);
e2e_test!(
  prove_4d_and,
  AndSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 4,
  /* M= */ 16,
  /* sparsity= */ 16
);
e2e_test!(
  prove_3d_range,
  RangeCheckSubtableStrategy::<40>,
  G1Projective,
  Fr,
  /* C= */ 3,
  /* M= */ 256,
  /* sparsity= */ 16
);
