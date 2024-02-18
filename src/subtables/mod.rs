use std::marker::{PhantomData, Sync};

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use merlin::Transcript;

use crate::{
  lasso::{densified::DensifiedRepresentation, memory_checking::GrandProducts},
  poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
  subprotocols::hyrax::{PolyCommitment, PolyCommitmentGens, PolyEvalProof},
  subprotocols::{hyrax::Hyrax, traits::PolynomialCommitmentScheme},
  utils::math::Math,
  utils::transcript::{AppendToTranscript, ProofTranscript},
  utils::{errors::ProofVerifyError, random::RandomTape},
};

#[cfg(feature = "multicore")]
use rayon::prelude::*;

pub mod and;
pub mod lt;
pub mod or;
pub mod range_check;
pub mod xor;

#[cfg(test)]
pub mod test;

pub trait SubtableStrategy<F: PrimeField, const C: usize, const M: usize> {
  const NUM_SUBTABLES: usize;
  const NUM_MEMORIES: usize;

  /// Materialize subtables indexed [1, ..., \alpha]
  fn materialize_subtables() -> [Vec<F>; Self::NUM_SUBTABLES];

  /// Evaluates the MLE of a subtable at the given point. Used by the verifier in memory-checking.
  ///
  /// Params
  /// - `subtable_index`: Which subtable to evaluate the MLE of. Ranges 0..ALPHA
  /// - `point`: Point at which to evaluate the MLE
  fn evaluate_subtable_mle(subtable_index: usize, point: &[F]) -> F;

  /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
  fn combine_lookups(vals: &[F; Self::NUM_MEMORIES]) -> F;

  /// The total degree of `g`, i.e. considering `combine_lookups` as a log(m)-variate polynomial.
  /// Determines the number of evaluation points in each sumcheck round.
  fn g_poly_degree() -> usize;

  /// Computes eq * g(T_1[k], ..., T_\alpha[k]) assuming the eq evaluation is the last element in vals
  fn combine_lookups_eq(vals: &[F; Self::NUM_MEMORIES + 1]) -> F {
    let mut table_evals: [F; Self::NUM_MEMORIES] = [F::zero(); Self::NUM_MEMORIES];
    table_evals.copy_from_slice(&vals[0..Self::NUM_MEMORIES]);
    Self::combine_lookups(&table_evals) * vals[Self::NUM_MEMORIES]
  }

  /// Total degree of eq * g(T_1[k], ..., T_\alpha[k])
  fn sumcheck_poly_degree() -> usize {
    Self::g_poly_degree() + 1
  }

  fn memory_to_subtable_index(memory_index: usize) -> usize {
    assert_eq!(Self::NUM_SUBTABLES * C, Self::NUM_MEMORIES);
    assert!(memory_index < Self::NUM_MEMORIES);
    memory_index % Self::NUM_SUBTABLES
  }

  fn memory_to_dimension_index(memory_index: usize) -> usize {
    assert_eq!(Self::NUM_SUBTABLES * C, Self::NUM_MEMORIES);
    assert!(memory_index < Self::NUM_MEMORIES);
    memory_index / Self::NUM_SUBTABLES
  }

  /// Converts subtables T_1, ..., T_{\alpha} and lookup indices nz_1, ..., nz_c
  /// into log(m)-variate "lookup polynomials" E_1, ..., E_{\alpha}.
  fn to_lookup_polys(
    subtable_entries: &[Vec<F>; Self::NUM_SUBTABLES],
    nz: &[Vec<usize>; C],
    s: usize,
  ) -> [DensePolynomial<F>; Self::NUM_MEMORIES] {
    std::array::from_fn(|i| {
      let mut subtable_lookups: Vec<F> = Vec::with_capacity(s);
      for j in 0..s {
        let subtable = &subtable_entries[Self::memory_to_subtable_index(i)];
        let nz = nz[Self::memory_to_dimension_index(i)][j];
        subtable_lookups.push(subtable[nz]);
      }
      DensePolynomial::new(subtable_lookups)
    })
  }
}

pub struct Subtables<F: PrimeField, const C: usize, const M: usize, S>
where
  S: SubtableStrategy<F, C, M>,
  [(); S::NUM_SUBTABLES]: Sized,
  [(); S::NUM_MEMORIES]: Sized,
{
  subtable_entries: [Vec<F>; S::NUM_SUBTABLES],
  pub lookup_polys: [DensePolynomial<F>; S::NUM_MEMORIES],
  pub combined_poly: DensePolynomial<F>,
  strategy: PhantomData<S>,
}

/// Stores the non-sparse evaluations of T[k] for each of the 'c'-dimensions as DensePolynomials, enables combination and commitment.
impl<F: PrimeField, const C: usize, const M: usize, S> Subtables<F, C, M, S>
where
  S: SubtableStrategy<F, C, M> + Sync,
  [(); S::NUM_SUBTABLES]: Sized,
  [(); S::NUM_MEMORIES]: Sized,
{
  /// Create new Subtables
  /// - `evaluations`: non-sparse evaluations of T[k] for each of the 'c'-dimensions as DensePolynomials
  pub fn new(nz: &[Vec<usize>; C], s: usize) -> Self {
    nz.iter().for_each(|nz_dim| assert_eq!(nz_dim.len(), s));
    let subtable_entries = S::materialize_subtables();
    let lookup_polys: [DensePolynomial<F>; S::NUM_MEMORIES] =
      S::to_lookup_polys(&subtable_entries, nz, s);
    let combined_poly = DensePolynomial::merge(&lookup_polys);

    Subtables {
      subtable_entries,
      lookup_polys,
      combined_poly,
      strategy: PhantomData,
    }
  }

  /// Converts subtables T_1, ..., T_{\alpha} and densified multilinear polynomial
  /// into grand products for memory-checking.
  #[tracing::instrument(skip_all, name = "Subtables.to_grand_products")]
  pub fn to_grand_products(
    &self,
    dense: &DensifiedRepresentation<F, C>,
    r_mem_check: &(F, F),
  ) -> Vec<GrandProducts<F>> {
    #[cfg(feature = "multicore")]
    {
      (0..S::NUM_MEMORIES)
        .into_par_iter()
        .map(|i| {
          let subtable = &self.subtable_entries[S::memory_to_subtable_index(i)];
          let j = S::memory_to_dimension_index(i);
          GrandProducts::new(
            subtable,
            &dense.dim[j],
            &dense.dim_usize[j],
            &dense.read[j],
            &dense.r#final[j],
            r_mem_check,
          )
        })
        .collect::<Vec<_>>()
    }

    #[cfg(not(feature = "multicore"))]
    {
      (0..S::NUM_MEMORIES)
        .map(|i| {
          let subtable = &self.subtable_entries[S::memory_to_subtable_index(i)];
          let j = S::memory_to_dimension_index(i);
          GrandProducts::new(
            subtable,
            &dense.dim[j],
            &dense.dim_usize[j],
            &dense.read[j],
            &dense.r#final[j],
            r_mem_check,
          )
        })
        .collect::<Vec<_>>()
    }
  }

  #[tracing::instrument(skip_all, name = "Subtables.commit")]
  pub fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
    gens: &PolyCommitmentGens<G>,
  ) -> CombinedTableCommitment<G> {
    let (comm_ops_val, _blinds) = Hyrax::<G>::commit(&self.combined_poly, (gens.clone(), None)).unwrap();
    CombinedTableCommitment { comm_ops_val }
  }

  #[tracing::instrument(skip_all, name = "Subtables.compute_sumcheck_claim")]
  pub fn compute_sumcheck_claim(&self, eq: &EqPolynomial<F>) -> F {
    let g_operands = self.lookup_polys.clone();
    let hypercube_size = g_operands[0].len();
    g_operands
      .iter()
      .for_each(|operand| assert_eq!(operand.len(), hypercube_size));

    let eq_evals = eq.evals();

    #[cfg(feature = "multicore")]
    let claim = (0..hypercube_size)
      .into_par_iter()
      .map(|k| {
        let g_operands: [F; S::NUM_MEMORIES] = std::array::from_fn(|j| g_operands[j][k]);
        // eq * g(T_1[k], ..., T_\alpha[k])
        eq_evals[k] * S::combine_lookups(&g_operands)
      })
      .sum();

    #[cfg(not(feature = "multicore"))]
    let claim = (0..hypercube_size)
      .map(|k| {
        let g_operands: [F; S::NUM_MEMORIES] = std::array::from_fn(|j| g_operands[j][k]);
        // eq * g(T_1[k], ..., T_\alpha[k])
        eq_evals[k] * S::combine_lookups(&g_operands)
      })
      .sum();

    claim
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CombinedTableCommitment<G: CurveGroup> {
  comm_ops_val: PolyCommitment<G>,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CombinedTableEvalProof<G: CurveGroup, const C: usize> {
  proof_table_eval: PolyEvalProof<G>,
}

impl<G: CurveGroup, const C: usize> CombinedTableEvalProof<G, C> {
  fn prove_single(
    joint_poly: &DensePolynomial<G::ScalarField>,
    r: &[G::ScalarField],
    evals: Vec<G::ScalarField>,
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> PolyEvalProof<G> {
    assert_eq!(
      joint_poly.get_num_vars(),
      r.len() + evals.len().log_2() as usize
    );

    // append the claimed evaluations to transcript
    <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", &evals);

    // n-to-1 reduction
    let (r_joint, eval_joint) = {
      let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
        transcript,
        b"challenge_combine_n_to_one",
        evals.len().log_2() as usize,
      );

      let mut poly_evals = DensePolynomial::new(evals);
      for i in (0..challenges.len()).rev() {
        poly_evals.bound_poly_var_bot(&challenges[i]);
      }
      assert_eq!(poly_evals.len(), 1);
      let joint_claim_eval = poly_evals[0];
      let mut r_joint = challenges;
      r_joint.extend(r);

      debug_assert_eq!(joint_poly.evaluate(&r_joint), joint_claim_eval);
      (r_joint, joint_claim_eval)
    };
    // decommit the joint polynomial at r_joint
    <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"joint_claim_eval", &eval_joint);

    let (proof_table_eval, _comm_table_eval) = Hyrax::prove(
      &joint_poly,
      &Some(eval_joint),
      &r_joint,
      (None, None, &gens, random_tape),
      transcript,
    )
    .unwrap();

    proof_table_eval
  }

  /// evalues both polynomials at r and produces a joint proof of opening
  #[tracing::instrument(skip_all, name = "CombinedEval.prove")]
  pub fn prove(
    combined_poly: &DensePolynomial<G::ScalarField>,
    eval_ops_val_vec: &[G::ScalarField],
    r: &[G::ScalarField],
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      CombinedTableEvalProof::<G, C>::protocol_name(),
    );

    let evals = {
      let mut evals = eval_ops_val_vec.to_vec();
      evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());
      evals
    };
    let proof_table_eval = CombinedTableEvalProof::<G, C>::prove_single(
      combined_poly,
      r,
      evals,
      gens,
      transcript,
      random_tape,
    );

    CombinedTableEvalProof { proof_table_eval }
  }

  fn verify_single(
    proof: &PolyEvalProof<G>,
    comm: &PolyCommitment<G>,
    r: &[G::ScalarField],
    evals: Vec<G::ScalarField>,
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    // append the claimed evaluations to transcript
    <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", &evals);

    // n-to-1 reduction
    let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
      transcript,
      b"challenge_combine_n_to_one",
      evals.len().log_2() as usize,
    );
    let mut poly_evals = DensePolynomial::new(evals);
    for i in (0..challenges.len()).rev() {
      poly_evals.bound_poly_var_bot(&challenges[i]);
    }
    assert_eq!(poly_evals.len(), 1);
    let joint_claim_eval = poly_evals[0];
    let mut r_joint = challenges;
    r_joint.extend(r);

    // decommit the joint polynomial at r_joint
    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"joint_claim_eval",
      &joint_claim_eval,
    );

    proof.verify_plain(gens, transcript, &r_joint, &joint_claim_eval, comm)
  }

  // verify evaluations of both polynomials at r
  pub fn verify(
    &self,
    r: &[G::ScalarField],
    evals: &[G::ScalarField],
    gens: &PolyCommitmentGens<G>,
    comm: &CombinedTableCommitment<G>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      CombinedTableEvalProof::<G, C>::protocol_name(),
    );
    let mut evals = evals.to_owned();
    evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());

    CombinedTableEvalProof::<G, C>::verify_single(
      &self.proof_table_eval,
      &comm.comm_ops_val,
      r,
      evals,
      gens,
      transcript,
    )
  }

  fn protocol_name() -> &'static [u8] {
    b"Lasso CombinedTableEvalProof"
  }
}

impl<G: CurveGroup> AppendToTranscript<G> for CombinedTableCommitment<G> {
  fn append_to_transcript<T: ProofTranscript<G>>(&self, label: &'static [u8], transcript: &mut T) {
    transcript.append_message(
      b"subtable_evals_commitment",
      b"begin_subtable_evals_commitment",
    );
    self.comm_ops_val.append_to_transcript(label, transcript);
    transcript.append_message(
      b"subtable_evals_commitment",
      b"end_subtable_evals_commitment",
    );
  }
}
