#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
use crate::lasso::densified::DensifiedRepresentation;
use crate::lasso::surge::{SparsePolyCommitmentGens, SparsePolynomialCommitment};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::identity_poly::IdentityPolynomial;
use crate::subprotocols::hyrax::Hyrax;
use crate::subprotocols::traits::PolynomialCommitmentScheme;
use crate::subprotocols::{
  grand_product::{BatchedGrandProductArgument, GrandProductCircuit},
  hyrax::PolyEvalProof,
};
use crate::subtables::{
  CombinedTableCommitment, CombinedTableEvalProof, SubtableStrategy, Subtables,
};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::random::RandomTape;
use crate::utils::transcript::ProofTranscript;

use ark_ec::CurveGroup;
use ark_ff::{Field, PrimeField};
use ark_serialize::*;
use ark_std::{One, Zero};
use merlin::Transcript;
use std::marker::Sync;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryCheckingProof<
  G: CurveGroup,
  const C: usize,
  const M: usize,
  S: SubtableStrategy<G::ScalarField, C, M> + Sync,
> where
  [(); S::NUM_MEMORIES]: Sized,
{
  proof_prod_layer: ProductLayerProof<G::ScalarField, { S::NUM_MEMORIES }>,
  proof_hash_layer: HashLayerProof<G, C, M, S>,
}

impl<
    G: CurveGroup,
    const C: usize,
    const M: usize,
    S: SubtableStrategy<G::ScalarField, C, M> + Sync,
  > MemoryCheckingProof<G, C, M, S>
where
  [(); S::NUM_SUBTABLES]: Sized,
  [(); S::NUM_MEMORIES]: Sized,
{
  /// Proves that E_i polynomials are well-formed, i.e., that E_i(j) equals T_i[dim_i(j)] for all j ∈ {0, 1}^{log(m)},
  /// using memory-checking techniques as described in Section 5 of the Lasso paper, or Section 7.2 of the Spartan paper.
  ///
  /// Params
  /// - `dense`: The densified representation of the sparse multilinear polynomial.
  /// - `r_mem_check`: (gamma, tau) – Parameters for Reed-Solomon fingerprinting (see `hash_func` closure).
  /// - `subtable_evaluations`: The subtable values read, i.e. T_i[nz(i)].
  /// - `gens`: Generates public parameters for polynomial commitments.
  /// - `transcript`: The proof transcript, used for Fiat-Shamir.
  /// - `random_tape`: Randomness for dense polynomial commitments.
  #[tracing::instrument(skip_all, name = "MemoryChecking.prove")]
  pub fn prove(
    dense: &DensifiedRepresentation<G::ScalarField, C>,
    r_mem_check: &(G::ScalarField, G::ScalarField),
    subtables: &Subtables<G::ScalarField, C, M, S>,
    gens: &SparsePolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let mut grand_products = subtables.to_grand_products(dense, r_mem_check);
    let (proof_prod_layer, rand_mem, rand_ops) =
      ProductLayerProof::prove::<G>(&mut grand_products, transcript);

    let proof_hash_layer = HashLayerProof::prove(
      (&rand_mem, &rand_ops),
      dense,
      subtables,
      gens,
      transcript,
      random_tape,
    );

    MemoryCheckingProof {
      proof_prod_layer,
      proof_hash_layer,
    }
  }

  /// Verifies that E_i polynomials are well-formed, i.e., that E_i(j) equals T_i[dim_i(j)] for all j ∈ {0, 1}^{log(m)},
  /// using memory-checking techniques as described in Section 5 of the Lasso paper, or Section 7.2 of the Spartan paper.
  ///
  /// Params
  /// - `comm`: The sparse polynomial commitment.
  /// - `comm_derefs`: The commitment to the E_i polynomials.
  /// - `gens`: Generates public parameters for polynomial commitments.
  /// - `r`: The evaluation point at which the Lasso commitment is being opened.
  /// - `r_mem_check`: (gamma, tau) – Parameters for Reed-Solomon fingerprinting (see `hash_func` closure).
  /// - `s`: Sparsity, i.e. the number of lookups.
  /// - `transcript`: The proof transcript, used for Fiat-Shamir.
  pub fn verify(
    &self,
    comm: &SparsePolynomialCommitment<G>,
    comm_derefs: &CombinedTableCommitment<G>,
    gens: &SparsePolyCommitmentGens<G>,
    r_mem_check: &(G::ScalarField, G::ScalarField),
    s: usize,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (r_hash, r_multiset_check) = r_mem_check;

    let num_ops = s.next_power_of_two();
    let num_cells = comm.m;

    let (claims_mem, rand_mem, claims_ops, rand_ops) = self
      .proof_prod_layer
      .verify::<G>(num_ops, num_cells, transcript)?;

    let claims: [(
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
    ); S::NUM_MEMORIES] = std::array::from_fn(|i| {
      (
        claims_mem[2 * i],     // init
        claims_ops[2 * i],     // read
        claims_ops[2 * i + 1], // write
        claims_mem[2 * i + 1], // final
      )
    });

    // verify the proof of hash layer
    self.proof_hash_layer.verify(
      (&rand_mem, &rand_ops),
      &claims,
      comm,
      gens,
      comm_derefs,
      r_hash,
      r_multiset_check,
      transcript,
    )?;

    Ok(())
  }

  fn protocol_name() -> &'static [u8] {
    b"Lasso MemoryCheckingProof"
  }
}

/// Contains grand product circuits to evaluate multi-set checks on memories.
/// Evaluating each circuit is equivalent to computing the hash/fingerprint
/// H_{\tau, \gamma} of the corresponding set.
#[derive(Debug)]
pub struct GrandProducts<F> {
  /// Corresponds to the Init_{row/col} hash in the Spartan paper.
  init: GrandProductCircuit<F>,
  /// Corresponds to the RS_{row/col} hash in the Spartan paper.
  read: GrandProductCircuit<F>,
  /// Corresponds to the WS_{row/col} hash in the Spartan paper.
  write: GrandProductCircuit<F>,
  /// Corresponds to the Audit_{row/col} hash in the Spartan paper.
  r#final: GrandProductCircuit<F>,
}

impl<F: PrimeField> GrandProducts<F> {
  /// Creates the grand product circuits used for memory checking.
  ///
  /// Params
  /// - `eval_table`: M-sized list of table entries
  /// - `dim_i`: log(s)-variate polynomial evaluating to the table index corresponding to each access.
  /// - `dim_i_usize`: Vector of table indices accessed, as `usize`s.
  /// - `read_i`: "Counter polynomial" for memory reads.
  /// - `final_i` "Counter polynomial" for the final memory state.
  /// - `r_mem_check`: (gamma, tau) – Parameters for Reed-Solomon fingerprinting.
  pub fn new(
    eval_table: &[F],
    dim_i: &DensePolynomial<F>,
    dim_i_usize: &[usize],
    read_i: &DensePolynomial<F>,
    final_i: &DensePolynomial<F>,
    r_mem_check: &(F, F),
  ) -> Self {
    let (
      grand_product_input_init,
      grand_product_input_read,
      grand_product_input_write,
      grand_product_input_final,
    ) = GrandProducts::build_grand_product_inputs(
      eval_table,
      dim_i,
      dim_i_usize,
      read_i,
      final_i,
      r_mem_check,
    );

    let prod_init = GrandProductCircuit::new(&grand_product_input_init);
    let prod_read = GrandProductCircuit::new(&grand_product_input_read);
    let prod_write = GrandProductCircuit::new(&grand_product_input_write);
    let prod_final = GrandProductCircuit::new(&grand_product_input_final);

    #[cfg(debug)]
    {
      let hashed_write_set: F = prod_init.evaluate() * prod_write.evaluate();
      let hashed_read_set: F = prod_read.evaluate() * prod_final.evaluate();
      // H(Init) * H(WS) ?= H(RS) * H(Audit)
      // analogous to H(WS) = H(RS) * H(S) in the Lasso paper
      debug_assert_eq!(hashed_read_set, hashed_write_set);
    }

    GrandProducts {
      init: prod_init,
      read: prod_read,
      write: prod_write,
      r#final: prod_final,
    }
  }

  /// Builds the multilinear polynomials that will serve as the inputs to the grand product circuits
  /// used for memory checking. Specifically, this function computes the hash (Reed-Solomon fingerprint)
  /// for each tuple in the "init", "read", "write", and "final" sets (named "Init", "WS", "RS", "Audit"
  /// in the Spartan paper).
  ///
  /// Params
  /// - `eval_table`: M-sized list of table entries
  /// - `dim_i`: log(s)-variate polynomial evaluating to the table index corresponding to each access.
  /// - `dim_i_usize`: Vector of table indices accessed, as `usize`s.
  /// - `read_i`: "Counter polynomial" for memory reads.
  /// - `final_i` "Counter polynomial" for the final memory state.
  /// - `r_mem_check`: (gamma, tau) – Parameters for Reed-Solomon fingerprinting (see `hash_func` closure).
  ///
  /// Returns
  /// - `(init, read, write, final)`: These are the memory polynomials as described in the Spartan paper.
  /// Note that the Lasso describes using `RS`, `WS`, and `S` (using fewer grand products for efficiency),
  /// but that they serve the same purpose: to prove/verify memory consistency.
  fn build_grand_product_inputs(
    eval_table: &[F],
    dim_i: &DensePolynomial<F>,
    dim_i_usize: &[usize],
    read_i: &DensePolynomial<F>,
    final_i: &DensePolynomial<F>,
    r_mem_check: &(F, F),
  ) -> (
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
  ) {
    let (gamma, tau) = r_mem_check;

    // hash(a, v, t) = t * gamma^2 + v * gamma + a - tau
    let hash_func = |a: &F, v: &F, t: &F| -> F { *t * gamma.square() + *v * *gamma + *a - tau };

    // init: M hash evaluations => log(M)-variate polynomial
    assert_eq!(eval_table.len(), final_i.len());
    let num_mem_cells = eval_table.len();
    let grand_product_input_init = DensePolynomial::new(
      (0..num_mem_cells)
        .map(|i| {
          // addr is given by i, init value is given by eval_table, and ts = 0
          hash_func(&F::from(i as u64), &eval_table[i], &F::zero())
        })
        .collect::<Vec<F>>(),
    );
    // final: M hash evaluations => log(M)-variate polynomial
    let grand_product_input_final = DensePolynomial::new(
      (0..num_mem_cells)
        .map(|i| {
          // addr is given by i, value is given by eval_table, and ts is given by audit_ts
          hash_func(&F::from(i as u64), &eval_table[i], &final_i[i])
        })
        .collect::<Vec<F>>(),
    );

    // TODO(#30): Parallelize

    // read: s hash evaluations => log(s)-variate polynomial
    assert_eq!(dim_i.len(), read_i.len());

    #[cfg(feature = "multicore")]
    let num_ops = (0..dim_i.len()).into_par_iter();
    #[cfg(not(feature = "multicore"))]
    let num_ops = 0..dim_i.len();
    let grand_product_input_read = DensePolynomial::new(
      num_ops
        .clone()
        .map(|i| {
          // addr is given by dim_i, value is given by eval_table, and ts is given by read_ts
          hash_func(&dim_i[i], &eval_table[dim_i_usize[i]], &read_i[i])
        })
        .collect::<Vec<F>>(),
    );
    // write: s hash evaluation => log(s)-variate polynomial
    let grand_product_input_write = DensePolynomial::new(
      num_ops
        .map(|i| {
          // addr is given by dim_i, value is given by eval_table, and ts is given by write_ts = read_ts + 1
          hash_func(
            &dim_i[i],
            &eval_table[dim_i_usize[i]],
            &(read_i[i] + F::one()),
          )
        })
        .collect::<Vec<F>>(),
    );

    (
      grand_product_input_init,
      grand_product_input_read,
      grand_product_input_write,
      grand_product_input_final,
    )
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct HashLayerProof<
  G: CurveGroup,
  const C: usize,
  const M: usize,
  S: SubtableStrategy<G::ScalarField, C, M>,
> where
  [(); S::NUM_MEMORIES]: Sized,
{
  eval_dim: [G::ScalarField; C],
  eval_read: [G::ScalarField; C],
  eval_final: [G::ScalarField; C],
  eval_derefs: [G::ScalarField; S::NUM_MEMORIES],
  proof_ops: PolyEvalProof<G>,
  proof_mem: PolyEvalProof<G>,
  proof_derefs: CombinedTableEvalProof<G, C>,
}

impl<G: CurveGroup, const C: usize, const M: usize, S: SubtableStrategy<G::ScalarField, C, M>>
  HashLayerProof<G, C, M, S>
where
  [(); S::NUM_SUBTABLES]: Sized,
  [(); S::NUM_MEMORIES]: Sized,
{
  #[tracing::instrument(skip_all, name = "HashLayer.prove")]
  fn prove(
    rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
    dense: &DensifiedRepresentation<G::ScalarField, C>,
    subtables: &Subtables<G::ScalarField, C, M, S>,
    gens: &SparsePolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (rand_mem, rand_ops) = rand;

    // decommit derefs at rand_ops
    let eval_derefs: [G::ScalarField; S::NUM_MEMORIES] =
      std::array::from_fn(|i| subtables.lookup_polys[i].evaluate(rand_ops));
    let proof_derefs = CombinedTableEvalProof::prove(
      &subtables.combined_poly,
      eval_derefs.as_ref(),
      rand_ops,
      &gens.gens_derefs,
      transcript,
      random_tape,
    );

    // form a single decommitment using comm_comb_ops
    let mut evals_ops: Vec<G::ScalarField> = Vec::new(); // moodlezoup: changed order of evals_ops

    let eval_dim: [G::ScalarField; C] = std::array::from_fn(|i| dense.dim[i].evaluate(rand_ops));
    let eval_read: [G::ScalarField; C] = std::array::from_fn(|i| dense.read[i].evaluate(rand_ops));
    let eval_final: [G::ScalarField; C] =
      std::array::from_fn(|i| dense.r#final[i].evaluate(rand_mem));

    evals_ops.extend(eval_dim);
    evals_ops.extend(eval_read);
    evals_ops.resize(evals_ops.len().next_power_of_two(), G::ScalarField::zero());

    <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claim_evals_ops", &evals_ops);

    let challenges_ops = <Transcript as ProofTranscript<G>>::challenge_vector(
      transcript,
      b"challenge_combine_n_to_one",
      evals_ops.len().log_2() as usize,
    );

    let mut poly_evals_ops = DensePolynomial::new(evals_ops);
    for i in (0..challenges_ops.len()).rev() {
      poly_evals_ops.bound_poly_var_bot(&challenges_ops[i]);
    }
    assert_eq!(poly_evals_ops.len(), 1);

    let joint_claim_eval_ops = poly_evals_ops[0];
    let mut r_joint_ops = challenges_ops;
    r_joint_ops.extend(rand_ops);
    debug_assert_eq!(
      dense.combined_l_variate_polys.evaluate(&r_joint_ops),
      joint_claim_eval_ops
    );

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"joint_claim_eval_ops",
      &joint_claim_eval_ops,
    );

    let (proof_ops, _) = Hyrax::prove(
      &dense.combined_l_variate_polys,
      &Some(joint_claim_eval_ops),
      &r_joint_ops,
      (None, None, &gens.gens_combined_l_variate, random_tape),
      transcript,
    )
    .unwrap();

    <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claim_evals_mem", &eval_final);
    let challenges_mem = <Transcript as ProofTranscript<G>>::challenge_vector(
      transcript,
      b"challenge_combine_two_to_one",
      eval_final.len().log_2() as usize,
    );

    let mut poly_evals_mem = DensePolynomial::new_padded(eval_final.to_vec());
    for i in (0..challenges_mem.len()).rev() {
      poly_evals_mem.bound_poly_var_bot(&challenges_mem[i]);
    }
    assert_eq!(poly_evals_mem.len(), 1);

    let joint_claim_eval_mem = poly_evals_mem[0];
    let mut r_joint_mem = challenges_mem;
    r_joint_mem.extend(rand_mem);
    debug_assert_eq!(
      dense.combined_log_m_variate_polys.evaluate(&r_joint_mem),
      joint_claim_eval_mem
    );

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"joint_claim_eval_mem",
      &joint_claim_eval_mem,
    );

    let (proof_mem, _) = Hyrax::prove(
      &dense.combined_log_m_variate_polys,
      &Some(joint_claim_eval_mem),
      &r_joint_mem,
      (None, None, &gens.gens_combined_log_m_variate, random_tape),
      transcript,
    )
    .unwrap();

    HashLayerProof {
      eval_dim,
      eval_read,
      eval_final,
      proof_ops,
      proof_mem,
      eval_derefs,
      proof_derefs,
    }
  }

  /// Checks that the Reed-Solomon fingerprints of init, read, write, and final multisets
  /// are as claimed by the final sumchecks of their respective grand product arguments.
  ///
  /// Params
  /// - `claims`: Fingerprint values of the init, read, write, and final multisets, as
  /// as claimed by their respective grand product arguments.
  /// - `eval_deref`: The evaluation E_i(r'''_i).
  /// - `eval_dim`: The evaluation dim_i(r'''_i).
  /// - `eval_read`: The evaluation read_i(r'''_i).
  /// - `eval_final`: The evaluation final_i(r''_i).
  /// - `init_addr`: The MLE of the memory addresses, evaluated at r''_i.
  /// - `init_memory`: The MLE of the initial memory values, evaluated at r''_i.
  /// - `r_i`: One chunk of the evaluation point at which the Lasso commitment is being opened.
  /// - `gamma`: Random value used to compute the Reed-Solomon fingerprint.
  /// - `tau`: Random value used to compute the Reed-Solomon fingerprint.
  fn check_reed_solomon_fingerprints(
    claims: &(
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
    ),
    eval_deref: &G::ScalarField,
    eval_dim: &G::ScalarField,
    eval_read: &G::ScalarField,
    eval_final: &G::ScalarField,
    init_addr: &G::ScalarField,
    init_memory: &G::ScalarField,
    gamma: &G::ScalarField,
    tau: &G::ScalarField,
  ) -> Result<(), ProofVerifyError> {
    // Computes the Reed-Solomon fingerprint of the tuple (a, v, t)
    let hash_func = |a: &G::ScalarField,
                     v: &G::ScalarField,
                     t: &G::ScalarField|
     -> G::ScalarField { *t * gamma.square() + *v * *gamma + *a - tau };
    // Note: this differs from the Lasso paper a little:
    // (t * gamma^2 + v * gamma + a) instead of (a * gamma^2 + v * gamma + t)

    let (claim_init, claim_read, claim_write, claim_final) = claims;

    // init
    let hash_init = hash_func(init_addr, init_memory, &G::ScalarField::zero());
    assert_eq!(&hash_init, claim_init); // verify the last claim of the `init` grand product sumcheck

    // read
    let hash_read = hash_func(eval_dim, eval_deref, eval_read);
    assert_eq!(hash_read, *claim_read); // verify the last claim of the `read` grand product sumcheck

    // write: shares addr, val with read
    let eval_write = *eval_read + G::ScalarField::one();
    let hash_write = hash_func(eval_dim, eval_deref, &eval_write);
    assert_eq!(hash_write, *claim_write); // verify the last claim of the `write` grand product sumcheck

    // final: shares addr and val with init
    let eval_final_addr = init_addr;
    let eval_final_val = init_memory;
    let hash_final = hash_func(eval_final_addr, eval_final_val, eval_final);
    assert_eq!(hash_final, *claim_final); // verify the last claim of the `final` grand product sumcheck

    Ok(())
  }

  fn verify(
    &self,
    rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
    grand_product_claims: &[(
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
      G::ScalarField,
    ); S::NUM_MEMORIES],
    comm: &SparsePolynomialCommitment<G>,
    gens: &SparsePolyCommitmentGens<G>,
    table_eval_commitment: &CombinedTableCommitment<G>,
    r_hash: &G::ScalarField,
    r_multiset_check: &G::ScalarField,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (rand_mem, rand_ops) = rand;

    // verify derefs at rand_ops
    // E_i(r_i''') ?= v_{E_i}
    self.proof_derefs.verify(
      rand_ops,
      &self.eval_derefs,
      &gens.gens_derefs,
      table_eval_commitment,
      transcript,
    )?;

    let mut evals_ops: Vec<G::ScalarField> = Vec::new();
    evals_ops.extend(self.eval_dim);
    evals_ops.extend(self.eval_read);
    evals_ops.resize(evals_ops.len().next_power_of_two(), G::ScalarField::zero());

    <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"claim_evals_ops", &evals_ops);

    let challenges_ops = <Transcript as ProofTranscript<G>>::challenge_vector(
      transcript,
      b"challenge_combine_n_to_one",
      evals_ops.len().log_2() as usize,
    );

    let mut poly_evals_ops = DensePolynomial::new(evals_ops);
    for i in (0..challenges_ops.len()).rev() {
      poly_evals_ops.bound_poly_var_bot(&challenges_ops[i]);
    }
    assert_eq!(poly_evals_ops.len(), 1);

    let joint_claim_eval_ops = poly_evals_ops[0];
    let mut r_joint_ops = challenges_ops;
    r_joint_ops.extend(rand_ops);
    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"joint_claim_eval_ops",
      &joint_claim_eval_ops,
    );

    // dim_i(r_i''') ?= v_i
    // read_i(r_i''') ?= v_{read_i}
    self.proof_ops.verify_plain(
      &gens.gens_combined_l_variate,
      transcript,
      &r_joint_ops,
      &joint_claim_eval_ops,
      &comm.l_variate_polys_commitment,
    )?;

    <Transcript as ProofTranscript<G>>::append_scalars(
      transcript,
      b"claim_evals_mem",
      &self.eval_final,
    );
    let challenges_mem = <Transcript as ProofTranscript<G>>::challenge_vector(
      transcript,
      b"challenge_combine_two_to_one",
      self.eval_final.len().log_2() as usize,
    );

    let mut poly_evals_mem = DensePolynomial::new_padded(self.eval_final.to_vec());
    for i in (0..challenges_mem.len()).rev() {
      poly_evals_mem.bound_poly_var_bot(&challenges_mem[i]);
    }
    assert_eq!(poly_evals_mem.len(), 1);

    let joint_claim_eval_mem = poly_evals_mem[0];
    let mut r_joint_mem = challenges_mem;
    r_joint_mem.extend(rand_mem);
    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"joint_claim_eval_mem",
      &joint_claim_eval_mem,
    );

    // final_i(r_i'') ?= v_{final_i}
    self.proof_mem.verify_plain(
      &gens.gens_combined_log_m_variate,
      transcript,
      &r_joint_mem,
      &joint_claim_eval_mem,
      &comm.log_m_variate_polys_commitment,
    )?;

    // verify the claims from the product layer
    let init_addr = IdentityPolynomial::new(rand_mem.len()).evaluate(rand_mem);
    for (i, grand_product_claim) in grand_product_claims.iter().enumerate() {
      let j = S::memory_to_dimension_index(i);
      let k = S::memory_to_subtable_index(i);
      // Check ALPHA memories / lookup polys / grand products
      // Only need 'C' indices / dimensions / read_timestamps / final_timestamps
      Self::check_reed_solomon_fingerprints(
        grand_product_claim,
        &self.eval_derefs[i],
        &self.eval_dim[j],
        &self.eval_read[j],
        &self.eval_final[j],
        &init_addr,
        &S::evaluate_subtable_mle(k, rand_mem),
        r_hash,
        r_multiset_check,
      )?;
    }
    Ok(())
  }

  fn protocol_name() -> &'static [u8] {
    b"Lasso HashLayerProof"
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct ProductLayerProof<F: PrimeField, const NUM_MEMORIES: usize> {
  grand_product_evals: [(F, F, F, F); NUM_MEMORIES],
  proof_mem: BatchedGrandProductArgument<F>,
  proof_ops: BatchedGrandProductArgument<F>,
}

impl<F: PrimeField, const NUM_MEMORIES: usize> ProductLayerProof<F, NUM_MEMORIES> {
  fn protocol_name() -> &'static [u8] {
    b"Lasso ProductLayerProof"
  }

  /// Performs grand product argument proofs required for memory-checking.
  /// Batches everything into two instances of BatchedGrandProductArgument.
  ///
  /// Params
  /// - `grand_products`: The grand product circuits whose evaluations are proven.
  /// - `transcript`: The proof transcript, used for Fiat-Shamir.
  #[tracing::instrument(skip_all, name = "ProductLayer.prove")]
  pub fn prove<G>(
    grand_products: &mut Vec<GrandProducts<F>>,
    transcript: &mut Transcript,
  ) -> (Self, Vec<F>, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let grand_product_evals: [(F, F, F, F); NUM_MEMORIES] = std::array::from_fn(|i| {
      let hash_init = grand_products[i].init.evaluate();
      let hash_read = grand_products[i].read.evaluate();
      let hash_write = grand_products[i].write.evaluate();
      let hash_final = grand_products[i].r#final.evaluate();

      assert_eq!(hash_init * hash_write, hash_read * hash_final);

      <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"claim_hash_init", &hash_init);
      <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"claim_hash_read", &hash_read);
      <Transcript as ProofTranscript<G>>::append_scalar(
        transcript,
        b"claim_hash_write",
        &hash_write,
      );
      <Transcript as ProofTranscript<G>>::append_scalar(
        transcript,
        b"claim_hash_final",
        &hash_final,
      );

      (hash_init, hash_read, hash_write, hash_final)
    });

    let mut read_write_grand_products: Vec<&mut GrandProductCircuit<F>> = grand_products
      .iter_mut()
      .flat_map(|grand_product| [&mut grand_product.read, &mut grand_product.write])
      .collect();

    let (proof_ops, rand_ops) =
      BatchedGrandProductArgument::<F>::prove::<G>(&mut read_write_grand_products, transcript);

    let mut init_final_grand_products: Vec<&mut GrandProductCircuit<F>> = grand_products
      .iter_mut()
      .flat_map(|grand_product| [&mut grand_product.init, &mut grand_product.r#final])
      .collect();

    // produce a batched proof of memory-related product circuits
    let (proof_mem, rand_mem) =
      BatchedGrandProductArgument::<F>::prove::<G>(&mut init_final_grand_products, transcript);

    let product_layer_proof = ProductLayerProof {
      grand_product_evals,
      proof_mem,
      proof_ops,
    };

    (product_layer_proof, rand_mem, rand_ops)
  }

  pub fn verify<G>(
    &self,
    num_ops: usize,
    num_cells: usize,
    transcript: &mut Transcript,
  ) -> Result<(Vec<F>, Vec<F>, Vec<F>, Vec<F>), ProofVerifyError>
  where
    G: CurveGroup<ScalarField = F>,
  {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    for (hash_init, hash_read, hash_write, hash_final) in self.grand_product_evals {
      // Multiset equality check
      assert_eq!(hash_init * hash_write, hash_read * hash_final);

      <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"claim_hash_init", &hash_init);
      <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"claim_hash_read", &hash_read);
      <Transcript as ProofTranscript<G>>::append_scalar(
        transcript,
        b"claim_hash_write",
        &hash_write,
      );
      <Transcript as ProofTranscript<G>>::append_scalar(
        transcript,
        b"claim_hash_final",
        &hash_final,
      );
    }

    let read_write_claims: Vec<F> = self
      .grand_product_evals
      .iter()
      .flat_map(|(_, hash_read, hash_write, _)| [*hash_read, *hash_write])
      .collect();

    let (claims_ops, rand_ops) =
      self
        .proof_ops
        .verify::<G, Transcript>(&read_write_claims, num_ops, transcript);

    let init_final_claims: Vec<F> = self
      .grand_product_evals
      .iter()
      .flat_map(|(hash_init, _, _, hash_final)| [*hash_init, *hash_final])
      .collect();

    let (claims_mem, rand_mem) =
      self
        .proof_mem
        .verify::<G, Transcript>(&init_final_claims, num_cells, transcript);

    Ok((claims_mem, rand_mem, claims_ops, rand_ops))
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;

  use super::*;

  #[test]
  fn test() {
    // Memory size: 8
    // Sparsity (num-ops): 4
    let eval_table = vec![
      Fr::from(10),
      Fr::from(11),
      Fr::from(12),
      Fr::from(13),
      Fr::from(14),
      Fr::from(15),
      Fr::from(16),
      Fr::from(17),
    ];
    let dim_i = DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(1), Fr::from(5)]);
    let dim_i_usize = vec![1usize, 2, 1, 5];
    let read_i = DensePolynomial::new(vec![Fr::from(0), Fr::from(0), Fr::from(1), Fr::from(0)]);
    let final_i = DensePolynomial::new(vec![
      Fr::from(0),
      Fr::from(2),
      Fr::from(1),
      Fr::from(0),
      Fr::from(0),
      Fr::from(1),
      Fr::from(0),
      Fr::from(0),
    ]);
    let r_mem_check = (Fr::from(100), Fr::from(200));

    let _gp = GrandProducts::new(
      &eval_table,
      &dim_i,
      &dim_i_usize,
      &read_i,
      &final_i,
      &r_mem_check,
    );
  }
}
