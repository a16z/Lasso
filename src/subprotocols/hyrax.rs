use super::dot_product::DotProductProofGens;
use super::traits::PolynomialCommitmentScheme;

use crate::poly::commitments::{Commitments, MultiCommitGens};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::subprotocols::dot_product::DotProductProofLog;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::random::RandomTape;
use crate::utils::transcript::ProofTranscript;
use crate::{msm::VariableBaseMSM, utils::transcript::AppendToTranscript};
use ark_ec::CurveGroup;
use ark_serialize::*;
use ark_std::Zero;
use merlin::Transcript;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::borrow::Borrow;
use std::marker::PhantomData;

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct PolyCommitment<G: CurveGroup> {
  C: Vec<G>,
}

impl<G: CurveGroup> AppendToTranscript<G> for PolyCommitment<G> {
  fn append_to_transcript<T: ProofTranscript<G>>(&self, label: &'static [u8], transcript: &mut T) {
    transcript.append_message(label, b"poly_commitment_begin");
    for i in 0..self.C.len() {
      transcript.append_point(b"poly_commitment_share", &self.C[i]);
    }
    transcript.append_message(label, b"poly_commitment_end");
  }
}

#[derive(Clone)]
pub struct PolyCommitmentBlinds<F> {
  blinds: Vec<F>,
}

#[derive(Clone)]
pub struct PolyCommitmentGens<G> {
  pub gens: DotProductProofGens<G>,
}

impl<G: CurveGroup> PolyCommitmentGens<G> {
  // the number of variables in the multilinear polynomial
  pub fn new(num_vars: usize, label: &'static [u8]) -> Self {
    let (_left, right) = EqPolynomial::<G::ScalarField>::compute_factored_lens(num_vars);
    let gens = DotProductProofGens::new(right.pow2(), label);
    PolyCommitmentGens { gens }
  }
}

#[derive(Debug, CanonicalSerialize, Clone, CanonicalDeserialize)]
pub struct PolyEvalProof<G: CurveGroup> {
  proof: DotProductProofLog<G>,
}

impl<G: CurveGroup> PolyEvalProof<G> {
  fn protocol_name() -> &'static [u8] {
    b"polynomial evaluation proof"
  }

  pub fn verify_plain(
    &self,
    gens: &PolyCommitmentGens<G>,
    transcript: &mut Transcript,
    r: &[G::ScalarField],
    Zr: &G::ScalarField,
    comm: &PolyCommitment<G>,
  ) -> Result<(), ProofVerifyError> {
    // compute a commitment to Zr with a blind of zero
    let C_Zr = Zr.commit(&G::ScalarField::zero(), &gens.gens.gens_1);

    // TODO: Make blinds an Option
    Hyrax::verify(
      &(
        comm.clone(),
        PolyCommitmentBlinds {
          blinds: vec![G::ScalarField::zero()],
        },
      ),
      &None,
      r.to_vec(),
      gens,
      transcript,
      (self.clone(), C_Zr),
    )
  }
}

#[cfg(feature = "multicore")]
fn commit_inner<G: CurveGroup>(
  poly: &DensePolynomial<G::ScalarField>,
  blinds: &[G::ScalarField],
  gens: &MultiCommitGens<G>,
) -> PolyCommitment<G> {
  let L_size = blinds.len();
  let R_size = poly.Z.len() / L_size;
  assert_eq!(L_size * R_size, poly.Z.len());
  let C = (0..L_size)
    .into_par_iter()
    .map(|i| {
      Commitments::batch_commit(
        poly.Z[R_size * i..R_size * (i + 1)].as_ref(),
        &blinds[i],
        gens,
      )
    })
    .collect();
  PolyCommitment { C }
}

#[cfg(not(feature = "multicore"))]
fn commit_inner<G: CurveGroup>(
  poly: &DensePolynomial<G::ScalarField>,
  blinds: &[F],
  gens: &MultiCommitGens<G>,
) -> PolyCommitment<G> {
  let L_size = blinds.len();
  let R_size = poly.Z.len() / L_size;
  assert_eq!(L_size * R_size, poly.Z.len());
  let C = (0..L_size)
    .map(|i| {
      Commitments::batch_commit(
        poly.Z[R_size * i..R_size * (i + 1)].as_ref(),
        &blinds[i],
        gens,
      )
    })
    .collect();
  PolyCommitment { C }
}

pub struct Hyrax<G: CurveGroup> {
  _phantom: PhantomData<G>,
}

impl<G: CurveGroup> PolynomialCommitmentScheme for Hyrax<G> {
  // TODO: remove/manage blinds somehow -> Option
  type Commitment = (PolyCommitment<G>, PolyCommitmentBlinds<G::ScalarField>);
  type Polynomial = DensePolynomial<G::ScalarField>;
  type Evaluation = Option<G::ScalarField>;
  type Challenge = Vec<G::ScalarField>;
  type Proof = (PolyEvalProof<G>, G);
  type Error = ProofVerifyError;

  type ProverKey<'p> = (
    Option<&'p PolyCommitmentBlinds<G::ScalarField>>,
    Option<&'p G::ScalarField>,
    &'p PolyCommitmentGens<G>,
    &'p mut RandomTape<G>,
  );
  type CommitmentKey<'c> = (PolyCommitmentGens<G>, Option<&'c mut RandomTape<G>>);
  type VerifierKey = PolyCommitmentGens<G>;

  #[tracing::instrument(skip_all, name = "DensePolynomial.commit")]
  fn commit<'a, 'c>(
    poly: &'a Self::Polynomial,
    ck: Self::CommitmentKey<'c>,
  ) -> Result<Self::Commitment, Self::Error> 
  {
    let n = poly.Z.len();
    let ell = poly.get_num_vars();
    let (gens, random_tape) = ck;
    assert_eq!(n, ell.pow2());

    let (left_num_vars, right_num_vars) =
      EqPolynomial::<G::ScalarField>::compute_factored_lens(ell);
    let L_size = left_num_vars.pow2();
    let R_size = right_num_vars.pow2();
    assert_eq!(L_size * R_size, n);

    let blinds = if let Some(t) = random_tape {
      PolyCommitmentBlinds {
        blinds: t.random_vector(b"poly_blinds", L_size),
      }
    } else {
      PolyCommitmentBlinds {
        blinds: vec![G::ScalarField::zero(); L_size],
      }
    };
    Ok((
      commit_inner(&poly, &blinds.blinds, &gens.gens.gens_n),
      blinds,
    ))
  }

  // Note this excludes commitments which introduces a concern that the proof would generate for polys and commitments not tied to one another
  fn prove<'a, 'p>(
    poly: &'a Self::Polynomial,
    //blinds_opt: Option<&PolyCommitmentBlinds<G::ScalarField>>,
    evals: &'a Self::Evaluation,     // evaluation of \widetilde{Z}(r)
    challenges: &'a Self::Challenge, // point at which the polynomial is evaluated
    //blind_Zr_opt: Option<&G::ScalarField>, // specifies a blind for Zr
    //gens: &PolyCommitmentGens<G>,
    //random_tape: &mut RandomTape<G>,
    pk: Self::ProverKey<'p>,
    transcript: &mut Transcript,
  ) -> Result<Self::Proof, Self::Error> 
    where
      Self::Challenge: 'a,
  {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      PolyEvalProof::<G>::protocol_name(),
    );

    let (blinds_opt, blind_Zr_opt, gens, random_tape) = pk;

    // assert vectors are of the right size
    assert_eq!(poly.get_num_vars(), challenges.len());

    let (left_num_vars, right_num_vars) =
      EqPolynomial::<G::ScalarField>::compute_factored_lens(challenges.len());
    let L_size = left_num_vars.pow2();
    let R_size = right_num_vars.pow2();

    let default_blinds = PolyCommitmentBlinds {
      blinds: vec![G::ScalarField::zero(); L_size],
    };
    let blinds = blinds_opt.map_or(&default_blinds, |p| &p);

    assert_eq!(blinds.blinds.len(), L_size);

    let zero = G::ScalarField::zero();
    let blind_Zr = blind_Zr_opt.map_or(&zero, |p| &p);

    // compute the L and R vectors
    let eq = EqPolynomial::new(challenges.to_vec());
    let (L, R) = eq.compute_factored_evals();
    assert_eq!(L.len(), L_size);
    assert_eq!(R.len(), R_size);

    // compute the vector underneath L*Z and the L*blinds
    // compute vector-matrix product between L and Z viewed as a matrix
    let LZ = poly.bound(&L);
    let LZ_blind: G::ScalarField = (0..L.len()).map(|i| blinds.blinds[i] * L[i]).sum();

    // a dot product proof of size R_size
    // TODO: how to remove this unwrap and still maintain clean interface
    let (proof, _C_LR, C_Zr_prime) = DotProductProofLog::prove(
      &gens.gens,
      transcript,
      random_tape,
      &LZ,
      &LZ_blind,
      &R,
      //TODO: fix this nasty unwrap
      &evals.unwrap(),
      blind_Zr,
    );

    Ok((PolyEvalProof { proof }, C_Zr_prime))
  }

  fn verify<'a>(
    commitments: &'a Self::Commitment,
    // Find a better way to handle this... perhaps verifier key???
    evals: &'a Self::Evaluation,
    challenges: Self::Challenge, // point at which the polynomial is evaluated
    vk: &'a Self::VerifierKey, // C_Zr commitment to \widetilde{Z}(r)
    transcript: &mut Transcript,
    proof: Self::Proof,
  ) -> Result<(), Self::Error> 
    where
      Self::Commitment: 'a,
      Self::VerifierKey: 'a,
  {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      PolyEvalProof::<G>::protocol_name(),
    );
    let (proof, C_Zr) = proof;
    let (comm, _) = commitments;
    let gens = vk.borrow();

    // compute L and R
    let eq = EqPolynomial::new(challenges);
    let (L, R) = eq.compute_factored_evals();

    // compute a weighted sum of commitments and L
    let C_affine = G::normalize_batch(&comm.C);

    let C_LZ = VariableBaseMSM::msm(C_affine.as_ref(), L.as_ref()).unwrap();
    Ok(
      proof
        .proof
        .verify(R.len(), &gens.gens, transcript, &R, &C_LZ, &C_Zr)
        .unwrap(),
    )
  }
}
