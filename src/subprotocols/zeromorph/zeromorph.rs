#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::{iter, marker::PhantomData};

use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::traits::PolynomialCommitmentScheme;
use crate::subprotocols::zeromorph::kzg::UniversalKzgSrs;
use crate::utils::transcript::ProofTranscript;
use ark_bn254::Bn254;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{batch_inversion, Field};
use ark_std::{iterable::Iterable, One, Zero, ops::Neg};
use itertools::Itertools;
use lazy_static::lazy_static;
use merlin::Transcript;
use rand_chacha::{
  rand_core::{RngCore, SeedableRng},
  ChaCha20Rng,
};
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[cfg(not(feature = "ark-msm"))]
use crate::msm::VariableBaseMSM;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

use super::kzg::UnivariateKZG;

const MAX_VARS: usize = 17;

lazy_static! {
  pub static ref ZEROMORPH_SRS: Arc<Mutex<ZeromorphSRS<Bn254>>> =
    Arc::new(Mutex::new(ZeromorphSRS::setup(
      1 << (MAX_VARS + 1),
      &mut ChaCha20Rng::from_seed(*b"ZEROMORPH_POLY_COMMITMENT_SCHEME")
    )));
}

#[derive(Debug, Clone, Default)]
pub struct ZeromorphSRS<P: Pairing>(UniversalKzgSrs<P>);

impl<P: Pairing> ZeromorphSRS<P> {
  pub fn setup<R: RngCore>(max_degree: usize, rng: &mut R) -> ZeromorphSRS<P> {
    ZeromorphSRS(UniversalKzgSrs::<P>::setup(max_degree, rng))
  }

  pub fn trim(
    &self,
    max_degree: usize,
  ) -> Result<(ZeromorphProverKey<P>, ZeromorphVerifierKey<P>), ZeromorphError> {
    let offset = self.0.g1_powers.len() - max_degree;
    if self.0.g1_powers.len() >= max_degree + offset {
      return Err(ZeromorphError::KeyLengthError(
        max_degree,
        self.0.g1_powers.len(),
      ));
    }
    let offset = self.0.g1_powers.len() - max_degree;
    let offset_g1_powers = self.0.g1_powers[offset..(offset + max_degree)].to_vec();
    Ok((
      ZeromorphProverKey {
        g1_powers: self.0.g1_powers.clone(),
        offset_g1_powers: offset_g1_powers,
      },
      ZeromorphVerifierKey {
        g1: self.0.g1_powers[0],
        g2: self.0.g2_powers[0],
        tau_2: self.0.g2_powers[1],
        tau_N_max_sub_2_N: self.0.g2_powers[offset],
      },
    ))
  }
}

#[derive(Clone, Debug)]
pub struct ZeromorphProverKey<P: Pairing> {
  pub g1_powers: Vec<P::G1Affine>,
  pub offset_g1_powers: Vec<P::G1Affine>,
}

#[derive(Copy, Clone, Debug)]
pub struct ZeromorphVerifierKey<P: Pairing> {
  pub g1: P::G1Affine,
  pub g2: P::G2Affine,
  pub tau_2: P::G2Affine,
  pub tau_N_max_sub_2_N: P::G2Affine,
}

#[derive(Clone, Debug)]
pub struct ZeromorphProof<P: Pairing> {
  pub pi: P::G1Affine,
  pub q_hat_com: P::G1Affine,
  pub q_k_com: Vec<P::G1Affine>,
}

#[derive(Error, Debug)]
pub enum ZeromorphError {
  #[error("Length Error: SRS Length: {0}, Key Length: {0}")]
  KeyLengthError(usize, usize),
}

fn compute_multilinear_quotients<P: Pairing>(
  poly: &DensePolynomial<P::ScalarField>,
  u_challenge: &[P::ScalarField],
) -> (Vec<UniPoly<P::ScalarField>>, P::ScalarField) {
  assert_eq!(poly.get_num_vars(), u_challenge.len());

  let mut g = poly.Z.to_vec();
  let mut quotients: Vec<_> = u_challenge
    .iter()
    .enumerate()
    .map(|(i, x_i)| {
      let (g_lo, g_hi) = g.split_at_mut(1 << (poly.get_num_vars() - 1 - i));
      let mut quotient = vec![P::ScalarField::zero(); g_lo.len()];

      #[cfg(feature = "multicore")]
      let quotient_iter = quotient.par_iter_mut();

      #[cfg(not(feature = "multicore"))]
      let quotient_iter = quotient.iter_mut();

      quotient_iter
        .zip_eq(&*g_lo)
        .zip_eq(&*g_hi)
        .for_each(|((q, g_lo), g_hi)| {
          *q = *g_hi - *g_lo;
        });

      #[cfg(feature = "multicore")]
      let g_lo_iter = g_lo.par_iter_mut();

      #[cfg(not(feature = "multicore"))]
      let g_lo_iter = g_lo.iter_mut();
      g_lo_iter.zip_eq(g_hi).for_each(|(g_lo, g_hi)| {
        *g_lo += (*g_hi - g_lo as &_) * x_i;
      });

      g.truncate(1 << (poly.get_num_vars() - 1 - i));

      UniPoly::from_coeff(quotient)
    })
    .collect();
  quotients.reverse();
  (quotients, g[0])
}

fn compute_batched_lifted_degree_quotient<P: Pairing>(
  n: usize,
  quotients: &Vec<UniPoly<P::ScalarField>>,
  y_challenge: &P::ScalarField,
) -> UniPoly<P::ScalarField> {
  // Compute \hat{q} = \sum_k y^k * X^{N - d_k - 1} * q_k
  let mut scalar = P::ScalarField::one(); // y^k
                                          // Rather than explicitly computing the shifts of q_k by N - d_k - 1 (i.e. multiplying q_k by X^{N - d_k - 1})
                                          // then accumulating them, we simply accumulate y^k*q_k into \hat{q} at the index offset N - d_k - 1
  let q_hat =
    quotients
      .iter()
      .enumerate()
      .fold(vec![P::ScalarField::zero(); n], |mut q_hat, (idx, q)| {
        #[cfg(feature = "multicore")]
        let q_hat_iter = q_hat[n - (1 << idx)..].par_iter_mut();

        #[cfg(not(feature = "multicore"))]
        let q_hat_iter = q_hat[n - (1 << idx)..].iter_mut();
        q_hat_iter.zip(&q.coeffs).for_each(|(q_hat, q)| {
          *q_hat += scalar * q;
        });
        scalar *= y_challenge;
        q_hat
      });

  UniPoly::from_coeff(q_hat)
}

fn eval_and_quotient_scalars<P: Pairing>(
  y_challenge: P::ScalarField,
  x_challenge: P::ScalarField,
  z_challenge: P::ScalarField,
  challenges: &[P::ScalarField],
) -> (P::ScalarField, (Vec<P::ScalarField>, Vec<P::ScalarField>)) {
  let num_vars = challenges.len();

  // squares of x = [x, x^2, .. x^{2^k}, .. x^{2^num_vars}]
  let squares_of_x: Vec<_> = iter::successors(Some(x_challenge), |&x| Some(x.square()))
    .take(num_vars + 1)
    .collect();

  // offsets of x =
  let offsets_of_x = {
    let mut offsets_of_x = squares_of_x
      .iter()
      .rev()
      .skip(1)
      .scan(P::ScalarField::one(), |acc, pow_x| {
        *acc *= pow_x;
        Some(*acc)
      })
      .collect::<Vec<_>>();
    offsets_of_x.reverse();
    offsets_of_x
  };

  let vs = {
    let v_numer = squares_of_x[num_vars] - P::ScalarField::one();
    let mut v_denoms = squares_of_x
      .iter()
      .map(|squares_of_x| *squares_of_x - P::ScalarField::one())
      .collect::<Vec<_>>();
    batch_inversion(&mut v_denoms);
    v_denoms
      .iter()
      .map(|v_denom| v_numer * v_denom)
      .collect::<Vec<_>>()
  };

  let q_scalars = iter::successors(Some(P::ScalarField::one()), |acc| Some(*acc * y_challenge))
    .take(num_vars)
    .zip_eq(offsets_of_x)
    .zip(squares_of_x)
    .zip(&vs)
    .zip_eq(&vs[1..])
    .zip_eq(challenges.iter().rev())
    .map(
      |(((((power_of_y, offset_of_x), square_of_x), v_i), v_j), u_i)| {
        (
          -(power_of_y * offset_of_x),
          -(z_challenge * (square_of_x * v_j - *u_i * v_i)),
        )
      },
    )
    .unzip();

  // -vs[0] * z = -z * (x^(2^num_vars) - 1) / (x - 1) = -z Φ_n(x)
  (-vs[0] * z_challenge, q_scalars)
}

pub trait ZMPolynomialCommitmentScheme {
  // Abstracting over Polynomial allows us to have batched and non-batched PCS
  type Polynomial;
  type Commitment;
  type Evaluation;
  type Challenge;
  type Proof;
  type Error;

  type ProverKey<'p>;
  type CommitmentKey<'c>;
  type VerifierKey;

  //TODO: convert to impl IntoIterator<Item = Self::Polynomial>
  fn commit<'c>(
    poly: &[Self::Polynomial],
    ck: &Self::CommitmentKey<'c>,
  ) -> Result<Vec<Self::Commitment>, Self::Error>;

  fn prove<'p>(
    poly: &[Self::Polynomial],
    evals: &[Self::Evaluation],
    challenges: &[Self::Challenge],
    pk: &Self::ProverKey<'p>,
    transcript: &mut Transcript,
  ) -> Result<Self::Proof, Self::Error>;

  fn verify(
    commitments: &[Self::Commitment],
    evals: &[Self::Evaluation],
    challenges: &[Self::Challenge],
    vk: &Self::VerifierKey,
    transcript: &mut Transcript,
    proof: Self::Proof,
  ) -> Result<(), Self::Error>;
}

pub struct Zeromorph<P: Pairing> {
  _phantom: PhantomData<P>,
}

impl<P: Pairing> ZMPolynomialCommitmentScheme for Zeromorph<P> 
where <P as Pairing>::G2Affine: Neg
{
  type Commitment = P::G1Affine;
  type Polynomial = DensePolynomial<P::ScalarField>;
  type Evaluation = P::ScalarField;
  type Challenge = P::ScalarField;
  type Proof = ZeromorphProof<P>;
  type Error = ZeromorphError;

  type ProverKey<'p> = ZeromorphProverKey<P>;
  type CommitmentKey<'c> = Vec<Self::Commitment>;
  type VerifierKey = ZeromorphVerifierKey<P>;

  fn commit<'c>(
    polys: &[Self::Polynomial],
    ck:  &Self::CommitmentKey<'c>,
  ) -> Result<Vec<Self::Commitment>, Self::Error>
    {
    // TODO: assert lengths are valid
    #[cfg(feature = "multicore")]
    let iter = polys.par_iter();
    #[cfg(not(feature = "multicore"))]
    let iter = polys.iter();
    Ok(
      iter
        .map(|poly| UnivariateKZG::<P>::commit(ck, &UniPoly::from_coeff(poly.Z.clone())).unwrap())
        .collect::<Vec<_>>(),
    )
  }

  fn prove<'p>(
    polys: &[Self::Polynomial],
    evals: &[Self::Evaluation],
    challenge: &[Self::Challenge],
    pk: &Self::ProverKey<'p>,
    transcript: &mut Transcript,
  ) -> Result<Self::Proof, Self::Error> 
  {
    let num_vars = challenge.len();
    let n: usize = 1 << num_vars;

    for (poly, eval) in polys.iter().zip_eq(evals.iter()) {
      // Note by evaluating we confirm the number of challenges is valid
      debug_assert_eq!(poly.evaluate(challenge), *eval);
    }

    // Generate batching challenge \rho and powers 1,...,\rho^{m-1}
    let rho = <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: rho");
    // Compute batching of unshifted polynomials f_i:
    let mut scalar = P::ScalarField::one();
    let (f_batched, batched_evaluation) = (0..polys.len()).fold(
      (
        DensePolynomial::new(vec![P::ScalarField::zero(); n]),
        P::ScalarField::zero(),
      ),
      |(mut f_batched, mut batched_evaluation), i| {
        f_batched += polys[i].clone() * scalar;
        batched_evaluation += scalar * evals[i];
        scalar *= rho;
        (f_batched, batched_evaluation)
      },
    );
    let mut pi_poly = UniPoly::from_coeff(f_batched.Z.clone());

    // Compute the multilinear quotients q_k = q_k(X_0, ..., X_{k-1})
    let (quotients, remainder) = compute_multilinear_quotients::<P>(&f_batched, challenge);
    debug_assert_eq!(quotients.len(), f_batched.get_num_vars());
    debug_assert_eq!(remainder, batched_evaluation);

    // Compute and send commitments C_{q_k} = [q_k], k = 0, ..., d-1
    let label = b"q_k_commitments";
    transcript.append_message(label, b"begin_append_vector");
    let q_k_commitments: Vec<_> = quotients
      .iter()
      .map(|q| {
        let q_k_commitment = UnivariateKZG::<P>::commit(&pk.g1_powers, q).unwrap();
        transcript.append_point(label, &q_k_commitment.into_group());
        q_k_commitment
      })
      .collect();
    transcript.append_message(label, b"end_append_vector");

    // Get challenge y
    let y_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: y");

    // Compute the batched, lifted-degree quotient \hat{q}
    let q_hat = compute_batched_lifted_degree_quotient::<P>(n, &quotients, &y_challenge);

    // Compute and send the commitment C_q = [\hat{q}]
    // commit at offset
    let offset = 1 << (quotients.len() - 1);
    let q_hat_com = UnivariateKZG::<P>::commit_offset(&pk.g1_powers, &q_hat, offset).unwrap();
    transcript.append_point(b"ZM: C_q_hat", &q_hat_com.into_group());

    // Get challenges x and z
    let x_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: x");
    let z_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: z");

    let (eval_scalar, (zeta_degree_check_q_scalars, z_zmpoly_q_scalars)) =
      eval_and_quotient_scalars::<P>(y_challenge, x_challenge, z_challenge, challenge);
    // f = z * x * poly.Z + q_hat + (-z * x * Φ_n(x) * e) + x * ∑_k (q_scalars_k * q_k)
    pi_poly *= &z_challenge;
    pi_poly += &q_hat;
    pi_poly[0] += &(batched_evaluation * eval_scalar);
    quotients
      .into_iter()
      .zip_eq(zeta_degree_check_q_scalars)
      .zip_eq(z_zmpoly_q_scalars)
      .for_each(|((mut q, zeta_degree_check_q_scalar), z_zmpoly_q_scalar)| {
        q *= &(zeta_degree_check_q_scalar + z_zmpoly_q_scalar);
        pi_poly += &q;
      });

    debug_assert_eq!(pi_poly.evaluate(&x_challenge), P::ScalarField::zero());

    // Compute the KZG opening proof pi_poly; -> TODO should abstract into separate trait
    let (pi, _) = UnivariateKZG::<P>::open(&pk.offset_g1_powers, &pi_poly, &x_challenge).unwrap();

    Ok(ZeromorphProof {
      pi,
      q_hat_com,
      q_k_com: q_k_commitments,
    })
  }

  fn verify(
    commitments: &[Self::Commitment],
    evals: &[Self::Evaluation],
    challenges: &[Self::Challenge],
    vk: &Self::VerifierKey,
    transcript: &mut Transcript,
    proof: Self::Proof,
  ) -> Result<(), Self::Error> 
  {
    debug_assert_eq!(evals.len(), commitments.len());
    let ZeromorphProof {
      pi,
      q_k_com,
      q_hat_com,
    } = proof;
    let pi = pi.into_group();
    let q_hat_com = q_hat_com.into_group();

    //Receive q_k commitments
    let label = b"q_k_commitments";
    transcript.append_message(label, b"begin_append_vector");
    q_k_com
      .iter()
      .for_each(|c| transcript.append_point(b"ZM: C_q_k", &c.into_group()));
    transcript.append_message(label, b"end_append_vector");

    // Compute powers of batching challenge rho
    let rho = <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: rho");

    // Compute batching of unshifted polynomials f_i:
    let mut scalar = P::ScalarField::one();
    let (batched_evaluation, batched_commitment) = evals.iter().zip_eq(commitments.iter()).fold(
      (P::ScalarField::zero(), P::G1::zero()),
      |(mut batched_evaluation, mut batched_commitment), (eval, commitment)| {
        batched_evaluation += scalar * eval;
        batched_commitment += *commitment * scalar;
        scalar *= rho;
        (batched_evaluation, batched_commitment)
      },
    );

    // Challenge y
    let y_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: y");

    // Receive commitment C_{q} -> Since our transcript does not support appending and receiving data we instead store these commitments in a zeromorph proof struct
    transcript.append_point(b"ZM: C_q_hat", &q_hat_com);

    // Challenge x, z
    let x_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: x");
    let z_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: z");

    let (eval_scalar, (mut q_scalars, zm_poly_q_scalars)) =
      eval_and_quotient_scalars::<P>(y_challenge, x_challenge, z_challenge, challenges);

    q_scalars
      .iter_mut()
      .zip_eq(zm_poly_q_scalars)
      .for_each(|(scalar, zm_poly_scalar)| {
        *scalar += zm_poly_scalar;
      });

    let scalars = [
      vec![
        P::ScalarField::one(),
        z_challenge,
        batched_evaluation * eval_scalar,
      ],
      q_scalars,
    ]
    .concat();

    let bases = [
      vec![q_hat_com.into(), batched_commitment.into(), vk.g1],
      q_k_com,
    ]
    .concat();
    let Zeta_z_com = <P::G1 as VariableBaseMSM>::msm(&bases, &scalars).unwrap();

    // e(pi, [tau]_2 - x * [1]_2) == e(C_{\zeta,Z}, -[X^(N_max - 2^n - 1)]_2) <==> e(C_{\zeta,Z} - x * pi, [X^{N_max - 2^n - 1}]_2) * e(-pi, [tau_2]) == 1
    let e = P::multi_pairing(&[Zeta_z_com, pi], &[vk.tau_2.into_group() - (vk.g2 * x_challenge), -vk.tau_N_max_sub_2_N.into_group()]);
    assert!(e.is_zero());
    Ok(())
  }
}

#[cfg(test)]
mod test {

  use super::*;
  use crate::utils::math::Math;
  use ark_bn254::{Bn254, Fr};
  use ark_ff::{BigInt, Zero};
  use ark_std::{test_rng, UniformRand};

  // Evaluate Phi_k(x) = \sum_{i=0}^k x^i using the direct inefficent formula
  fn phi<P: Pairing>(challenge: &P::ScalarField, subscript: usize) -> P::ScalarField {
    let len = (1 << subscript) as u64;
    (0..len)
      .into_iter()
      .fold(P::ScalarField::zero(), |mut acc, i| {
        //Note this is ridiculous DevX
        acc += challenge.pow(BigInt::<1>::from(i));
        acc
      })
  }

  #[test]
  fn prove_verify_single() {
    let max_vars = 8;
    let mut rng = test_rng();
    let srs = ZEROMORPH_SRS.lock().unwrap();

    for num_vars in 3..max_vars {
      // Setup
      let (pk, vk) = {
        let poly_size = 1 << (num_vars + 1);
        srs.trim(poly_size - 1).unwrap()
      };
      let polys = DensePolynomial::new(
        (0..(1 << num_vars))
          .map(|_| Fr::rand(&mut rng))
          .collect::<Vec<_>>(),
      );
      let challenges = (0..num_vars)
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>();
      let evals = polys.evaluate(&challenges);

      // Commit and open
      let commitments = Zeromorph::<Bn254>::commit(&[polys.clone()], &pk.g1_powers).unwrap();

      let mut prover_transcript = Transcript::new(b"example");
      let proof = Zeromorph::<Bn254>::prove(
        &[polys],
        &[evals],
        &challenges,
        &pk,
        &mut prover_transcript,
      )
      .unwrap();

      let mut verifier_transcript = Transcript::new(b"example");
      Zeromorph::<Bn254>::verify(
        &commitments,
        &[evals],
        &challenges,
        &vk,
        &mut verifier_transcript,
        proof,
      )
      .unwrap();

      //TODO: check both random oracles are synced
    }
  }

  #[test]
  fn prove_verify_batched() {
    let max_vars = 16;
    let mut rng = test_rng();
    let num_polys = 8;
    let srs = ZEROMORPH_SRS.lock().unwrap();

    for num_vars in 3..max_vars {
      // Setup
      let (pk, vk) = {
        let poly_size = 1 << (num_vars + 1);
        srs.trim(poly_size - 1).unwrap()
      };
      let polys: Vec<DensePolynomial<Fr>> = (0..num_polys)
        .map(|_| {
          DensePolynomial::new(
            (0..(1 << num_vars))
              .map(|_| Fr::rand(&mut rng))
              .collect::<Vec<_>>(),
          )
        })
        .collect::<Vec<_>>();
      let challenges = (0..num_vars)
        .into_iter()
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>();
      let evals = polys
        .clone()
        .into_iter()
        .map(|poly| poly.evaluate(&challenges))
        .collect::<Vec<_>>();

      // Commit and open
      let commitments = Zeromorph::<Bn254>::commit(&polys, &pk.g1_powers).unwrap();

      let mut prover_transcript = Transcript::new(b"example");
      let proof =
        Zeromorph::<Bn254>::prove(&polys, &evals, &challenges, &pk, &mut prover_transcript).unwrap();

      let mut verifier_transcript = Transcript::new(b"example");
      Zeromorph::<Bn254>::verify(
        &commitments,
        &evals,
        &challenges,
        &vk,
        &mut verifier_transcript,
        proof,
      )
      .unwrap();

      //TODO: check both random oracles are synced
    }
  }

  /// Test for computing qk given multilinear f
  /// Given 𝑓(𝑋₀, …, 𝑋ₙ₋₁), and `(𝑢, 𝑣)` such that \f(\u) = \v, compute `qₖ(𝑋₀, …, 𝑋ₖ₋₁)`
  /// such that the following identity holds:
  ///
  /// `𝑓(𝑋₀, …, 𝑋ₙ₋₁) − 𝑣 = ∑ₖ₌₀ⁿ⁻¹ (𝑋ₖ − 𝑢ₖ) qₖ(𝑋₀, …, 𝑋ₖ₋₁)`
  #[test]
  fn quotient_construction() {
    // Define size params
    let num_vars = 4;
    let n: u64 = 1 << num_vars;

    // Construct a random multilinear polynomial f, and (u,v) such that f(u) = v
    let mut rng = test_rng();
    let multilinear_f =
      DensePolynomial::new((0..n).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>());
    let u_challenge = (0..num_vars)
      .into_iter()
      .map(|_| Fr::rand(&mut rng))
      .collect::<Vec<_>>();
    let v_evaluation = multilinear_f.evaluate(&u_challenge);

    // Compute multilinear quotients `qₖ(𝑋₀, …, 𝑋ₖ₋₁)`
    let (quotients, constant_term) =
      compute_multilinear_quotients::<Bn254>(&multilinear_f, &u_challenge);

    // Assert the constant term is equal to v_evaluation
    assert_eq!(
      constant_term, v_evaluation,
      "The constant term equal to the evaluation of the polynomial at challenge point."
    );

    //To demonstrate that q_k was properly constructd we show that the identity holds at a random multilinear challenge
    // i.e. 𝑓(𝑧) − 𝑣 − ∑ₖ₌₀ᵈ⁻¹ (𝑧ₖ − 𝑢ₖ)𝑞ₖ(𝑧) = 0
    let z_challenge = (0..num_vars)
      .map(|_| Fr::rand(&mut rng))
      .collect::<Vec<_>>();

    let mut res = multilinear_f.evaluate(&z_challenge);
    res -= v_evaluation;

    for (k, q_k_uni) in quotients.iter().enumerate() {
      let z_partial = &z_challenge[&z_challenge.len() - k..];
      //This is a weird consequence of how things are done.. the univariate polys are of the multilinear commitment in lagrange basis. Therefore we evaluate as multilinear
      let q_k = DensePolynomial::new(q_k_uni.coeffs.clone());
      let q_k_eval = q_k.evaluate(z_partial);

      res -= (z_challenge[z_challenge.len() - k - 1] - u_challenge[z_challenge.len() - k - 1])
        * q_k_eval;
    }
    assert!(res.is_zero());
  }

  /// Test for construction of batched lifted degree quotient:
  ///  ̂q = ∑ₖ₌₀ⁿ⁻¹ yᵏ Xᵐ⁻ᵈᵏ⁻¹ ̂qₖ, 𝑑ₖ = deg(̂q), 𝑚 = 𝑁
  #[test]
  fn batched_lifted_degree_quotient() {
    let num_vars = 3;
    let n = 1 << num_vars;

    // Define mock qₖ with deg(qₖ) = 2ᵏ⁻¹
    let q_0 = UniPoly::from_coeff(vec![Fr::one()]);
    let q_1 = UniPoly::from_coeff(vec![Fr::from(2u64), Fr::from(3u64)]);
    let q_2 = UniPoly::from_coeff(vec![
      Fr::from(4u64),
      Fr::from(5u64),
      Fr::from(6u64),
      Fr::from(7u64),
    ]);
    let quotients = vec![q_0, q_1, q_2];

    let mut rng = test_rng();
    let y_challenge = Fr::rand(&mut rng);

    //Compute batched quptient  ̂q
    let batched_quotient =
      compute_batched_lifted_degree_quotient::<Bn254>(n, &quotients, &y_challenge);

    //Explicitly define q_k_lifted = X^{N-2^k} * q_k and compute the expected batched result
    let q_0_lifted = UniPoly::from_coeff(vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::one(),
    ]);
    let q_1_lifted = UniPoly::from_coeff(vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::from(2u64),
      Fr::from(3u64),
    ]);
    let q_2_lifted = UniPoly::from_coeff(vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::from(4u64),
      Fr::from(5u64),
      Fr::from(6u64),
      Fr::from(7u64),
    ]);

    //Explicitly compute  ̂q i.e. RLC of lifted polys
    let mut batched_quotient_expected = UniPoly::from_coeff(vec![Fr::zero(); n]);

    batched_quotient_expected += &q_0_lifted;
    batched_quotient_expected += &(q_1_lifted * y_challenge);
    batched_quotient_expected += &(q_2_lifted * (y_challenge * y_challenge));
    assert_eq!(batched_quotient, batched_quotient_expected);
  }

  /// evaluated quotient \zeta_x
  ///
  /// 𝜁 = 𝑓 − ∑ₖ₌₀ⁿ⁻¹𝑦ᵏ𝑥ʷˢ⁻ʷ⁺¹𝑓ₖ  = 𝑓 − ∑_{d ∈ {d₀, ..., dₙ₋₁}} X^{d* - d + 1}  − ∑{k∶ dₖ=d} yᵏ fₖ , where d* = lifted degree
  ///
  /// 𝜁 =  ̂q - ∑ₖ₌₀ⁿ⁻¹ yᵏ Xᵐ⁻ᵈᵏ⁻¹ ̂qₖ, m = N
  #[test]
  fn partially_evaluated_quotient_zeta() {
    let num_vars = 3;
    let n: u64 = 1 << num_vars;

    let mut rng = test_rng();
    let x_challenge = Fr::rand(&mut rng);
    let y_challenge = Fr::rand(&mut rng);

    let challenges: Vec<_> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
    let z_challenge = Fr::rand(&mut rng);

    let (_, (zeta_x_scalars, _)) =
      eval_and_quotient_scalars::<Bn254>(y_challenge, x_challenge, z_challenge, &challenges);

    // To verify we manually compute zeta using the computed powers and expected
    // 𝜁 =  ̂q - ∑ₖ₌₀ⁿ⁻¹ yᵏ Xᵐ⁻ᵈᵏ⁻¹ ̂qₖ, m = N
    assert_eq!(
      zeta_x_scalars[0],
      -x_challenge.pow(BigInt::<1>::from((n - 1) as u64))
    );

    assert_eq!(
      zeta_x_scalars[1],
      -y_challenge * x_challenge.pow(BigInt::<1>::from((n - 1 - 1) as u64))
    );

    assert_eq!(
      zeta_x_scalars[2],
      -y_challenge * y_challenge * x_challenge.pow(BigInt::<1>::from((n - 3 - 1) as u64))
    );
  }

  /// Test efficiently computing 𝛷ₖ(x) = ∑ᵢ₌₀ᵏ⁻¹xⁱ
  /// 𝛷ₖ(𝑥) = ∑ᵢ₌₀ᵏ⁻¹𝑥ⁱ = (𝑥²^ᵏ − 1) / (𝑥 − 1)
  #[test]
  fn phi_n_x_evaluation() {
    const N: u64 = 8u64;
    let log_N = (N as usize).log_2();

    // 𝛷ₖ(𝑥)
    let mut rng = test_rng();
    let x_challenge = Fr::rand(&mut rng);

    let efficient = (x_challenge.pow(BigInt::<1>::from((1 << log_N) as u64)) - Fr::one())
      / (x_challenge - Fr::one());
    let expected: Fr = phi::<Bn254>(&x_challenge, log_N);
    assert_eq!(efficient, expected);
  }

  /// Test efficiently computing 𝛷ₖ(x) = ∑ᵢ₌₀ᵏ⁻¹xⁱ
  /// 𝛷ₙ₋ₖ₋₁(𝑥²^ᵏ⁺¹) = (𝑥²^ⁿ − 1) / (𝑥²^ᵏ⁺¹ − 1)
  #[test]
  fn phi_n_k_1_x_evaluation() {
    const N: u64 = 8u64;
    let log_N = (N as usize).log_2();

    // 𝛷ₖ(𝑥)
    let mut rng = test_rng();
    let x_challenge = Fr::rand(&mut rng);
    let k = 2;

    //𝑥²^ᵏ⁺¹
    let x_pow = x_challenge.pow(BigInt::<1>::from((1 << (k + 1)) as u64));

    //(𝑥²^ⁿ − 1) / (𝑥²^ᵏ⁺¹ − 1)
    let efficient =
      (x_challenge.pow(BigInt::<1>::from((1 << log_N) as u64)) - Fr::one()) / (x_pow - Fr::one());
    let expected: Fr = phi::<Bn254>(&x_challenge, log_N - k - 1);
    assert_eq!(efficient, expected);
  }

  /// Test construction of 𝑍ₓ
  /// 𝑍ₓ =  ̂𝑓 − 𝑣 ∑ₖ₌₀ⁿ⁻¹(𝑥²^ᵏ𝛷ₙ₋ₖ₋₁(𝑥ᵏ⁺¹)− 𝑢ₖ𝛷ₙ₋ₖ(𝑥²^ᵏ)) ̂qₖ
  #[test]
  fn partially_evaluated_quotient_z_x() {
    let num_vars = 3;

    // Construct a random multilinear polynomial f, and (u,v) such that f(u) = v.
    let mut rng = test_rng();
    let challenges: Vec<_> = (0..num_vars)
      .into_iter()
      .map(|_| Fr::rand(&mut rng))
      .collect();

    let u_rev = {
      let mut res = challenges.clone();
      res.reverse();
      res
    };

    let x_challenge = Fr::rand(&mut rng);
    let y_challenge = Fr::rand(&mut rng);
    let z_challenge = Fr::rand(&mut rng);

    // Construct Z_x scalars
    let (_, (_, z_x_scalars)) =
      eval_and_quotient_scalars::<Bn254>(y_challenge, x_challenge, z_challenge, &challenges);

    for k in 0..num_vars {
      let x_pow_2k = x_challenge.pow(BigInt::<1>::from((1 << k) as u64)); // x^{2^k}
      let x_pow_2kp1 = x_challenge.pow(BigInt::<1>::from((1 << (k + 1)) as u64)); // x^{2^{k+1}}
                                                                                  // x^{2^k} * \Phi_{n-k-1}(x^{2^{k+1}}) - u_k *  \Phi_{n-k}(x^{2^k})
      let mut scalar = x_pow_2k * &phi::<Bn254>(&x_pow_2kp1, num_vars - k - 1)
        - u_rev[k] * &phi::<Bn254>(&x_pow_2k, num_vars - k);
      scalar *= z_challenge;
      scalar *= Fr::from(-1);
      assert_eq!(z_x_scalars[k], scalar);
    }
  }
}
