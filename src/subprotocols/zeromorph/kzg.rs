use ark_bn254::g1;
use ark_ec::scalar_mul::fixed_base::FixedBase;
use std::{borrow::Borrow, marker::PhantomData};

use crate::msm::VariableBaseMSM;
use crate::poly::unipoly::UniPoly;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::PrimeField;
use ark_std::One;
use ark_std::UniformRand;
use rand_chacha::rand_core::RngCore;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KZGError {
  #[error("Length Error: SRS Length: {0}, Key Length: {0}")]
  KeyLengthError(usize, usize),
  #[error("Length Error: Commitment Key Length: {0}, Polynomial Degree {0}")]
  CommitLengthError(usize, usize),
  #[error("Failed to compute quotient polynomial due to polynomial division")]
  PolyDivisionError,
}

#[derive(Debug, Clone, Default)]
pub struct UniversalKzgSrs<P: Pairing> {
  pub g1_powers: Vec<P::G1Affine>,
  pub g2_powers: Vec<P::G2Affine>,
}

#[derive(Clone, Debug)]
pub struct KZGProverKey<P: Pairing> {
  /// generators
  pub g1_powers: Vec<P::G1Affine>,
}

pub struct KZGVerifierKey<P: Pairing> {
  /// The generator of G1.
  pub g1: P::G1Affine,
  /// The generator of G2.
  pub g2: P::G2Affine,
  /// tau times the above generator of G2.
  pub tau_2: P::G2Affine,
}

impl<P: Pairing> UniversalKzgSrs<P> {
  pub fn setup<R: RngCore>(max_degree: usize, rng: &mut R) -> UniversalKzgSrs<P> {
    let tau = P::ScalarField::rand(rng);
    let g1 = P::G1::rand(rng);
    let g2 = P::G2::rand(rng);

    let tau_powers: Vec<_> = (0..=max_degree)
      .scan(tau, |state, _| {
        let val = *state;
        *state *= &tau;
        Some(val)
      })
      .collect();

    let window_size = FixedBase::get_mul_window_size(max_degree);
    let scalar_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;

    //TODO: gate with rayon
    let g1_table = FixedBase::get_window_table(scalar_bits, window_size, g1);
    let g2_table = FixedBase::get_window_table(scalar_bits, window_size, g2);
    let g1_powers_projective = FixedBase::msm(scalar_bits, window_size, &g1_table, &tau_powers);
    let g2_powers_projective = FixedBase::msm(scalar_bits, window_size, &g2_table, &tau_powers);
    let g1_powers = P::G1::normalize_batch(&g1_powers_projective);
    let g2_powers = P::G2::normalize_batch(&g2_powers_projective);

    UniversalKzgSrs {
      g1_powers,
      g2_powers,
    }
  }

  pub fn get_prover_key(&self, key_size: usize) -> Result<Vec<P::G1Affine>, KZGError> {
    if self.g1_powers.len() < key_size {
      return Err(KZGError::KeyLengthError(self.g1_powers.len(), key_size));
    }
    Ok(self.g1_powers[..=key_size].to_vec())
  }

  pub fn get_verifier_key(&self, key_size: usize) -> Result<KZGVerifierKey<P>, KZGError> {
    if self.g1_powers.len() < key_size {
      return Err(KZGError::KeyLengthError(self.g1_powers.len(), key_size));
    }
    Ok(KZGVerifierKey {
      g1: self.g1_powers[0],
      g2: self.g2_powers[0],
      tau_2: self.g2_powers[1],
    })
  }

  pub fn trim(&self, key_size: usize) -> Result<(Vec<P::G1Affine>, KZGVerifierKey<P>), KZGError> {
    if self.g1_powers.len() < key_size {
      return Err(KZGError::KeyLengthError(self.g1_powers.len(), key_size));
    }
    let g1_powers = self.g1_powers[..=key_size].to_vec();

    let pk = g1_powers;
    let vk = KZGVerifierKey {
      g1: self.g1_powers[0],
      g2: self.g2_powers[0],
      tau_2: self.g2_powers[1],
    };
    Ok((pk, vk))
  }
}

pub struct UnivariateKZG<P> {
  phantom: PhantomData<P>,
}

impl<P: Pairing> UnivariateKZG<P> {
  pub fn commit_offset(
    g1_powers: &Vec<P::G1Affine>,
    poly: &UniPoly<P::ScalarField>,
    offset: usize,
  ) -> Result<P::G1Affine, KZGError> {

    if poly.degree() > g1_powers.len() {
      return Err(KZGError::CommitLengthError(poly.degree(), g1_powers.len()));
    }

    let scalars = poly.coeffs.as_slice();
    let bases = g1_powers.as_slice();

    let com =
      <P::G1 as VariableBaseMSM>::msm(&bases[offset..scalars.len()], &poly.coeffs[offset..])
        .unwrap();

    Ok(com.into_affine())
  }

  pub fn commit(
    g1_powers: &Vec<P::G1Affine>,
    poly: &UniPoly<P::ScalarField>,
  ) -> Result<P::G1Affine, KZGError> {

    if poly.degree() > g1_powers.len() {
      return Err(KZGError::CommitLengthError(poly.degree(), g1_powers.len()));
    }
    let com = <P::G1 as VariableBaseMSM>::msm(
      &g1_powers.as_slice()[..poly.coeffs.len()],
      poly.coeffs.as_slice(),
    )
    .unwrap();
    Ok(com.into_affine())
  }

  pub fn open(
    g1_powers: impl Borrow<Vec<P::G1Affine>>,
    polynomial: &UniPoly<P::ScalarField>,
    point: &P::ScalarField,
  ) -> Result<(P::G1Affine, P::ScalarField), KZGError> {
    let g1_powers = g1_powers.borrow();
    let divisor = UniPoly {
      coeffs: vec![-*point, P::ScalarField::one()],
    };
    let witness_polynomial = polynomial
      .divide_with_q_and_r(&divisor)
      .map(|(q, _r)| q)
      .ok_or(KZGError::PolyDivisionError)?;
    let proof = <P::G1 as VariableBaseMSM>::msm(
      &g1_powers.as_slice()[..witness_polynomial.coeffs.len()],
      witness_polynomial.coeffs.as_slice(),
    )
    .unwrap();
    let evaluation = polynomial.evaluate(point);

    Ok((proof.into_affine(), evaluation))
  }

  fn verify(
    vk: impl Borrow<KZGVerifierKey<P>>,
    commitment: &P::G1Affine,
    point: &P::ScalarField,
    proof: &P::G1Affine,
    evaluation: &P::ScalarField,
  ) -> Result<bool, KZGError> {
    let vk = vk.borrow();

    let lhs = P::pairing(
      commitment.into_group() - vk.g1.into_group() * evaluation,
      vk.g2,
    );
    let rhs = P::pairing(proof, vk.tau_2.into_group() - (vk.g2 * point));
    Ok(lhs == rhs)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::poly::unipoly::UniPoly;
  use ark_bn254::{Bn254, Fr};

  use ark_std::{
    rand::{Rng, SeedableRng},
    UniformRand,
  };
  use rand_chacha::{rand_core::RngCore, ChaCha20Rng};

  fn random<P: Pairing, R: RngCore>(degree: usize, mut rng: &mut R) -> UniPoly<P::ScalarField> {
    let coeffs = (0..=degree)
      .map(|_| P::ScalarField::rand(&mut rng))
      .collect::<Vec<_>>();
    UniPoly::from_coeff(coeffs)
  }

  #[test]
  fn commit_prove_verify() -> Result<(), KZGError> {
    let seed = b"11111111111111111111111111111111";
    for _ in 0..100 {
      let mut rng = &mut ChaCha20Rng::from_seed(*seed);
      let degree = rng.gen_range(2..20);

      let pp = UniversalKzgSrs::<Bn254>::setup(degree, &mut rng);
      let (ck, vk) = pp.trim(degree).unwrap();
      let p = random::<Bn254, ChaCha20Rng>(degree, rng);
      let comm = UnivariateKZG::<Bn254>::commit(&ck, &p)?;
      let point = Fr::rand(rng);
      let (proof, value) = UnivariateKZG::<Bn254>::open(&ck, &p, &point)?;
      assert!(
        UnivariateKZG::<Bn254>::verify(&vk, &comm, &point, &proof, &value)?,
        "proof was incorrect for max_degree = {}, polynomial_degree = {}",
        degree,
        p.degree(),
      );
    }
    Ok(())
  }
}
