#![allow(dead_code)]

use std::cmp::Ordering;
use std::ops::{AddAssign, Index, IndexMut, Mul, MulAssign};

use super::commitments::{Commitments, MultiCommitGens};
use crate::utils::gaussian_elimination::gaussian_elimination;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::*;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

// ax^2 + bx + c stored as vec![c,b,a]
// ax^3 + bx^2 + cx + d stored as vec![d,c,b,a]
#[derive(Debug, Clone, PartialEq)]
pub struct UniPoly<F> {
  pub coeffs: Vec<F>,
}

// ax^2 + bx + c stored as vec![c,a]
// ax^3 + bx^2 + cx + d stored as vec![d,b,a]
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct CompressedUniPoly<F: PrimeField> {
  coeffs_except_linear_term: Vec<F>,
}

impl<F: PrimeField> UniPoly<F> {
  #[allow(dead_code)]
  pub fn from_coeff(coeffs: Vec<F>) -> Self {
    UniPoly { coeffs }
  }

  pub fn from_evals(evals: &[F]) -> Self {
    UniPoly {
      coeffs: Self::vandermonde_interpolation(evals),
    }
  }

  fn zero() -> Self {
    Self::from_coeff(Vec::new())
  }

  fn vandermonde_interpolation(evals: &[F]) -> Vec<F> {
    let n = evals.len();
    let xs: Vec<F> = (0..n).map(|x| F::from(x as u64)).collect();

    let mut vandermonde: Vec<Vec<F>> = Vec::with_capacity(n);
    for i in 0..n {
      let mut row = Vec::with_capacity(n);
      let x = xs[i];
      row.push(F::one());
      row.push(x);
      for j in 2..n {
        row.push(row[j - 1] * x);
      }
      row.push(evals[i]);
      vandermonde.push(row);
    }

    gaussian_elimination(&mut vandermonde)
  }

  /// Divide self by another polynomial, and returns the
  /// quotient and remainder.
  pub fn divide_with_q_and_r(&self, divisor: &Self) -> Option<(Self, Self)> {
    if self.is_zero() {
      Some((Self::zero(), Self::zero()))
    } else if divisor.is_zero() {
      None
    } else if self.degree() < divisor.degree() {
      Some((Self::zero(), self.clone()))
    } else {
      // Now we know that self.degree() >= divisor.degree();
      let mut quotient = vec![F::ZERO; self.degree() - divisor.degree() + 1];
      let mut remainder: Self = self.clone();
      // Can unwrap here because we know self is not zero.
      let divisor_leading_inv = divisor.leading_coefficient().unwrap().inverse().unwrap();
      while !remainder.is_zero() && remainder.degree() >= divisor.degree() {
        let cur_q_coeff = *remainder.leading_coefficient().unwrap() * divisor_leading_inv;
        let cur_q_degree = remainder.degree() - divisor.degree();
        quotient[cur_q_degree] = cur_q_coeff;

        for (i, div_coeff) in divisor.coeffs.iter().enumerate() {
          remainder.coeffs[cur_q_degree + i] -= &(cur_q_coeff * div_coeff);
        }
        while let Some(true) = remainder.coeffs.last().map(|c| c == &F::ZERO) {
          remainder.coeffs.pop();
        }
      }
      Some((Self::from_coeff(quotient), remainder))
    }
  }

  pub fn degree(&self) -> usize {
    self.coeffs.len() - 1
  }

  pub fn as_vec(&self) -> Vec<F> {
    self.coeffs.clone()
  }

  pub fn len(&self) -> usize {
    self.coeffs.len()
  }

  pub fn eval_at_zero(&self) -> F {
    self.coeffs[0]
  }

  pub fn eval_at_one(&self) -> F {
    (0..self.coeffs.len()).map(|i| self.coeffs[i]).sum()
  }

  pub fn evaluate(&self, r: &F) -> F {
    let mut eval = self.coeffs[0];
    let mut power = *r;
    for i in 1..self.coeffs.len() {
      eval += power * self.coeffs[i];
      power *= r;
    }
    eval
  }

  pub fn compress(&self) -> CompressedUniPoly<F> {
    let coeffs_except_linear_term = [&self.coeffs[..1], &self.coeffs[2..]].concat();
    assert_eq!(coeffs_except_linear_term.len() + 1, self.coeffs.len());
    CompressedUniPoly {
      coeffs_except_linear_term,
    }
  }

  pub fn commit<G: CurveGroup<ScalarField = F>>(&self, gens: &MultiCommitGens<G>, blind: &F) -> G {
    Commitments::batch_commit(&self.coeffs, blind, gens)
  }

  fn is_zero(&self) -> bool {
    self.coeffs.is_empty() || self.coeffs.iter().all(|c| c == &F::zero())
  }

  fn truncate_leading_zeros(&mut self) {
    while self.coeffs.last().map_or(false, |c| c == &F::zero()) {
      self.coeffs.pop();
    }
  }

  fn leading_coefficient(&self) -> Option<&F> {
    self.coeffs.last()
  }
}

impl<F: PrimeField> AddAssign<&F> for UniPoly<F> {
  fn add_assign(&mut self, rhs: &F) {
    #[cfg(feature = "multicore")]
    let iter = self.coeffs.par_iter_mut();
    #[cfg(not(feature = "multicore"))]
    let iter = self.coeffs.iter_mut();
    iter.for_each(|c| *c += rhs);
  }
}

impl<F: PrimeField> MulAssign<&F> for UniPoly<F> {
  fn mul_assign(&mut self, rhs: &F) {
    #[cfg(feature = "multicore")]
    let iter = self.coeffs.par_iter_mut();
    #[cfg(not(feature = "multicore"))]
    let iter = self.coeffs.iter_mut();
    iter.for_each(|c| *c *= rhs);
  }
}

impl<F: PrimeField> Mul<F> for UniPoly<F> {
  type Output = Self;

  fn mul(self, rhs: F) -> Self {
    #[cfg(feature = "multicore")]
    let iter = self.coeffs.into_par_iter();
    #[cfg(not(feature = "multicore"))]
    let iter = self.coeffs.iter();
    Self::from_coeff(iter.map(|c| c * rhs).collect::<Vec<_>>())
  }
}

impl<F: PrimeField> Mul<&F> for UniPoly<F> {
  type Output = Self;

  fn mul(self, rhs: &F) -> Self {
    #[cfg(feature = "multicore")]
    let iter = self.coeffs.into_par_iter();
    #[cfg(not(feature = "multicore"))]
    let iter = self.coeffs.iter();
    Self::from_coeff(iter.map(|c| c * rhs).collect::<Vec<_>>())
  }
}

impl<F: PrimeField> AddAssign<&Self> for UniPoly<F> {
  fn add_assign(&mut self, rhs: &Self) {
    let ordering = self.coeffs.len().cmp(&rhs.coeffs.len());
    #[allow(clippy::disallowed_methods)]
    for (lhs, rhs) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
      *lhs += rhs;
    }
    if matches!(ordering, Ordering::Less) {
      self
        .coeffs
        .extend(rhs.coeffs[self.coeffs.len()..].iter().cloned());
    }
  }
}

impl<F: PrimeField> AsRef<Vec<F>> for UniPoly<F> {
  fn as_ref(&self) -> &Vec<F> {
    &self.coeffs
  }
}

impl<F: PrimeField> Index<usize> for UniPoly<F> {
  type Output = F;

  #[inline(always)]
  fn index(&self, _index: usize) -> &F {
    &(self.coeffs[_index])
  }
}

impl<F: PrimeField> IndexMut<usize> for UniPoly<F> {
  #[inline(always)]
  fn index_mut(&mut self, index: usize) -> &mut F {
    &mut (self.coeffs[index])
  }
}

impl<F: PrimeField> CompressedUniPoly<F> {
  // we require eval(0) + eval(1) = hint, so we can solve for the linear term as:
  // linear_term = hint - 2 * constant_term - deg2 term - deg3 term
  pub fn decompress(&self, hint: &F) -> UniPoly<F> {
    let mut linear_term =
      *hint - self.coeffs_except_linear_term[0] - self.coeffs_except_linear_term[0];
    for i in 1..self.coeffs_except_linear_term.len() {
      linear_term -= self.coeffs_except_linear_term[i];
    }

    let mut coeffs = vec![self.coeffs_except_linear_term[0], linear_term];
    coeffs.extend(&self.coeffs_except_linear_term[1..]);
    assert_eq!(self.coeffs_except_linear_term.len() + 1, coeffs.len());
    UniPoly { coeffs }
  }
}

impl<G: CurveGroup> AppendToTranscript<G> for UniPoly<G::ScalarField> {
  fn append_to_transcript<T: ProofTranscript<G>>(&self, label: &'static [u8], transcript: &mut T) {
    transcript.append_message(label, b"UniPoly_begin");
    for i in 0..self.coeffs.len() {
      transcript.append_scalar(b"coeff", &self.coeffs[i]);
    }
    transcript.append_message(label, b"UniPoly_end");
  }
}

#[cfg(test)]
mod tests {

  use super::*;
  use ark_curve25519::Fr;

  #[test]
  fn test_from_evals_quad() {
    test_from_evals_quad_helper::<Fr>()
  }

  fn test_from_evals_quad_helper<F: PrimeField>() {
    // polynomial is 2x^2 + 3x + 1
    let e0 = F::one();
    let e1 = F::from(6u64);
    let e2 = F::from(15u64);
    let evals = vec![e0, e1, e2];
    let poly = UniPoly::from_evals(&evals);

    assert_eq!(poly.eval_at_zero(), e0);
    assert_eq!(poly.eval_at_one(), e1);
    assert_eq!(poly.coeffs.len(), 3);
    assert_eq!(poly.coeffs[0], F::one());
    assert_eq!(poly.coeffs[1], F::from(3u64));
    assert_eq!(poly.coeffs[2], F::from(2u64));

    let hint = e0 + e1;
    let compressed_poly = poly.compress();
    let decompressed_poly = compressed_poly.decompress(&hint);
    for i in 0..decompressed_poly.coeffs.len() {
      assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
    }

    let e3 = F::from(28u64);
    assert_eq!(poly.evaluate(&F::from(3u64)), e3);
  }

  #[test]
  fn test_from_evals_cubic() {
    test_from_evals_cubic_helper::<Fr>()
  }
  fn test_from_evals_cubic_helper<F: PrimeField>() {
    // polynomial is x^3 + 2x^2 + 3x + 1
    let e0 = F::one();
    let e1 = F::from(7u64);
    let e2 = F::from(23u64);
    let e3 = F::from(55u64);
    let evals = vec![e0, e1, e2, e3];
    let poly = UniPoly::from_evals(&evals);

    assert_eq!(poly.eval_at_zero(), e0);
    assert_eq!(poly.eval_at_one(), e1);
    assert_eq!(poly.coeffs.len(), 4);
    assert_eq!(poly.coeffs[0], F::one());
    assert_eq!(poly.coeffs[1], F::from(3u64));
    assert_eq!(poly.coeffs[2], F::from(2u64));
    assert_eq!(poly.coeffs[3], F::one());

    let hint = e0 + e1;
    let compressed_poly = poly.compress();
    let decompressed_poly = compressed_poly.decompress(&hint);
    for i in 0..decompressed_poly.coeffs.len() {
      assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
    }

    let e4 = F::from(109u64);
    assert_eq!(poly.evaluate(&F::from(4u64)), e4);
  }
}
