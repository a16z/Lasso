#![allow(dead_code)]

use ark_ff::PrimeField;

use ark_std::test_rng;
#[cfg(feature = "multicore")]
use rayon::prelude::*;

use crate::poly::dense_mlpoly::DensePolynomial;

#[cfg(test)]
pub mod test;

pub mod errors;
pub mod gaussian_elimination;
pub mod instruction_utils;
pub mod math;
pub mod transcript;
pub mod thread;

/// Converts an integer value to a bitvector (all values {0,1}) of field elements.
/// Note: ordering has the MSB in the highest index. All of the following represent the integer 1:
/// - [1]
/// - [0, 0, 1]
/// - [0, 0, 0, 0, 0, 0, 0, 1]
/// ```ignore
/// use liblasso::utils::index_to_field_bitvector;
/// # use ark_bn254::Fr;
/// # use ark_std::{One, Zero};
/// let zero = Fr::zero();
/// let one = Fr::one();
///
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 1), vec![one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 3), vec![zero, zero, one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 7), vec![zero, zero, zero, zero, zero, zero, one]);
/// ```
pub fn index_to_field_bitvector<F: PrimeField>(value: usize, bits: usize) -> Vec<F> {
    assert!(value < 1 << bits);

    let mut bitvector: Vec<F> = Vec::with_capacity(bits);

    for i in (0..bits).rev() {
        if (value >> i) & 1 == 1 {
            bitvector.push(F::one());
        } else {
            bitvector.push(F::zero());
        }
    }
    bitvector
}

/// Convert Vec<F> which should represent a bitvector to a packed string of bits {0, 1, ?}
pub fn ff_bitvector_dbg<F: PrimeField>(f: &Vec<F>) -> String {
    let mut result = "".to_owned();
    for bit in f {
        if *bit == F::one() {
            result.push('1');
        } else if *bit == F::zero() {
            result.push('0');
        } else {
            result.push('?');
        }
    }
    result
}

#[tracing::instrument(skip_all)]
pub fn compute_dotproduct<F: PrimeField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .map(|(a_i, b_i)| *a_i * b_i)
        .sum()
}

/// Compute dotproduct optimized for values being 0 / 1
#[tracing::instrument(skip_all)]
pub fn compute_dotproduct_low_optimized<F: PrimeField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .map(|(a_i, b_i)| mul_0_1_optimized(a_i, b_i))
        .sum()
}

#[inline(always)]
pub fn mul_0_1_optimized<F: PrimeField>(a: &F, b: &F) -> F {
    if a.is_zero() || b.is_zero() {
        F::zero()
    } else if a.is_one() {
        *b
    } else if b.is_one() {
        *a
    } else {
        // TODO(sragss): Negative 1
        *a * b
    }
}

#[inline(always)]
pub fn mul_0_optimized<F: PrimeField>(likely_zero: &F, x: &F) -> F {
    if likely_zero.is_zero() {
        F::zero()
    } else {
        *likely_zero * x
    }
}

/// Checks if `num` is a power of 2.
pub fn is_power_of_two(num: usize) -> bool {
    num != 0 && (num & (num - 1)) == 0
}

/// Splits `item` into two chunks of `num_bits` size where each is less than 2^num_bits.
/// Ex: split_bits(0b101_000, 3) -> (101, 000)
pub fn split_bits(item: usize, num_bits: usize) -> (usize, usize) {
    let max_value = (1 << num_bits) - 1; // Calculate the maximum value that can be represented with num_bits

    let low_chunk = item & max_value; // Extract the lower bits
    let high_chunk = (item >> num_bits) & max_value; // Shift the item to the right and extract the next set of bits

    (high_chunk, low_chunk)
}

pub fn gen_random_point<F: PrimeField>(memory_bits: usize) -> Vec<F> {
    let mut rng = test_rng();
    let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
    for _ in 0..memory_bits {
        r_i.push(F::rand(&mut rng));
    }
    r_i
}

#[inline]
#[tracing::instrument(skip_all, name = "split_poly_flagged")]
pub fn split_poly_flagged<F: PrimeField>(
    poly: &DensePolynomial<F>,
    flags: &DensePolynomial<F>,
) -> (Vec<F>, Vec<F>) {
    let poly_evals: &[F] = poly.evals_ref();
    let len = poly_evals.len();
    let half = len / 2;
    let mut left: Vec<F> = Vec::with_capacity(half);
    let mut right: Vec<F> = Vec::with_capacity(half);

    for i in 0..len {
        if flags[i].is_zero() {
            if i < half {
                left.push(F::one());
            } else {
                right.push(F::one());
            }
        } else {
            if i < half {
                left.push(poly_evals[i]);
            } else {
                right.push(poly_evals[i]);
            }
        }
    }
    (left, right)
}

pub fn count_poly_zeros<F: PrimeField>(poly: &DensePolynomial<F>) -> usize {
    let mut count = 0;
    for i in 0..poly.len() {
        if poly[i].is_zero() {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split() {
        assert_eq!(split_bits(0b00_01, 2), (0, 1));
        assert_eq!(split_bits(0b10_01, 2), (2, 1));
    }
}
