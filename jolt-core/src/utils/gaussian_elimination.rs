// Inspired by: https://github.com/TheAlgorithms/Rust/blob/master/src/math/gaussian_elimination.rs
// Gaussian Elimination of Quadratic Matrices
// Takes an augmented matrix as input, returns vector of results
// Wikipedia reference: augmented matrix: https://en.wikipedia.org/wiki/Augmented_matrix
// Wikipedia reference: algorithm: https://en.wikipedia.org/wiki/Gaussian_elimination

use ark_ff::PrimeField;

pub fn gaussian_elimination<F: PrimeField>(matrix: &mut [Vec<F>]) -> Vec<F> {
    let size = matrix.len();
    assert_eq!(size, matrix[0].len() - 1);

    for i in 0..size - 1 {
        for j in i..size - 1 {
            echelon(matrix, i, j);
        }
    }

    for i in (1..size).rev() {
        eliminate(matrix, i);
    }

    // Disable cargo clippy warnings about needless range loops.
    // Checking the diagonal like this is simpler than any alternative.
    #[allow(clippy::needless_range_loop)]
    for i in 0..size {
        if matrix[i][i] == F::zero() {
            println!("Infinitely many solutions");
        }
    }

    let mut result: Vec<F> = vec![F::zero(); size];
    for i in 0..size {
        result[i] = matrix[i][size] / matrix[i][i];
    }
    result
}

fn echelon<F: PrimeField>(matrix: &mut [Vec<F>], i: usize, j: usize) {
    let size = matrix.len();
    if matrix[i][i] == F::zero() {
    } else {
        let factor = matrix[j + 1][i] / matrix[i][i];
        (i..size + 1).for_each(|k| {
            let tmp = matrix[i][k];
            matrix[j + 1][k] -= factor * tmp;
        });
    }
}

fn eliminate<F: PrimeField>(matrix: &mut [Vec<F>], i: usize) {
    let size = matrix.len();
    if matrix[i][i] == F::zero() {
    } else {
        for j in (1..i + 1).rev() {
            let factor = matrix[j - 1][i] / matrix[i][i];
            for k in (0..size + 1).rev() {
                let tmp = matrix[i][k];
                matrix[j - 1][k] -= factor * tmp;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use super::gaussian_elimination;
    use ark_std::{One, Zero};

    #[test]
    fn test_gauss() {
        let mut matrix: Vec<Vec<Fr>> = vec![
            vec![Fr::one(), Fr::zero(), Fr::zero(), Fr::from(2u64)],
            vec![Fr::one(), Fr::one(), Fr::one(), Fr::from(17u64)],
            vec![Fr::one(), Fr::from(2u64), Fr::from(4u64), Fr::from(38u64)],
        ];
        let result = vec![Fr::from(2u64), Fr::from(12u64), Fr::from(3u64)];
        assert_eq!(gaussian_elimination(&mut matrix), result);
    }
}
