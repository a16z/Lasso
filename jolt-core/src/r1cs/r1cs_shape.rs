use std::cmp::max;

use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

use crate::utils::mul_0_1_optimized;

use super::spartan::{IndexablePoly, SpartanError};

/// A type that holds the shape of the R1CS matrices
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSShape<F: PrimeField> {
    pub(crate) num_cons: usize,
    pub(crate) num_vars: usize,
    pub(crate) num_io: usize,
    pub(crate) A: Vec<(usize, usize, F)>,
    pub(crate) B: Vec<(usize, usize, F)>,
    pub(crate) C: Vec<(usize, usize, F)>,
}

impl<F: PrimeField> R1CSShape<F> {
    /// Create an object of type `R1CSShape` from the explicitly specified R1CS matrices
    #[tracing::instrument(skip_all, name = "R1CSShape::new")]
    pub fn new(
        num_cons: usize,
        num_vars: usize,
        num_io: usize,
        A: &[(usize, usize, F)],
        B: &[(usize, usize, F)],
        C: &[(usize, usize, F)],
    ) -> Result<R1CSShape<F>, SpartanError> {
        let is_valid = |num_cons: usize,
                        num_vars: usize,
                        num_io: usize,
                        M: &[(usize, usize, F)]|
         -> Result<(), SpartanError> {
            let res = (0..M.len())
                .map(|i| {
                    let (row, col, _val) = M[i];
                    if row >= num_cons || col > num_io + num_vars {
                        Err(SpartanError::InvalidIndex)
                    } else {
                        Ok(())
                    }
                })
                .collect::<Result<Vec<()>, SpartanError>>();

            if res.is_err() {
                Err(SpartanError::InvalidIndex)
            } else {
                Ok(())
            }
        };

        let res_A = is_valid(num_cons, num_vars, num_io, A);
        let res_B = is_valid(num_cons, num_vars, num_io, B);
        let res_C = is_valid(num_cons, num_vars, num_io, C);

        if res_A.is_err() || res_B.is_err() || res_C.is_err() {
            return Err(SpartanError::InvalidIndex);
        }

        let shape = R1CSShape {
            num_cons,
            num_vars,
            num_io,
            A: A.to_owned(),
            B: B.to_owned(),
            C: C.to_owned(),
        };

        // pad the shape
        Ok(shape.pad())
    }

    // Checks regularity conditions on the R1CSShape, required in Spartan-class SNARKs
    // Panics if num_cons, num_vars, or num_io are not powers of two, or if num_io > num_vars
    #[inline]
    pub(crate) fn check_regular_shape(&self) {
        assert_eq!(self.num_cons.next_power_of_two(), self.num_cons);
        assert_eq!(self.num_vars.next_power_of_two(), self.num_vars);
        assert!(self.num_io < self.num_vars);
    }

    #[tracing::instrument(skip_all, name = "R1CSShape::multiply_vec")]
    pub fn multiply_vec(&self, z: &[F]) -> Result<(Vec<F>, Vec<F>, Vec<F>), SpartanError> {
        if z.len() != self.num_io + self.num_vars + 1 {
            return Err(SpartanError::InvalidWitnessLength);
        }

        // computes a product between a sparse matrix `M` and a vector `z`
        // This does not perform any validation of entries in M (e.g., if entries in `M` reference indexes outside the range of `z`)
        // This is safe since we know that `M` is valid
        let sparse_matrix_vec_product =
            |M: &Vec<(usize, usize, F)>, num_rows: usize, z: &[F]| -> Vec<F> {
                // Parallelism strategy below splits the (row, column, value) tuples into num_threads different chunks.
                // It is assumed that the tuples are (row, column) ordered. We exploit this fact to create a mutex over
                // each of the chunks and assume that only one of the threads will be writing to each chunk at a time
                // due to ordering.

                let num_threads = rayon::current_num_threads() * 4; // Enable work stealing incase of thread work imbalance
                let thread_chunk_size = M.len() / num_threads;
                let row_chunk_size = (num_rows as f64 / num_threads as f64).ceil() as usize;

                let mut chunks: Vec<std::sync::Mutex<Vec<F>>> = Vec::with_capacity(num_threads);
                let mut remaining_rows = num_rows;
                (0..num_threads).for_each(|i| {
                    if i == num_threads - 1 {
                        // the final chunk may be smaller
                        let inner = std::sync::Mutex::new(vec![F::zero(); remaining_rows]);
                        chunks.push(inner);
                    } else {
                        let inner = std::sync::Mutex::new(vec![F::ZERO; row_chunk_size]);
                        chunks.push(inner);
                        remaining_rows -= row_chunk_size;
                    }
                });

                let get_chunk = |row_index: usize| -> usize { row_index / row_chunk_size };
                let get_index = |row_index: usize| -> usize { row_index % row_chunk_size };

                let span = tracing::span!(tracing::Level::TRACE, "all_chunks_multiplication");
                let _enter = span.enter();
                M.par_chunks(thread_chunk_size)
                    .for_each(|sub_matrix: &[(usize, usize, F)]| {
                        let (init_row, init_col, init_val) = sub_matrix[0];
                        let mut prev_chunk_index = get_chunk(init_row);
                        let curr_row_index = get_index(init_row);
                        let mut curr_chunk = chunks[prev_chunk_index].lock().unwrap();

                        curr_chunk[curr_row_index] += init_val * z[init_col];

                        let span_a = tracing::span!(tracing::Level::TRACE, "chunk_multiplication");
                        let _enter_b = span_a.enter();
                        for (row, col, val) in sub_matrix.iter().skip(1) {
                            let curr_chunk_index = get_chunk(*row);
                            if prev_chunk_index != curr_chunk_index {
                                // only unlock the mutex again if required
                                drop(curr_chunk); // drop the curr_chunk before waiting for the next to avoid race condition
                                let new_chunk = chunks[curr_chunk_index].lock().unwrap();
                                curr_chunk = new_chunk;

                                prev_chunk_index = curr_chunk_index;
                            }

                            if z[*col].is_zero() {
                                continue;
                            }

                            let m = if z[*col].eq(&F::one()) {
                                *val
                            } else if val.eq(&F::one()) {
                                z[*col]
                            } else {
                                *val * z[*col]
                            };
                            curr_chunk[get_index(*row)] += m;
                        }
                    });
                drop(_enter);
                drop(span);

                let span_a = tracing::span!(tracing::Level::TRACE, "chunks_mutex_unwrap");
                let _enter_a = span_a.enter();
                // TODO(sragss): Mutex unwrap takes about 30% of the time due to clone, likely unnecessary.
                let mut flat_chunks: Vec<F> = Vec::with_capacity(num_rows);
                for chunk in chunks {
                    let inner_vec = chunk.into_inner().unwrap();
                    flat_chunks.extend(inner_vec.iter());
                }
                drop(_enter_a);
                drop(span_a);

                flat_chunks
            };

        let (Az, (Bz, Cz)) = rayon::join(
            || sparse_matrix_vec_product(&self.A, self.num_cons, z),
            || {
                rayon::join(
                    || sparse_matrix_vec_product(&self.B, self.num_cons, z),
                    || sparse_matrix_vec_product(&self.C, self.num_cons, z),
                )
            },
        );

        Ok((Az, Bz, Cz))
    }

    #[tracing::instrument(skip_all, name = "R1CSShape::multiply_vec_uniform")]
    pub fn multiply_vec_uniform<P: IndexablePoly<F>>(
        &self,
        full_witness_vector: &P,
        io: &[F],
        num_steps: usize,
    ) -> Result<(Vec<F>, Vec<F>, Vec<F>), SpartanError> {
        if full_witness_vector.len() + io.len() != (self.num_io + self.num_vars) * num_steps {
            return Err(SpartanError::InvalidWitnessLength);
        }

        // Simulates the `z` vector containing the full satisfying assignment
        //     [W, 1, X]
        // without actually concatenating W and X, which would be expensive.
        let virtual_z_vector = |index: usize| {
            if index < full_witness_vector.len() {
                full_witness_vector[index]
            } else if index == full_witness_vector.len() {
                F::one()
            } else {
                io[index - full_witness_vector.len() - 1]
            }
        };

        // Pre-processes matrix to return the indices of the start of each row
        let get_row_pointers = |M: &Vec<(usize, usize, F)>| -> Vec<usize> {
            let mut indptr = vec![0; self.num_cons + 1];
            for &(row, _, _) in M {
                indptr[row + 1] += 1;
            }
            for i in 0..self.num_cons {
                indptr[i + 1] += indptr[i];
            }
            indptr
        };

        // Multiplies one constraint (row from small M) and its uniform copies with the vector z into result
        let multiply_row_vec_uniform =
            |R: &[(usize, usize, F)], result: &mut [F], num_steps: usize| {
                for &(_, col, val) in R {
                    if col == self.num_vars {
                        result.par_iter_mut().for_each(|x| *x += val);
                    } else {
                        result.par_iter_mut().enumerate().for_each(|(i, x)| {
                            let z_index = col * num_steps + i;
                            *x += mul_0_1_optimized(&val, &virtual_z_vector(z_index));
                        });
                    }
                }
            };

        // computes a product between a sparse uniform matrix represented by `M` and a vector `z`
        let sparse_matrix_vec_product_uniform =
            |M: &Vec<(usize, usize, F)>, num_rows: usize| -> Vec<F> {
                let row_pointers = get_row_pointers(M);

                let mut result: Vec<F> = vec![F::zero(); num_steps * num_rows];

                let span = tracing::span!(
                    tracing::Level::TRACE,
                    "sparse_matrix_vec_product_uniform::multiply_row_vecs"
                );
                let _enter = span.enter();
                result
                    .par_chunks_mut(num_steps)
                    .enumerate()
                    .for_each(|(row_index, row_output)| {
                        let row = &M[row_pointers[row_index]..row_pointers[row_index + 1]];
                        multiply_row_vec_uniform(row, row_output, num_steps);
                    });

                result
            };

        let (mut Az, (mut Bz, mut Cz)) = rayon::join(
            || sparse_matrix_vec_product_uniform(&self.A, self.num_cons),
            || {
                rayon::join(
                    || sparse_matrix_vec_product_uniform(&self.B, self.num_cons),
                    || sparse_matrix_vec_product_uniform(&self.C, self.num_cons),
                )
            },
        );

        // pad each Az, Bz, Cz to the next power of 2
        let m = max(Az.len(), max(Bz.len(), Cz.len())).next_power_of_two();
        rayon::join(
            || Az.resize(m, F::zero()),
            || rayon::join(|| Bz.resize(m, F::zero()), || Cz.resize(m, F::zero())),
        );

        Ok((Az, Bz, Cz))
    }

    /// Pads the R1CSShape so that the number of variables is a power of two
    /// Renumbers variables to accomodate padded variables
    pub fn pad(&self) -> Self {
        // equalize the number of variables and constraints
        let m = max(self.num_vars, self.num_cons).next_power_of_two();

        // check if the provided R1CSShape is already as required
        if self.num_vars == m && self.num_cons == m {
            return self.clone();
        }

        // check if the number of variables are as expected, then
        // we simply set the number of constraints to the next power of two
        if self.num_vars == m {
            return R1CSShape {
                num_cons: m,
                num_vars: m,
                num_io: self.num_io,
                A: self.A.clone(),
                B: self.B.clone(),
                C: self.C.clone(),
            };
        }

        // otherwise, we need to pad the number of variables and renumber variable accesses
        let num_vars_padded = m;
        let num_cons_padded = m;
        let apply_pad = |M: &[(usize, usize, F)]| -> Vec<(usize, usize, F)> {
            M.par_iter()
                .map(|(r, c, v)| {
                    (
                        *r,
                        if c >= &self.num_vars {
                            c + num_vars_padded - self.num_vars
                        } else {
                            *c
                        },
                        *v,
                    )
                })
                .collect::<Vec<_>>()
        };

        let A_padded = apply_pad(&self.A);
        let B_padded = apply_pad(&self.B);
        let C_padded = apply_pad(&self.C);

        R1CSShape {
            num_cons: num_cons_padded,
            num_vars: num_vars_padded,
            num_io: self.num_io,
            A: A_padded,
            B: B_padded,
            C: C_padded,
        }
    }

    /// Same as above but can have different length of constraints and num_variables
    pub fn pad_vars(&self) -> Self {
        // equalize the number of variables and constraints
        // let m = max(self.num_vars, self.num_cons).next_power_of_two();
        let m_vars = self.num_vars.next_power_of_two();
        //let m_cons = self.num_cons.next_power_of_two();
        let m_cons = self.num_cons;

        // check if the provided R1CSShape is already as required
        if self.num_vars == m_vars && self.num_cons == m_cons {
            return self.clone();
        }

        // check if the number of variables are as expected, then
        // we simply set the number of constraints to the next power of two
        if self.num_vars == m_vars {
            return R1CSShape {
                num_cons: m_cons,
                num_vars: m_vars,
                num_io: self.num_io,
                A: self.A.clone(),
                B: self.B.clone(),
                C: self.C.clone(),
            };
        }

        // otherwise, we need to pad the number of variables and renumber variable accesses
        let num_vars_padded = m_vars;
        let num_cons_padded = m_cons;
        let apply_pad = |M: &[(usize, usize, F)]| -> Vec<(usize, usize, F)> {
            M.par_iter()
                .map(|(r, c, v)| {
                    (
                        *r,
                        if c >= &self.num_vars {
                            c + num_vars_padded - self.num_vars
                        } else {
                            *c
                        },
                        *v,
                    )
                })
                .collect::<Vec<_>>()
        };

        let A_padded = apply_pad(&self.A);
        let B_padded = apply_pad(&self.B);
        let C_padded = apply_pad(&self.C);

        R1CSShape {
            num_cons: num_cons_padded,
            num_vars: num_vars_padded,
            num_io: self.num_io,
            A: A_padded,
            B: B_padded,
            C: C_padded,
        }
    }

    // TODO(sragss / arasuarun): Fix and use for single step unit testing.
    // Checks if the R1CS instance is satisfiable given a witness and its shape
    // pub fn is_sat<G: CurveGroup<ScalarField = F>>(
    //   &self,
    //   ck: &CommitmentKey<G>,
    //   U: &R1CSInstance<G>,
    //   W: &R1CSWitness<G>,
    // ) -> Result<(), SpartanError> {
    //   assert_eq!(W.W.len(), self.num_vars);
    //   assert_eq!(U.X.len(), self.num_io);

    //   // verify if Az * Bz = u*Cz
    //   let res_eq: bool = {
    //     let z = concat(vec![W.W.clone(), vec![F::one()], U.X.clone()]);
    //     let (Az, Bz, Cz) = self.multiply_vec(&z)?;
    //     assert_eq!(Az.len(), self.num_cons);
    //     assert_eq!(Bz.len(), self.num_cons);
    //     assert_eq!(Cz.len(), self.num_cons);

    //     let res: usize = (0..self.num_cons)
    //       .map(|i| usize::from(Az[i] * Bz[i] != Cz[i]))
    //       .sum();

    //     res == 0
    //   };

    //   // verify if comm_W is a commitment to W
    //   let res_comm: bool = U.comm_W == CE::<G>::commit(ck, &W.W);

    //   if res_eq && res_comm {
    //     Ok(())
    //   } else {
    //     Err(SpartanError::UnSat)
    //   }
    // }
}
