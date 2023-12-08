#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::poly::commitments::MultiCommitGens;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::dot_product::DotProductProof;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::*;
use ark_std::One;
use merlin::Transcript;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[cfg(not(feature = "ark-msm"))]
use crate::msm::VariableBaseMSM;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum CubicSumcheckType {
    // eq * A * B
    Prod,

    // eq * A * B, optimized for high probability (A, B) = 1
    ProdOnes,

    // eq *(A * flags + (1 - flags))
    Flags,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CubicSumcheckParams<F: PrimeField> {
    poly_As: Vec<DensePolynomial<F>>,
    poly_Bs: Vec<DensePolynomial<F>>,

    // TODO(JOLT-41): Consider swapping to iterator references for `poly_As` / `poly_Bs`
    a_to_b: Vec<usize>,

    poly_eq: DensePolynomial<F>,

    pub num_rounds: usize,

    pub sumcheck_type: CubicSumcheckType,
}

impl<F: PrimeField> CubicSumcheckParams<F> {
    pub fn new_prod(
        poly_lefts: Vec<DensePolynomial<F>>,
        poly_rights: Vec<DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_lefts.len(), poly_rights.len());
        debug_assert_eq!(poly_lefts[0].len(), poly_rights[0].len());
        debug_assert_eq!(poly_lefts[0].len(), poly_eq.len());

        let a_to_b = (0..poly_lefts.len()).map(|i| i).collect();

        CubicSumcheckParams {
            poly_As: poly_lefts,
            poly_Bs: poly_rights,
            a_to_b,
            poly_eq,
            num_rounds,
            sumcheck_type: CubicSumcheckType::Prod,
        }
    }

    pub fn new_prod_ones(
        poly_lefts: Vec<DensePolynomial<F>>,
        poly_rights: Vec<DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_lefts.len(), poly_rights.len());
        debug_assert_eq!(poly_lefts[0].len(), poly_rights[0].len());
        debug_assert_eq!(poly_lefts[0].len(), poly_eq.len());

        let a_to_b = (0..poly_lefts.len()).map(|i| i).collect();

        CubicSumcheckParams {
            poly_As: poly_lefts,
            poly_Bs: poly_rights,
            a_to_b,
            poly_eq,
            num_rounds,
            sumcheck_type: CubicSumcheckType::ProdOnes,
        }
    }
    /// flag_map: poly_leaves length vector mapping between poly_leaves indices and flag indices.
    pub fn new_flags(
        poly_leaves: Vec<DensePolynomial<F>>,
        poly_flags: Vec<DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        flag_map: Vec<usize>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_leaves.len(), flag_map.len());
        debug_assert_eq!(poly_leaves[0].len(), poly_flags[0].len());
        debug_assert_eq!(poly_leaves[0].len(), poly_eq.len());

        CubicSumcheckParams {
            poly_As: poly_leaves,
            poly_Bs: poly_flags,
            a_to_b: flag_map,
            poly_eq,
            num_rounds,
            sumcheck_type: CubicSumcheckType::Flags,
        }
    }

    #[inline]
    pub fn combine(&self, a: &F, b: &F, c: &F) -> F {
        match self.sumcheck_type {
            CubicSumcheckType::Prod => Self::combine_prod(a, b, c),
            CubicSumcheckType::ProdOnes => Self::combine_prod(a, b, c),
            CubicSumcheckType::Flags => Self::combine_flags(a, b, c),
        }
    }

    #[inline]
    pub fn combine_prod(l: &F, r: &F, eq: &F) -> F {
        if *l == F::one() && *r == F::one() {
            *eq
        } else if *l == F::one() {
            *r * eq
        } else if *r == F::one() {
            *l * eq
        } else {
            *l * r * eq
        }
    }

    #[inline]
    pub fn combine_flags(h: &F, flag: &F, eq: &F) -> F {
        if *flag == F::zero() {
            *eq
        } else if *flag == F::one() {
            *eq * *h
        } else {
            *eq * (*flag * h + (F::one() + flag.neg()))
        }
    }

    pub fn pairs_iter(
        &self,
    ) -> impl Iterator<
        Item = (
            &DensePolynomial<F>,
            &DensePolynomial<F>,
            &DensePolynomial<F>,
        ),
    > {
        self.poly_As.iter().enumerate().map(move |(i, a)| {
            let b_idx = match self.sumcheck_type {
                CubicSumcheckType::Prod => i,
                CubicSumcheckType::ProdOnes => i,
                CubicSumcheckType::Flags => self.a_to_b[i],
            };

            let b = &self.poly_Bs[b_idx];
            let c = &self.poly_eq;
            (a, b, c)
        })
    }

    pub fn pairs_par_iter(
        &self,
    ) -> impl ParallelIterator<
        Item = (
            &DensePolynomial<F>,
            &DensePolynomial<F>,
            &DensePolynomial<F>,
        ),
    > {
        self.poly_As.par_iter().enumerate().map(move |(i, a)| {
            let b_idx = match self.sumcheck_type {
                CubicSumcheckType::Prod => i,
                CubicSumcheckType::ProdOnes => i,
                CubicSumcheckType::Flags => self.a_to_b[i],
            };

            let b = &self.poly_Bs[b_idx];
            let c = &self.poly_eq;
            (a, b, c)
        })
    }

    pub fn apply_bound_poly_var_top(&mut self, r_j: &F) {
        let mut all_polys_iter: Vec<&mut DensePolynomial<F>> = self.poly_As.iter_mut()
        .chain(self.poly_Bs.iter_mut())
        .chain(std::iter::once(&mut self.poly_eq))
        .collect();

        all_polys_iter.par_iter_mut().for_each(|poly| poly.bound_poly_var_top(&r_j));
    }

    pub fn get_final_evals(&self) -> (Vec<F>, Vec<F>, F) {
        debug_assert_eq!(self.poly_As[0].len(), 1);
        debug_assert_eq!(self.poly_Bs[0].len(), 1);
        debug_assert_eq!(self.poly_eq.len(), 1);

        let poly_A_final: Vec<F> = (0..self.poly_As.len())
            .map(|i| self.poly_As[i][0])
            .collect();

        let poly_B_final: Vec<F> = (0..self.poly_As.len())
            .map(|i| self.poly_Bs[self.a_to_b[i]][0])
            .collect();

        let poly_eq_final = self.poly_eq[0];

        (poly_A_final, poly_B_final, poly_eq_final)
    }
}

impl<F: PrimeField> SumcheckInstanceProof<F> {
    /// Create a sumcheck proof for polynomial(s) of arbitrary degree.
    ///
    /// Params
    /// - `claim`: Claimed sumcheck evaluation (note: currently unused)
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `polys`: Dense polynomials to combine and sumcheck
    /// - `comb_func`: Function used to combine each polynomial evaluation
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (SumcheckInstanceProof, r_eval_point, final_evals)
    /// - `r_eval_point`: Final random point of evaluation
    /// - `final_evals`: Each of the polys evaluated at `r_eval_point`
    #[tracing::instrument(skip_all, name = "Sumcheck.prove")]
    pub fn prove_arbitrary<Func, G, T: ProofTranscript<G>>(
        _claim: &F,
        num_rounds: usize,
        polys: &mut Vec<DensePolynomial<F>>,
        comb_func: Func,
        combined_degree: usize,
        transcript: &mut T,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        Func: Fn(&[F]) -> F + std::marker::Sync,
        G: CurveGroup<ScalarField = F>,
    {
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _round in 0..num_rounds {
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            let mut eval_points = vec![F::zero(); combined_degree + 1];

            let mle_half = polys[0].len() / 2;

            #[cfg(feature = "multicore")]
            let iterator = (0..mle_half).into_par_iter();

            #[cfg(not(feature = "multicore"))]
            let iterator = 0..mle_half;

            let accum: Vec<Vec<F>> = iterator
                .map(|poly_term_i| {
                    let mut accum = vec![F::zero(); combined_degree + 1];
                    // Evaluate P({0, ..., |g(r)|})

                    // TODO(#28): Optimize
                    // Tricks can be used here for low order bits {0,1} but general premise is a running sum for each
                    // of the m terms in the Dense multilinear polynomials. Formula is:
                    // half = | D_{n-1} | / 2
                    // D_n(index, r) = D_{n-1}[half + index] + r * (D_{n-1}[half + index] - D_{n-1}[index])

                    // eval 0: bound_func is A(low)
                    let params_zero: Vec<F> = polys.iter().map(|poly| poly[poly_term_i]).collect();
                    accum[0] += comb_func(&params_zero);

                    // TODO(#28): Can be computed from prev_round_claim - eval_point_0
                    let params_one: Vec<F> = polys
                        .iter()
                        .map(|poly| poly[mle_half + poly_term_i])
                        .collect();
                    accum[1] += comb_func(&params_one);

                    // D_n(index, r) = D_{n-1}[half + index] + r * (D_{n-1}[half + index] - D_{n-1}[index])
                    // D_n(index, 0) = D_{n-1}[LOW]
                    // D_n(index, 1) = D_{n-1}[HIGH]
                    // D_n(index, 2) = D_{n-1}[HIGH] + (D_{n-1}[HIGH] - D_{n-1}[LOW])
                    // D_n(index, 3) = D_{n-1}[HIGH] + (D_{n-1}[HIGH] - D_{n-1}[LOW]) + (D_{n-1}[HIGH] - D_{n-1}[LOW])
                    // ...
                    let mut existing_term = params_one;
                    for eval_i in 2..(combined_degree + 1) {
                        let mut poly_evals = vec![F::zero(); polys.len()];
                        for poly_i in 0..polys.len() {
                            let poly = &polys[poly_i];
                            poly_evals[poly_i] = existing_term[poly_i]
                                + poly[mle_half + poly_term_i]
                                - poly[poly_term_i];
                        }

                        accum[eval_i] += comb_func(&poly_evals);
                        existing_term = poly_evals;
                    }
                    accum
                })
                .collect();

            #[cfg(feature = "multicore")]
            eval_points
                .par_iter_mut()
                .enumerate()
                .for_each(|(poly_i, eval_point)| {
                    *eval_point = accum
                        .par_iter()
                        .take(mle_half)
                        .map(|mle| mle[poly_i])
                        .sum::<F>();
                });

            #[cfg(not(feature = "multicore"))]
            for (poly_i, eval_point) in eval_points.iter_mut().enumerate() {
                for mle in accum.iter().take(mle_half) {
                    *eval_point += mle[poly_i];
                }
            }

            let round_uni_poly = UniPoly::from_evals(&eval_points);

            // append the prover's message to the transcript
            <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(
                &round_uni_poly,
                b"poly",
                transcript,
            );
            let r_j = transcript.challenge_scalar(b"challenge_nextround");
            r.push(r_j);

            // bound all tables to the verifier's challenege
            for poly in polys.iter_mut() {
                poly.bound_poly_var_top(&r_j);
            }
            compressed_polys.push(round_uni_poly.compress());
        }

        let final_evals = polys.iter().map(|poly| poly[0]).collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }

    #[tracing::instrument(skip_all, name = "Sumcheck.prove_batched")]
    pub fn prove_cubic_batched<G>(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut Transcript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F))
    where
        G: CurveGroup<ScalarField = F>,
    {
        match params.sumcheck_type {
            CubicSumcheckType::Prod => Self::prove_cubic_batched_prod::<G>(claim, params, coeffs, transcript),
            CubicSumcheckType::ProdOnes => Self::prove_cubic_batched_prod_ones::<G>(claim, params, coeffs, transcript),
            CubicSumcheckType::Flags => Self::prove_cubic_batched_flags::<G>(claim, params, coeffs, transcript)
        }
    }

    #[tracing::instrument(skip_all, name = "Sumcheck.prove_cubic_batched_prod")]
    pub fn prove_cubic_batched_prod<G>(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut Transcript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F))
    where
        G: CurveGroup<ScalarField = F>,
    {
        let mut params = params;

        let mut e = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _j in 0..params.num_rounds {
            let len = params.poly_As[0].len() / 2;
            let eq = &params.poly_eq;
            let eq_evals: Vec<(F, F, F)> = (0..len)
                .into_par_iter()
                .map(|i| {
                    let low = i;
                    let high = len + i;

                    let eval_point_0 = eq[low];
                    let m_eq = eq[high] - eq[low];
                    let eval_point_2 = eq[high] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();

            // TODO(sragss): OPTIMIZATION IDEAS
            // - Optimize for 1s! 
            // - Compute 'r' bindings from 'm_a' / 'm_b

            let is_one = |f: &F| -> bool {
                f.is_one()
            };

            let _span = tracing::span!(tracing::Level::TRACE, "eval_loop");
            let _enter = _span.enter();
            let evals: Vec<(F, F, F)> = (0..params.poly_As.len()).into_par_iter()
                .map(|batch_index| {
                    let poly_A = &params.poly_As[batch_index];
                    let poly_B = &params.poly_Bs[batch_index];
                    let len = poly_A.len() / 2;

                    // In the case of a flagged tree, the majority of the leaves will be 1s, optimize for this case.
                    let (eval_point_0, eval_point_2, eval_point_3) = (0..len).into_iter()
                        .map(|mle_index| {
                            let low = mle_index;
                            let high = len + mle_index;

                            let eval_point_0: F = eq_evals[low].0 * poly_A[low] * poly_B[low];

                            let m_a = poly_A[high] - poly_A[low];
                            let m_b = poly_B[high] - poly_B[low];

                            let point_2_A = poly_A[high] + m_a;
                            let point_3_A = point_2_A + m_a;

                            let point_2_B = poly_B[high] + m_b;
                            let point_3_B = point_2_B + m_b;

                            let eval_point_2 = eq_evals[low].1 * point_2_A * point_2_B;
                            let eval_point_3 = eq_evals[low].2 * point_3_A * point_3_B;

                            (eval_point_0, eval_point_2, eval_point_3)
                        })
                        // For parallel
                        // .reduce(
                        //     || (F::zero(), F::zero(), F::zero()),
                        //     |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                        // );
                        // For normal
                        .fold(
                            (F::zero(), F::zero(), F::zero()),
                            |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                        );

                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();
            drop(_enter);
            drop(_span);

            let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
            let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
            let evals_combined_3 = (0..evals.len()).map(|i| evals[i].2 * coeffs[i]).sum();

            let evals = vec![
                evals_combined_0,
                e - evals_combined_0,
                evals_combined_2,
                evals_combined_3,
            ];
            let poly = UniPoly::from_evals(&evals);

            // append the prover's message to the transcript
            <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(&poly, b"poly", transcript);

            //derive the verifier's challenge for the next round
            let r_j = <Transcript as ProofTranscript<G>>::challenge_scalar(
                transcript,
                b"challenge_nextround",
            );
            r.push(r_j);

            // bound all tables to the verifier's challenege
            let _span = tracing::span!(tracing::Level::TRACE, "binding");
            let _enter = _span.enter();

            // params.apply_bound_poly_var_top(&r_j);
            let mut poly_iter: Vec<&mut DensePolynomial<F>> = params.poly_As.iter_mut()
                .chain(params.poly_Bs.iter_mut())
                .collect();

            poly_iter.par_iter_mut().for_each(|poly| poly.bound_poly_var_top(&r_j));
            params.poly_eq.bound_poly_var_top(&r_j);

            drop(_enter);
            drop(_span);

            e = poly.evaluate(&r_j);
            cubic_polys.push(poly.compress());
        }

        let claims_prod = params.get_final_evals();

        (SumcheckInstanceProof::new(cubic_polys), r, claims_prod)
    }

    #[tracing::instrument(skip_all, name = "Sumcheck.prove_cubic_batched_prod_ones")]
    pub fn prove_cubic_batched_prod_ones<G>(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut Transcript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F))
    where
        G: CurveGroup<ScalarField = F>,
    {
        let mut params = params;

        let mut e = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _j in 0..params.num_rounds {
            let len = params.poly_As[0].len() / 2;
            let eq = &params.poly_eq;
            let eq_evals: Vec<(F, F, F)> = (0..len)
                .into_par_iter()
                .map(|i| {
                    let low = i;
                    let high = len + i;

                    let eval_point_0 = eq[low];
                    let m_eq = eq[high] - eq[low];
                    let eval_point_2 = eq[high] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();

            // TODO(sragss): OPTIMIZATION IDEAS
            // - Compute 'r' bindings from 'm_a' / 'm_b

            let _span = tracing::span!(tracing::Level::TRACE, "eval_loop");
            let _enter = _span.enter();
            let evals: Vec<(F, F, F)> = (0..params.poly_As.len())
                .into_par_iter()
                .with_max_len(4)
                .map(|batch_index| {
                    let poly_A = &params.poly_As[batch_index];
                    let poly_B = &params.poly_Bs[batch_index];
                    let len = poly_A.len() / 2;

                    // In the case of a flagged tree, the majority of the leaves will be 1s, optimize for this case.
                    let (eval_point_0, eval_point_2, eval_point_3) = (0..len)
                        .map(|mle_index| {
                            let low = mle_index;
                            let high = len + mle_index;

                            // Optimized version of the product for the high probability that A[low], A[high], B[low], B[high] == 1

                            let a_low_one = poly_A[low].is_one();
                            let a_high_one = poly_A[high].is_one();
                            let b_low_one = poly_B[low].is_one();
                            let b_high_one = poly_B[high].is_one();

                            let eval_point_0: F = if a_low_one && b_low_one {
                                eq_evals[low].0
                            } 
                            else if a_low_one {
                                poly_B[low] * eq_evals[low].0
                            } 
                            else if b_low_one {
                                poly_A[low] * eq_evals[low].0
                            } 
                            else {
                                poly_A[low] * poly_B[low] * eq_evals[low].0
                            };

                            let m_a_zero = a_low_one && a_high_one;
                            let m_b_zero = b_low_one && b_high_one;

                            let (eval_point_2, eval_point_3) = if m_a_zero && m_b_zero {
                                (eq_evals[low].1, eq_evals[low].2)
                            } 
                            else if m_a_zero {
                                let m_b = poly_B[high] - poly_B[low];
                                let point_2_B = poly_B[high] + m_b;
                                let point_3_B = point_2_B + m_b;

                                let eval_point_2 = eq_evals[low].1 * point_2_B;
                                let eval_point_3 = eq_evals[low].2 * point_3_B;
                                (eval_point_2, eval_point_3)
                            } else if m_b_zero {
                                let m_a = poly_A[high] - poly_A[low];
                                let point_2_A = poly_A[high] + m_a;
                                let point_3_A = point_2_A + m_a;

                                let eval_point_2 = eq_evals[low].1 * point_2_A;
                                let eval_point_3 = eq_evals[low].2 * point_3_A;
                                (eval_point_2, eval_point_3)
                            } 
                            else {
                                let m_a = poly_A[high] - poly_A[low];
                                let m_b = poly_B[high] - poly_B[low];

                                let point_2_A = poly_A[high] + m_a;
                                let point_3_A = point_2_A + m_a;

                                let point_2_B = poly_B[high] + m_b;
                                let point_3_B = point_2_B + m_b;

                                let eval_point_2 = eq_evals[low].1 * point_2_A * point_2_B;
                                let eval_point_3 = eq_evals[low].2 * point_3_A * point_3_B;
                                (eval_point_2, eval_point_3)
                            };

                            (eval_point_0, eval_point_2, eval_point_3)
                        })
                        // For parallel
                        // .reduce(
                        //     || (F::zero(), F::zero(), F::zero()),
                        //     |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                        // );
                        // For normal
                        .fold(
                            (F::zero(), F::zero(), F::zero()),
                            |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                        );

                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();
            drop(_enter);
            drop(_span);

            let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
            let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
            let evals_combined_3 = (0..evals.len()).map(|i| evals[i].2 * coeffs[i]).sum();

            let evals = vec![
                evals_combined_0,
                e - evals_combined_0,
                evals_combined_2,
                evals_combined_3,
            ];
            let poly = UniPoly::from_evals(&evals);

            // append the prover's message to the transcript
            <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(&poly, b"poly", transcript);

            //derive the verifier's challenge for the next round
            let r_j = <Transcript as ProofTranscript<G>>::challenge_scalar(
                transcript,
                b"challenge_nextround",
            );
            r.push(r_j);

            // bound all tables to the verifier's challenege
            let _span = tracing::span!(tracing::Level::TRACE, "binding");
            let _enter = _span.enter();

            // params.apply_bound_poly_var_top(&r_j);
            let mut poly_iter: Vec<&mut DensePolynomial<F>> = params.poly_As.iter_mut()
                .chain(params.poly_Bs.iter_mut())
                .collect();

            poly_iter.par_iter_mut().for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j));
            params.poly_eq.bound_poly_var_top(&r_j);

            drop(_enter);
            drop(_span);

            e = poly.evaluate(&r_j);
            cubic_polys.push(poly.compress());
        }

        let claims_prod = params.get_final_evals();

        (SumcheckInstanceProof::new(cubic_polys), r, claims_prod)
    }

    #[tracing::instrument(skip_all, name = "Sumcheck.prove_batched_special_fork_flags")]
    pub fn prove_cubic_batched_flags<G>(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut Transcript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F))
    where
        G: CurveGroup<ScalarField = F>,
    {
        let mut params = params;

        let mut e = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _j in 0..params.num_rounds {

            let len = params.poly_As[0].len() / 2;
            let eq_span = tracing::span!(tracing::Level::TRACE, "eq_evals");
            let _eq_enter = eq_span.enter();
            let eq_evals: Vec<(F, F, F)> = (0..len)
                .into_par_iter()
                .map(|i| {
                    let low = i;
                    let high = len + i;

                    let eq = &params.poly_eq;

                    let eval_point_0 = eq[low];
                    let m_eq = eq[high] - eq[low];
                    let eval_point_2 = eq[high] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();
            drop(_eq_enter);
            drop(eq_span);

            let flag_span = tracing::span!(tracing::Level::TRACE, "flag_evals");
            let _flag_enter = flag_span.enter();
            // Batch<MLEIndex<(eval_0, eval_2, eval_3)>>
            let flag_evals: Vec<Vec<(F, F, F)>> = (0..params.poly_Bs.len())
                .into_par_iter()
                .map(|batch_index| {
                    let mle_evals: Vec<(F, F, F)> = (0..len)
                        .map(|mle_index| {
                            let low = mle_index;
                            let high = len + mle_index;

                            let poly = &params.poly_Bs[batch_index];

                            let eval_point_0 = poly[low];
                            let m_eq = poly[high] - poly[low];
                            let (eval_point_2, eval_point_3) = if m_eq.is_zero() {
                                (poly[high], poly[high])
                            } else {
                                let eval_point_2 = poly[high] + m_eq;
                                let eval_point_3 = eval_point_2 + m_eq;
                                (eval_point_2, eval_point_3)
                            };

                            (eval_point_0, eval_point_2, eval_point_3)
                        })
                        .collect();
                    mle_evals
                })
                .collect();
            drop(_flag_enter);
            drop(flag_span);

            let evals_span = tracing::span!(tracing::Level::TRACE, "evals");
            let _evals_enter = evals_span.enter();
            let evals: Vec<(F, F, F)> = (0..params.poly_As.len())
                .into_par_iter()
                .map(|batch_index| {
                    let eval: (F, F, F) = (0..len)
                        .map(|mle_index| {
                            let low = mle_index;
                            let high = len + mle_index;

                            let eq_eval = eq_evals[low];
                            let flag_eval = flag_evals[params.a_to_b[batch_index]][mle_index];
                            let poly_eval = &params.poly_As[batch_index];

                            let eval_point_0 = params.combine(&poly_eval[low], &flag_eval.0, &eq_eval.0);

                            // let eval_point_0 = if flag_eval.0.is_zero() {
                            //     eq_eval.0
                            // } else if flag_eval.0.is_one() {
                            //     eq_eval.0 * poly_eval[low]
                            // } else {
                            //     eq_eval.0 * (flag_eval.0 * poly_eval[low] + (F::one() - flag_eval.0))
                            // };

                            // let (eval_point_2, opt_poly_2_res): (F, Option<(F, F)>) = if flag_eval.1.is_zero() {
                            //     (eq_eval.1, None)
                            // } else if flag_eval.1.is_one() {
                            //     let poly_m = poly_eval[high] - poly_eval[low];
                            //     let poly_2 = poly_eval[high] + poly_m;
                            //     (eq_eval.1 * poly_2, Some((poly_2, poly_m)))
                            // } else {
                            //     let poly_m = poly_eval[high] - poly_eval[low];
                            //     let poly_2 = poly_eval[high] + poly_m;
                            //     (eq_eval.1 * (flag_eval.1 * poly_2 + (F::one() - flag_eval.1)), Some((poly_2, poly_m)))
                            // };

                            // let eval_point_3 = if let Some((poly_2, poly_m)) = opt_poly_2_res {
                            //     if flag_eval.2.is_zero() {
                            //         eq_eval.2 // TODO(sragss): Path may never happen
                            //     } else if flag_eval.2.is_one() {
                            //         let poly_3 = poly_2 + poly_m;
                            //         eq_eval.2 * poly_3
                            //     } else {
                            //         let poly_3 = poly_2 + poly_m;
                            //         (eq_eval.2 * (flag_eval.2 * poly_3 + (F::one() - flag_eval.2)))
                            //     }
                            // } else {
                            //     eq_eval.2
                            // };

                            let poly_m = poly_eval[high] - poly_eval[low];
                            let poly_2 = poly_eval[high] + poly_m;
                            let poly_3 = poly_2 + poly_m;

                            let eval_point_2 = params.combine(&poly_2, &flag_eval.1, &eq_eval.1);
                            let eval_point_3 = params.combine(&poly_3, &flag_eval.2, &eq_eval.2);

                            (eval_point_0, eval_point_2, eval_point_3)
                        })
                        .fold(
                            (F::zero(), F::zero(), F::zero()),
                            |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                        );

                        eval
                })
                .collect();
            drop(_evals_enter);
            drop(evals_span);

            let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
            let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
            let evals_combined_3 = (0..evals.len()).map(|i| evals[i].2 * coeffs[i]).sum();

            let evals = vec![
                evals_combined_0,
                e - evals_combined_0,
                evals_combined_2,
                evals_combined_3,
            ];
            let poly = UniPoly::from_evals(&evals);

            // append the prover's message to the transcript
            <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(&poly, b"poly", transcript);

            //derive the verifier's challenge for the next round
            let r_j = <Transcript as ProofTranscript<G>>::challenge_scalar(
                transcript,
                b"challenge_nextround",
            );
            r.push(r_j);

            // bound all tables to the verifier's challenege
            let bound_span = tracing::span!(tracing::Level::TRACE, "apply_bound_poly_var_top");
            let _bound_enter = bound_span.enter();
            params.apply_bound_poly_var_top(&r_j);
            drop(_bound_enter);
            drop(bound_span);

            e = poly.evaluate(&r_j);
            cubic_polys.push(poly.compress());
        }

        let claims_prod = params.get_final_evals();

        (SumcheckInstanceProof::new(cubic_polys), r, claims_prod)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct SumcheckInstanceProof<F: PrimeField> {
    compressed_polys: Vec<CompressedUniPoly<F>>,
}

impl<F: PrimeField> SumcheckInstanceProof<F> {
    pub fn new(compressed_polys: Vec<CompressedUniPoly<F>>) -> SumcheckInstanceProof<F> {
        SumcheckInstanceProof { compressed_polys }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    pub fn verify<G, T: ProofTranscript<G>>(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut T,
    ) -> Result<(F, Vec<F>), ProofVerifyError>
    where
        G: CurveGroup<ScalarField = F>,
    {
        let mut e = claim;
        let mut r: Vec<F> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            let poly = self.compressed_polys[i].decompress(&e);

            // verify degree bound
            if poly.degree() != degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    poly.degree(),
                ));
            }

            // check if G_k(0) + G_k(1) = e
            assert_eq!(poly.eval_at_zero() + poly.eval_at_one(), e);

            // append the prover's message to the transcript
            <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(&poly, b"poly", transcript);

            //derive the verifier's challenge for the next round
            let r_i = transcript.challenge_scalar(b"challenge_nextround");

            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i
            e = poly.evaluate(&r_i);
        }

        Ok((e, r))
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct ZKSumcheckInstanceProof<G: CurveGroup> {
    comm_polys: Vec<G>,
    comm_evals: Vec<G>,
    proofs: Vec<DotProductProof<G>>,
}

#[allow(dead_code)]
impl<G: CurveGroup> ZKSumcheckInstanceProof<G> {
    pub fn new(comm_polys: Vec<G>, comm_evals: Vec<G>, proofs: Vec<DotProductProof<G>>) -> Self {
        ZKSumcheckInstanceProof {
            comm_polys,
            comm_evals,
            proofs,
        }
    }

    pub fn verify(
        &self,
        comm_claim: &G,
        num_rounds: usize,
        degree_bound: usize,
        gens_1: &MultiCommitGens<G>,
        gens_n: &MultiCommitGens<G>,
        transcript: &mut Transcript,
    ) -> Result<(G, Vec<G::ScalarField>), ProofVerifyError> {
        // verify degree bound
        assert_eq!(gens_n.n, degree_bound + 1);

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.comm_polys.len(), num_rounds);
        assert_eq!(self.comm_evals.len(), num_rounds);

        let mut r: Vec<G::ScalarField> = Vec::new();
        for i in 0..self.comm_polys.len() {
            let comm_poly = &self.comm_polys[i];

            // append the prover's polynomial to the transcript
            <Transcript as ProofTranscript<G>>::append_point(transcript, b"comm_poly", comm_poly);

            //derive the verifier's challenge for the next round
            let r_i = <Transcript as ProofTranscript<G>>::challenge_scalar(
                transcript,
                b"challenge_nextround",
            );

            // verify the proof of sum-check and evals
            let res = {
                let comm_claim_per_round = if i == 0 {
                    comm_claim
                } else {
                    &self.comm_evals[i - 1]
                };
                let comm_eval = &self.comm_evals[i];

                // add two claims to transcript
                <Transcript as ProofTranscript<G>>::append_point(
                    transcript,
                    b"comm_claim_per_round",
                    comm_claim_per_round,
                );
                <Transcript as ProofTranscript<G>>::append_point(
                    transcript,
                    b"comm_eval",
                    comm_eval,
                );

                // produce two weights
                let w = <Transcript as ProofTranscript<G>>::challenge_vector(
                    transcript,
                    b"combine_two_claims_to_one",
                    2,
                );

                // compute a weighted sum of the RHS
                let bases = vec![comm_claim_per_round.into_affine(), comm_eval.into_affine()];

                let comm_target = VariableBaseMSM::msm(bases.as_ref(), w.as_ref()).unwrap();

                let a = {
                    // the vector to use to decommit for sum-check test
                    let a_sc = {
                        let mut a = vec![G::ScalarField::one(); degree_bound + 1];
                        a[0] += G::ScalarField::one();
                        a
                    };

                    // the vector to use to decommit for evaluation
                    let a_eval = {
                        let mut a = vec![G::ScalarField::one(); degree_bound + 1];
                        for j in 1..a.len() {
                            a[j] = a[j - 1] * r_i;
                        }
                        a
                    };

                    // take weighted sum of the two vectors using w
                    assert_eq!(a_sc.len(), a_eval.len());
                    (0..a_sc.len())
                        .map(|i| w[0] * a_sc[i] + w[1] * a_eval[i])
                        .collect::<Vec<G::ScalarField>>()
                };

                self.proofs[i]
                    .verify(
                        gens_1,
                        gens_n,
                        transcript,
                        &a,
                        &self.comm_polys[i],
                        &comm_target,
                    )
                    .is_ok()
            };
            if !res {
                return Err(ProofVerifyError::InternalError);
            }

            r.push(r_i);
        }

        Ok((self.comm_evals[self.comm_evals.len() - 1], r))
    }
}

pub mod bench {
    use super::*;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::poly::eq_poly::EqPolynomial;
    use crate::subprotocols::sumcheck::{CubicSumcheckParams, SumcheckInstanceProof};
    use crate::utils::index_to_field_bitvector;
    use ark_curve25519::{EdwardsProjective, Fr};
    use ark_std::{rand::Rng, test_rng, One, UniformRand, Zero};
    use criterion::black_box;

    pub fn sumcheck_bench(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    ) {
        let num_vars = 16;
        let mut rng = test_rng();

        // PLAIN
        let r1 = vec![Fr::rand(&mut rng); num_vars];
        let r2 = vec![Fr::rand(&mut rng); num_vars];
        let r3 = vec![Fr::rand(&mut rng); num_vars];
        let eq1 = DensePolynomial::new(EqPolynomial::new(r1).evals());
        let left = DensePolynomial::new(EqPolynomial::new(r2).evals());
        let right = DensePolynomial::new(EqPolynomial::new(r3).evals());
        let params = CubicSumcheckParams::new_prod(
            vec![eq1.clone()],
            vec![left.clone()],
            right.clone(),
            num_vars,
        );

        let mut claim = Fr::zero();
        for i in 0..num_vars {
            let eval1 = eq1.evaluate(&index_to_field_bitvector(i, num_vars));
            let eval2 = left.evaluate(&index_to_field_bitvector(i, num_vars));
            let eval3 = right.evaluate(&index_to_field_bitvector(i, num_vars));

            claim += eval1 * eval2 * eval3;
        }

        let coeffs = vec![Fr::one()];

        group.bench_function("sumcheck unbatched 2^16", |b| {
            b.iter(|| {
                let mut transcript = Transcript::new(b"test_transcript");
                let params = black_box(params.clone());
                let (proof, r, evals) = SumcheckInstanceProof::prove_cubic_batched::<
                    EdwardsProjective,
                >(&claim, params, &coeffs, &mut transcript);
            })
        });

        // FLAGGED
        let num_leaves = 1 << num_vars;
        let mut vals1 = vec![Fr::rand(&mut rng); num_leaves];
        let mut vals2 = vec![Fr::rand(&mut rng); num_leaves];
        // Set approximately half of the leaves to 1
        for _ in 0..num_vars / 2 {
            let rand_index = rng.gen_range(0..num_leaves);
            vals1[rand_index] = Fr::one();
            vals2[rand_index] = Fr::one();
        }
        let poly_a = DensePolynomial::new(vals1);
        let poly_b = DensePolynomial::new(vals2);
        let r = vec![Fr::rand(&mut rng); num_vars];
        let eq = DensePolynomial::new(EqPolynomial::new(r).evals());
        let params = CubicSumcheckParams::new_prod(
            vec![poly_a.clone()],
            vec![poly_b.clone()],
            eq.clone(),
            num_vars,
        );

        let mut claim = Fr::zero();
        for i in 0..num_vars {
            let eval1 = poly_a.evaluate(&index_to_field_bitvector(i, num_vars));
            let eval2 = poly_b.evaluate(&index_to_field_bitvector(i, num_vars));
            let eval3 = eq.evaluate(&index_to_field_bitvector(i, num_vars));

            claim += eval1 * eval2 * eval3;
        }

        let coeffs = vec![Fr::one()];

        group.bench_function("sumcheck unbatched (ones) 2^16", |b| {
            b.iter(|| {
                let mut transcript = Transcript::new(b"test_transcript");
                let params = black_box(params.clone());
                let (proof, r, evals) = SumcheckInstanceProof::prove_cubic_batched::<
                    EdwardsProjective,
                >(&claim, params, &coeffs, &mut transcript);
            })
        });

        // BATCHED
        let batch_size = 10;
        let num_vars = 14;

        let r_eq = vec![Fr::rand(&mut rng); num_vars];
        let eq = DensePolynomial::new(EqPolynomial::new(r_eq).evals());

        let mut poly_as = Vec::with_capacity(batch_size);
        let mut poly_bs = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let ra = vec![Fr::rand(&mut rng); num_vars];
            let rb = vec![Fr::rand(&mut rng); num_vars];
            let a = DensePolynomial::new(EqPolynomial::new(ra).evals());
            let b = DensePolynomial::new(EqPolynomial::new(rb).evals());
            poly_as.push(a);
            poly_bs.push(b);
        }
        let params =
            CubicSumcheckParams::new_prod(poly_as.clone(), poly_bs.clone(), eq.clone(), num_vars);
        let coeffs = vec![Fr::rand(&mut rng); batch_size];

        let mut joint_claim = Fr::zero();
        for batch_i in 0..batch_size {
            let mut claim = Fr::zero();
            for var_i in 0..num_vars {
                let eval_a = poly_as[batch_i].evaluate(&index_to_field_bitvector(var_i, num_vars));
                let eval_b = poly_bs[batch_i].evaluate(&index_to_field_bitvector(var_i, num_vars));
                let eval_eq = eq.evaluate(&index_to_field_bitvector(var_i, num_vars));

                claim += eval_a * eval_b * eval_eq;
            }
            joint_claim += coeffs[batch_i] * claim;
        }

        group.bench_function("sumcheck 10xbatched 2^14", |b| {
            b.iter(|| {
                let mut transcript = Transcript::new(b"test_transcript");
                let params = black_box(params.clone());
                let (proof, r, evals) = SumcheckInstanceProof::prove_cubic_batched::<
                    EdwardsProjective,
                >(
                    &joint_claim, params, &coeffs, &mut transcript
                );
            })
        });
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::test::TestTranscript;
    use crate::{poly::eq_poly::EqPolynomial, utils::math::Math};
    use ark_curve25519::{EdwardsProjective as G1Projective, Fr};
    use ark_ff::Zero;

    #[test]
    fn sumcheck_arbitrary_cubic() {
        // Create three dense polynomials (all the same)
        let num_vars = 3;
        let num_evals = num_vars.pow2();
        let mut evals: Vec<Fr> = Vec::with_capacity(num_evals);
        for i in 0..num_evals {
            evals.push(Fr::from(8 + i as u64));
        }

        let A: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());
        let B: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());
        let C: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());

        let mut claim = Fr::zero();
        for i in 0..num_evals {
            use crate::utils::index_to_field_bitvector;

            claim += A.evaluate(&index_to_field_bitvector(i, num_vars))
                * B.evaluate(&index_to_field_bitvector(i, num_vars))
                * C.evaluate(&index_to_field_bitvector(i, num_vars));
        }
        let mut polys = vec![A.clone(), B.clone(), C.clone()];

        let comb_func_prod =
            |polys: &[Fr]| -> Fr { polys.iter().fold(Fr::one(), |acc, poly| acc * *poly) };

        let r = vec![Fr::from(3), Fr::from(1), Fr::from(3)]; // point 0,0,0 within the boolean hypercube

        let mut transcript: TestTranscript<Fr> = TestTranscript::new(r.clone(), vec![]);
        let (proof, prove_randomness, _final_poly_evals) =
            SumcheckInstanceProof::<Fr>::prove_arbitrary::<_, G1Projective, _>(
                &claim,
                num_vars,
                &mut polys,
                comb_func_prod,
                3,
                &mut transcript,
            );

        let mut transcript: TestTranscript<Fr> = TestTranscript::new(r.clone(), vec![]);
        let verify_result = proof.verify::<G1Projective, _>(claim, num_vars, 3, &mut transcript);
        assert!(verify_result.is_ok());

        let (verify_evaluation, verify_randomness) = verify_result.unwrap();
        assert_eq!(prove_randomness, verify_randomness);
        assert_eq!(prove_randomness, r);

        // Consider this the opening proof to a(r) * b(r) * c(r)
        let a = A.evaluate(prove_randomness.as_slice());
        let b = B.evaluate(prove_randomness.as_slice());
        let c = C.evaluate(prove_randomness.as_slice());

        let oracle_query = a * b * c;
        assert_eq!(verify_evaluation, oracle_query);
    }

    #[test]
    fn flags_special_trivial() {
        let factorial =
            DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]);
        let flags = DensePolynomial::new(vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()]);
        let r = vec![Fr::one(), Fr::one()];
        let eq = DensePolynomial::new(EqPolynomial::new(r.clone()).evals());
        let num_rounds = 2;

        let claim = Fr::from(4); // r points eq to the 1,1 eval
        let coeffs = vec![Fr::one()];

        let comb_func = |h: &Fr, f: &Fr, eq: &Fr| eq * &(h * f + (&Fr::one() - f));

        let cubic_sumcheck_params = CubicSumcheckParams::new_flags(
            vec![factorial.clone()],
            vec![flags.clone()],
            eq.clone(),
            vec![0],
            num_rounds,
        );

        let mut transcript = Transcript::new(b"test_transcript");
        let (proof, prove_randomness, _evals) =
            SumcheckInstanceProof::prove_cubic_batched::<G1Projective>(
                &claim,
                cubic_sumcheck_params,
                &coeffs,
                &mut transcript,
            );

        let mut transcript = Transcript::new(b"test_transcript");
        let verify_result = proof.verify::<G1Projective, _>(claim, 2, 3, &mut transcript);
        assert!(verify_result.is_ok());

        let (verify_evaluation, verify_randomness) = verify_result.unwrap();
        assert_eq!(prove_randomness, verify_randomness);

        let factorial_eval = factorial.evaluate(prove_randomness.as_slice());
        let flag_eval = flags.evaluate(prove_randomness.as_slice());
        let eq_eval = eq.evaluate(prove_randomness.as_slice());
        let oracle_query = comb_func(&factorial_eval, &flag_eval, &eq_eval);
        assert_eq!(verify_evaluation, oracle_query);
    }

    #[test]
    fn flags_special_non_trivial() {
        // H(r_0, r_1, r_2) = sum_{x \in {0,1} ^ 3}{ eq(r, x) \cdot [flags(x) * h(x) + 1 - flags(x)]}
        // Inside the boolean hypercube H(i) = flags(i) * h(i) + 1 - flags(i)
        // Which means if flags(i) = 1, H(i) = h(i)
        //             if flags(i) = 0, H(i) = 1
        // In reality we'll perform sumcheck to transform H(r_0, r_1, r_2) to evaluations of eq(r_0, r_1, r_2, r_3, r_4, r_5), h(r_3, r_4, r_5), flags(r_3, r_4, r_5)
        // where (r_3, r_4, r_5) are generated over the course of sumcheck.
        // The verifier can check this by computing eq(...), h(...), flags(...) on their own.

        let h = DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]);
        let flags = DensePolynomial::new(vec![Fr::one(), Fr::zero(), Fr::one(), Fr::one()]);
        let r = vec![Fr::from(100), Fr::from(200)];
        let eq = DensePolynomial::new(EqPolynomial::new(r.clone()).evals());
        let num_rounds = 2;

        let mut claim = Fr::zero();
        let num_evals = 4;
        let num_vars = 2;
        for i in 0..num_evals {
            use crate::utils::index_to_field_bitvector;

            let h_eval = h.evaluate(&index_to_field_bitvector(i, num_vars));
            let flag_eval = flags.evaluate(&index_to_field_bitvector(i, num_vars));
            let eq_eval = eq.evaluate(&index_to_field_bitvector(i, num_vars));

            claim += eq_eval * (flag_eval * h_eval + Fr::one() - flag_eval);
        }

        let coeffs = vec![Fr::one()]; // TODO(sragss): Idk how to make this work in the case of non-one coefficients.

        let comb_func = |h: &Fr, f: &Fr, eq: &Fr| eq * &(h * f + (&Fr::one() - f));

        let cubic_sumcheck_params = CubicSumcheckParams::new_flags(
            vec![h.clone()],
            vec![flags.clone()],
            eq.clone(),
            vec![0],
            num_rounds,
        );

        let mut transcript = Transcript::new(b"test_transcript");
        let (proof, prove_randomness, prove_evals) =
            SumcheckInstanceProof::prove_cubic_batched::<G1Projective>(
                &claim,
                cubic_sumcheck_params,
                &coeffs,
                &mut transcript,
            );

        // Prover eval: unwrap and combine
        let (leaf_eval, flag_eval, eq_eval) = prove_evals;
        assert_eq!(leaf_eval.len(), 1);
        assert_eq!(flag_eval.len(), 1);
        let leaf_eval = leaf_eval[0];
        let flag_eval = flag_eval[0];
        let prove_fingerprint_eval = flag_eval * leaf_eval + Fr::one() - flag_eval;
        let prove_eval = eq_eval * (flag_eval * leaf_eval + Fr::one() - flag_eval);

        let mut transcript = Transcript::new(b"test_transcript");
        let verify_result = proof.verify::<G1Projective, _>(claim, 2, 3, &mut transcript);
        assert!(verify_result.is_ok());

        let (verify_evaluation, verify_randomness) = verify_result.unwrap();
        assert_eq!(prove_randomness, verify_randomness);
        assert_eq!(verify_evaluation, prove_eval);

        let h_eval = h.evaluate(prove_randomness.as_slice());
        let flag_eval = flags.evaluate(prove_randomness.as_slice());
        let eq_eval = eq.evaluate(prove_randomness.as_slice());
        let oracle_query = comb_func(&h_eval, &flag_eval, &eq_eval);
        assert_eq!(verify_evaluation, oracle_query);

        let fingerprint_oracle_query = flag_eval * h_eval + Fr::one() - flag_eval;
        assert_eq!(prove_fingerprint_eval, fingerprint_oracle_query);
    }
}
