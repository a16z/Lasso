#![allow(clippy::too_many_arguments)]
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::{self, compute_dotproduct};

use super::commitments::{Commitments, MultiCommitGens};
use crate::subprotocols::combined_table_proof::CombinedTableCommitment;
use crate::subprotocols::dot_product::{DotProductProofGens, DotProductProofLog};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::random::RandomTape;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::AffineRepr;
use ark_ec::CurveGroup;
use ark_ff::BigInteger;
use ark_ff::PrimeField;
use ark_serialize::*;
use ark_std::{test_rng, UniformRand, Zero};
use core::ops::Index;
use merlin::Transcript;
use std::ops::AddAssign;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[cfg(not(feature = "ark-msm"))]
use crate::msm::VariableBaseMSM;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct DensePolynomial<F> {
    num_vars: usize, // the number of variables in the multilinear polynomial
    len: usize,
    Z: Vec<F>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

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

pub struct PolyCommitmentBlinds<F> {
    blinds: Vec<F>,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize, PartialEq)]
pub struct PolyCommitment<G: CurveGroup> {
    C: Vec<G>,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ConstPolyCommitment<G: CurveGroup> {
    C: G,
}

pub enum CommitHint {
    Normal,
    Flags,
    Small,
}

impl<F: PrimeField> DensePolynomial<F> {
    pub fn new(Z: Vec<F>) -> Self {
        assert!(
            utils::is_power_of_two(Z.len()),
            "Dense multi-linear polynomials must be made from a power of 2"
        );

        DensePolynomial {
            num_vars: Z.len().log_2(),
            len: Z.len(),
            Z,
        }
    }

    pub fn new_padded(evals: Vec<F>) -> Self {
        // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
        let mut poly_evals = evals;
        while !(utils::is_power_of_two(poly_evals.len())) {
            poly_evals.push(F::zero());
        }

        DensePolynomial {
            num_vars: poly_evals.len().log_2(),
            len: poly_evals.len(),
            Z: poly_evals,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn clone(&self) -> Self {
        Self::new(self.Z[0..self.len].to_vec())
    }

    pub fn split(&self, idx: usize) -> (Self, Self) {
        assert!(idx < self.len());
        (
            Self::new(self.Z[..idx].to_vec()),
            Self::new(self.Z[idx..2 * idx].to_vec()),
        )
    }

    #[cfg(feature = "multicore")]
    #[tracing::instrument(skip_all, name = "DensePolynomial.commit_inner")]
    fn commit_inner<G: CurveGroup<ScalarField = F>>(
        &self,
        blinds: &[F],
        gens: &MultiCommitGens<G>,
    ) -> PolyCommitment<G> {
        let L_size = blinds.len();
        let R_size = self.Z.len() / L_size;
        assert_eq!(L_size * R_size, self.Z.len());

        let gens = CurveGroup::normalize_batch(&gens.G);

        let C = (0..L_size)
            .into_par_iter()
            .map(|i| {
                Commitments::batch_commit(
                    self.Z[R_size * i..R_size * (i + 1)].as_ref(),
                    &gens,
                )
            })
            .collect();
        PolyCommitment { C }
    }

    #[cfg(not(feature = "multicore"))]
    fn commit_inner<G: CurveGroup<ScalarField = F>>(
        &self,
        blinds: &[F],
        gens: &MultiCommitGens<G>,
    ) -> PolyCommitment<G> {
        let L_size = blinds.len();
        let R_size = self.Z.len() / L_size;
        assert_eq!(L_size * R_size, self.Z.len());
        let C = (0..L_size)
            .map(|i| {
                Commitments::batch_commit(
                    self.Z[R_size * i..R_size * (i + 1)].as_ref(),
                    &blinds[i],
                    gens,
                )
            })
            .collect();
        PolyCommitment { C }
    }

    pub fn commit<G>(
        &self,
        gens: &PolyCommitmentGens<G>,
        random_tape: Option<&mut RandomTape<G>>,
    ) -> (PolyCommitment<G>, PolyCommitmentBlinds<F>)
    where
        G: CurveGroup<ScalarField = F>,
    {
        let n = self.Z.len();
        let ell = self.get_num_vars();
        assert_eq!(n, ell.pow2());

        let (left_num_vars, right_num_vars) = EqPolynomial::<F>::compute_factored_lens(ell);
        let L_size = left_num_vars.pow2();
        let R_size = right_num_vars.pow2();
        assert_eq!(L_size * R_size, n);

        let blinds = if let Some(t) = random_tape {
            PolyCommitmentBlinds {
                blinds: t.random_vector(b"poly_blinds", L_size),
            }
        } else {
            PolyCommitmentBlinds {
                blinds: vec![F::zero(); L_size],
            }
        };

        (self.commit_inner(&blinds.blinds, &gens.gens.gens_n), blinds)
    }

    /// Commit given the lagrange coefficients are 0 / 1
    pub fn commit_with_hint<G>(
        &self,
        gens: &PolyCommitmentGens<G>,
        hint: CommitHint,
    ) -> PolyCommitment<G>
    where
        G: CurveGroup<ScalarField = F>,
    {
        let n = self.Z.len();
        let ell = self.get_num_vars();
        assert_eq!(n, ell.pow2());

        let (left_num_vars, right_num_vars) = EqPolynomial::<F>::compute_factored_lens(ell);
        let L_size = left_num_vars.pow2();
        let R_size = right_num_vars.pow2();
        assert_eq!(L_size * R_size, n);

        let gens: Vec<G::Affine> = CurveGroup::normalize_batch(&gens.gens.gens_n.G);

        let C = (0..L_size)
            .into_par_iter()
            .map(|i| {
                let scalars = self.Z[R_size * i..R_size * (i + 1)].as_ref();
                match hint {
                    CommitHint::Normal => Commitments::batch_commit(scalars, &gens),
                    CommitHint::Flags => Self::flags_msm(scalars, &gens),
                    CommitHint::Small => {
                        let bigints: Vec<_> = scalars.iter().map(|s| s.into_bigint()).collect();
                        Self::sm_msm(&bigints, &gens)
                    }
                }
            })
            .collect();
        PolyCommitment { C }
    }

    /// Special MSM where all scalar values are 0 / 1 – does not verify.
    fn flags_msm<G: CurveGroup>(scalars: &[G::ScalarField], bases: &[G::Affine]) -> G {
        assert_eq!(scalars.len(), bases.len());
        let result = scalars
            .into_iter()
            .enumerate()
            .filter(|(_index, scalar)| !scalar.is_zero())
            .map(|(index, scalar)| bases[index])
            .sum();

        result
    }

    pub fn sm_msm<V: VariableBaseMSM>(
        scalars: &[<V::ScalarField as PrimeField>::BigInt],
        bases: &[V::MulBase],
    ) -> V {
        assert_eq!(scalars.len(), bases.len());
        let num_buckets: usize = 1 << 16; // TODO(sragss): This should be passed in / dependent on M = N^{1/C}

        // #[cfg(test)]
        // scalars.for_each(|scalar| {
        //     assert!(scalar < V::ScalarField::from(num_buckets as u64).into_bigint())
        // });

        // Assign things to buckets based on the scalar
        let mut buckets: Vec<V> = vec![V::zero(); num_buckets];
        scalars.into_iter().enumerate().for_each(|(index, scalar)| {
            let bucket_index: u64 = scalar.as_ref()[0];
            buckets[bucket_index as usize] += bases[index];
        });

        let mut result = V::zero();
        let mut running_sum = V::zero();
        buckets
            .into_iter()
            .skip(1)
            .enumerate()
            .rev()
            .for_each(|(index, bucket)| {
                running_sum += bucket;
                result += running_sum;
            });
        result
    }

    #[tracing::instrument(skip_all, name = "DensePolynomial.bound")]
    pub fn bound(&self, L: &[F]) -> Vec<F> {
        let (left_num_vars, right_num_vars) =
            EqPolynomial::<F>::compute_factored_lens(self.get_num_vars());
        let L_size = left_num_vars.pow2();
        let R_size = right_num_vars.pow2();

        #[cfg(feature = "multicore")]
        let bound_vals = (0..R_size)
            .into_par_iter()
            .map(|i| {
                (0..L_size)
                    .into_iter()
                    .map(|j| L[j] * self.Z[j * R_size + i])
                    .sum()
            })
            .collect();

        #[cfg(not(feature = "multicore"))]
        let bound_vals = (0..R_size)
            .map(|i| (0..L_size).map(|j| L[j] * self.Z[j * R_size + i]).sum())
            .collect();

        bound_vals
    }

    pub fn bound_poly_var_top(&mut self, r: &F) {
        let n = self.len() / 2;

        for i in 0..n {
            let m = self.Z[i + n] - self.Z[i];
            self.Z[i] += *r * m;
        }
        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_top_many_ones(&mut self, r: &F) {
        let n = self.len() / 2;

        for i in 0..n {
            let m = self.Z[i + n] - self.Z[i];
            if !m.is_zero() {
                self.Z[i] += *r * m;
            }
        }
        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_top_pre_compute(&mut self, r: &F, pre_compute: &[F]) {
        let n = self.len() / 2;
        assert_eq!(pre_compute.len(), n);

        for i in 0..n {
            // let m = self.Z[i + n] - self.Z[i];
            let m = pre_compute[i];
            if !m.is_zero() {
                self.Z[i] += *r * m;
            }
        }
        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_bot(&mut self, r: &F) {
        let n = self.len() / 2;
        for i in 0..n {
            self.Z[i] = self.Z[2 * i] + *r * (self.Z[2 * i + 1] - self.Z[2 * i]);
        }
        self.num_vars -= 1;
        self.len = n;
    }

    // returns Z(r) in O(n) time
    pub fn evaluate(&self, r: &[F]) -> F {
        // r must have a value for each variable
        assert_eq!(r.len(), self.get_num_vars());
        let chis = EqPolynomial::new(r.to_vec()).evals();
        assert_eq!(chis.len(), self.Z.len());
        compute_dotproduct(&self.Z, &chis)
    }

    pub fn evaluate_at_chi(&self, chis: &Vec<F>) -> F {
        compute_dotproduct(&self.Z, &chis)
    }

    fn vec(&self) -> &Vec<F> {
        &self.Z
    }

    pub fn evals(&self) -> Vec<F> {
        self.Z.clone()
    }

    pub fn extend(&mut self, other: &DensePolynomial<F>) {
        assert_eq!(self.Z.len(), self.len);
        let other_vec = other.vec();
        assert_eq!(other_vec.len(), self.len);
        self.Z.extend(other_vec);
        self.num_vars += 1;
        self.len *= 2;
        assert_eq!(self.Z.len(), self.len);
    }

    pub fn merge<T>(polys: &Vec<T>) -> DensePolynomial<F>
    where
        T: AsRef<DensePolynomial<F>>,
    {
        let mut Z: Vec<F> = Vec::new();
        for poly in polys.iter() {
            Z.extend(poly.as_ref().vec().iter());
        }

        // pad the polynomial with zero polynomial at the end
        Z.resize(Z.len().next_power_of_two(), F::zero());

        DensePolynomial::new(Z)
    }

    pub fn combined_commit<G>(
        &self,
        label: &'static [u8],
    ) -> (PolyCommitmentGens<G>, CombinedTableCommitment<G>)
    where
        G: CurveGroup<ScalarField = F>,
    {
        let generators = PolyCommitmentGens::new(self.num_vars, label);
        let (joint_commitment, _) = self.commit(&generators, None);
        (generators, CombinedTableCommitment::new(joint_commitment))
    }

    pub fn combined_commit_with_hint<G>(
        &self,
        label: &'static [u8],
    ) -> (PolyCommitmentGens<G>, CombinedTableCommitment<G>)
    where
        G: CurveGroup<ScalarField = F>,
    {
        let generators = PolyCommitmentGens::new(self.num_vars, label);
        let joint_commitment = self.commit_with_hint(&generators, CommitHint::Normal);
        (generators, CombinedTableCommitment::new(joint_commitment))
    }

    pub fn from_usize(Z: &[usize]) -> Self {
        DensePolynomial::new(
            (0..Z.len())
                .map(|i| F::from(Z[i] as u64))
                .collect::<Vec<F>>(),
        )
    }

    pub fn from_u64(Z: &[u64]) -> Self {
        DensePolynomial::new((0..Z.len()).map(|i| F::from(Z[i])).collect::<Vec<F>>())
    }
}

impl<F> Index<usize> for DensePolynomial<F> {
    type Output = F;

    #[inline(always)]
    fn index(&self, _index: usize) -> &F {
        &(self.Z[_index])
    }
}

impl<F> AsRef<DensePolynomial<F>> for DensePolynomial<F> {
    fn as_ref(&self) -> &DensePolynomial<F> {
        self
    }
}

impl<G: CurveGroup> AppendToTranscript<G> for PolyCommitment<G> {
    fn append_to_transcript<T: ProofTranscript<G>>(
        &self,
        label: &'static [u8],
        transcript: &mut T,
    ) {
        transcript.append_message(label, b"poly_commitment_begin");
        for i in 0..self.C.len() {
            transcript.append_point(b"poly_commitment_share", &self.C[i]);
        }
        transcript.append_message(label, b"poly_commitment_end");
    }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PolyEvalProof<G: CurveGroup> {
    proof: DotProductProofLog<G>,
}

impl<G: CurveGroup> PolyEvalProof<G> {
    fn protocol_name() -> &'static [u8] {
        b"polynomial evaluation proof"
    }

    #[tracing::instrument(skip_all, name = "DensePolyEval.prove")]
    pub fn prove(
        poly: &DensePolynomial<G::ScalarField>,
        blinds_opt: Option<&PolyCommitmentBlinds<G::ScalarField>>,
        r: &[G::ScalarField], // point at which the polynomial is evaluated
        Zr: &G::ScalarField,  // evaluation of \widetilde{Z}(r)
        blind_Zr_opt: Option<&G::ScalarField>, // specifies a blind for Zr
        gens: &PolyCommitmentGens<G>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> (PolyEvalProof<G>, G) {
        <Transcript as ProofTranscript<G>>::append_protocol_name(
            transcript,
            PolyEvalProof::<G>::protocol_name(),
        );

        // assert vectors are of the right size
        assert_eq!(poly.get_num_vars(), r.len());

        let (left_num_vars, right_num_vars) =
            EqPolynomial::<G::ScalarField>::compute_factored_lens(r.len());
        let L_size = left_num_vars.pow2();
        let R_size = right_num_vars.pow2();

        let default_blinds = PolyCommitmentBlinds {
            blinds: vec![G::ScalarField::zero(); L_size],
        };
        let blinds = blinds_opt.map_or(&default_blinds, |p| p);

        assert_eq!(blinds.blinds.len(), L_size);

        let zero = G::ScalarField::zero();
        let blind_Zr = blind_Zr_opt.map_or(&zero, |p| p);

        // compute the L and R vectors
        let eq = EqPolynomial::new(r.to_vec());
        let (L, R) = eq.compute_factored_evals();
        assert_eq!(L.len(), L_size);
        assert_eq!(R.len(), R_size);

        // compute the vector underneath L*Z and the L*blinds
        // compute vector-matrix product between L and Z viewed as a matrix
        let LZ = poly.bound(&L);
        let LZ_blind: G::ScalarField = (0..L.len()).map(|i| blinds.blinds[i] * L[i]).sum();

        // a dot product proof of size R_size
        let (proof, _C_LR, C_Zr_prime) = DotProductProofLog::prove(
            &gens.gens,
            transcript,
            random_tape,
            &LZ,
            &LZ_blind,
            &R,
            Zr,
            blind_Zr,
        );

        (PolyEvalProof { proof }, C_Zr_prime)
    }

    pub fn verify(
        &self,
        gens: &PolyCommitmentGens<G>,
        transcript: &mut Transcript,
        r: &[G::ScalarField], // point at which the polynomial is evaluated
        C_Zr: &G,             // commitment to \widetilde{Z}(r)
        comm: &PolyCommitment<G>,
    ) -> Result<(), ProofVerifyError> {
        <Transcript as ProofTranscript<G>>::append_protocol_name(
            transcript,
            PolyEvalProof::<G>::protocol_name(),
        );

        // compute L and R
        let eq: EqPolynomial<_> = EqPolynomial::new(r.to_vec());
        let (L, R) = eq.compute_factored_evals();

        // compute a weighted sum of commitments and L
        let C_affine = G::normalize_batch(&comm.C);

        let C_LZ = VariableBaseMSM::msm(C_affine.as_ref(), L.as_ref()).unwrap();

        self.proof
            .verify(R.len(), &gens.gens, transcript, &R, &C_LZ, C_Zr)
    }

    pub fn verify_plain(
        &self,
        gens: &PolyCommitmentGens<G>,
        transcript: &mut Transcript,
        r: &[G::ScalarField], // point at which the polynomial is evaluated
        Zr: &G::ScalarField,  // evaluation \widetilde{Z}(r)
        comm: &PolyCommitment<G>,
    ) -> Result<(), ProofVerifyError> {
        // compute a commitment to Zr with a blind of zero
        let C_Zr = Zr.commit(&G::ScalarField::zero(), &gens.gens.gens_1);

        self.verify(gens, transcript, r, &C_Zr, comm)
    }
}

impl<F: PrimeField> AddAssign<&DensePolynomial<F>> for DensePolynomial<F> {
    fn add_assign(&mut self, rhs: &DensePolynomial<F>) {
        assert_eq!(self.num_vars, rhs.num_vars);
        assert_eq!(self.len, rhs.len);
        let summed_evaluations: Vec<F> = self.Z.iter().zip(&rhs.Z).map(|(a, b)| *a + *b).collect();

        *self = Self {
            num_vars: self.num_vars,
            len: self.len,
            Z: summed_evaluations,
        }
    }
}

pub mod bench {
    use super::*;
    use crate::utils::gen_random_point;
    use ark_curve25519::{EdwardsProjective, Fr};
    use ark_std::{rand::Rng, test_rng, One, Zero};
    use criterion::{black_box, measurement::WallTime, BenchmarkGroup};

    pub fn dense_ml_poly_bench(group: &mut BenchmarkGroup<'_, WallTime>) {
        let evals: Vec<Fr> = gen_random_point::<Fr>(1 << 10);
        let poly = DensePolynomial::new(evals.clone());

        let r: Vec<Fr> = gen_random_point::<Fr>(10);

        group.bench_function("evaluate", |b| {
            b.iter(|| {
                let result = black_box(poly.evaluate(&r));
                black_box(result);
            })
        });

        let log_sizes = [10, 16];
        for &log_size in &log_sizes {
            group.bench_function(format!("commit {}", log_size), |b| {
                b.iter_with_setup(
                    || init_commit_bench(log_size),
                    |(gens, poly)| {
                        black_box(run_commit_bench(gens, poly));
                    },
                )
            });
        }
    }

    pub fn init_commit_bench(
        log_size: usize,
    ) -> (PolyCommitmentGens<EdwardsProjective>, DensePolynomial<Fr>) {
        let evals: Vec<Fr> = gen_random_point::<Fr>(1 << log_size);
        let gens = PolyCommitmentGens::new(log_size, b"test_gens");
        let poly = DensePolynomial::new(evals.clone());
        (gens, poly)
    }

    /// Gets a commitment benchmark for evaluations that are not random field elements but rather 0/1
    pub fn init_commit_bench_ones(
        log_size: usize,
        pct_ones: f64,
    ) -> (PolyCommitmentGens<EdwardsProjective>, DensePolynomial<Fr>) {
        let mut evals: Vec<Fr> = Vec::with_capacity(1 << log_size);
        let mut rng = test_rng();
        for _ in 0..(1 << log_size) {
            let val = if rng.gen::<f64>() < pct_ones {
                Fr::one()
            } else {
                Fr::zero()
            };
            evals.push(val);
        }

        let gens = PolyCommitmentGens::new(log_size, b"test_gens");
        let poly = DensePolynomial::new(evals.clone());
        (gens, poly)
    }

    pub fn init_commit_small(
        log_size: usize,
        max_size: usize,
    ) -> (PolyCommitmentGens<EdwardsProjective>, DensePolynomial<Fr>) {
        let mut evals: Vec<Fr> = Vec::with_capacity(1 << log_size);
        let mut rng = test_rng();
        for _ in 0..(1 << log_size) {
            let val = Fr::from(rng.gen::<u64>() % (max_size as u64));
            evals.push(val);
        }

        let gens = PolyCommitmentGens::new(log_size, b"test_gens");
        let poly = DensePolynomial::new(evals.clone());
        (gens, poly)
    }

    pub fn run_commit_bench(
        gens: PolyCommitmentGens<EdwardsProjective>,
        poly: DensePolynomial<Fr>,
    ) {
        let result = black_box(poly.commit::<EdwardsProjective>(&gens, None));
        black_box(result);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::subprotocols::dot_product::DotProductProof;
    use ark_curve25519::EdwardsProjective as G1Projective;
    use ark_curve25519::Fr;
    use ark_std::One;
    use ark_std::UniformRand;
    use ark_std::{rand::Rng, test_rng};

    fn evaluate_with_LR<G: CurveGroup>(
        Z: &[G::ScalarField],
        r: &[G::ScalarField],
    ) -> G::ScalarField {
        let eq = EqPolynomial::<G::ScalarField>::new(r.to_vec());
        let (L, R) = eq.compute_factored_evals();

        let ell = r.len();
        // ensure ell is even
        assert!(ell % 2 == 0);
        // compute n = 2^\ell
        let n = ell.pow2();
        // compute m = sqrt(n) = 2^{\ell/2}
        let m = n.square_root();

        // compute vector-matrix product between L and Z viewed as a matrix
        let LZ = (0..m)
            .map(|i| (0..m).map(|j| L[j] * Z[j * m + i]).sum())
            .collect::<Vec<G::ScalarField>>();

        // compute dot product between LZ and R
        DotProductProof::<G>::compute_dotproduct(&LZ, &R)
    }

    #[test]
    fn check_polynomial_evaluation() {
        check_polynomial_evaluation_helper::<G1Projective>()
    }

    fn check_polynomial_evaluation_helper<G: CurveGroup>() {
        // Z = [1, 2, 1, 4]
        let Z = vec![
            G::ScalarField::one(),
            G::ScalarField::from(2u64),
            G::ScalarField::one(),
            G::ScalarField::from(4u64),
        ];

        // r = [4,3]
        let r = vec![G::ScalarField::from(4u64), G::ScalarField::from(3u64)];

        let eval_with_LR = evaluate_with_LR::<G>(&Z, &r);
        let poly = DensePolynomial::new(Z);

        let eval = poly.evaluate(&r);
        assert_eq!(eval, G::ScalarField::from(28u64));
        assert_eq!(eval_with_LR, eval);
    }

    pub fn compute_factored_chis_at_r<F: PrimeField>(r: &[F]) -> (Vec<F>, Vec<F>) {
        let mut L: Vec<F> = Vec::new();
        let mut R: Vec<F> = Vec::new();

        let ell = r.len();
        assert!(ell % 2 == 0); // ensure ell is even
        let n = ell.pow2();
        let m = n.square_root();

        // compute row vector L
        for i in 0..m {
            let mut chi_i = F::one();
            for j in 0..ell / 2 {
                let bit_j = ((m * i) & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::one() - r[j];
                }
            }
            L.push(chi_i);
        }

        // compute column vector R
        for i in 0..m {
            let mut chi_i = F::one();
            for j in ell / 2..ell {
                let bit_j = (i & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::one() - r[j];
                }
            }
            R.push(chi_i);
        }
        (L, R)
    }

    pub fn compute_chis_at_r<F: PrimeField>(r: &[F]) -> Vec<F> {
        let ell = r.len();
        let n = ell.pow2();
        let mut chis: Vec<F> = Vec::new();
        for i in 0..n {
            let mut chi_i = F::one();
            for j in 0..r.len() {
                let bit_j = (i & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::one() - r[j];
                }
            }
            chis.push(chi_i);
        }
        chis
    }

    pub fn compute_outerproduct<F: PrimeField>(L: &[F], R: &[F]) -> Vec<F> {
        assert_eq!(L.len(), R.len());
        (0..L.len())
            .map(|i| (0..R.len()).map(|j| L[i] * R[j]).collect::<Vec<F>>())
            .collect::<Vec<Vec<F>>>()
            .into_iter()
            .flatten()
            .collect::<Vec<F>>()
    }

    #[test]
    fn check_memoized_chis() {
        check_memoized_chis_helper::<G1Projective>()
    }

    fn check_memoized_chis_helper<G: CurveGroup>() {
        let mut prng = test_rng();

        let s = 10;
        let mut r: Vec<G::ScalarField> = Vec::new();
        for _i in 0..s {
            r.push(G::ScalarField::rand(&mut prng));
        }
        let chis = compute_chis_at_r::<G::ScalarField>(&r);
        let chis_m = EqPolynomial::<G::ScalarField>::new(r).evals();
        assert_eq!(chis, chis_m);
    }

    #[test]
    fn check_factored_chis() {
        check_factored_chis_helper::<Fr>()
    }

    fn check_factored_chis_helper<F: PrimeField>() {
        let mut prng = test_rng();

        let s = 10;
        let mut r: Vec<F> = Vec::new();
        for _i in 0..s {
            r.push(F::rand(&mut prng));
        }
        let chis = EqPolynomial::new(r.clone()).evals();
        let (L, R) = EqPolynomial::new(r).compute_factored_evals();
        let O = compute_outerproduct(&L, &R);
        assert_eq!(chis, O);
    }

    #[test]
    fn check_memoized_factored_chis() {
        check_memoized_factored_chis_helper::<Fr>()
    }

    fn check_memoized_factored_chis_helper<F: PrimeField>() {
        let mut prng = test_rng();

        let s = 10;
        let mut r: Vec<F> = Vec::new();
        for _i in 0..s {
            r.push(F::rand(&mut prng));
        }
        let (L, R) = compute_factored_chis_at_r(&r);
        let eq = EqPolynomial::new(r);
        let (L2, R2) = eq.compute_factored_evals();
        assert_eq!(L, L2);
        assert_eq!(R, R2);
    }

    #[test]
    fn check_polynomial_commit() {
        check_polynomial_commit_helper::<G1Projective>()
    }

    fn check_polynomial_commit_helper<G: CurveGroup>() {
        let Z = vec![
            G::ScalarField::one(),
            G::ScalarField::from(2u64),
            G::ScalarField::one(),
            G::ScalarField::from(4u64),
        ];
        let poly = DensePolynomial::new(Z);

        // r = [4,3]
        let r = vec![G::ScalarField::from(4u64), G::ScalarField::from(3u64)];
        let eval = poly.evaluate(&r);
        assert_eq!(eval, G::ScalarField::from(28u64));

        let gens = PolyCommitmentGens::<G>::new(poly.get_num_vars(), b"test-two");
        let (poly_commitment, blinds) = poly.commit(&gens, None);

        let mut random_tape = RandomTape::new(b"proof");
        let mut prover_transcript = Transcript::new(b"example");
        let (proof, C_Zr) = PolyEvalProof::prove(
            &poly,
            Some(&blinds),
            &r,
            &eval,
            None,
            &gens,
            &mut prover_transcript,
            &mut random_tape,
        );

        let mut verifier_transcript = Transcript::new(b"example");

        assert!(proof
            .verify(&gens, &mut verifier_transcript, &r, &C_Zr, &poly_commitment)
            .is_ok());
    }

    #[test]
    fn evaluation() {
        let num_evals = 4;
        let mut evals: Vec<Fr> = Vec::with_capacity(num_evals);
        for _ in 0..num_evals {
            evals.push(Fr::from(8));
        }
        let dense_poly: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());

        // Evaluate at 3:
        // (0, 0) = 1
        // (0, 1) = 1
        // (1, 0) = 1
        // (1, 1) = 1
        // g(x_0,x_1) => c_0*(1 - x_0)(1 - x_1) + c_1*(1-x_0)(x_1) + c_2*(x_0)(1-x_1) + c_3*(x_0)(x_1)
        // g(3, 4) = 8*(1 - 3)(1 - 4) + 8*(1-3)(4) + 8*(3)(1-4) + 8*(3)(4) = 48 + -64 + -72 + 96  = 8
        // g(5, 10) = 8*(1 - 5)(1 - 10) + 8*(1 - 5)(10) + 8*(5)(1-10) + 8*(5)(10) = 96 + -16 + -72 + 96  = 8
        assert_eq!(
            dense_poly.evaluate(vec![Fr::from(3), Fr::from(4)].as_slice()),
            Fr::from(8)
        );
    }

    #[test]
    fn sm_msm_parity() {
        use ark_curve25519::{EdwardsAffine as G1Affine, EdwardsProjective as G1Projective, Fr};
        let mut rng = test_rng();
        let bases = vec![
            G1Affine::rand(&mut rng),
            G1Affine::rand(&mut rng),
            G1Affine::rand(&mut rng),
        ];
        let scalars = vec![Fr::from(3), Fr::from(2), Fr::from(1)];
        let expected_result = bases[0] + bases[0] + bases[0] + bases[1] + bases[1] + bases[2];
        assert_eq!(bases[0] + bases[0] + bases[0], bases[0] * scalars[0]);
        let expected_result_b =
            bases[0] * scalars[0] + bases[1] * scalars[1] + bases[2] * scalars[2];
        assert_eq!(expected_result, expected_result_b);

        let calc_result_a: G1Projective = VariableBaseMSM::msm(&bases, &scalars).unwrap();
        assert_eq!(calc_result_a, expected_result);

        let scalars_bigint: Vec<_> = scalars
            .into_iter()
            .map(|scalar| scalar.into_bigint())
            .collect();
        let calc_result_b: G1Projective = DensePolynomial::<Fr>::sm_msm(&scalars_bigint, &bases);
        assert_eq!(calc_result_b, expected_result);
    }

    #[test]
    fn commit_with_hint_parity() {
        let log_size = 6;
        let max_size = 1 << 4;

        let mut evals: Vec<Fr> = Vec::with_capacity(1 << log_size);
        let mut rng = test_rng();
        for _ in 0..(1 << log_size) {
            let val = Fr::from(rng.gen::<u64>() % (max_size as u64));
            evals.push(val);
        }

        let gens: PolyCommitmentGens<G1Projective> =
            PolyCommitmentGens::new(log_size, b"test_gens");
        let poly: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());

        let a = poly.commit_with_hint(&gens, CommitHint::Normal);
        let b = poly.commit_with_hint(&gens, CommitHint::Small);

        assert_eq!(a, b);
    }
}
