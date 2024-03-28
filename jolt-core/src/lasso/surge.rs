use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use merlin::Transcript;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::marker::{PhantomData, Sync};

use crate::{
    jolt::instruction::JoltInstruction,
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        hyrax::matrix_dimensions,
        identity_poly::IdentityPolynomial,
        pedersen::PedersenGenerators,
        structured_poly::{BatchablePolynomials, StructuredOpeningProof},
    },
    subprotocols::{
        concatenated_commitment::{
            ConcatenatedPolynomialCommitment, ConcatenatedPolynomialOpeningProof,
        },
        sumcheck::SumcheckInstanceProof,
    },
    utils::{errors::ProofVerifyError, math::Math, mul_0_1_optimized, transcript::ProofTranscript},
};

use super::memory_checking::NoPreprocessing;

pub struct SurgePolys<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    _group: PhantomData<G>,
    pub dim: Vec<DensePolynomial<F>>,
    pub read_cts: Vec<DensePolynomial<F>>,
    pub final_cts: Vec<DensePolynomial<F>>,
    pub E_polys: Vec<DensePolynomial<F>>,
}

pub struct BatchedSurgePolynomials<F: PrimeField> {
    pub batched_dim_read: DensePolynomial<F>,
    pub batched_final: DensePolynomial<F>,
    pub batched_E: DensePolynomial<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeCommitment<G: CurveGroup> {
    pub dim_read_commitment: ConcatenatedPolynomialCommitment<G>,
    pub final_commitment: ConcatenatedPolynomialCommitment<G>,
    pub E_commitment: ConcatenatedPolynomialCommitment<G>,
}

impl<F, G> BatchablePolynomials<G> for SurgePolys<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type BatchedPolynomials = BatchedSurgePolynomials<F>;
    type Commitment = SurgeCommitment<G>;

    #[tracing::instrument(skip_all, name = "SurgePolys::batch")]
    fn batch(&self) -> Self::BatchedPolynomials {
        let (batched_dim_read, (batched_final, batched_E)) = rayon::join(
            || DensePolynomial::merge(self.dim.iter().chain(&self.read_cts)),
            || {
                rayon::join(
                    || DensePolynomial::merge(&self.final_cts),
                    || DensePolynomial::merge(&self.E_polys),
                )
            },
        );

        Self::BatchedPolynomials {
            batched_dim_read,
            batched_final,
            batched_E,
        }
    }

    #[tracing::instrument(skip_all, name = "SurgePolys::commit")]
    fn commit(
        &self,
        batched_polys: &Self::BatchedPolynomials,
        pedersen_generators: &PedersenGenerators<G>,
    ) -> Self::Commitment {
        let dim_read_commitment = batched_polys
            .batched_dim_read
            .combined_commit(pedersen_generators);
        let final_commitment = batched_polys
            .batched_final
            .combined_commit(pedersen_generators);
        let E_commitment = batched_polys.batched_E.combined_commit(pedersen_generators);

        Self::Commitment {
            dim_read_commitment,
            final_commitment,
            E_commitment,
        }
    }
}

type PrimarySumcheckOpenings<F> = Vec<F>;

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> StructuredOpeningProof<F, G, SurgePolys<F, G>>
    for PrimarySumcheckOpenings<F>
{
    #[tracing::instrument(skip_all, name = "PrimarySumcheckOpenings::open")]
    fn open(polynomials: &SurgePolys<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        polynomials
            .E_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect()
    }

    #[tracing::instrument(skip_all, name = "PrimarySumcheckOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &SurgePolys<F, G>,
        batched_polynomials: &BatchedSurgePolynomials<F>,
        opening_point: &Vec<F>,
        E_poly_openings: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        ConcatenatedPolynomialOpeningProof::prove(
            &batched_polynomials.batched_E,
            opening_point,
            E_poly_openings,
            transcript,
        )
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &SurgeCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        opening_proof.verify(opening_point, &self, &commitment.E_commitment, transcript)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeReadWriteOpenings<F>
where
    F: PrimeField,
{
    dim_openings: Vec<F>,    // C-sized
    read_openings: Vec<F>,   // C-sized
    E_poly_openings: Vec<F>, // NUM_MEMORIES-sized
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeReadWriteOpeningProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    dim_read_opening_proof: ConcatenatedPolynomialOpeningProof<G>,
    E_poly_opening_proof: ConcatenatedPolynomialOpeningProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, SurgePolys<F, G>> for SurgeReadWriteOpenings<F>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Proof = SurgeReadWriteOpeningProof<F, G>;

    #[tracing::instrument(skip_all, name = "SurgeReadWriteOpenings::open")]
    fn open(polynomials: &SurgePolys<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        let evaluate = |poly: &DensePolynomial<F>| -> F { poly.evaluate_at_chi(&chis) };
        Self {
            dim_openings: polynomials.dim.par_iter().map(evaluate).collect(),
            read_openings: polynomials.read_cts.par_iter().map(evaluate).collect(),
            E_poly_openings: polynomials.E_polys.par_iter().map(evaluate).collect(),
        }
    }

    #[tracing::instrument(skip_all, name = "SurgeReadWriteOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &SurgePolys<F, G>,
        batched_polynomials: &BatchedSurgePolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        let mut dim_read_openings = [
            openings.dim_openings.as_slice(),
            openings.read_openings.as_slice(),
        ]
        .concat()
        .to_vec();
        dim_read_openings.resize(dim_read_openings.len().next_power_of_two(), F::zero());

        let dim_read_opening_proof = ConcatenatedPolynomialOpeningProof::prove(
            &batched_polynomials.batched_dim_read,
            &opening_point,
            &dim_read_openings,
            transcript,
        );
        let E_poly_opening_proof = ConcatenatedPolynomialOpeningProof::prove(
            &batched_polynomials.batched_E,
            &opening_point,
            &openings.E_poly_openings,
            transcript,
        );

        SurgeReadWriteOpeningProof {
            dim_read_opening_proof,
            E_poly_opening_proof,
        }
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &SurgeCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let mut dim_read_openings = [self.dim_openings.as_slice(), self.read_openings.as_slice()]
            .concat()
            .to_vec();
        dim_read_openings.resize(dim_read_openings.len().next_power_of_two(), F::zero());

        opening_proof.dim_read_opening_proof.verify(
            opening_point,
            &dim_read_openings,
            &commitment.dim_read_commitment,
            transcript,
        )?;

        opening_proof.E_poly_opening_proof.verify(
            opening_point,
            &self.E_poly_openings,
            &commitment.E_commitment,
            transcript,
        )?;

        Ok(())
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeFinalOpenings<F, Instruction, const C: usize, const M: usize>
where
    F: PrimeField,
    Instruction: JoltInstruction + Default,
{
    _instruction: PhantomData<Instruction>,
    final_openings: Vec<F>,       // C-sized
    a_init_final: Option<F>,      // Computed by verifier
    v_init_final: Option<Vec<F>>, // Computed by verifier
}

impl<F, G, Instruction, const C: usize, const M: usize>
    StructuredOpeningProof<F, G, SurgePolys<F, G>> for SurgeFinalOpenings<F, Instruction, C, M>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Instruction: JoltInstruction + Default,
{
    type Preprocessing = SurgePreprocessing<F, Instruction, C, M>;

    #[tracing::instrument(skip_all, name = "SurgeFinalOpenings::open")]
    fn open(polynomials: &SurgePolys<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        let final_openings = polynomials
            .final_cts
            .par_iter()
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect();
        Self {
            _instruction: PhantomData,
            final_openings,
            a_init_final: None,
            v_init_final: None,
        }
    }

    #[tracing::instrument(skip_all, name = "SurgeFinalOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &SurgePolys<F, G>,
        batched_polynomials: &BatchedSurgePolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        ConcatenatedPolynomialOpeningProof::prove(
            &batched_polynomials.batched_final,
            &opening_point,
            &openings.final_openings,
            transcript,
        )
    }

    fn compute_verifier_openings(&mut self, _: &Self::Preprocessing, opening_point: &Vec<F>) {
        self.a_init_final =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
        self.v_init_final = Some(
            Instruction::default()
                .subtables(C, M)
                .iter()
                .map(|(subtable, _)| subtable.evaluate_mle(opening_point))
                .collect(),
        );
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &SurgeCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        opening_proof.verify(
            opening_point,
            &self.final_openings,
            &commitment.final_commitment,
            transcript,
        )
    }
}

impl<F, G, Instruction, const C: usize, const M: usize> MemoryCheckingProver<F, G, SurgePolys<F, G>>
    for SurgeProof<F, G, Instruction, C, M>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Instruction: JoltInstruction + Default + Sync,
{
    type Preprocessing = SurgePreprocessing<F, Instruction, C, M>;
    type ReadWriteOpenings = SurgeReadWriteOpenings<F>;
    type InitFinalOpenings = SurgeFinalOpenings<F, Instruction, C, M>;

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - tau
    }

    #[tracing::instrument(skip_all, name = "Surge::compute_leaves")]
    fn compute_leaves(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        polynomials: &SurgePolys<F, G>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) {
        let gamma_squared = gamma.square();
        let num_lookups = polynomials.dim[0].len();

        let read_write_leaves = (0..Self::num_memories())
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                let read_fingerprints: Vec<F> = (0..num_lookups)
                    .map(|i| {
                        mul_0_1_optimized(&polynomials.read_cts[dim_index][i], &gamma_squared)
                            + mul_0_1_optimized(&polynomials.E_polys[memory_index][i], gamma)
                            + polynomials.dim[dim_index][i]
                            - *tau
                    })
                    .collect();
                let write_fingerprints = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| *read_fingerprint + gamma_squared)
                    .collect();

                vec![
                    DensePolynomial::new(read_fingerprints),
                    DensePolynomial::new(write_fingerprints),
                ]
            })
            .collect();

        let init_final_leaves = (0..Self::num_memories())
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                let subtable_index = Self::memory_to_subtable_index(memory_index);
                // TODO(moodlezoup): Only need one init polynomial per subtable
                let init_fingerprints: Vec<F> = (0..M)
                    .map(|i| {
                        // 0 * gamma^2 +
                        mul_0_1_optimized(
                            &preprocessing.materialized_subtables[subtable_index][i],
                            gamma,
                        ) + F::from_u64(i as u64).unwrap()
                            - *tau
                    })
                    .collect();
                let final_fingerprints = init_fingerprints
                    .iter()
                    .enumerate()
                    .map(|(i, init_fingerprint)| {
                        *init_fingerprint
                            + mul_0_1_optimized(
                                &polynomials.final_cts[dim_index][i],
                                &gamma_squared,
                            )
                    })
                    .collect();

                vec![
                    DensePolynomial::new(init_fingerprints),
                    DensePolynomial::new(final_fingerprints),
                ]
            })
            .collect();

        (read_write_leaves, init_final_leaves)
    }

    fn protocol_name() -> &'static [u8] {
        b"Surge memory checking"
    }
}

impl<F, G, Instruction, const C: usize, const M: usize>
    MemoryCheckingVerifier<F, G, SurgePolys<F, G>> for SurgeProof<F, G, Instruction, C, M>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Instruction: JoltInstruction + Default + Sync,
{
    fn read_tuples(
        _preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..Self::num_memories())
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    openings.dim_openings[dim_index],
                    openings.E_poly_openings[memory_index],
                    openings.read_openings[dim_index],
                )
            })
            .collect()
    }
    fn write_tuples(
        _preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..Self::num_memories())
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    openings.dim_openings[dim_index],
                    openings.E_poly_openings[memory_index],
                    openings.read_openings[dim_index] + F::one(),
                )
            })
            .collect()
    }
    fn init_tuples(
        _preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Self::num_memories())
            .map(|memory_index| {
                (
                    a_init,
                    v_init[Self::memory_to_subtable_index(memory_index)],
                    F::zero(),
                )
            })
            .collect()
    }
    fn final_tuples(
        _preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Self::num_memories())
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    a_init,
                    v_init[Self::memory_to_subtable_index(memory_index)],
                    openings.final_openings[dim_index],
                )
            })
            .collect()
    }
}

pub struct SurgePrimarySumcheck<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    sumcheck_proof: SumcheckInstanceProof<F>,
    num_rounds: usize,
    claimed_evaluation: F,
    openings: PrimarySumcheckOpenings<F>,
    opening_proof: ConcatenatedPolynomialOpeningProof<G>,
}

pub struct SurgePreprocessing<F, Instruction, const C: usize, const M: usize>
where
    F: PrimeField,
    Instruction: JoltInstruction + Default,
{
    _instruction: PhantomData<Instruction>,
    materialized_subtables: Vec<Vec<F>>,
}

pub struct SurgeProof<F, G, Instruction, const C: usize, const M: usize>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Instruction: JoltInstruction + Default,
{
    /// Commitments to all polynomials
    commitment: SurgeCommitment<G>,

    /// Primary collation sumcheck proof
    primary_sumcheck: SurgePrimarySumcheck<F, G>,

    memory_checking: MemoryCheckingProof<
        G,
        SurgePolys<F, G>,
        SurgeReadWriteOpenings<F>,
        SurgeFinalOpenings<F, Instruction, C, M>,
    >,
}

impl<F, Instruction, const C: usize, const M: usize> SurgePreprocessing<F, Instruction, C, M>
where
    F: PrimeField,
    Instruction: JoltInstruction + Default + Sync,
{
    #[tracing::instrument(skip_all, name = "Surge::preprocess")]
    pub fn preprocess() -> Self {
        let instruction = Instruction::default();

        let materialized_subtables = instruction
            .subtables(C, M)
            .par_iter()
            .map(|(subtable, _)| subtable.materialize(M))
            .collect();

        Self {
            _instruction: PhantomData,
            materialized_subtables,
        }
    }
}

impl<F, G, Instruction, const C: usize, const M: usize> SurgeProof<F, G, Instruction, C, M>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Instruction: JoltInstruction + Default + Sync,
{
    #[tracing::instrument(skip_all, name = "Surge::preprocess")]

    fn num_memories() -> usize {
        C * Instruction::default().subtables::<F>(C, M).len()
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, NUM_SUBTABLES)
    fn memory_to_subtable_index(i: usize) -> usize {
        i / C
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, C)
    fn memory_to_dimension_index(i: usize) -> usize {
        i % C
    }

    fn protocol_name() -> &'static [u8] {
        b"Surge"
    }

    /// Computes the maximum number of group generators needed to commit to Surge polynomials
    /// using Hyrax, given `M` and the maximum number of lookups.
    pub fn num_generators(max_num_lookups: usize) -> usize {
        let dim_read_num_vars = (max_num_lookups * 2 * C).log_2();
        let final_num_vars = (M * C).log_2();
        let E_num_vars = (max_num_lookups * Self::num_memories()).log_2();

        let max_num_vars =
            std::cmp::max(std::cmp::max(dim_read_num_vars, final_num_vars), E_num_vars);
        matrix_dimensions(max_num_vars, 1).1
    }

    #[tracing::instrument(skip_all, name = "Surge::prove")]
    pub fn prove(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        ops: Vec<Instruction>,
        transcript: &mut Transcript,
    ) -> Self {
        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        let num_lookups = ops.len().next_power_of_two();
        let polynomials = Self::construct_polys(preprocessing, &ops);
        let batched_polys: BatchedSurgePolynomials<F> = polynomials.batch();
        // TODO(sragss): Move upstream
        let pedersen_generators =
            PedersenGenerators::new(Self::num_generators(num_lookups), b"LassoV1");
        let commitment = polynomials.commit(&batched_polys, &pedersen_generators);

        let num_rounds = num_lookups.log_2();
        let instruction = Instruction::default();

        // TODO(sragss): Commit some of this stuff to transcript?

        // Primary sumcheck
        let r_primary_sumcheck = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"primary_sumcheck",
            num_rounds,
        );
        let eq: DensePolynomial<F> =
            DensePolynomial::new(EqPolynomial::new(r_primary_sumcheck.to_vec()).evals());
        let sumcheck_claim: F = Self::compute_primary_sumcheck_claim(&polynomials, &eq);

        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"sumcheck_claim",
            &sumcheck_claim,
        );
        let mut combined_sumcheck_polys = polynomials.E_polys.clone();
        combined_sumcheck_polys.push(eq);

        let combine_lookups_eq = |vals: &[F]| -> F {
            let vals_no_eq: &[F] = &vals[0..(vals.len() - 1)];
            let eq = vals[vals.len() - 1];
            instruction.combine_lookups(vals_no_eq, C, M) * eq
        };

        let (primary_sumcheck_proof, r_z, _) =
            SumcheckInstanceProof::<F>::prove_arbitrary::<_, G, Transcript>(
                &sumcheck_claim,
                num_rounds,
                &mut combined_sumcheck_polys,
                combine_lookups_eq,
                instruction.g_poly_degree(C) + 1, // combined degree + eq term
                transcript,
            );

        let sumcheck_openings = PrimarySumcheckOpenings::open(&polynomials, &r_z); // TODO: use return value from prove_arbitrary?
                                                                                   // Create a single opening proof for the E polynomials
        let sumcheck_opening_proof = PrimarySumcheckOpenings::prove_openings(
            &polynomials,
            &batched_polys,
            &r_z,
            &sumcheck_openings,
            transcript,
        );

        let primary_sumcheck = SurgePrimarySumcheck {
            claimed_evaluation: sumcheck_claim,
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            openings: sumcheck_openings,
            opening_proof: sumcheck_opening_proof,
        };

        let memory_checking = SurgeProof::prove_memory_checking(
            &preprocessing,
            &polynomials,
            &batched_polys,
            transcript,
        );

        SurgeProof {
            commitment,
            primary_sumcheck,
            memory_checking,
        }
    }

    pub fn verify(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        proof: SurgeProof<F, G, Instruction, C, M>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());
        let instruction = Instruction::default();

        let r_primary_sumcheck = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"primary_sumcheck",
            proof.primary_sumcheck.num_rounds,
        );

        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"sumcheck_claim",
            &proof.primary_sumcheck.claimed_evaluation,
        );
        let primary_sumcheck_poly_degree = instruction.g_poly_degree(C) + 1;
        let (claim_last, r_z) = proof
            .primary_sumcheck
            .sumcheck_proof
            .verify::<G, Transcript>(
                proof.primary_sumcheck.claimed_evaluation,
                proof.primary_sumcheck.num_rounds,
                primary_sumcheck_poly_degree,
                transcript,
            )?;

        let eq_eval = EqPolynomial::new(r_primary_sumcheck.to_vec()).evaluate(&r_z);
        assert_eq!(
            eq_eval * instruction.combine_lookups(&proof.primary_sumcheck.openings, C, M),
            claim_last,
            "Primary sumcheck check failed."
        );

        proof.primary_sumcheck.openings.verify_openings(
            &proof.primary_sumcheck.opening_proof,
            &proof.commitment,
            &r_z,
            transcript,
        )?;

        Self::verify_memory_checking(
            preprocessing,
            proof.memory_checking,
            &proof.commitment,
            transcript,
        )
    }

    #[tracing::instrument(skip_all, name = "Surge::construct_polys")]
    fn construct_polys(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        ops: &Vec<Instruction>,
    ) -> SurgePolys<F, G> {
        let num_lookups = ops.len().next_power_of_two();
        let mut dim_usize: Vec<Vec<usize>> = vec![vec![0; num_lookups]; C];

        let mut read_cts = vec![vec![0usize; num_lookups]; C];
        let mut final_cts = vec![vec![0usize; M]; C];
        let log_M = ark_std::log2(M) as usize;

        for (op_index, op) in ops.iter().enumerate() {
            let access_sequence = op.to_indices(C, log_M);
            assert_eq!(access_sequence.len(), C);

            for dimension_index in 0..C {
                let memory_address = access_sequence[dimension_index];
                debug_assert!(memory_address < M);

                dim_usize[dimension_index][op_index] = memory_address;

                let ts = final_cts[dimension_index][memory_address];
                read_cts[dimension_index][op_index] = ts;
                let write_timestamp = ts + 1;
                final_cts[dimension_index][memory_address] = write_timestamp;
            }
        }

        // num_ops is padded to the nearest power of 2 for the usage of DensePolynomial. We cannot just fill
        // in zeros for read_cts and final_cts as this implicitly specifies a read at address 0. The prover
        // and verifier plumbing assume write_ts(r) = read_ts(r) + 1. This will not hold unless we update
        // the final_cts for these phantom reads.
        for fake_ops_index in ops.len()..num_lookups {
            for dimension_index in 0..C {
                let memory_address = 0;
                let ts = final_cts[dimension_index][memory_address];
                read_cts[dimension_index][fake_ops_index] = ts;
                let write_timestamp = ts + 1;
                final_cts[dimension_index][memory_address] = write_timestamp;
            }
        }

        let dim: Vec<DensePolynomial<F>> = dim_usize
            .iter()
            .map(|dim| DensePolynomial::from_usize(dim))
            .collect();
        let read_cts: Vec<DensePolynomial<F>> = read_cts
            .iter()
            .map(|read| DensePolynomial::from_usize(read))
            .collect();
        let final_cts: Vec<DensePolynomial<F>> = final_cts
            .iter()
            .map(|fin| DensePolynomial::from_usize(fin))
            .collect();

        // Construct E
        let mut E_i_evals = Vec::with_capacity(Self::num_memories());
        for E_index in 0..Self::num_memories() {
            let mut E_evals = Vec::with_capacity(num_lookups);
            for op_index in 0..num_lookups {
                let dimension_index = Self::memory_to_dimension_index(E_index);
                let subtable_index = Self::memory_to_subtable_index(E_index);

                let eval_index = dim_usize[dimension_index][op_index];
                let eval = preprocessing.materialized_subtables[subtable_index][eval_index];
                E_evals.push(eval);
            }
            E_i_evals.push(E_evals);
        }
        let E_poly: Vec<DensePolynomial<F>> = E_i_evals
            .iter()
            .map(|E| DensePolynomial::new(E.to_vec()))
            .collect();

        SurgePolys {
            _group: PhantomData,
            dim,
            read_cts,
            final_cts,
            E_polys: E_poly,
        }
    }

    #[tracing::instrument(skip_all, name = "Surge::compute_primary_sumcheck_claim")]
    fn compute_primary_sumcheck_claim(polys: &SurgePolys<F, G>, eq: &DensePolynomial<F>) -> F {
        let g_operands = &polys.E_polys;
        let hypercube_size = g_operands[0].len();
        g_operands
            .iter()
            .for_each(|operand| assert_eq!(operand.len(), hypercube_size));

        let instruction = Instruction::default();

        (0..hypercube_size)
            .into_par_iter()
            .map(|eval_index| {
                let g_operands: Vec<F> = (0..Self::num_memories())
                    .map(|memory_index| g_operands[memory_index][eval_index])
                    .collect();
                eq[eval_index] * instruction.combine_lookups(&g_operands, C, M)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use merlin::Transcript;

    use super::SurgePreprocessing;
    use crate::{jolt::instruction::xor::XORInstruction, lasso::surge::SurgeProof};
    use ark_curve25519::{EdwardsProjective, Fr};

    #[test]
    fn e2e() {
        let ops = vec![
            XORInstruction(12, 12),
            XORInstruction(12, 82),
            XORInstruction(12, 12),
            XORInstruction(25, 12),
        ];
        const C: usize = 8;
        const M: usize = 1 << 8;

        let mut transcript = Transcript::new(b"test_transcript");
        let preprocessing = SurgePreprocessing::preprocess();
        let proof = SurgeProof::<Fr, EdwardsProjective, XORInstruction, C, M>::prove(
            &preprocessing,
            ops,
            &mut transcript,
        );

        let mut transcript = Transcript::new(b"test_transcript");
        SurgeProof::verify(&preprocessing, proof, &mut transcript).expect("should work");
    }

    #[test]
    fn e2e_non_pow_2() {
        let ops = vec![
            XORInstruction(0, 1),
            XORInstruction(101, 101),
            XORInstruction(202, 1),
            XORInstruction(220, 1),
            XORInstruction(220, 1),
        ];
        const C: usize = 2;
        const M: usize = 1 << 8;

        let mut transcript = Transcript::new(b"test_transcript");
        let preprocessing = SurgePreprocessing::preprocess();
        let proof = SurgeProof::<Fr, EdwardsProjective, XORInstruction, C, M>::prove(
            &preprocessing,
            ops,
            &mut transcript,
        );

        let mut transcript = Transcript::new(b"test_transcript");
        SurgeProof::verify(&preprocessing, proof, &mut transcript).expect("should work");
    }
}
