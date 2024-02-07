use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::interleave;
use merlin::Transcript;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::any::TypeId;
use std::marker::PhantomData;
use strum::{EnumCount, IntoEnumIterator};
use tracing::trace_span;

use rayon::prelude::*;

use crate::lasso::memory_checking::MultisetHashes;
use crate::poly::hyrax::HyraxGenerators;
use crate::utils::{mul_0_1_optimized, split_poly_flagged};
use crate::{
    jolt::{
        instruction::{JoltInstruction, Opcode},
        subtable::LassoSubtable,
    },
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        structured_poly::{BatchablePolynomials, StructuredOpeningProof},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::{
        batched_commitment::{BatchedPolynomialCommitment, BatchedPolynomialOpeningProof},
        grand_product::{BatchedGrandProductCircuit, GrandProductCircuit},
        sumcheck::SumcheckInstanceProof,
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};

/// All polynomials associated with Jolt instruction lookups.
pub struct InstructionPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    _group: PhantomData<G>,
    /// `C` sized vector of `DensePolynomials` whose evaluations correspond to
    /// indices at which the memories will be evaluated. Each `DensePolynomial` has size
    /// `m` (# lookups).
    pub dim: Vec<DensePolynomial<F>>,

    /// `C` sized vector of `DensePolynomials` whose evaluations correspond to
    /// read access counts to the memory. Each `DensePolynomial` has size `m` (# lookups).
    pub read_cts: Vec<DensePolynomial<F>>,

    /// `C` sized vector of `DensePolynomials` whose evaluations correspond to
    /// final access counts to the memory. Each `DensePolynomial` has size M, AKA subtable size.
    pub final_cts: Vec<DensePolynomial<F>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// the evaluation of memory accessed at each step of the CPU. Each `DensePolynomial` has
    /// size `m` (# lookups).
    pub E_polys: Vec<DensePolynomial<F>>,

    /// Polynomial encodings for flag polynomials for each instruction.
    /// If using a single instruction this will be empty.
    /// NUM_INSTRUCTIONS sized, each polynomial of length 'm' (# lookups).
    ///
    /// Stored independently for use in sumcheck, combined into single DensePolynomial for commitment.
    pub instruction_flag_polys: Vec<DensePolynomial<F>>,
}

/// Batched version of InstructionPolynomials.
pub struct BatchedInstructionPolynomials<F: PrimeField> {
    /// dim_i and read_cts_i polynomials, batched together.
    batched_dim_read: DensePolynomial<F>,
    /// final_cts_i polynomials, batched together.
    batched_final: DensePolynomial<F>,
    /// E_i polynomials, batched together.
    batched_E: DensePolynomial<F>,
    /// flag polynomials, batched together.
    batched_flag: DensePolynomial<F>,
}

/// Commitments to BatchedInstructionPolynomials.
pub struct InstructionCommitment<G: CurveGroup> {
    /// Commitment to dim_i and read_cts_i polynomials.
    pub dim_read_commitment: BatchedPolynomialCommitment<G>,
    /// Commitment to final_cts_i polynomials.
    pub final_commitment: BatchedPolynomialCommitment<G>,
    /// Commitment to E_i polynomials.
    pub E_commitment: BatchedPolynomialCommitment<G>,
    /// Commitment to flag polynomials.
    pub instruction_flag_commitment: BatchedPolynomialCommitment<G>,
}

/// Contains generators used to commit to InstructionPolynomials.
pub struct InstructionCommitmentGenerators<G: CurveGroup> {
    pub dim_read_commitment_gens: HyraxGenerators<G>,
    pub final_commitment_gens: HyraxGenerators<G>,
    pub E_commitment_gens: HyraxGenerators<G>,
    pub flag_commitment_gens: HyraxGenerators<G>,
}

// TODO: macro?
impl<F, G> BatchablePolynomials for InstructionPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type BatchedPolynomials = BatchedInstructionPolynomials<F>;
    type Commitment = InstructionCommitment<G>;

    #[tracing::instrument(skip_all, name = "InstructionPolynomials::batch")]
    fn batch(&self) -> Self::BatchedPolynomials {
        let (batched_dim_read, (batched_final, batched_E, batched_flag)) = rayon::join(
            || DensePolynomial::merge(self.dim.iter().chain(&self.read_cts)),
            || {
                let batched_final = DensePolynomial::merge(&self.final_cts);
                let (batched_E, batched_flag) = rayon::join(
                    || DensePolynomial::merge(&self.E_polys),
                    || DensePolynomial::merge(&self.instruction_flag_polys),
                );
                (batched_final, batched_E, batched_flag)
            },
        );

        Self::BatchedPolynomials {
            batched_dim_read,
            batched_final,
            batched_E,
            batched_flag,
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionPolynomials::commit")]
    fn commit(batched_polys: &Self::BatchedPolynomials) -> Self::Commitment {
        let dim_read_commitment = batched_polys
            .batched_dim_read
            .combined_commit(b"BatchedInstructionPolynomials.dim_read");
        let final_commitment = batched_polys
            .batched_final
            .combined_commit(b"BatchedInstructionPolynomials.final_cts");
        let E_commitment = batched_polys
            .batched_E
            .combined_commit(b"BatchedInstructionPolynomials.E_poly");
        let instruction_flag_commitment = batched_polys
            .batched_flag
            .combined_commit(b"BatchedInstructionPolynomials.flag");

        Self::Commitment {
            dim_read_commitment,
            final_commitment,
            E_commitment,
            instruction_flag_commitment,
        }
    }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
/// Polynomial openings associated with the "primary sumcheck" of Jolt instruction lookups.
struct PrimarySumcheckOpenings<F>
where
    F: PrimeField,
{
    /// Evaluations of the E_i polynomials at the opening point. Vector is of length NUM_MEMORIES.
    E_poly_openings: Vec<F>,
    /// Evaluations of the flag polynomials at the opening point. Vector is of length NUM_INSTRUCTIONS.
    flag_openings: Vec<F>,
}

struct PrimarySumcheckOpeningProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    E_poly_opening_proof: BatchedPolynomialOpeningProof<G>,
    flag_opening_proof: BatchedPolynomialOpeningProof<G>,
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>>
    StructuredOpeningProof<F, G, InstructionPolynomials<F, G>> for PrimarySumcheckOpenings<F>
{
    type Proof = PrimarySumcheckOpeningProof<F, G>;

    fn open(_polynomials: &InstructionPolynomials<F, G>, _opening_point: &Vec<F>) -> Self {
        unimplemented!("Openings are output by sumcheck protocol");
    }

    #[tracing::instrument(skip_all, name = "PrimarySumcheckOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedInstructionPolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        let E_poly_opening_proof = BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_E,
            opening_point,
            &openings.E_poly_openings,
            transcript,
        );
        let flag_opening_proof = BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_flag,
            opening_point,
            &openings.flag_openings,
            transcript,
        );

        PrimarySumcheckOpeningProof {
            E_poly_opening_proof,
            flag_opening_proof,
        }
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &InstructionCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        opening_proof.E_poly_opening_proof.verify(
            opening_point,
            &self.E_poly_openings,
            &commitment.E_commitment,
            transcript,
        )?;
        opening_proof.flag_opening_proof.verify(
            opening_point,
            &self.flag_openings,
            &commitment.instruction_flag_commitment,
            transcript,
        )?;

        Ok(())
    }
}

pub struct InstructionReadWriteOpenings<F>
where
    F: PrimeField,
{
    /// Evaluations of the dim_i polynomials at the opening point. Vector is of length C.
    dim_openings: Vec<F>,
    /// Evaluations of the read_cts_i polynomials at the opening point. Vector is of length NUM_MEMORIES.
    read_openings: Vec<F>,
    /// Evaluations of the E_i polynomials at the opening point. Vector is of length NUM_MEMORIES.
    E_poly_openings: Vec<F>,
    /// Evaluations of the flag polynomials at the opening point. Vector is of length NUM_INSTRUCTIONS.
    flag_openings: Vec<F>,
}

pub struct InstructionReadWriteOpeningProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    dim_read_opening_proof: BatchedPolynomialOpeningProof<G>,
    E_poly_opening_proof: BatchedPolynomialOpeningProof<G>,
    flag_opening_proof: BatchedPolynomialOpeningProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, InstructionPolynomials<F, G>>
    for InstructionReadWriteOpenings<F>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Proof = InstructionReadWriteOpeningProof<F, G>;

    #[tracing::instrument(skip_all, name = "InstructionReadWriteOpenings::open")]
    fn open(polynomials: &InstructionPolynomials<F, G>, opening_point: &Vec<F>) -> Self {
        // All of these evaluations share the lagrange basis polynomials.
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();

        let dim_openings = polynomials
            .dim
            .par_iter()
            .map(|poly| poly.evaluate_at_chi_low_optimized(&chis))
            .collect();
        let read_openings = polynomials
            .read_cts
            .par_iter()
            .map(|poly| poly.evaluate_at_chi_low_optimized(&chis))
            .collect();
        let E_poly_openings = polynomials
            .E_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi_low_optimized(&chis))
            .collect();
        let flag_openings = polynomials
            .instruction_flag_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi_low_optimized(&chis))
            .collect();

        Self {
            dim_openings,
            read_openings,
            E_poly_openings,
            flag_openings,
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionReadWriteOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedInstructionPolynomials<F>,
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

        let dim_read_opening_proof = BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_dim_read,
            &opening_point,
            &dim_read_openings,
            transcript,
        );
        let E_poly_opening_proof = BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_E,
            &opening_point,
            &openings.E_poly_openings,
            transcript,
        );
        let flag_opening_proof = BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_flag,
            &opening_point,
            &openings.flag_openings,
            transcript,
        );

        InstructionReadWriteOpeningProof {
            dim_read_opening_proof,
            E_poly_opening_proof,
            flag_opening_proof,
        }
    }

    fn verify_openings(
        &self,
        openings_proof: &Self::Proof,
        commitment: &InstructionCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let mut dim_read_openings = [self.dim_openings.as_slice(), self.read_openings.as_slice()]
            .concat()
            .to_vec();
        dim_read_openings.resize(dim_read_openings.len().next_power_of_two(), F::zero());

        openings_proof.dim_read_opening_proof.verify(
            opening_point,
            &dim_read_openings,
            &commitment.dim_read_commitment,
            transcript,
        )?;

        openings_proof.E_poly_opening_proof.verify(
            opening_point,
            &self.E_poly_openings,
            &commitment.E_commitment,
            transcript,
        )?;

        openings_proof.flag_opening_proof.verify(
            opening_point,
            &self.flag_openings,
            &commitment.instruction_flag_commitment,
            transcript,
        )?;
        Ok(())
    }
}

pub struct InstructionFinalOpenings<F, Subtables>
where
    F: PrimeField,
    Subtables: LassoSubtable<F> + IntoEnumIterator,
{
    _subtables: PhantomData<Subtables>,
    /// Evaluations of the final_cts_i polynomials at the opening point. Vector is of length NUM_MEMORIES.
    final_openings: Vec<F>,
    /// Evaluation of the a_init/final polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    a_init_final: Option<F>,
    /// Evaluation of the v_init/final polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    v_init_final: Option<Vec<F>>,
}

impl<F, G, Subtables> StructuredOpeningProof<F, G, InstructionPolynomials<F, G>>
    for InstructionFinalOpenings<F, Subtables>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Subtables: LassoSubtable<F> + IntoEnumIterator,
{
    #[tracing::instrument(skip_all, name = "InstructionFinalOpenings::open")]
    fn open(polynomials: &InstructionPolynomials<F, G>, opening_point: &Vec<F>) -> Self {
        // All of these evaluations share the lagrange basis polynomials.
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        let final_openings = polynomials
            .final_cts
            .par_iter()
            .map(|final_cts_i| final_cts_i.evaluate_at_chi_low_optimized(&chis))
            .collect();
        Self {
            _subtables: PhantomData,
            final_openings,
            a_init_final: None,
            v_init_final: None,
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionFinalOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedInstructionPolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_final,
            &opening_point,
            &openings.final_openings,
            transcript,
        )
    }

    fn compute_verifier_openings(&mut self, opening_point: &Vec<F>) {
        self.a_init_final =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
        self.v_init_final = Some(
            Subtables::iter()
                .map(|subtable| subtable.evaluate_mle(opening_point))
                .collect(),
        );
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &InstructionCommitment<G>,
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

impl<F, G, InstructionSet, Subtables, const C: usize, const M: usize>
    MemoryCheckingProver<F, G, InstructionPolynomials<F, G>>
    for InstructionLookups<F, G, InstructionSet, Subtables, C, M>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount,
    Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>,
{
    type ReadWriteOpenings = InstructionReadWriteOpenings<F>;
    type InitFinalOpenings = InstructionFinalOpenings<F, Subtables>;

    type MemoryTuple = (F, F, F, Option<F>); // (a, v, t, flag)

    fn fingerprint(inputs: &(F, F, F, Option<F>), gamma: &F, tau: &F) -> F {
        let (a, v, t, flag) = *inputs;
        match flag {
            Some(val) => val * (t * gamma.square() + v * *gamma + a - tau) + F::one() - val,
            None => t * gamma.square() + v * *gamma + a - tau,
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionLookups::compute_leaves")]
    fn compute_leaves(
        &self,
        polynomials: &InstructionPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) {
        let gamma_squared = gamma.square();

        let read_write_leaves = (0..Self::NUM_MEMORIES)
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);

                let read_fingerprints: Vec<F> = (0..self.num_lookups)
                    .map(|i| {
                        let a = &polynomials.dim[dim_index][i];
                        let v = &polynomials.E_polys[memory_index][i];
                        let t = &polynomials.read_cts[memory_index][i];
                        mul_0_1_optimized(t, &gamma_squared) + mul_0_1_optimized(v, gamma) + a - tau
                    })
                    .collect();
                let write_fingerprints = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| *read_fingerprint + gamma_squared)
                    .collect();
                [
                    DensePolynomial::new(read_fingerprints),
                    DensePolynomial::new(write_fingerprints),
                ]
            })
            .collect();

        let init_final_leaves: Vec<DensePolynomial<F>> = self
            .materialized_subtables
            .par_iter()
            .enumerate()
            .flat_map_iter(|(subtable_index, subtable)| {
                let init_fingerprints: Vec<F> = (0..M)
                    .map(|i| {
                        let a = &F::from(i as u64);
                        let v = &subtable[i];
                        // let t = F::zero();
                        // Compute h(a,v,t) where t == 0
                        mul_0_1_optimized(v, gamma) + a - tau
                    })
                    .collect();

                let final_leaves: Vec<DensePolynomial<F>> = (0..C)
                    .map(|dim_index| {
                        let memory_index = C * subtable_index + dim_index;
                        let final_cts = &polynomials.final_cts[memory_index];
                        let final_fingerprints = (0..M)
                            .map(|i| {
                                init_fingerprints[i]
                                    + mul_0_1_optimized(&final_cts[i], &gamma_squared)
                            })
                            .collect();
                        DensePolynomial::new(final_fingerprints)
                    })
                    .collect();

                let mut polys = Vec::with_capacity(C + 1);
                polys.push(DensePolynomial::new(init_fingerprints));
                polys.extend(final_leaves);
                polys
            })
            .collect();

        (read_write_leaves, init_final_leaves)
    }

    fn interleave_hashes(multiset_hashes: &MultisetHashes<F>) -> (Vec<F>, Vec<F>) {
        // R W R W R W ...
        let read_write_hashes = interleave(
            multiset_hashes.read_hashes.clone(),
            multiset_hashes.write_hashes.clone(),
        )
        .collect();

        // I F F F F I F F F F ...
        let mut init_final_hashes = Vec::with_capacity(
            multiset_hashes.init_hashes.len() + multiset_hashes.final_hashes.len(),
        );
        for subtable_index in 0..Self::NUM_SUBTABLES {
            init_final_hashes.push(multiset_hashes.init_hashes[subtable_index]);
            init_final_hashes.extend_from_slice(
                &multiset_hashes.final_hashes[C * subtable_index..C * (subtable_index + 1)],
            );
        }

        (read_write_hashes, init_final_hashes)
    }

    fn uninterleave_hashes(
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        assert_eq!(read_write_hashes.len(), 2 * Self::NUM_MEMORIES);
        assert_eq!(
            init_final_hashes.len(),
            Self::NUM_SUBTABLES + Self::NUM_MEMORIES
        );

        let mut read_hashes = Vec::with_capacity(Self::NUM_MEMORIES);
        let mut write_hashes = Vec::with_capacity(Self::NUM_MEMORIES);
        for i in 0..Self::NUM_MEMORIES {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        let mut init_hashes = Vec::with_capacity(Self::NUM_SUBTABLES);
        let mut final_hashes = Vec::with_capacity(Self::NUM_MEMORIES);
        let mut init_final_hashes = init_final_hashes.iter();
        for _ in 0..Self::NUM_SUBTABLES {
            // I
            init_hashes.push(*init_final_hashes.next().unwrap());
            // F F F F
            for _ in 0..C {
                final_hashes.push(*init_final_hashes.next().unwrap());
            }
        }

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        }
    }

    fn check_multiset_equality(multiset_hashes: &MultisetHashes<F>) {
        assert_eq!(multiset_hashes.init_hashes.len(), Self::NUM_SUBTABLES);
        assert_eq!(multiset_hashes.read_hashes.len(), Self::NUM_MEMORIES);
        assert_eq!(multiset_hashes.write_hashes.len(), Self::NUM_MEMORIES);
        assert_eq!(multiset_hashes.final_hashes.len(), Self::NUM_MEMORIES);

        (0..Self::NUM_MEMORIES).into_par_iter().for_each(|i| {
            let read_hash = multiset_hashes.read_hashes[i];
            let write_hash = multiset_hashes.write_hashes[i];
            let init_hash = multiset_hashes.init_hashes[Self::memory_to_subtable_index(i)];
            let final_hash = multiset_hashes.final_hashes[i];
            assert_eq!(
                init_hash * write_hash,
                final_hash * read_hash,
                "Multiset hashes don't match"
            );
        });
    }

    /// Overrides default implementation to handle flags
    #[tracing::instrument(skip_all, name = "InstructionLookups::read_write_grand_product")]
    fn read_write_grand_product(
        &self,
        polynomials: &InstructionPolynomials<F, G>,
        read_write_leaves: Vec<DensePolynomial<F>>,
    ) -> (BatchedGrandProductCircuit<F>, Vec<F>) {
        assert_eq!(read_write_leaves.len(), 2 * Self::NUM_MEMORIES);

        let _span = trace_span!("InstructionLookups: construct circuits");
        let _enter = _span.enter();

        let subtable_flag_polys = Self::subtable_flag_polys(&polynomials.instruction_flag_polys);

        let read_write_circuits = read_write_leaves
            .par_iter()
            .enumerate()
            .map(|(i, leaves_poly)| {
                // Split while cloning to save on future cloning in GrandProductCircuit
                let subtable_index = Self::memory_to_subtable_index(i / 2);
                let flag = &subtable_flag_polys[subtable_index];
                let (toggled_leaves_l, toggled_leaves_r) = split_poly_flagged(&leaves_poly, &flag);
                GrandProductCircuit::new_split(
                    DensePolynomial::new(toggled_leaves_l),
                    DensePolynomial::new(toggled_leaves_r),
                )
            })
            .collect::<Vec<GrandProductCircuit<F>>>();

        drop(_enter);
        drop(_span);

        let _span = trace_span!("InstructionLookups: compute hashes");
        let _enter = _span.enter();

        let read_write_hashes: Vec<F> = read_write_circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();

        drop(_enter);
        drop(_span);

        let _span = trace_span!("InstructionLookups: the rest");
        let _enter = _span.enter();

        // self.memory_to_subtable map has to be expanded because we've doubled the number of "grand products memorys": [read_0, write_0, ... read_NUM_MEMOREIS, write_NUM_MEMORIES]
        let expanded_flag_map: Vec<usize> = (0..2 * Self::NUM_MEMORIES)
            .map(|i| Self::memory_to_subtable_index(i / 2))
            .collect();

        // Prover has access to subtable_flag_polys, which are uncommitted, but verifier can derive from instruction_flag commitments.
        let batched_circuits = BatchedGrandProductCircuit::new_batch_flags(
            read_write_circuits,
            subtable_flag_polys,
            expanded_flag_map,
            read_write_leaves,
        );

        drop(_enter);
        drop(_span);

        (batched_circuits, read_write_hashes)
    }

    fn protocol_name() -> &'static [u8] {
        b"Instruction lookups memory checking"
    }
}

impl<F, G, InstructionSet, Subtables, const C: usize, const M: usize>
    MemoryCheckingVerifier<F, G, InstructionPolynomials<F, G>>
    for InstructionLookups<F, G, InstructionSet, Subtables, C, M>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount,
    Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>,
{
    fn read_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
        let subtable_flags = Self::subtable_flags(&openings.flag_openings);
        (0..Self::NUM_MEMORIES)
            .map(|memory_index| {
                let subtable_index = Self::memory_to_subtable_index(memory_index);
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    openings.dim_openings[dim_index],
                    openings.E_poly_openings[memory_index],
                    openings.read_openings[memory_index],
                    Some(subtable_flags[subtable_index]),
                )
            })
            .collect()
    }
    fn write_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
        Self::read_tuples(openings)
            .iter()
            .map(|(a, v, t, flag)| (*a, *v, *t + F::one(), *flag))
            .collect()
    }
    fn init_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Self::NUM_SUBTABLES)
            .map(|subtable_index| (a_init, v_init[subtable_index], F::zero(), None))
            .collect()
    }
    fn final_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Self::NUM_MEMORIES)
            .map(|memory_index| {
                (
                    a_init,
                    v_init[Self::memory_to_subtable_index(memory_index)],
                    openings.final_openings[memory_index],
                    None,
                )
            })
            .collect()
    }
}

/// Proof of instruction lookups for a single Jolt program execution.
pub struct InstructionLookupsProof<F, G, Subtables>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Subtables: LassoSubtable<F> + IntoEnumIterator,
{
    /// "Primary" sumcheck, i.e. proving \sum_x \tilde{eq}(r, x) * \sum_i flag_i(x) * g_i(E_1(x), ..., E_\alpha(x))
    primary_sumcheck: PrimarySumcheck<F, G>,

    /// Memory checking proof, showing that E_i polynomials are well-formed.
    memory_checking: MemoryCheckingProof<
        G,
        InstructionPolynomials<F, G>,
        InstructionReadWriteOpenings<F>,
        InstructionFinalOpenings<F, Subtables>,
    >,
}

pub struct PrimarySumcheck<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    sumcheck_proof: SumcheckInstanceProof<F>,
    num_rounds: usize,
    claimed_evaluation: F,
    openings: PrimarySumcheckOpenings<F>,
    opening_proof: PrimarySumcheckOpeningProof<F, G>,
}

pub struct InstructionLookups<F, G, InstructionSet, Subtables, const C: usize, const M: usize>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount,
    Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>,
{
    _field: PhantomData<F>,
    _group: PhantomData<G>,
    _instructions: PhantomData<InstructionSet>,
    _subtables: PhantomData<Subtables>,
    ops: Vec<InstructionSet>,
    materialized_subtables: Vec<Vec<F>>,
    num_lookups: usize,
}

impl<F, G, InstructionSet, Subtables, const C: usize, const M: usize>
    InstructionLookups<F, G, InstructionSet, Subtables, C, M>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount,
    Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>,
{
    const NUM_SUBTABLES: usize = Subtables::COUNT;
    const NUM_INSTRUCTIONS: usize = InstructionSet::COUNT;
    const NUM_MEMORIES: usize = C * Subtables::COUNT;

    #[tracing::instrument(skip_all, name = "InstructionLookups::new")]
    pub fn new(ops: Vec<InstructionSet>) -> Self {
        let materialized_subtables = Self::materialize_subtables();
        let num_lookups = ops.len().next_power_of_two();

        Self {
            _field: PhantomData,
            _group: PhantomData,
            _instructions: PhantomData,
            _subtables: PhantomData,
            ops,
            materialized_subtables,
            num_lookups,
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionLookups::prove_lookups")]
    pub fn prove_lookups(
        &self,
        transcript: &mut Transcript,
    ) -> (
        InstructionLookupsProof<F, G, Subtables>,
        InstructionPolynomials<F, G>,
        InstructionCommitment<G>,
    ) {
        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        let polynomials = self.polynomialize();
        let batched_polys = polynomials.batch();
        let commitment = InstructionPolynomials::commit(&batched_polys);

        commitment
            .E_commitment
            .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

        let r_eq = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"Jolt instruction lookups",
            self.ops.len().log_2(),
        );

        let eq = EqPolynomial::new(r_eq.to_vec());
        let sumcheck_claim = Self::compute_sumcheck_claim(&self.ops, &polynomials.E_polys, &eq);

        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"claim_eval_scalar_product",
            &sumcheck_claim,
        );

        let mut eq_poly = DensePolynomial::new(EqPolynomial::new(r_eq).evals());
        let num_rounds = self.ops.len().log_2();

        // TODO: compartmentalize all primary sumcheck logic

        let (primary_sumcheck_proof, r_primary_sumcheck, flag_evals, E_evals) =
            Self::prove_primary_sumcheck(
                &F::zero(),
                num_rounds,
                &mut eq_poly,
                &polynomials.E_polys,
                &polynomials.instruction_flag_polys,
                Self::sumcheck_poly_degree(),
                transcript,
            );

        // Create a single opening proof for the flag_evals and memory_evals
        let sumcheck_openings = PrimarySumcheckOpenings {
            E_poly_openings: E_evals,
            flag_openings: flag_evals,
        };
        let sumcheck_opening_proof = PrimarySumcheckOpenings::prove_openings(
            &batched_polys,
            &r_primary_sumcheck,
            &sumcheck_openings,
            transcript,
        );

        let primary_sumcheck = PrimarySumcheck {
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            claimed_evaluation: sumcheck_claim,
            openings: sumcheck_openings,
            opening_proof: sumcheck_opening_proof,
        };

        let memory_checking = self.prove_memory_checking(&polynomials, &batched_polys, transcript);

        (
            InstructionLookupsProof {
                primary_sumcheck,
                memory_checking,
            },
            polynomials,
            commitment,
        )
    }

    pub fn verify(
        proof: InstructionLookupsProof<F, G, Subtables>,
        commitment: InstructionCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        commitment
            .E_commitment
            .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

        let r_eq = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"Jolt instruction lookups",
            proof.primary_sumcheck.num_rounds,
        );

        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"claim_eval_scalar_product",
            &proof.primary_sumcheck.claimed_evaluation,
        );

        // TODO: compartmentalize all primary sumcheck logic
        let (claim_last, r_primary_sumcheck) = proof
            .primary_sumcheck
            .sumcheck_proof
            .verify::<G, Transcript>(
                proof.primary_sumcheck.claimed_evaluation,
                proof.primary_sumcheck.num_rounds,
                Self::sumcheck_poly_degree(),
                transcript,
            )?;

        // Verify that eq(r, r_z) * [f_1(r_z) * g(E_1(r_z)) + ... + f_F(r_z) * E_F(r_z))] = claim_last
        let eq_eval = EqPolynomial::new(r_eq.to_vec()).evaluate(&r_primary_sumcheck);
        assert_eq!(
            eq_eval
                * Self::combine_lookups(
                    &proof.primary_sumcheck.openings.E_poly_openings,
                    &proof.primary_sumcheck.openings.flag_openings,
                ),
            claim_last,
            "Primary sumcheck check failed."
        );

        proof.primary_sumcheck.openings.verify_openings(
            &proof.primary_sumcheck.opening_proof,
            &commitment,
            &r_primary_sumcheck,
            transcript,
        )?;

        Self::verify_memory_checking(proof.memory_checking, &commitment, transcript)?;

        Ok(())
    }

    /// Constructs the polynomials used in the primary sumcheck and memory checking.
    #[tracing::instrument(skip_all, name = "InstructionLookups::polynomialize")]
    fn polynomialize(&self) -> InstructionPolynomials<F, G> {
        let m: usize = self.ops.len().next_power_of_two();

        let subtable_lookup_indices: Vec<Vec<usize>> = Self::subtable_lookup_indices(&self.ops);

        let instruction_to_memory_indices_map: Vec<Vec<usize>> = InstructionSet::iter()
            .map(|op| Self::instruction_to_memory_indices(&op))
            .collect();
        let polys: Vec<(DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>)> = (0
            ..Self::NUM_MEMORIES)
            .into_par_iter()
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                let subtable_index = Self::memory_to_subtable_index(memory_index);
                let access_sequence: &Vec<usize> = &subtable_lookup_indices[dim_index];

                let mut final_cts_i = vec![0usize; M];
                let mut read_cts_i = vec![0usize; m];
                let mut subtable_lookups = vec![F::zero(); m];

                for (j, op) in self.ops.iter().enumerate() {
                    let memories_used: &Vec<usize> =
                        &instruction_to_memory_indices_map[op.to_opcode() as usize];
                    if memories_used.contains(&memory_index) {
                        let memory_address = access_sequence[j];
                        debug_assert!(memory_address < M);

                        let counter = final_cts_i[memory_address];
                        read_cts_i[j] = counter;
                        final_cts_i[memory_address] = counter + 1;
                        subtable_lookups[j] =
                            self.materialized_subtables[subtable_index][memory_address];
                    }
                }

                (
                    DensePolynomial::from_usize(&read_cts_i),
                    DensePolynomial::from_usize(&final_cts_i),
                    DensePolynomial::new(subtable_lookups),
                )
            })
            .collect();

        // Vec<(DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>)> -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>)
        let (read_cts, final_cts, E_polys): (
            Vec<DensePolynomial<F>>,
            Vec<DensePolynomial<F>>,
            Vec<DensePolynomial<F>>,
        ) = polys.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut read_acc, mut final_acc, mut E_acc), (read, f, E)| {
                read_acc.push(read);
                final_acc.push(f);
                E_acc.push(E);
                (read_acc, final_acc, E_acc)
            },
        );

        let dim: Vec<DensePolynomial<F>> = (0..C)
            .into_par_iter()
            .map(|i| {
                let access_sequence: &Vec<usize> = &subtable_lookup_indices[i];
                DensePolynomial::from_usize(access_sequence)
            })
            .collect();

        let mut instruction_flag_bitvectors: Vec<Vec<usize>> =
            vec![vec![0usize; m]; Self::NUM_INSTRUCTIONS];
        for (j, op) in self.ops.iter().enumerate() {
            let opcode_index = op.to_opcode() as usize;
            instruction_flag_bitvectors[opcode_index][j] = 1;
        }

        let instruction_flag_polys: Vec<DensePolynomial<F>> = instruction_flag_bitvectors
            .iter()
            .map(|flag_bitvector| DensePolynomial::from_usize(&flag_bitvector))
            .collect();

        InstructionPolynomials {
            _group: PhantomData,
            dim,
            read_cts,
            final_cts,
            instruction_flag_polys,
            E_polys,
        }
    }

    /// Prove Jolt primary sumcheck including instruction collation.
    ///
    /// Computes \sum{ eq(r,x) * [ flags_0(x) * g_0(E(x)) + flags_1(x) * g_1(E(x)) + ... + flags_{NUM_INSTRUCTIONS}(E(x)) * g_{NUM_INSTRUCTIONS}(E(x)) ]}
    /// via the sumcheck protocol.
    /// Note: These E(x) terms differ from term to term depending on the memories used in the instruction.
    ///
    /// Returns: (SumcheckProof, Random evaluation point, claimed evaluations of polynomials)
    ///
    /// Params:
    /// - `claim`: Claimed sumcheck evaluation.
    /// - `num_rounds`: Number of rounds to run sumcheck. Corresponds to the number of free bits or free variables in the polynomials.
    /// - `memory_polys`: Each of the `E` polynomials or "dereferenced memory" polynomials.
    /// - `flag_polys`: Each of the flag selector polynomials describing which instruction is used at a given step of the CPU.
    /// - `degree`: Degree of the inner sumcheck polynomial. Corresponds to number of evaluation points per round.
    /// - `transcript`: Fiat-shamir transcript.
    #[tracing::instrument(skip_all, name = "InstructionLookups::prove_primary_sumcheck")]
    fn prove_primary_sumcheck(
        _claim: &F,
        num_rounds: usize,
        eq_poly: &mut DensePolynomial<F>,
        memory_polys: &Vec<DensePolynomial<F>>,
        flag_polys: &Vec<DensePolynomial<F>>,
        degree: usize,
        transcript: &mut Transcript,
    ) -> (SumcheckInstanceProof<F>, Vec<F>, Vec<F>, Vec<F>) {
        // Check all polys are the same size
        let poly_len = eq_poly.len();
        for index in 0..Self::NUM_MEMORIES {
            debug_assert_eq!(memory_polys[index].len(), poly_len);
        }
        for index in 0..Self::NUM_INSTRUCTIONS {
            debug_assert_eq!(flag_polys[index].len(), poly_len);
        }

        let instruction_to_memory_indices_map: Vec<Vec<usize>> = InstructionSet::iter()
            .map(|op| Self::instruction_to_memory_indices(&op))
            .collect();

        let mut random_vars: Vec<F> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        let num_eval_points = degree + 1;

        let round_uni_poly = Self::primary_sumcheck_inner_loop(
            &eq_poly,
            &flag_polys,
            &memory_polys,
            num_eval_points,
            &instruction_to_memory_indices_map,
        );
        compressed_polys.push(round_uni_poly.compress());
        let r_j = Self::update_primary_sumcheck_transcript(round_uni_poly, transcript);
        random_vars.push(r_j);

        let _bind_span = trace_span!("BindPolys");
        let _bind_enter = _bind_span.enter();
        eq_poly.bound_poly_var_top(&r_j);
        let mut flag_polys_updated: Vec<DensePolynomial<F>> = flag_polys
            .par_iter()
            .map(|poly| poly.new_poly_from_bound_poly_var_top_flags(&r_j))
            .collect();
        let mut memory_polys_updated: Vec<DensePolynomial<F>> = memory_polys
            .par_iter()
            .map(|poly| poly.new_poly_from_bound_poly_var_top(&r_j))
            .collect();
        drop(_bind_enter);
        drop(_bind_span);

        for _round in 1..num_rounds {
            let round_uni_poly = Self::primary_sumcheck_inner_loop(
                &eq_poly,
                &flag_polys_updated,
                &memory_polys_updated,
                num_eval_points,
                &instruction_to_memory_indices_map,
            );
            compressed_polys.push(round_uni_poly.compress());
            let r_j = Self::update_primary_sumcheck_transcript(round_uni_poly, transcript);
            random_vars.push(r_j);

            // Bind all polys
            let _bind_span = trace_span!("BindPolys");
            let _bind_enter = _bind_span.enter();
            eq_poly.bound_poly_var_top(&r_j);
            flag_polys_updated
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j));
            memory_polys_updated
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j));

            drop(_bind_enter);
            drop(_bind_span);
        } // End rounds

        // Pass evaluations at point r back in proof:
        // - flags(r) * NUM_INSTRUCTIONS
        // - E(r) * NUM_SUBTABLES

        // Polys are fully defined so we can just take the first (and only) evaluation
        // let flag_evals = (0..flag_polys.len()).map(|i| flag_polys[i][0]).collect();
        let flag_evals = (0..flag_polys_updated.len())
            .map(|i| flag_polys_updated[i][0])
            .collect();
        let memory_evals = (0..memory_polys_updated.len())
            .map(|i| memory_polys_updated[i][0])
            .collect();

        (
            SumcheckInstanceProof::new(compressed_polys),
            random_vars,
            flag_evals,
            memory_evals,
        )
    }

    #[tracing::instrument(skip_all, name = "InstructionLookups::primary_sumcheck_inner_loop")]
    fn primary_sumcheck_inner_loop(
        eq_poly: &DensePolynomial<F>,
        flag_polys: &Vec<DensePolynomial<F>>,
        memory_polys: &Vec<DensePolynomial<F>>,
        num_eval_points: usize,
        instruction_to_memory_indices_map: &Vec<Vec<usize>>,
    ) -> UniPoly<F> {
        let mle_len = eq_poly.len();
        let mle_half = mle_len / 2;

        let evaluate_mles_iterator = (0..mle_half).into_par_iter();

        // Loop over half MLE size (size of MLE next round)
        //   - Compute evaluations of eq, flags, E, at p {0, 1, ..., degree}:
        //       eq(p, _boolean_hypercube_), flags(p, _boolean_hypercube_), E(p, _boolean_hypercube_)
        // After: Sum over MLE elements (with combine)

        let evaluations: Vec<F> = evaluate_mles_iterator
            .map(|low_index| {
                let high_index = mle_half + low_index;

                let mut eq_evals: Vec<F> = vec![F::zero(); num_eval_points];
                let mut multi_flag_evals: Vec<Vec<F>> =
                    vec![vec![F::zero(); Self::NUM_INSTRUCTIONS]; num_eval_points];
                let mut multi_memory_evals: Vec<Vec<F>> =
                    vec![vec![F::zero(); Self::NUM_MEMORIES]; num_eval_points];

                eq_evals[0] = eq_poly[low_index];
                eq_evals[1] = eq_poly[high_index];
                let eq_m = eq_poly[high_index] - eq_poly[low_index];
                for eval_index in 2..num_eval_points {
                    let eq_eval = eq_evals[eval_index - 1] + eq_m;
                    eq_evals[eval_index] = eq_eval;
                }

                // TODO: Exactly one flag across NUM_INSTRUCTIONS is non-zero
                for flag_instruction_index in 0..Self::NUM_INSTRUCTIONS {
                    multi_flag_evals[0][flag_instruction_index] =
                        flag_polys[flag_instruction_index][low_index];
                    multi_flag_evals[1][flag_instruction_index] =
                        flag_polys[flag_instruction_index][high_index];
                    let flag_m = flag_polys[flag_instruction_index][high_index]
                        - flag_polys[flag_instruction_index][low_index];
                    for eval_index in 2..num_eval_points {
                        let flag_eval =
                            multi_flag_evals[eval_index - 1][flag_instruction_index] + flag_m;
                        multi_flag_evals[eval_index][flag_instruction_index] = flag_eval;
                    }
                }

                // TODO: Some of these intermediates need not be computed if flags is computed
                for memory_index in 0..Self::NUM_MEMORIES {
                    multi_memory_evals[0][memory_index] = memory_polys[memory_index][low_index];

                    multi_memory_evals[1][memory_index] = memory_polys[memory_index][high_index];
                    let memory_m = memory_polys[memory_index][high_index]
                        - memory_polys[memory_index][low_index];
                    for eval_index in 2..num_eval_points {
                        multi_memory_evals[eval_index][memory_index] =
                            multi_memory_evals[eval_index - 1][memory_index] + memory_m;
                    }
                }

                // Accumulate inner terms.
                // S({0,1,... num_eval_points}) = eq * [ INNER TERMS ]
                //            = eq[000] * [ flags_0[000] * g_0(E_0)[000] + flags_1[000] * g_1(E_1)[000]]
                //            + eq[001] * [ flags_0[001] * g_0(E_0)[001] + flags_1[001] * g_1(E_1)[001]]
                //            + ...
                //            + eq[111] * [ flags_0[111] * g_0(E_0)[111] + flags_1[111] * g_1(E_1)[111]]
                let mut inner_sum = vec![F::zero(); num_eval_points];
                for instruction in InstructionSet::iter() {
                    let instruction_index = instruction.to_opcode() as usize;
                    let memory_indices: &Vec<usize> =
                        &instruction_to_memory_indices_map[instruction_index];

                    for eval_index in 0..num_eval_points {
                        let flag_eval = multi_flag_evals[eval_index][instruction_index];
                        if flag_eval == F::zero() {
                            continue;
                        }; // Early exit if no contribution.

                        let terms: Vec<F> = memory_indices
                            .iter()
                            .map(|memory_index| multi_memory_evals[eval_index][*memory_index])
                            .collect();
                        let instruction_collation_eval = instruction.combine_lookups(&terms, C, M);

                        // TODO(sragss): Could sum all shared inner terms before multiplying by the flag eval
                        inner_sum[eval_index] += flag_eval * instruction_collation_eval;
                    }
                }
                let evaluations: Vec<F> = (0..num_eval_points)
                    .map(|eval_index| eq_evals[eval_index] * inner_sum[eval_index])
                    .collect();
                evaluations
            })
            .reduce(
                || vec![F::zero(); num_eval_points],
                |running, new| {
                    debug_assert_eq!(running.len(), new.len());
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(r, n)| *r + n)
                        .collect()
                },
            );

        let round_uni_poly = UniPoly::from_evals(&evaluations);
        round_uni_poly
    }

    fn update_primary_sumcheck_transcript(
        round_uni_poly: UniPoly<F>,
        transcript: &mut Transcript,
    ) -> F {
        <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(
            &round_uni_poly,
            b"poly",
            transcript,
        );
        let r_j = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"challenge_nextround",
        );
        r_j
    }

    #[tracing::instrument(skip_all, name = "InstructionLookups::compute_sumcheck_claim")]
    fn compute_sumcheck_claim(
        ops: &Vec<InstructionSet>,
        E_polys: &Vec<DensePolynomial<F>>,
        eq: &EqPolynomial<F>,
    ) -> F {
        #[cfg(test)]
        {
            let m = ops.len().next_power_of_two();
            E_polys.iter().for_each(|E_i| assert_eq!(E_i.len(), m));
        }

        let eq_evals = eq.evals();

        let instruction_to_memory_indices_map: Vec<Vec<usize>> = InstructionSet::iter()
            .map(|op| Self::instruction_to_memory_indices(&op))
            .collect();

        let claim = ops
            .par_iter()
            .enumerate()
            .map(|(k, op)| {
                let memory_indices = &instruction_to_memory_indices_map[op.to_opcode() as usize];
                let filtered_operands: Vec<F> = memory_indices
                    .iter()
                    .map(|memory_index| E_polys[*memory_index][k])
                    .collect();

                let collation_eval = op.combine_lookups(&filtered_operands, C, M);
                eq_evals[k] * collation_eval
            })
            .sum();

        claim
    }

    /// Combines the subtable values given by `vals` and the flag values given by `flags`.
    /// This function corresponds to the "primary" sumcheck expression:
    /// \sum_x \tilde{eq}(r, x) * \sum_i flag_i(x) * g_i(E_1(x), ..., E_\alpha(x))
    /// where `vals` corresponds to E_1, ..., E_\alpha,
    /// and `flags` corresponds to the flag_i's
    fn combine_lookups(vals: &[F], flags: &[F]) -> F {
        assert_eq!(vals.len(), Self::NUM_MEMORIES);
        assert_eq!(flags.len(), Self::NUM_INSTRUCTIONS);

        let mut sum = F::zero();
        for instruction in InstructionSet::iter() {
            let instruction_index = instruction.to_opcode() as usize;
            let memory_indices = Self::instruction_to_memory_indices(&instruction);
            let mut filtered_operands = Vec::with_capacity(memory_indices.len());
            for index in memory_indices {
                filtered_operands.push(vals[index]);
            }
            sum += flags[instruction_index] * instruction.combine_lookups(&filtered_operands, C, M);
        }

        sum
    }

    /// Converts instruction flag values into subtable flag vales. A subtable flag value
    /// can be computed by summing over the instructions that use that subtable: if a given execution step
    /// accesses the subtable, it must be executing exactly one of those instructions.
    fn subtable_flags(instruction_flags: &Vec<F>) -> Vec<F> {
        let mut subtable_flags = vec![F::zero(); Self::NUM_SUBTABLES];
        for (i, instruction) in InstructionSet::iter().enumerate() {
            let instruction_subtables: Vec<Subtables> = instruction
                .subtables::<F>(C)
                .iter()
                .map(|subtable| Subtables::from(subtable.subtable_id()))
                .collect();
            for subtable in instruction_subtables {
                let subtable_index: usize = subtable.into();
                subtable_flags[subtable_index] += &instruction_flags[i];
            }
        }
        subtable_flags
    }

    /// Converts instruction flag polynomials into subtable flag polynomials. A subtable flag polynomial
    /// can be computed by summing over the instructions that use that subtable: if a given execution step
    /// accesses the subtable, it must be executing exactly one of those instructions.
    fn subtable_flag_polys(
        instruction_flag_polys: &Vec<DensePolynomial<F>>,
    ) -> Vec<DensePolynomial<F>> {
        let m = instruction_flag_polys[0].len();
        let subtable_flag_polys = (0..Self::NUM_SUBTABLES)
            .into_par_iter()
            .map(|subtable_index| {
                let mut subtable_poly = DensePolynomial::new(vec![F::zero(); m]);
                for (i, instruction) in InstructionSet::iter().enumerate() {
                    if instruction.subtables::<F>(C).iter().any(|subtable| {
                        Subtables::from(subtable.subtable_id()).into() == subtable_index
                    }) {
                        // TODO(JOLT-81): Do not DensePolynomial<F>::add_assign to compute this value.
                        subtable_poly += &instruction_flag_polys[i];
                    }
                }
                subtable_poly
            })
            .collect();
        subtable_flag_polys
    }

    /// Converts an instruction into the memory indices that it "accesses". Each instruction uses some
    /// subset of `Subtables`, and each subtable in turn maps to some contiguous range of memory indices.
    fn instruction_to_memory_indices(op: &InstructionSet) -> Vec<usize> {
        let instruction_subtables: Vec<Subtables> = op
            .subtables::<F>(C)
            .iter()
            .map(|subtable| Subtables::from(subtable.subtable_id()))
            .collect();

        let mut memory_indices = Vec::with_capacity(C * instruction_subtables.len());
        for subtable in instruction_subtables {
            let index: usize = subtable.into();
            memory_indices.extend((C * index)..(C * (index + 1)));
        }

        memory_indices
    }

    /// Returns the sumcheck polynomial degree for the "primary" sumcheck. Since the primary sumcheck expression
    /// is \sum_x \tilde{eq}(r, x) * \sum_i flag_i(x) * g_i(E_1(x), ..., E_\alpha(x)), the degree is
    /// the max over all the instructions' `g_i` polynomial degrees, plus two (one for \tilde{eq}, one for flag_i)
    fn sumcheck_poly_degree() -> usize {
        InstructionSet::iter()
            .map(|instruction| instruction.g_poly_degree(C))
            .max()
            .unwrap()
            + 2 // eq and flag
    }

    /// Materializes all subtables used by this Jolt instance.
    #[tracing::instrument(skip_all, name = "InstructionLookups.materialize_subtables")]
    fn materialize_subtables() -> Vec<Vec<F>> {
        Subtables::iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|subtable| subtable.materialize(M))
            .collect()
    }

    /// Converts each instruction in `ops` into its corresponding subtable lookup indices.
    /// The output is `C` vectors, each of length `m`.
    fn subtable_lookup_indices(ops: &Vec<InstructionSet>) -> Vec<Vec<usize>> {
        let m = ops.len().next_power_of_two();
        let log_M = M.log_2();
        let chunked_indices: Vec<Vec<usize>> =
            ops.iter().map(|op| op.to_indices(C, log_M)).collect();

        let mut subtable_lookup_indices: Vec<Vec<usize>> = Vec::with_capacity(C);
        for i in 0..C {
            let mut access_sequence: Vec<usize> =
                chunked_indices.iter().map(|chunks| chunks[i]).collect();
            access_sequence.resize(m, 0);
            subtable_lookup_indices.push(access_sequence);
        }
        subtable_lookup_indices
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
        b"Jolt instruction lookups"
    }
}
