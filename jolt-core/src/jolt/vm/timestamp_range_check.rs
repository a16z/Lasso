use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use common::constants::MEMORY_OPS_PER_INSTRUCTION;
use itertools::interleave;
use merlin::Transcript;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[cfg(test)]
use std::collections::HashSet;
use std::{iter::zip, marker::PhantomData};
use tracing::trace_span;

use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
        NoPreprocessing,
    },
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
        grand_product::{
            BatchedGrandProductArgument, BatchedGrandProductCircuit, GrandProductCircuit,
        },
    },
    utils::{errors::ProofVerifyError, math::Math, mul_0_1_optimized, transcript::ProofTranscript},
};

use super::read_write_memory::{BatchedMemoryPolynomials, MemoryCommitment, ReadWriteMemory};

pub struct RangeCheckPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    _group: PhantomData<G>,
    pub read_timestamps: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION],
    pub read_cts_read_timestamp: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    pub read_cts_global_minus_read: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    pub final_cts_read_timestamp: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    pub final_cts_global_minus_read: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
}

impl<F, G> RangeCheckPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::new")]
    pub fn new(read_timestamps: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION]) -> Self {
        let M = read_timestamps[0].len();

        #[cfg(test)]
        let mut init_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
        #[cfg(test)]
        {
            for i in 0..M {
                init_tuples.insert((i as u64, i as u64, 0u64));
            }
        }

        let read_and_final_cts: Vec<[Vec<u64>; 4]> = (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .map(|i| {
                let mut read_cts_read_timestamp: Vec<u64> = vec![0; M];
                let mut read_cts_global_minus_read: Vec<u64> = vec![0; M];
                let mut final_cts_read_timestamp: Vec<u64> = vec![0; M];
                let mut final_cts_global_minus_read: Vec<u64> = vec![0; M];

                for (j, read_timestamp) in read_timestamps[i].iter().enumerate() {
                    read_cts_read_timestamp[j] = final_cts_read_timestamp[*read_timestamp as usize];
                    final_cts_read_timestamp[*read_timestamp as usize] += 1;
                    let lookup_index = j - *read_timestamp as usize;
                    read_cts_global_minus_read[j] = final_cts_global_minus_read[lookup_index];
                    final_cts_global_minus_read[lookup_index] += 1;
                }

                #[cfg(test)]
                {
                    let global_minus_read_timestamps = &read_timestamps[i]
                        .iter()
                        .enumerate()
                        .map(|(j, timestamp)| j as u64 - *timestamp)
                        .collect();

                    for (lookup_indices, read_cts, final_cts) in [
                        (
                            &read_timestamps[i],
                            &read_cts_read_timestamp,
                            &final_cts_read_timestamp,
                        ),
                        (
                            &global_minus_read_timestamps,
                            &read_cts_global_minus_read,
                            &final_cts_global_minus_read,
                        ),
                    ]
                    .iter()
                    {
                        let mut read_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
                        let mut write_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
                        for (v, t) in lookup_indices.iter().zip(read_cts.iter()) {
                            read_tuples.insert((*v, *v, *t));
                            write_tuples.insert((*v, *v, *t + 1));
                        }

                        let mut final_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
                        for (i, t) in final_cts.iter().enumerate() {
                            final_tuples.insert((i as u64, i as u64, *t));
                        }

                        let init_write: HashSet<_> = init_tuples.union(&write_tuples).collect();
                        let read_final: HashSet<_> = read_tuples.union(&final_tuples).collect();
                        let set_difference: Vec<_> =
                            init_write.symmetric_difference(&read_final).collect();
                        assert_eq!(set_difference.len(), 0);
                    }
                }

                [
                    read_cts_read_timestamp,
                    read_cts_global_minus_read,
                    final_cts_read_timestamp,
                    final_cts_global_minus_read,
                ]
            })
            .collect();

        let read_cts_read_timestamp = read_and_final_cts
            .par_iter()
            .map(|cts| DensePolynomial::from_u64(&cts[0]))
            .collect::<Vec<DensePolynomial<F>>>()
            .try_into()
            .unwrap();
        let read_cts_global_minus_read = read_and_final_cts
            .par_iter()
            .map(|cts| DensePolynomial::from_u64(&cts[1]))
            .collect::<Vec<DensePolynomial<F>>>()
            .try_into()
            .unwrap();
        let final_cts_read_timestamp = read_and_final_cts
            .par_iter()
            .map(|cts| DensePolynomial::from_u64(&cts[2]))
            .collect::<Vec<DensePolynomial<F>>>()
            .try_into()
            .unwrap();
        let final_cts_global_minus_read = read_and_final_cts
            .par_iter()
            .map(|cts| DensePolynomial::from_u64(&cts[3]))
            .collect::<Vec<DensePolynomial<F>>>()
            .try_into()
            .unwrap();

        Self {
            _group: PhantomData,
            read_timestamps,
            read_cts_read_timestamp,
            read_cts_global_minus_read,
            final_cts_read_timestamp,
            final_cts_global_minus_read,
        }
    }
}

pub type RangeCheckCommitment<G> = ConcatenatedPolynomialCommitment<G>;
pub type BatchedRangeCheckPolynomials<F> = DensePolynomial<F>;

impl<F, G> BatchablePolynomials<G> for RangeCheckPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type BatchedPolynomials = BatchedRangeCheckPolynomials<F>;
    type Commitment = RangeCheckCommitment<G>;

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::batch")]
    fn batch(&self) -> Self::BatchedPolynomials {
        DensePolynomial::merge(
            self.read_cts_read_timestamp
                .iter()
                .chain(self.read_cts_global_minus_read.iter())
                .chain(self.final_cts_read_timestamp.iter())
                .chain(self.final_cts_global_minus_read.iter()),
        )
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::commit")]
    fn commit(
        &self,
        batched_polys: &Self::BatchedPolynomials,
        pedersen_generators: &PedersenGenerators<G>,
    ) -> Self::Commitment {
        batched_polys.combined_commit(pedersen_generators)
    }
}

pub struct RangeCheckOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    read_cts_read_timestamp: [F; MEMORY_OPS_PER_INSTRUCTION],
    read_cts_global_minus_read: [F; MEMORY_OPS_PER_INSTRUCTION],
    final_cts_read_timestamp: [F; MEMORY_OPS_PER_INSTRUCTION],
    final_cts_global_minus_read: [F; MEMORY_OPS_PER_INSTRUCTION],
    memory_t_read: [F; MEMORY_OPS_PER_INSTRUCTION],
    memory_t_write: [F; MEMORY_OPS_PER_INSTRUCTION],
    identity_poly_opening: Option<F>,
}

pub struct RangeCheckOpeningProof<G>
where
    G: CurveGroup,
{
    range_check_opening_proof: ConcatenatedPolynomialOpeningProof<G>,
    memory_poly_opening_proof: Option<ConcatenatedPolynomialOpeningProof<G>>,
}

impl<F, G> StructuredOpeningProof<F, G, RangeCheckPolynomials<F, G>> for RangeCheckOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Proof = RangeCheckOpeningProof<G>;

    #[tracing::instrument(skip_all, name = "RangeCheckReadWriteOpenings::open")]
    fn open(_polynomials: &RangeCheckPolynomials<F, G>, _opening_point: &Vec<F>) -> Self {
        unimplemented!("Openings are computed in TimestampValidityProof::prove");
    }

    #[tracing::instrument(skip_all, name = "RangeCheckReadWriteOpenings::prove_openings")]
    fn prove_openings(
        _polynomials: &RangeCheckPolynomials<F, G>,
        batched_polynomials: &BatchedRangeCheckPolynomials<F>,
        opening_point: &Vec<F>,
        openings: &RangeCheckOpenings<F, G>,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        let range_check_openings: Vec<F> = openings
            .read_cts_read_timestamp
            .into_iter()
            .chain(openings.read_cts_global_minus_read.into_iter())
            .chain(openings.final_cts_read_timestamp.into_iter())
            .chain(openings.final_cts_global_minus_read.into_iter())
            .collect();
        let range_check_opening_proof = ConcatenatedPolynomialOpeningProof::prove(
            &batched_polynomials,
            opening_point,
            &range_check_openings,
            transcript,
        );
        RangeCheckOpeningProof {
            range_check_opening_proof,
            memory_poly_opening_proof: None,
        }
    }

    fn compute_verifier_openings(&mut self, _: &NoPreprocessing, opening_point: &Vec<F>) {
        self.identity_poly_opening =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &RangeCheckCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let range_check_openings: Vec<F> = self
            .read_cts_read_timestamp
            .into_iter()
            .chain(self.read_cts_global_minus_read.into_iter())
            .chain(self.final_cts_read_timestamp.into_iter())
            .chain(self.final_cts_global_minus_read.into_iter())
            .collect();
        opening_proof.range_check_opening_proof.verify(
            opening_point,
            &range_check_openings,
            &commitment,
            transcript,
        )
    }
}

impl<F, G> MemoryCheckingProver<F, G, RangeCheckPolynomials<F, G>> for TimestampValidityProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type ReadWriteOpenings = RangeCheckOpenings<F, G>;
    type InitFinalOpenings = RangeCheckOpenings<F, G>;

    fn prove_memory_checking(
        _: &NoPreprocessing,
        _polynomials: &RangeCheckPolynomials<F, G>,
        _batched_polys: &BatchedRangeCheckPolynomials<F>,
        _transcript: &mut Transcript,
    ) -> MemoryCheckingProof<
        G,
        RangeCheckPolynomials<F, G>,
        Self::ReadWriteOpenings,
        Self::InitFinalOpenings,
    > {
        unimplemented!("Use TimestampValidityProof::prove instead");
    }

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - tau
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::compute_leaves")]
    fn compute_leaves(
        _: &NoPreprocessing,
        polynomials: &RangeCheckPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) {
        let M = polynomials.read_timestamps[0].len();
        let gamma_squared = gamma.square();

        let read_write_leaves = (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .flat_map(|i| {
                let read_fingerprints_0: Vec<F> = (0..M)
                    .into_par_iter()
                    .map(|j| {
                        let read_timestamp =
                            F::from_u64(polynomials.read_timestamps[i][j]).unwrap();
                        polynomials.read_cts_read_timestamp[i][j] * gamma_squared
                            + read_timestamp * gamma
                            + read_timestamp
                            - tau
                    })
                    .collect();
                let write_fingeprints_0 = read_fingerprints_0
                    .par_iter()
                    .map(|read_fingerprint| *read_fingerprint + gamma_squared)
                    .collect();

                let read_fingerprints_1: Vec<F> = (0..M)
                    .into_par_iter()
                    .map(|j| {
                        let global_minus_read =
                            F::from_u64(j as u64 - polynomials.read_timestamps[i][j]).unwrap();
                        polynomials.read_cts_global_minus_read[i][j] * gamma_squared
                            + global_minus_read * gamma
                            + global_minus_read
                            - tau
                    })
                    .collect();
                let write_fingeprints_1 = read_fingerprints_1
                    .par_iter()
                    .map(|read_fingerprint| *read_fingerprint + gamma_squared)
                    .collect();

                [
                    DensePolynomial::new(read_fingerprints_0),
                    DensePolynomial::new(write_fingeprints_0),
                    DensePolynomial::new(read_fingerprints_1),
                    DensePolynomial::new(write_fingeprints_1),
                ]
            })
            .collect();

        let init_fingerprints = (0..M)
            .into_par_iter()
            .map(|i| {
                let index = F::from_u64(i as u64).unwrap();
                // 0 * gamma^2 +
                index * gamma + index - tau
            })
            .collect();
        let init_leaves = DensePolynomial::new(init_fingerprints);

        let final_leaves: Vec<DensePolynomial<F>> = (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .flat_map(|i| {
                let final_fingerprints_0 = (0..M)
                    .into_par_iter()
                    .map(|j| {
                        mul_0_1_optimized(
                            &polynomials.final_cts_read_timestamp[i][j],
                            &gamma_squared,
                        ) + init_leaves[j]
                    })
                    .collect();

                let final_fingerprints_1 = (0..M)
                    .into_par_iter()
                    .map(|j| {
                        mul_0_1_optimized(
                            &polynomials.final_cts_global_minus_read[i][j],
                            &gamma_squared,
                        ) + init_leaves[j]
                    })
                    .collect();

                [
                    DensePolynomial::new(final_fingerprints_0),
                    DensePolynomial::new(final_fingerprints_1),
                ]
            })
            .collect();

        let mut init_final_leaves = final_leaves;
        init_final_leaves.push(init_leaves);

        (read_write_leaves, init_final_leaves)
    }

    fn interleave_hashes(
        _: &NoPreprocessing,
        multiset_hashes: &MultisetHashes<F>,
    ) -> (Vec<F>, Vec<F>) {
        let read_write_hashes = interleave(
            multiset_hashes.read_hashes.clone(),
            multiset_hashes.write_hashes.clone(),
        )
        .collect();
        let mut init_final_hashes = multiset_hashes.final_hashes.clone();
        init_final_hashes.extend(multiset_hashes.init_hashes.clone());

        (read_write_hashes, init_final_hashes)
    }

    fn uninterleave_hashes(
        _: &NoPreprocessing,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        let num_memories = 2 * MEMORY_OPS_PER_INSTRUCTION;

        assert_eq!(read_write_hashes.len(), 2 * num_memories);
        let mut read_hashes = Vec::with_capacity(num_memories);
        let mut write_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        assert_eq!(init_final_hashes.len(), num_memories + 1);
        let mut final_hashes = init_final_hashes;
        let init_hash = final_hashes.pop().unwrap();

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes: vec![init_hash],
            final_hashes,
        }
    }

    fn check_multiset_equality(_: &NoPreprocessing, multiset_hashes: &MultisetHashes<F>) {
        let num_memories = 2 * MEMORY_OPS_PER_INSTRUCTION;
        assert_eq!(multiset_hashes.read_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.write_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.final_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.init_hashes.len(), 1);

        (0..num_memories).into_par_iter().for_each(|i| {
            let read_hash = multiset_hashes.read_hashes[i];
            let write_hash = multiset_hashes.write_hashes[i];
            let init_hash = multiset_hashes.init_hashes[0];
            let final_hash = multiset_hashes.final_hashes[i];
            assert_eq!(
                init_hash * write_hash,
                final_hash * read_hash,
                "Multiset hashes don't match"
            );
        });
    }

    fn protocol_name() -> &'static [u8] {
        b"Timestamp validity proof memory checking"
    }
}

impl<F, G> MemoryCheckingVerifier<F, G, RangeCheckPolynomials<F, G>>
    for TimestampValidityProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    fn verify_memory_checking(
        _: &NoPreprocessing,
        mut _proof: MemoryCheckingProof<
            G,
            RangeCheckPolynomials<F, G>,
            Self::ReadWriteOpenings,
            Self::InitFinalOpenings,
        >,
        _commitments: &RangeCheckCommitment<G>,
        _transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        unimplemented!("Use TimestampValidityProof::verify instead");
    }

    fn read_tuples(
        _: &NoPreprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let t_read_openings = openings.memory_t_read;

        (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_iter()
            .flat_map(|i| {
                [
                    (
                        t_read_openings[i],
                        t_read_openings[i],
                        openings.read_cts_read_timestamp[i],
                    ),
                    (
                        openings.identity_poly_opening.unwrap() - t_read_openings[i],
                        openings.identity_poly_opening.unwrap() - t_read_openings[i],
                        openings.read_cts_global_minus_read[i],
                    ),
                ]
            })
            .collect()
    }

    fn write_tuples(
        _: &NoPreprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let t_read_openings = openings.memory_t_read;

        (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_iter()
            .flat_map(|i| {
                [
                    (
                        t_read_openings[i],
                        t_read_openings[i],
                        openings.read_cts_read_timestamp[i] + F::one(),
                    ),
                    (
                        openings.identity_poly_opening.unwrap() - t_read_openings[i],
                        openings.identity_poly_opening.unwrap() - t_read_openings[i],
                        openings.read_cts_global_minus_read[i] + F::one(),
                    ),
                ]
            })
            .collect()
    }

    fn init_tuples(
        _: &NoPreprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.identity_poly_opening.unwrap(),
            openings.identity_poly_opening.unwrap(),
            F::zero(),
        )]
    }

    fn final_tuples(
        _: &NoPreprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_iter()
            .flat_map(|i| {
                [
                    (
                        openings.identity_poly_opening.unwrap(),
                        openings.identity_poly_opening.unwrap(),
                        openings.final_cts_read_timestamp[i],
                    ),
                    (
                        openings.identity_poly_opening.unwrap(),
                        openings.identity_poly_opening.unwrap(),
                        openings.final_cts_global_minus_read[i],
                    ),
                ]
            })
            .collect()
    }
}

pub struct TimestampValidityProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    multiset_hashes: MultisetHashes<F>,
    openings: RangeCheckOpenings<F, G>,
    opening_proof: RangeCheckOpeningProof<G>,
    batched_grand_product: BatchedGrandProductArgument<F>,
    pub commitment: RangeCheckCommitment<G>,
}

impl<F, G> TimestampValidityProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    #[tracing::instrument(skip_all, name = "TimestampValidityProof::prove")]
    pub fn prove(
        read_timestamps: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION],
        memory_polynomials: &ReadWriteMemory<F, G>,
        batched_memory_polynomials: &BatchedMemoryPolynomials<F>,
        generators: &PedersenGenerators<G>,
        transcript: &mut Transcript,
    ) -> Self {
        let range_check_polys: RangeCheckPolynomials<F, G> =
            RangeCheckPolynomials::new(read_timestamps);
        let batched_range_check_polys = range_check_polys.batch();
        let range_check_commitment = RangeCheckPolynomials::commit(
            &range_check_polys,
            &batched_range_check_polys,
            &generators,
        );
        let (batched_grand_product, multiset_hashes, r_grand_product) =
            TimestampValidityProof::prove_grand_products(&range_check_polys, transcript);

        let chis = EqPolynomial::new(r_grand_product.to_vec()).evals();
        let mut openings = range_check_polys
            .read_cts_read_timestamp
            .par_iter()
            .chain(range_check_polys.read_cts_global_minus_read.par_iter())
            .chain(range_check_polys.final_cts_read_timestamp.par_iter())
            .chain(range_check_polys.final_cts_global_minus_read.par_iter())
            .chain(memory_polynomials.t_read.par_iter())
            .chain(memory_polynomials.t_write.par_iter())
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect::<Vec<F>>()
            .into_iter();

        let read_cts_read_timestamp: [F; MEMORY_OPS_PER_INSTRUCTION] =
            openings.next_chunk().unwrap();
        let read_cts_global_minus_read = openings.next_chunk().unwrap();
        let final_cts_read_timestamp = openings.next_chunk().unwrap();
        let final_cts_global_minus_read = openings.next_chunk().unwrap();
        let memory_t_read = openings.next_chunk().unwrap();
        let memory_t_write = openings.next_chunk().unwrap();

        let openings = RangeCheckOpenings {
            read_cts_read_timestamp,
            read_cts_global_minus_read,
            final_cts_read_timestamp,
            final_cts_global_minus_read,
            memory_t_read,
            memory_t_write,
            identity_poly_opening: None,
        };

        let mut opening_proof = RangeCheckOpenings::prove_openings(
            &range_check_polys,
            &batched_range_check_polys,
            &r_grand_product,
            &openings,
            transcript,
        );
        opening_proof.memory_poly_opening_proof = Some(ConcatenatedPolynomialOpeningProof::prove(
            &batched_memory_polynomials.batched_t_read_write,
            &r_grand_product,
            &openings
                .memory_t_read
                .into_iter()
                .chain(openings.memory_t_write.into_iter())
                .collect::<Vec<_>>(),
            transcript,
        ));

        Self {
            multiset_hashes,
            openings,
            opening_proof,
            batched_grand_product,
            commitment: range_check_commitment,
        }
    }

    fn prove_grand_products(
        polynomials: &RangeCheckPolynomials<F, G>,
        transcript: &mut Transcript,
    ) -> (BatchedGrandProductArgument<F>, MultisetHashes<F>, Vec<F>) {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking gamma",
        );
        let tau: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking tau",
        );

        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        // fka "ProductLayerProof"
        let (read_write_leaves, init_final_leaves) =
            TimestampValidityProof::compute_leaves(&NoPreprocessing, polynomials, &gamma, &tau);

        let _span = trace_span!("TimestampValidityProof: construct grand product circuits");
        let _enter = _span.enter();

        // R1, W1, R2, W2, ... R14, W14, F1, F2, ... F14, I
        let circuits: Vec<GrandProductCircuit<F>> = read_write_leaves
            .par_iter()
            .chain(init_final_leaves.par_iter())
            .map(|leaves_poly| GrandProductCircuit::new(leaves_poly))
            .collect();

        drop(_enter);
        drop(_span);

        let hashes: Vec<F> = circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();
        let (read_write_hashes, init_final_hashes) = hashes.split_at(read_write_leaves.len());
        let multiset_hashes = TimestampValidityProof::<F, G>::uninterleave_hashes(
            &NoPreprocessing,
            read_write_hashes.to_vec(),
            init_final_hashes.to_vec(),
        );
        TimestampValidityProof::<F, G>::check_multiset_equality(&NoPreprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript::<G>(transcript);

        let batched_circuit = BatchedGrandProductCircuit::new_batch(circuits);

        let _span = trace_span!("TimestampValidityProof: prove grand products");
        let _enter = _span.enter();
        let (batched_grand_product, r_grand_product) =
            BatchedGrandProductArgument::prove::<G>(batched_circuit, transcript);
        drop(_enter);
        drop(_span);

        (batched_grand_product, multiset_hashes, r_grand_product)
    }

    pub fn verify(
        &mut self,
        memory_commitment: &MemoryCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking gamma",
        );
        let tau: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking tau",
        );

        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        // Multiset equality checks
        TimestampValidityProof::<F, G>::check_multiset_equality(
            &NoPreprocessing,
            &self.multiset_hashes,
        );
        self.multiset_hashes.append_to_transcript::<G>(transcript);

        let (read_write_hashes, init_final_hashes) =
            TimestampValidityProof::<F, G>::interleave_hashes(
                &NoPreprocessing,
                &self.multiset_hashes,
            );
        let concatenated_hashes = [read_write_hashes, init_final_hashes].concat();
        let (grand_product_claims, r_grand_product) = self
            .batched_grand_product
            .verify::<G, Transcript>(&concatenated_hashes, transcript);

        self.openings.verify_openings(
            &self.opening_proof,
            &self.commitment,
            &r_grand_product,
            transcript,
        )?;
        self.opening_proof
            .memory_poly_opening_proof
            .as_ref()
            .unwrap()
            .verify(
                &r_grand_product,
                &self
                    .openings
                    .memory_t_read
                    .into_iter()
                    .chain(self.openings.memory_t_write.into_iter())
                    .collect::<Vec<_>>(),
                &memory_commitment.t_read_write_commitments,
                transcript,
            )?;

        self.openings
            .compute_verifier_openings(&NoPreprocessing, &r_grand_product);

        let read_hashes: Vec<_> =
            TimestampValidityProof::read_tuples(&NoPreprocessing, &self.openings)
                .iter()
                .map(|tuple| TimestampValidityProof::<F, G>::fingerprint(tuple, &gamma, &tau))
                .collect();
        let write_hashes: Vec<_> =
            TimestampValidityProof::write_tuples(&NoPreprocessing, &self.openings)
                .iter()
                .map(|tuple| TimestampValidityProof::<F, G>::fingerprint(tuple, &gamma, &tau))
                .collect();
        let init_hashes: Vec<_> =
            TimestampValidityProof::init_tuples(&NoPreprocessing, &self.openings)
                .iter()
                .map(|tuple| TimestampValidityProof::<F, G>::fingerprint(tuple, &gamma, &tau))
                .collect();
        let final_hashes: Vec<_> =
            TimestampValidityProof::final_tuples(&NoPreprocessing, &self.openings)
                .iter()
                .map(|tuple| TimestampValidityProof::<F, G>::fingerprint(tuple, &gamma, &tau))
                .collect();

        assert_eq!(
            grand_product_claims.len(),
            6 * MEMORY_OPS_PER_INSTRUCTION + 1
        );
        let (read_write_claims, init_final_claims) =
            grand_product_claims.split_at(4 * MEMORY_OPS_PER_INSTRUCTION);

        let multiset_hashes = MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        };
        let (read_write_hashes, init_final_hashes) =
            TimestampValidityProof::<F, G>::interleave_hashes(&NoPreprocessing, &multiset_hashes);

        for (claim, fingerprint) in zip(read_write_claims, read_write_hashes) {
            assert_eq!(*claim, fingerprint);
        }
        for (claim, fingerprint) in zip(init_final_claims, init_final_hashes) {
            assert_eq!(*claim, fingerprint);
        }

        Ok(())
    }

    /// Computes the maximum number of group generators needed to commit to timestamp
    /// range-check polynomials using Hyrax, given the maximum trace length.
    pub fn num_generators(max_trace_length: usize) -> usize {
        let max_trace_length = max_trace_length.next_power_of_two();
        let batch_num_vars = (max_trace_length * MEMORY_OPS_PER_INSTRUCTION * 4).log_2();
        matrix_dimensions(batch_num_vars, 1).1
    }

    fn protocol_name() -> &'static [u8] {
        b"Timestamp validity proof memory checking"
    }
}

#[cfg(test)]
mod tests {
    use crate::jolt::vm::read_write_memory::{random_memory_trace, RandomInstruction, ReadWriteMemoryPreprocessing};

    use super::*;
    use ark_curve25519::{EdwardsProjective, Fr};
    use common::rv_trace::{ELFInstruction, JoltDevice};
    use rand_core::SeedableRng;

    #[test]
    fn timestamp_range_check() {
        const MEMORY_SIZE: usize = 1 << 16;
        const NUM_OPS: usize = 1 << 8;
        const BYTECODE_SIZE: usize = 1 << 8;

        let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);
        let bytecode = (0..BYTECODE_SIZE)
            .map(|i| ELFInstruction::random(i, &mut rng))
            .collect();
        let memory_trace = random_memory_trace(&bytecode, MEMORY_SIZE, NUM_OPS, &mut rng);

        let mut transcript: Transcript = Transcript::new(b"test_transcript");

        let preprocessing = ReadWriteMemoryPreprocessing::preprocess(&bytecode, JoltDevice::new());
        let (rw_memory, read_timestamps): (ReadWriteMemory<Fr, EdwardsProjective>, _) =
            ReadWriteMemory::new(&preprocessing, memory_trace, &mut transcript);
        let batched_polys = rw_memory.batch();
        let generators = PedersenGenerators::new(1 << 10, b"Test generators");
        let commitments = rw_memory.commit(&batched_polys, &generators);

        let mut timestamp_validity_proof = TimestampValidityProof::prove(
            read_timestamps,
            &rw_memory,
            &batched_polys,
            &generators,
            &mut transcript,
        );

        let mut transcript: Transcript = Transcript::new(b"test_transcript");
        assert!(TimestampValidityProof::verify(
            &mut timestamp_validity_proof,
            &commitments,
            &mut transcript,
        )
        .is_ok());
    }
}
