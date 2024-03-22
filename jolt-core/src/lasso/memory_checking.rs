#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::poly::{
    dense_mlpoly::DensePolynomial,
    structured_poly::{BatchablePolynomials, StructuredOpeningProof},
};
use crate::subprotocols::grand_product::{
    BatchedGrandProductArgument, BatchedGrandProductCircuit, GrandProductCircuit,
};
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::ProofTranscript;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use itertools::interleave;
use merlin::Transcript;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::iter::zip;
use std::marker::PhantomData;

pub struct MultisetHashes<F: PrimeField> {
    /// Multiset hash of "read" tuples
    pub read_hashes: Vec<F>,
    /// Multiset hash of "write" tuples
    pub write_hashes: Vec<F>,
    /// Multiset hash of "init" tuples
    pub init_hashes: Vec<F>,
    /// Multiset hash of "final" tuples
    pub final_hashes: Vec<F>,
}

impl<F: PrimeField> MultisetHashes<F> {
    pub fn append_to_transcript<G: CurveGroup<ScalarField = F>>(
        &self,
        transcript: &mut Transcript,
    ) {
        <Transcript as ProofTranscript<G>>::append_scalars(
            transcript,
            b"Read multiset hashes",
            &self.read_hashes,
        );
        <Transcript as ProofTranscript<G>>::append_scalars(
            transcript,
            b"Write multiset hashes",
            &self.write_hashes,
        );
        <Transcript as ProofTranscript<G>>::append_scalars(
            transcript,
            b"Init multiset hashes",
            &self.init_hashes,
        );
        <Transcript as ProofTranscript<G>>::append_scalars(
            transcript,
            b"Final multiset hashes",
            &self.final_hashes,
        );
    }
}

pub struct MemoryCheckingProof<G, Polynomials, ReadWriteOpenings, InitFinalOpenings>
where
    G: CurveGroup,
    Polynomials: BatchablePolynomials<G> + ?Sized,
    ReadWriteOpenings: StructuredOpeningProof<G::ScalarField, G, Polynomials>,
    InitFinalOpenings: StructuredOpeningProof<G::ScalarField, G, Polynomials>,
{
    pub _polys: PhantomData<Polynomials>,
    /// Read/write/init/final multiset hashes for each memory
    pub multiset_hashes: MultisetHashes<G::ScalarField>,
    /// The read and write grand products for every memory has the same size,
    /// so they can be batched.
    pub read_write_grand_product: BatchedGrandProductArgument<G::ScalarField>,
    /// The init and final grand products for every memory has the same size,
    /// so they can be batched.
    pub init_final_grand_product: BatchedGrandProductArgument<G::ScalarField>,
    /// The opening proofs associated with the read/write grand product.
    pub read_write_openings: ReadWriteOpenings,
    pub read_write_opening_proof: ReadWriteOpenings::Proof,
    /// The opening proofs associated with the init/final grand product.
    pub init_final_openings: InitFinalOpenings,
    pub init_final_opening_proof: InitFinalOpenings::Proof,
}

// Empty struct to represent that no preprocessing data is used.
pub struct NoPreprocessing;

pub trait MemoryCheckingProver<F, G, Polynomials>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Polynomials: BatchablePolynomials<G> + std::marker::Sync,
    Self: std::marker::Sync,
{
    type Preprocessing = NoPreprocessing;
    type ReadWriteOpenings: StructuredOpeningProof<
        F,
        G,
        Polynomials,
        Preprocessing = NoPreprocessing,
    >;
    type InitFinalOpenings: StructuredOpeningProof<
        F,
        G,
        Polynomials,
        Preprocessing = Self::Preprocessing,
    >;
    /// The data associated with each memory slot. A triple (a, v, t) by default.
    type MemoryTuple = (F, F, F);

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::prove_memory_checking")]
    /// Generates a memory checking proof for the given committed polynomials.
    fn prove_memory_checking(
        preprocessing: &Self::Preprocessing,
        polynomials: &Polynomials,
        batched_polys: &Polynomials::BatchedPolynomials,
        transcript: &mut Transcript,
    ) -> MemoryCheckingProof<G, Polynomials, Self::ReadWriteOpenings, Self::InitFinalOpenings> {
        // TODO(JOLT-62): Make sure Polynomials::Commitment have been posted to transcript.

        // fka "ProductLayerProof"
        let (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ) = Self::prove_grand_products(preprocessing, polynomials, transcript);

        let read_write_openings = Self::ReadWriteOpenings::open(polynomials, &r_read_write);
        let read_write_opening_proof = Self::ReadWriteOpenings::prove_openings(
            polynomials,
            batched_polys,
            &r_read_write,
            &read_write_openings,
            transcript,
        );
        let init_final_openings = Self::InitFinalOpenings::open(polynomials, &r_init_final);
        let init_final_opening_proof = Self::InitFinalOpenings::prove_openings(
            polynomials,
            batched_polys,
            &r_init_final,
            &init_final_openings,
            transcript,
        );

        MemoryCheckingProof {
            _polys: PhantomData,
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
            read_write_openings,
            read_write_opening_proof,
            init_final_openings,
            init_final_opening_proof,
        }
    }

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::prove_grand_products")]
    /// Proves the grand products for the memory checking multisets (init, read, write, final).
    fn prove_grand_products(
        preprocessing: &Self::Preprocessing,
        polynomials: &Polynomials,
        transcript: &mut Transcript,
    ) -> (
        BatchedGrandProductArgument<F>,
        BatchedGrandProductArgument<F>,
        MultisetHashes<F>,
        Vec<F>,
        Vec<F>,
    ) {
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
            Self::compute_leaves(preprocessing, polynomials, &gamma, &tau);
        let (read_write_circuit, read_write_hashes) =
            Self::read_write_grand_product(preprocessing, polynomials, read_write_leaves);
        let (init_final_circuit, init_final_hashes) =
            Self::init_final_grand_product(preprocessing, polynomials, init_final_leaves);

        let multiset_hashes =
            Self::uninterleave_hashes(preprocessing, read_write_hashes, init_final_hashes);
        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript::<G>(transcript);

        let (read_write_grand_product, r_read_write) =
            BatchedGrandProductArgument::prove::<G>(read_write_circuit, transcript);
        let (init_final_grand_product, r_init_final) =
            BatchedGrandProductArgument::prove::<G>(init_final_circuit, transcript);
        (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        )
    }

    /// Constructs a batched grand product circuit for the read and write multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::read_write_grand_product")]
    fn read_write_grand_product(
        preprocessing: &Self::Preprocessing,
        _polynomials: &Polynomials,
        read_write_leaves: Vec<DensePolynomial<F>>,
    ) -> (BatchedGrandProductCircuit<F>, Vec<F>) {
        let read_write_circuits: Vec<GrandProductCircuit<F>> = read_write_leaves
            .par_iter()
            .map(|leaves| GrandProductCircuit::new(&leaves))
            .collect();
        let read_write_hashes: Vec<F> = read_write_circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();

        (
            BatchedGrandProductCircuit::new_batch(read_write_circuits),
            read_write_hashes,
        )
    }

    /// Constructs a batched grand product circuit for the init and final multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::init_final_grand_product")]
    fn init_final_grand_product(
        preprocessing: &Self::Preprocessing,
        _polynomials: &Polynomials,
        init_final_leaves: Vec<DensePolynomial<F>>,
    ) -> (BatchedGrandProductCircuit<F>, Vec<F>) {
        let init_final_circuits: Vec<GrandProductCircuit<F>> = init_final_leaves
            .par_iter()
            .map(|leaves| GrandProductCircuit::new(&leaves))
            .collect();
        let init_final_hashes: Vec<F> = init_final_circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();

        (
            BatchedGrandProductCircuit::new_batch(init_final_circuits),
            init_final_hashes,
        )
    }

    fn interleave_hashes(
        _preprocessing: &Self::Preprocessing,
        multiset_hashes: &MultisetHashes<F>,
    ) -> (Vec<F>, Vec<F>) {
        let read_write_hashes = interleave(
            multiset_hashes.read_hashes.clone(),
            multiset_hashes.write_hashes.clone(),
        )
        .collect();
        let init_final_hashes = interleave(
            multiset_hashes.init_hashes.clone(),
            multiset_hashes.final_hashes.clone(),
        )
        .collect();

        (read_write_hashes, init_final_hashes)
    }

    fn uninterleave_hashes(
        _preprocessing: &Self::Preprocessing,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        assert_eq!(read_write_hashes.len() % 2, 0);
        let num_memories = read_write_hashes.len() / 2;

        let mut read_hashes = Vec::with_capacity(num_memories);
        let mut write_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        let mut init_hashes = Vec::with_capacity(num_memories);
        let mut final_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            init_hashes.push(init_final_hashes[2 * i]);
            final_hashes.push(init_final_hashes[2 * i + 1]);
        }

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        }
    }

    fn check_multiset_equality(
        _preprocessing: &Self::Preprocessing,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        let num_memories = multiset_hashes.read_hashes.len();
        assert_eq!(multiset_hashes.final_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.write_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.init_hashes.len(), num_memories);

        (0..num_memories).into_par_iter().for_each(|i| {
            let read_hash = multiset_hashes.read_hashes[i];
            let write_hash = multiset_hashes.write_hashes[i];
            let init_hash = multiset_hashes.init_hashes[i];
            let final_hash = multiset_hashes.final_hashes[i];
            assert_eq!(
                init_hash * write_hash,
                final_hash * read_hash,
                "Multiset hashes don't match"
            );
        });
    }

    /// Computes the MLE of the leaves of the read, write, init, and final grand product circuits,
    /// one of each type per memory.
    /// Returns: (interleaved read/write leaves, interleaved init/final leaves)
    fn compute_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &Polynomials,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>);

    /// Computes the Reed-Solomon fingerprint (parametrized by `gamma` and `tau`) of the given memory `tuple`.
    /// Each individual "leaf" of a grand product circuit (as computed by `read_leaves`, etc.) should be
    /// one such fingerprint.
    fn fingerprint(tuple: &Self::MemoryTuple, gamma: &F, tau: &F) -> F;
    /// Name of the memory checking instance, used for Fiat-Shamir.
    fn protocol_name() -> &'static [u8];
}

pub trait MemoryCheckingVerifier<F, G, Polynomials>:
    MemoryCheckingProver<F, G, Polynomials>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Polynomials: BatchablePolynomials<G> + std::marker::Sync,
{
    /// Verifies a memory checking proof, given its associated polynomial `commitment`.
    fn verify_memory_checking(
        preprocessing: &Self::Preprocessing,
        mut proof: MemoryCheckingProof<
            G,
            Polynomials,
            Self::ReadWriteOpenings,
            Self::InitFinalOpenings,
        >,
        commitments: &Polynomials::Commitment,
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

        Self::check_multiset_equality(preprocessing, &proof.multiset_hashes);
        proof.multiset_hashes.append_to_transcript::<G>(transcript);

        let (read_write_hashes, init_final_hashes) =
            Self::interleave_hashes(preprocessing, &proof.multiset_hashes);

        let (claims_read_write, r_read_write) = proof
            .read_write_grand_product
            .verify::<G, Transcript>(&read_write_hashes, transcript);
        let (claims_init_final, r_init_final) = proof
            .init_final_grand_product
            .verify::<G, Transcript>(&init_final_hashes, transcript);

        proof.read_write_openings.verify_openings(
            &proof.read_write_opening_proof,
            commitments,
            &r_read_write,
            transcript,
        )?;
        proof.init_final_openings.verify_openings(
            &proof.init_final_opening_proof,
            commitments,
            &r_init_final,
            transcript,
        )?;

        proof
            .read_write_openings
            .compute_verifier_openings(&NoPreprocessing, &r_read_write);
        proof
            .init_final_openings
            .compute_verifier_openings(preprocessing, &r_init_final);

        Self::check_fingerprints(
            preprocessing,
            claims_read_write,
            claims_init_final,
            &proof.read_write_openings,
            &proof.init_final_openings,
            &gamma,
            &tau,
        );

        Ok(())
    }

    /// Computes "read" memory tuples (one per memory) from the given `openings`.
    fn read_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "write" memory tuples (one per memory) from the given `openings`.
    fn write_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "init" memory tuples (one per memory) from the given `openings`.
    fn init_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "final" memory tuples (one per memory) from the given `openings`.
    fn final_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple>;

    /// Checks that the claimed multiset hashes (output by grand product) are consistent with the
    /// openings given by `read_write_openings` and `init_final_openings`.
    fn check_fingerprints(
        preprocessing: &Self::Preprocessing,
        claims_read_write: Vec<F>,
        claims_init_final: Vec<F>,
        read_write_openings: &Self::ReadWriteOpenings,
        init_final_openings: &Self::InitFinalOpenings,
        gamma: &F,
        tau: &F,
    ) {
        let read_hashes: Vec<_> = Self::read_tuples(preprocessing, read_write_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();
        let write_hashes: Vec<_> = Self::write_tuples(preprocessing, read_write_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();
        let init_hashes: Vec<_> = Self::init_tuples(preprocessing, init_final_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();
        let final_hashes: Vec<_> = Self::final_tuples(preprocessing, init_final_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();
        assert_eq!(
            read_hashes.len() + write_hashes.len(),
            claims_read_write.len()
        );
        assert_eq!(
            init_hashes.len() + final_hashes.len(),
            claims_init_final.len()
        );

        let multiset_hashes = MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        };
        let (read_write_hashes, init_final_hashes) =
            Self::interleave_hashes(preprocessing, &multiset_hashes);

        for (claim, fingerprint) in zip(claims_read_write, read_write_hashes) {
            assert_eq!(claim, fingerprint);
        }
        for (claim, fingerprint) in zip(claims_init_final, init_final_hashes) {
            assert_eq!(claim, fingerprint);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::poly::pedersen::PedersenGenerators;

    use super::*;
    use ark_curve25519::{EdwardsProjective, Fr};
    use ark_ff::Field;
    use ark_std::{One, Zero};

    #[test]
    fn product_layer_proof_trivial() {
        struct NormalMems {
            a_ops: DensePolynomial<Fr>,

            v_ops: DensePolynomial<Fr>,
            v_mems: DensePolynomial<Fr>,

            t_reads: DensePolynomial<Fr>,
            t_finals: DensePolynomial<Fr>,
        }
        struct FakeType();
        struct FakeOpeningProof();
        impl StructuredOpeningProof<Fr, EdwardsProjective, NormalMems> for FakeOpeningProof {
            fn open(_: &NormalMems, _: &Vec<Fr>) -> Self {
                unimplemented!()
            }
            fn prove_openings(
                _: &NormalMems,
                _: &FakeType,
                _: &Vec<Fr>,
                _: &Self,
                _: &mut Transcript,
            ) -> Self::Proof {
                unimplemented!()
            }
            fn verify_openings(
                &self,
                _: &Self::Proof,
                _: &FakeType,
                _: &Vec<Fr>,
                _: &mut Transcript,
            ) -> Result<(), ProofVerifyError> {
                unimplemented!()
            }
        }

        impl BatchablePolynomials<EdwardsProjective> for NormalMems {
            type Commitment = FakeType;
            type BatchedPolynomials = FakeType;

            fn batch(&self) -> Self::BatchedPolynomials {
                unimplemented!()
            }
            fn commit(
                &self,
                _batched_polys: &Self::BatchedPolynomials,
                _generators: &PedersenGenerators<EdwardsProjective>,
            ) -> Self::Commitment {
                unimplemented!()
            }
        }

        struct TestProver {}
        impl MemoryCheckingProver<Fr, EdwardsProjective, NormalMems> for TestProver {
            type ReadWriteOpenings = FakeOpeningProof;
            type InitFinalOpenings = FakeOpeningProof;

            type MemoryTuple = (Fr, Fr, Fr);

            #[rustfmt::skip]
            fn compute_leaves(
                _: &NoPreprocessing,
                polynomials: &NormalMems,
                gamma: &Fr,
                tau: &Fr,
            ) -> (
                Vec<DensePolynomial<Fr>>,
                Vec<DensePolynomial<Fr>>,
            ) {
                let read_leaves = DensePolynomial::new(
                    (0..polynomials.a_ops.len()).map(|i| {
                        Self::fingerprint(
                            &(polynomials.a_ops[i], polynomials.v_ops[i], polynomials.t_reads[i]),
                            gamma,
                            tau,
                        )
                    }).collect(),
                );
                let write_leaves = DensePolynomial::new(
                    (0..polynomials.a_ops.len()).map(|i| {
                        Self::fingerprint(
                            &(polynomials.a_ops[i], polynomials.v_ops[i], polynomials.t_reads[i] + Fr::one()),
                            gamma,
                            tau,
                        )
                    }).collect(),
                );
                let init_leaves = DensePolynomial::new(
                    (0..polynomials.v_mems.len()).map(|i| {
                        Self::fingerprint(
                            &(Fr::from(i as u64), polynomials.v_mems[i], Fr::zero()),
                            gamma,
                            tau,
                        )
                    }).collect(),
                );
                let final_leaves = DensePolynomial::new(
                    (0..polynomials.v_mems.len()).map(|i| {
                        Self::fingerprint(
                            &(Fr::from(i as u64), polynomials.v_mems[i], polynomials.t_finals[i]),
                            gamma,
                            tau,
                        )
                    }).collect(),
                );
                (vec![read_leaves, write_leaves], vec![init_leaves, final_leaves])
            }

            fn fingerprint(tuple: &Self::MemoryTuple, gamma: &Fr, tau: &Fr) -> Fr {
                let (a, v, t) = tuple;
                t * &gamma.square() + v * gamma + a - tau
            }

            fn protocol_name() -> &'static [u8] {
                b"protocol_name"
            }
        }
        // Imagine a size-8 range-check table (addresses and values just ascending), with 4 lookups
        let v_mems = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
        ];

        // 2 lookups into the last 2 elements of memory each
        let a_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
        let v_ops = a_ops.clone();

        let t_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
        let t_finals = vec![
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::from(2),
            Fr::from(2),
        ];

        let a_ops = DensePolynomial::new(a_ops);
        let v_ops = DensePolynomial::new(v_ops);
        let v_mems = DensePolynomial::new(v_mems);
        let t_reads = DensePolynomial::new(t_reads);
        let t_finals = DensePolynomial::new(t_finals);
        let polys = NormalMems {
            a_ops,
            v_ops,
            v_mems,
            t_reads,
            t_finals,
        };

        // Prove
        let mut transcript = Transcript::new(b"test_transcript");
        let (proof_rw, proof_if, multiset_hashes, r_rw, r_if) =
            TestProver::prove_grand_products(&NoPreprocessing, &polys, &mut transcript);

        // Verify
        let mut transcript = Transcript::new(b"test_transcript");
        let _gamma: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking gamma",
        );
        let _tau: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking tau",
        );
        <Transcript as ProofTranscript<EdwardsProjective>>::append_protocol_name(
            &mut transcript,
            TestProver::protocol_name(),
        );
        multiset_hashes.append_to_transcript::<EdwardsProjective>(&mut transcript);
        let (interleaved_read_write_hashes, interleaved_init_final_hashes) =
            TestProver::interleave_hashes(&NoPreprocessing, &multiset_hashes);

        let (_claims_rw, r_rw_verify) = proof_rw
            .verify::<EdwardsProjective, _>(&interleaved_read_write_hashes, &mut transcript);
        assert_eq!(r_rw_verify, r_rw);

        let (_claims_if, r_if_verify) = proof_if
            .verify::<EdwardsProjective, _>(&interleaved_init_final_hashes, &mut transcript);
        assert_eq!(r_if_verify, r_if);
    }

    fn get_difference<T: Clone + Eq + std::hash::Hash>(vec1: &[T], vec2: &[T]) -> Vec<T> {
        let set1: HashSet<_> = vec1.iter().cloned().collect();
        let set2: HashSet<_> = vec2.iter().cloned().collect();
        set1.symmetric_difference(&set2).cloned().collect()
    }

    #[test]
    fn product_layer_proof_batched() {
        // Define a GrandProduct circuit that can be batched across 2 memories
        struct Polys {
            a_0_ops: DensePolynomial<Fr>,
            a_1_ops: DensePolynomial<Fr>,

            v_0_ops: DensePolynomial<Fr>,
            v_1_ops: DensePolynomial<Fr>,
            v_mems: DensePolynomial<Fr>,

            t_0_reads: DensePolynomial<Fr>,
            t_1_reads: DensePolynomial<Fr>,

            t_0_finals: DensePolynomial<Fr>,
            t_1_finals: DensePolynomial<Fr>,
        }

        struct FakeType();
        struct FakeOpeningProof();
        impl StructuredOpeningProof<Fr, EdwardsProjective, Polys> for FakeOpeningProof {
            fn open(_: &Polys, _: &Vec<Fr>) -> Self {
                unimplemented!()
            }
            fn prove_openings(
                _: &Polys,
                _: &FakeType,
                _: &Vec<Fr>,
                _: &Self,
                _: &mut Transcript,
            ) -> Self::Proof {
                unimplemented!()
            }
            fn verify_openings(
                &self,
                _: &Self::Proof,
                _: &FakeType,
                _: &Vec<Fr>,
                _: &mut Transcript,
            ) -> Result<(), ProofVerifyError> {
                unimplemented!()
            }
        }

        impl BatchablePolynomials<EdwardsProjective> for Polys {
            type Commitment = FakeType;
            type BatchedPolynomials = FakeType;

            fn batch(&self) -> Self::BatchedPolynomials {
                unimplemented!()
            }
            fn commit(
                &self,
                _batched_polys: &Self::BatchedPolynomials,
                _generators: &PedersenGenerators<EdwardsProjective>,
            ) -> Self::Commitment {
                unimplemented!()
            }
        }

        struct TestProver {}
        impl MemoryCheckingProver<Fr, EdwardsProjective, Polys> for TestProver {
            type ReadWriteOpenings = FakeOpeningProof;
            type InitFinalOpenings = FakeOpeningProof;

            type MemoryTuple = (Fr, Fr, Fr);

            #[rustfmt::skip]
            fn compute_leaves(
                _: &NoPreprocessing,
                polynomials: &Polys,
                gamma: &Fr,
                tau: &Fr,
            ) -> (
                Vec<DensePolynomial<Fr>>,
                Vec<DensePolynomial<Fr>>,
            ) {
                let read_leaves: Vec<DensePolynomial<Fr>> = [0, 1].iter().map(|memory_index| {
                    DensePolynomial::new(
                        (0..polynomials.a_0_ops.len()).map(|leaf_index| {
                            let tuple = match memory_index {
                                0 => (polynomials.a_0_ops[leaf_index], polynomials.v_0_ops[leaf_index], polynomials.t_0_reads[leaf_index]),
                                1 => (polynomials.a_1_ops[leaf_index], polynomials.v_1_ops[leaf_index], polynomials.t_1_reads[leaf_index]),
                                _ => unimplemented!(),
                            };
                            Self::fingerprint(&tuple, gamma, tau)
                        }).collect(),
                    )
                }).collect();
                let write_leaves: Vec<DensePolynomial<Fr>> = [0, 1].iter().map(|memory_index| {
                    DensePolynomial::new(
                        (0..polynomials.a_0_ops.len()).map(|leaf_index| {
                            let tuple = match memory_index {
                                0 => (polynomials.a_0_ops[leaf_index], polynomials.v_0_ops[leaf_index], polynomials.t_0_reads[leaf_index] + Fr::one()),
                                1 => (polynomials.a_1_ops[leaf_index], polynomials.v_1_ops[leaf_index], polynomials.t_1_reads[leaf_index] + Fr::one()),
                                _ => unimplemented!(),
                            };
                            Self::fingerprint(&tuple, gamma, tau)
                        }).collect(),
                    )
                }).collect();
                let init_leaves: Vec<DensePolynomial<Fr>> = [0, 1].iter().map(|memory_index| {
                    DensePolynomial::new(
                        (0..polynomials.v_mems.len()).map(|leaf_index| {
                            let tuple = match memory_index {
                                0 | 1 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], Fr::zero()),
                                _ => unimplemented!(),
                            };
                            Self::fingerprint(&tuple, gamma, tau)
                        }).collect(),
                    )
                }).collect();
                let final_leaves: Vec<DensePolynomial<Fr>> = [0, 1].iter().map(|memory_index| {
                    DensePolynomial::new(
                        (0..polynomials.v_mems.len()).map(|leaf_index| {
                            let tuple = match memory_index {
                                0 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], polynomials.t_0_finals[leaf_index]),
                                1 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], polynomials.t_1_finals[leaf_index]),
                                _ => unimplemented!(),
                            };
                            Self::fingerprint(&tuple, gamma, tau)
                        }).collect(),
                    )
                }).collect();
                (interleave(read_leaves, write_leaves).collect(), interleave(init_leaves, final_leaves).collect())
            }

            fn fingerprint(tuple: &Self::MemoryTuple, gamma: &Fr, tau: &Fr) -> Fr {
                let (a, v, t) = tuple;
                t * &gamma.square() + v * gamma + a - tau
            }

            fn protocol_name() -> &'static [u8] {
                b"protocol_name"
            }
        }

        // Imagine a 2 memories. Size-8 range-check table (addresses and values just ascending), with 4 lookups into each
        let v_mems = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
        ];

        // 2 lookups into the last 2 elements of memory each
        let a_0_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
        let a_1_ops = vec![Fr::from(0), Fr::from(1), Fr::from(0), Fr::from(2)];
        let v_0_ops = a_0_ops.clone();
        let v_1_ops = a_1_ops.clone();

        let t_0_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
        let t_1_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::zero()];
        let t_0_finals = vec![
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::from(2),
            Fr::from(2),
        ];
        let t_1_finals = vec![
            Fr::from(2),
            Fr::one(),
            Fr::one(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
        ];

        let a_0_ops = DensePolynomial::new(a_0_ops);
        let a_1_ops = DensePolynomial::new(a_1_ops);
        let v_0_ops = DensePolynomial::new(v_0_ops);
        let v_1_ops = DensePolynomial::new(v_1_ops);
        let v_mems = DensePolynomial::new(v_mems);
        let t_0_reads = DensePolynomial::new(t_0_reads);
        let t_1_reads = DensePolynomial::new(t_1_reads);
        let t_0_finals = DensePolynomial::new(t_0_finals);
        let t_1_finals = DensePolynomial::new(t_1_finals);
        let polys = Polys {
            a_0_ops,
            a_1_ops,
            v_0_ops,
            v_1_ops,
            v_mems,
            t_0_reads,
            t_1_reads,
            t_0_finals,
            t_1_finals,
        };

        // Check leaves match
        let (gamma, tau) = (&Fr::from(100), &Fr::from(35));
        let (read_write_leaves, init_final_leaves) =
            TestProver::compute_leaves(&NoPreprocessing, &polys, gamma, tau);

        [0, 1].into_iter().for_each(|i| {
            let read_leaves = &read_write_leaves[2 * i];
            let write_leaves = &read_write_leaves[2 * i + 1];
            let init_leaves = &init_final_leaves[2 * i];
            let final_leaves = &init_final_leaves[2 * i + 1];
            let read_final_leaves = vec![read_leaves.evals(), final_leaves.evals()].concat();
            let init_write_leaves = vec![init_leaves.evals(), write_leaves.evals()].concat();
            let difference: Vec<Fr> = get_difference(&read_final_leaves, &init_write_leaves);
            assert_eq!(difference.len(), 0);
        });

        // Prove
        let mut transcript = Transcript::new(b"test_transcript");
        let (proof_rw, proof_if, multiset_hashes, r_rw, r_if) =
            TestProver::prove_grand_products(&NoPreprocessing, &polys, &mut transcript);

        // Verify
        let mut transcript = Transcript::new(b"test_transcript");
        let _gamma: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking gamma",
        );
        let _tau: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking tau",
        );
        <Transcript as ProofTranscript<EdwardsProjective>>::append_protocol_name(
            &mut transcript,
            TestProver::protocol_name(),
        );
        multiset_hashes.append_to_transcript::<EdwardsProjective>(&mut transcript);
        let (interleaved_read_write_hashes, interleaved_init_final_hashes) =
            TestProver::interleave_hashes(&NoPreprocessing, &multiset_hashes);

        let (_claims_rw, r_rw_verify) = proof_rw
            .verify::<EdwardsProjective, _>(&interleaved_read_write_hashes, &mut transcript);
        assert_eq!(r_rw_verify, r_rw);

        let (_claims_if, r_if_verify) = proof_if
            .verify::<EdwardsProjective, _>(&interleaved_init_final_hashes, &mut transcript);
        assert_eq!(r_if_verify, r_if);
    }

    #[test]
    fn product_layer_proof_flags_no_reuse() {
        // Define a GrandProduct circuit that can be batched across 2 memories
        struct FlagPolys {
            a_0_ops: DensePolynomial<Fr>,
            a_1_ops: DensePolynomial<Fr>,

            v_0_ops: DensePolynomial<Fr>,
            v_1_ops: DensePolynomial<Fr>,
            v_mems: DensePolynomial<Fr>,

            t_0_reads: DensePolynomial<Fr>,
            t_1_reads: DensePolynomial<Fr>,

            t_0_finals: DensePolynomial<Fr>,
            t_1_finals: DensePolynomial<Fr>,

            flags_0: DensePolynomial<Fr>,
            flags_1: DensePolynomial<Fr>,
        }

        struct FakeType();
        struct FakeOpeningProof();
        impl StructuredOpeningProof<Fr, EdwardsProjective, FlagPolys> for FakeOpeningProof {
            fn open(_: &FlagPolys, _: &Vec<Fr>) -> Self {
                unimplemented!()
            }
            fn prove_openings(
                _: &FlagPolys,
                _: &FakeType,
                _: &Vec<Fr>,
                _: &Self,
                _: &mut Transcript,
            ) -> Self::Proof {
                unimplemented!()
            }
            fn verify_openings(
                &self,
                _: &Self::Proof,
                _: &FakeType,
                _: &Vec<Fr>,
                _: &mut Transcript,
            ) -> Result<(), ProofVerifyError> {
                unimplemented!()
            }
        }

        impl BatchablePolynomials<EdwardsProjective> for FlagPolys {
            type Commitment = FakeType;
            type BatchedPolynomials = FakeType;

            fn batch(&self) -> Self::BatchedPolynomials {
                unimplemented!()
            }
            fn commit(
                &self,
                _batched_polys: &Self::BatchedPolynomials,
                _generators: &PedersenGenerators<EdwardsProjective>,
            ) -> Self::Commitment {
                unimplemented!()
            }
        }

        struct TestProver {}
        impl MemoryCheckingProver<Fr, EdwardsProjective, FlagPolys> for TestProver {
            type ReadWriteOpenings = FakeOpeningProof;
            type InitFinalOpenings = FakeOpeningProof;

            type MemoryTuple = (Fr, Fr, Fr, Option<Fr>);

            #[rustfmt::skip]
            fn compute_leaves(
                _: &NoPreprocessing,
                polynomials: &FlagPolys,
                gamma: &Fr,
                tau: &Fr,
            ) -> (
                Vec<DensePolynomial<Fr>>,
                Vec<DensePolynomial<Fr>>,
            ) {
                let read_leaves: Vec<DensePolynomial<Fr>> = [0, 1].iter().map(|memory_index| {
                    DensePolynomial::new(
                        (0..polynomials.a_0_ops.len()).map(|leaf_index| {
                            let tuple = match memory_index {
                                0 => (polynomials.a_0_ops[leaf_index], polynomials.v_0_ops[leaf_index], polynomials.t_0_reads[leaf_index], None),
                                1 => (polynomials.a_1_ops[leaf_index], polynomials.v_1_ops[leaf_index], polynomials.t_1_reads[leaf_index], None),
                                _ => unimplemented!(),
                            };
                            Self::fingerprint(&tuple, gamma, tau)
                        }).collect(),
                    )
                }).collect();
                let write_leaves: Vec<DensePolynomial<Fr>> = [0, 1].iter().map(|memory_index| {
                    DensePolynomial::new(
                        (0..polynomials.a_0_ops.len()).map(|leaf_index| {
                            let tuple = match memory_index {
                                0 => (polynomials.a_0_ops[leaf_index], polynomials.v_0_ops[leaf_index], polynomials.t_0_reads[leaf_index] + Fr::one(), None),
                                1 => (polynomials.a_1_ops[leaf_index], polynomials.v_1_ops[leaf_index], polynomials.t_1_reads[leaf_index] + Fr::one(), None),
                                _ => unimplemented!(),
                            };
                            Self::fingerprint(&tuple, gamma, tau)
                        }).collect(),
                    )
                }).collect();
                let init_leaves: Vec<DensePolynomial<Fr>> = [0, 1].iter().map(|memory_index| {
                    DensePolynomial::new(
                        (0..polynomials.v_mems.len()).map(|leaf_index| {
                            let tuple = match memory_index {
                                0 | 1 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], Fr::zero(), None),
                                _ => unimplemented!(),
                            };
                            Self::fingerprint(&tuple, gamma, tau)
                        }).collect(),
                    )
                }).collect();
                let final_leaves: Vec<DensePolynomial<Fr>> =[0, 1].iter().map(|memory_index| {
                    DensePolynomial::new(
                        (0..polynomials.v_mems.len()).map(|leaf_index| {
                            let tuple = match memory_index {
                                0 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], polynomials.t_0_finals[leaf_index], None),
                                1 => (Fr::from(leaf_index as u64), polynomials.v_mems[leaf_index], polynomials.t_1_finals[leaf_index], None),
                                _ => unimplemented!(),
                            };
                            Self::fingerprint(&tuple, gamma, tau)
                        }).collect(),
                    )
                }).collect();
                (interleave(read_leaves, write_leaves).collect(), interleave(init_leaves, final_leaves).collect())
            }

            fn fingerprint(tuple: &Self::MemoryTuple, gamma: &Fr, tau: &Fr) -> Fr {
                let (a, v, t, flag) = *tuple;
                match flag {
                    Some(val) => {
                        val * (t * gamma.square() + v * *gamma + a - tau) + Fr::one() - val
                    }
                    None => t * gamma.square() + v * *gamma + a - tau,
                }
            }

            // FLAGS OVERRIDES

            // Override read_write_grand product to call BatchedGrandProductCircuit::new_batch_flags and insert our additional toggling layer.
            fn read_write_grand_product(
                _: &NoPreprocessing,
                polynomials: &FlagPolys,
                read_write_leaves: Vec<DensePolynomial<Fr>>,
            ) -> (BatchedGrandProductCircuit<Fr>, Vec<Fr>) {
                // Generate "flagged" leaves for the second to last layer. Input to normal Grand Products
                let num_memories = 2;
                let mut circuits = Vec::with_capacity(2 * num_memories);
                let mut read_hashes = vec![];
                let mut write_hashes = vec![];

                for i in 0..num_memories {
                    let mut toggled_read_fingerprints = read_write_leaves[2 * i].evals();
                    let mut toggled_write_fingerprints = read_write_leaves[2 * i + 1].evals();

                    let subtable_index = i;
                    for leaf_index in 0..polynomials.a_0_ops.len() {
                        let flag = match subtable_index {
                            0 => polynomials.flags_0[leaf_index],
                            1 => polynomials.flags_1[leaf_index],
                            _ => unimplemented!(),
                        };
                        if flag == Fr::zero() {
                            toggled_read_fingerprints[leaf_index] = Fr::one();
                            toggled_write_fingerprints[leaf_index] = Fr::one();
                        }
                    }

                    let read_circuit =
                        GrandProductCircuit::new(&DensePolynomial::new(toggled_read_fingerprints));
                    let write_circuit =
                        GrandProductCircuit::new(&DensePolynomial::new(toggled_write_fingerprints));
                    read_hashes.push(read_circuit.evaluate());
                    write_hashes.push(write_circuit.evaluate());
                    circuits.push(read_circuit);
                    circuits.push(write_circuit);
                }

                let expanded_flag_map = vec![0, 0, 1, 1];
                let batched_circuits = BatchedGrandProductCircuit::new_batch_flags(
                    circuits,
                    vec![polynomials.flags_0.clone(), polynomials.flags_1.clone()],
                    expanded_flag_map,
                    read_write_leaves,
                );

                (
                    batched_circuits,
                    interleave(read_hashes, write_hashes).collect(),
                )
            }

            fn protocol_name() -> &'static [u8] {
                b"protocol_name"
            }
        }

        // Imagine a 2 memories. Size-8 range-check table (addresses and values just ascending), with 4 lookups into each
        let v_mems = vec![
            Fr::from(0),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(5),
            Fr::from(6),
            Fr::from(7),
        ];

        // 2 lookups into the last 2 elements of memory each
        let a_0_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
        let a_1_ops = vec![Fr::from(0), Fr::from(1), Fr::from(0), Fr::from(2)];
        let v_0_ops = a_0_ops.clone();
        let v_1_ops = a_1_ops.clone();

        let flags_0 = vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()];
        let flags_1 = vec![
            Fr::one(),
            Fr::zero(), // Flagged off!
            Fr::one(),
            Fr::one(),
        ];

        let t_0_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
        let t_1_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::zero()];
        let t_0_finals = vec![
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::from(2),
            Fr::from(2),
        ];
        let t_1_finals = vec![
            Fr::from(2),
            Fr::zero(), // Flagged off!
            Fr::one(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
        ];

        let a_0_ops = DensePolynomial::new(a_0_ops);
        let a_1_ops = DensePolynomial::new(a_1_ops);
        let v_0_ops = DensePolynomial::new(v_0_ops);
        let v_1_ops = DensePolynomial::new(v_1_ops);
        let v_mems = DensePolynomial::new(v_mems);
        let t_0_reads = DensePolynomial::new(t_0_reads);
        let t_1_reads = DensePolynomial::new(t_1_reads);
        let t_0_finals = DensePolynomial::new(t_0_finals);
        let t_1_finals = DensePolynomial::new(t_1_finals);
        let flags_0 = DensePolynomial::new(flags_0);
        let flags_1 = DensePolynomial::new(flags_1);
        let polys = FlagPolys {
            a_0_ops,
            a_1_ops,
            v_0_ops,
            v_1_ops,
            v_mems,
            t_0_reads,
            t_1_reads,
            t_0_finals,
            t_1_finals,
            flags_0,
            flags_1,
        };

        // Prove
        let mut transcript = Transcript::new(b"test_transcript");
        let (proof_rw, proof_if, multiset_hashes, r_rw, r_if) =
            TestProver::prove_grand_products(&NoPreprocessing, &polys, &mut transcript);

        // Verify
        let mut transcript = Transcript::new(b"test_transcript");
        let _gamma: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking gamma",
        );
        let _tau: Fr = <Transcript as ProofTranscript<EdwardsProjective>>::challenge_scalar(
            &mut transcript,
            b"Memory checking tau",
        );
        <Transcript as ProofTranscript<EdwardsProjective>>::append_protocol_name(
            &mut transcript,
            TestProver::protocol_name(),
        );
        multiset_hashes.append_to_transcript::<EdwardsProjective>(&mut transcript);
        let (interleaved_read_write_hashes, interleaved_init_final_hashes) =
            TestProver::interleave_hashes(&NoPreprocessing, &multiset_hashes);

        let (_claims_rw, r_rw_verify) = proof_rw
            .verify::<EdwardsProjective, _>(&interleaved_read_write_hashes, &mut transcript);
        assert_eq!(r_rw_verify, r_rw);

        let (_claims_if, r_if_verify) = proof_if
            .verify::<EdwardsProjective, _>(&interleaved_init_final_hashes, &mut transcript);
        assert_eq!(r_if_verify, r_if);
    }
}
