use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use halo2curves::group::Curve;
use merlin::Transcript;
use rand::rngs::StdRng;
use rand_core::RngCore;
use std::{collections::HashMap, marker::PhantomData};

use crate::jolt::trace::{rv::RVTraceRow, JoltProvableTrace};
use crate::lasso::memory_checking::NoPreprocessing;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::hyrax::{
    matrix_dimensions, BatchedHyraxOpeningProof, HyraxCommitment, HyraxGenerators,
};
use crate::poly::pedersen::PedersenGenerators;
use common::constants::{BYTES_PER_INSTRUCTION, NUM_R1CS_POLYS, RAM_START_ADDRESS, REGISTER_COUNT};
use common::RV32IM;
use common::{to_ram_address, ELFInstruction};

use rayon::prelude::*;

use crate::{
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{
        dense_mlpoly::DensePolynomial,
        identity_poly::IdentityPolynomial,
        structured_poly::{BatchablePolynomials, StructuredOpeningProof},
    },
    subprotocols::concatenated_commitment::{
        ConcatenatedPolynomialCommitment, ConcatenatedPolynomialOpeningProof,
    },
    utils::{errors::ProofVerifyError, is_power_of_two, math::Math},
};

pub type BytecodeProof<F, G> = MemoryCheckingProof<
    G,
    BytecodePolynomials<F, G>,
    BytecodeReadWriteOpenings<F>,
    BytecodeInitFinalOpenings<F>,
>;

#[derive(Debug, Clone, PartialEq)]
pub struct ELFRow {
    /// Memory address as read from the ELF.
    address: usize,
    /// Opcode of the instruction as read from the ELF.
    opcode: u64,
    /// Index of the destination register for this instruction (0 if register is unused).
    rd: u64,
    /// Index of the first source register for this instruction (0 if register is unused).
    rs1: u64,
    /// Index of the second source register for this instruction (0 if register is unused).
    rs2: u64,
    /// "Immediate" value for this instruction (0 if unused).
    imm: u64,
}

impl ELFRow {
    pub fn new(address: usize, opcode: u64, rd: u64, rs1: u64, rs2: u64, imm: u64) -> Self {
        Self {
            address,
            opcode,
            rd,
            rs1,
            rs2,
            imm,
        }
    }

    pub fn no_op(address: usize) -> Self {
        Self {
            address,
            opcode: 0,
            rd: 0,
            rs1: 0,
            rs2: 0,
            imm: 0,
        }
    }

    pub fn random(index: usize, rng: &mut StdRng) -> Self {
        Self {
            address: to_ram_address(index),
            opcode: rng.next_u64() % 64, // Roughly how many opcodes there are
            rd: rng.next_u64() % REGISTER_COUNT,
            rs1: rng.next_u64() % REGISTER_COUNT,
            rs2: rng.next_u64() % REGISTER_COUNT,
            imm: rng.next_u64() % (1 << 20), // U-format instructions have 20-bit imm values
        }
    }

    fn circuit_flags_packed<F: PrimeField>(&self) -> F {
        let circuit_flags: Vec<F> = RVTraceRow::new(
            0,
            RV32IM::from_repr(self.opcode as u8).unwrap(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .to_circuit_flags();

        let circuit_flags_bits: Vec<bool> = circuit_flags
            .iter()
            .map(|x| if x.is_zero() { false } else { true })
            .collect();

        println!("opcode as u8: {:?}", (self.opcode as u8));
        println!(
            "opcode: {:?}",
            RV32IM::from_repr(self.opcode as u8).unwrap()
        );
        println!("bits: {:?}", circuit_flags_bits);

        let mut bytes = [0u8; 2];
        for (idx, bit) in circuit_flags_bits.into_iter().enumerate() {
            let byte = idx / 8;
            let shift = idx % 8;
            bytes[byte] |= (bit as u8) << shift;
        }

        println!("bytes: {:?}", bytes);

        F::from_le_bytes_mod_order(&bytes)
    }
}

pub fn random_bytecode_trace(
    bytecode: &Vec<ELFRow>,
    num_ops: usize,
    rng: &mut StdRng,
) -> Vec<ELFRow> {
    let mut trace: Vec<ELFRow> = Vec::with_capacity(num_ops);
    for _ in 0..num_ops {
        trace.push(bytecode[rng.next_u64() as usize % bytecode.len()].clone());
    }
    trace
}

// TODO(JOLT-74): Consolidate ELFInstruction and ELFRow
impl From<&ELFInstruction> for ELFRow {
    fn from(value: &ELFInstruction) -> Self {
        Self::new(
            value.address as usize,
            value.opcode as u64,
            value.rd.unwrap_or(0),
            value.rs1.unwrap_or(0),
            value.rs2.unwrap_or(0),
            value.imm.unwrap_or(0) as u64, // imm is always cast to its 32-bit repr, signed or unsigned
        )
    }
}

/// Polynomial representation of bytecode as expected by Jolt –– each bytecode address maps to a
/// tuple, containing information about the instruction and its operands.
#[derive(Clone)]
pub struct FiveTuplePoly<F: PrimeField> {
    /// MLE of all opcodes in the bytecode.
    opcode: DensePolynomial<F>,
    /// MLE of all destination register indices in the bytecode.
    rd: DensePolynomial<F>,
    /// MLE of all first source register indices in the bytecode.
    rs1: DensePolynomial<F>,
    /// MLE of all second source register indices in the bytecode.
    rs2: DensePolynomial<F>,
    /// MLE of all immediate values in the bytecode.
    imm: DensePolynomial<F>,
}

impl<F: PrimeField> FiveTuplePoly<F> {
    #[tracing::instrument(skip_all, name = "FiveTuplePoly::from_elf")]
    fn from_elf(elf: &Vec<ELFRow>) -> Self {
        let len = elf.len().next_power_of_two();
        let mut opcodes = vec![0u64; len];
        let mut rds = vec![0u64; len];
        let mut rs1s = vec![0u64; len];
        let mut rs2s = vec![0u64; len];
        let mut imms = vec![0u64; len];

        elf.iter().enumerate().for_each(|(index, row)| {
            opcodes[index] = row.opcode;
            rds[index] = row.rd;
            rs1s[index] = row.rs1;
            rs2s[index] = row.rs2;
            imms[index] = row.imm;
        });

        let (opcode, rd, rs1, rs2, imm) = common::par_join_5!(
            || DensePolynomial::from_u64(&opcodes),
            || DensePolynomial::from_u64(&rds),
            || DensePolynomial::from_u64(&rs1s),
            || DensePolynomial::from_u64(&rs2s),
            || DensePolynomial::from_u64(&imms)
        );

        FiveTuplePoly {
            opcode,
            rd,
            rs1,
            rs2,
            imm,
        }
    }

    #[tracing::instrument(skip_all, name = "FiveTuplePoly::evaluate")]
    /// Evaluates the 5-tuple poly at the lagrange basis `chi`
    fn evaluate_at_chi(&self, chis: &Vec<F>) -> Vec<F> {
        vec![
            self.opcode.evaluate_at_chi(chis),
            self.rd.evaluate_at_chi(chis),
            self.rs1.evaluate_at_chi(chis),
            self.rs2.evaluate_at_chi(chis),
            self.imm.evaluate_at_chi(chis),
        ]
    }

    fn from_elf_r1cs(elf: &Vec<ELFRow>) -> Vec<F> {
        let len = elf.len();

        // pack: [opcode_1, ..., opcode_len, rd_1, ... rd_len, .... imm_1, ... imm_len]
        let mut packed = vec![F::zero(); 5 * len];

        let opcode_index = |index: usize| -> usize { index };
        let rs1_index = |index: usize| -> usize { len + index };
        let rs2_index = |index: usize| -> usize { 2 * len + index };
        let rd_index = |index: usize| -> usize { 3 * len + index };
        let imm_index = |index: usize| -> usize { 4 * len + index };

        // TODO(arasuarun): handle circuit flags here and not in prove_r1cs()
        // let mut circuit_flags = Vec::with_capacity(len * 15);

        for (index, row) in elf.iter().enumerate() {
            packed[opcode_index(index)] = F::from(row.opcode);
            packed[rs1_index(index)] = F::from(row.rs1);
            packed[rs2_index(index)] = F::from(row.rs2);
            packed[rd_index(index)] = F::from(row.rd);
            packed[imm_index(index)] = F::from(row.imm);
        }

        packed
    }
}

pub struct BytecodePolynomials<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    _group: PhantomData<G>,
    /// MLE of read/write addresses. For offline memory checking, each read is paired with a "virtual" write,
    /// so the read addresses and write addresses are the same.
    a_read_write: DensePolynomial<F>,
    /// MLE of read/write values. For offline memory checking, each read is paired with a "virtual" write,
    /// so the read values and write values are the same. There are multiple values (opcode, rd, rs1, etc.)
    /// associated with each memory address, so `v_read_write` comprises multiple polynomials.
    v_read_write: FiveTuplePoly<F>,
    /// MLE of init/final values. Bytecode is read-only data, so the final memory values are unchanged from
    /// the intiial memory values. There are multiple values (opcode, rd, rs1, etc.)
    /// associated with each memory address, so `v_init_final` comprises multiple polynomials.
    v_init_final: FiveTuplePoly<F>,
    /// MLE of the read timestamps.
    t_read: DensePolynomial<F>,
    /// MLE of the final timestamps.
    t_final: DensePolynomial<F>,
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> BytecodePolynomials<F, G> {
    #[tracing::instrument(skip_all, name = "BytecodePolynomials::new")]
    pub fn new(mut bytecode: Vec<ELFRow>, mut trace: Vec<ELFRow>) -> Self {
        Self::validate_bytecode(&bytecode, &trace);
        Self::preprocess(&mut bytecode, &mut trace);
        let max_bytecode_address = bytecode.iter().map(|instr| instr.address).max().unwrap();

        // Preprocessing should deal with padding.
        assert!(is_power_of_two(bytecode.len()));
        assert!(is_power_of_two(trace.len()));

        let num_ops = trace.len().next_power_of_two();
        // Bytecode addresses are 0-indexed, so we add one to `max_bytecode_address`
        let code_size = (max_bytecode_address + 1).next_power_of_two();

        let mut a_read_write_usize: Vec<usize> = vec![0; num_ops];
        let mut read_cts: Vec<usize> = vec![0; num_ops];
        let mut final_cts: Vec<usize> = vec![0; code_size];

        for (trace_index, trace) in trace.iter().enumerate() {
            let address = trace.address;
            debug_assert!(address < code_size);
            a_read_write_usize[trace_index] = address;
            let counter = final_cts[address];
            read_cts[trace_index] = counter;
            final_cts[address] = counter + 1;
        }

        let (v_read_write, v_init_final, a_read_write, t_read, t_final) = common::par_join_5!(
            || FiveTuplePoly::from_elf(&trace),
            || FiveTuplePoly::from_elf(&bytecode),
            || DensePolynomial::from_usize(&a_read_write_usize),
            || DensePolynomial::from_usize(&read_cts),
            || DensePolynomial::from_usize(&final_cts)
        );

        Self {
            _group: PhantomData,
            a_read_write,
            v_read_write,
            v_init_final,
            t_read,
            t_final,
        }
    }

    pub fn get_polys_r1cs(&self) -> (Vec<F>, Vec<F>) {
        let a_read_write_evals = self.a_read_write.evals().clone();
        let v_read_write_evals = [
            self.v_read_write.opcode.evals(),
            self.v_read_write.rs1.evals(),
            self.v_read_write.rs2.evals(),
            self.v_read_write.rd.evals(),
            self.v_read_write.imm.evals(),
        ].concat();

        (a_read_write_evals, v_read_write_evals)
    }

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::new")]
    pub fn r1cs_polys_from_bytecode(
        mut bytecode: Vec<ELFRow>,
        mut trace: Vec<ELFRow>,
    ) -> [Vec<F>; 3] {
        // As R1CS isn't padded, measure length here before padding is applied.
        let num_ops: usize = trace.len();

        Self::validate_bytecode(&bytecode, &trace);
        Self::preprocess(&mut bytecode, &mut trace);

        // ignore the padding
        let trace = trace.drain(0..num_ops).collect::<Vec<ELFRow>>();

        let max_bytecode_address = bytecode.iter().map(|instr| instr.address).max().unwrap();

        // Bytecode addresses are 0-indexed, so we add one to `max_bytecode_address`
        let code_size = (max_bytecode_address + 1).next_power_of_two();

        let mut a_read_write_usize: Vec<usize> = vec![0; num_ops];
        let mut read_cts: Vec<usize> = vec![0; num_ops];
        let mut final_cts: Vec<usize> = vec![0; code_size];

        for (trace_index, trace) in trace.iter().take(num_ops).enumerate() {
            let address = trace.address * 4 + RAM_START_ADDRESS as usize;
            // debug_assert!(address < code_size);
            a_read_write_usize[trace_index] = trace.address;
            let counter = final_cts[trace.address];
            read_cts[trace_index] = counter;
            final_cts[trace.address] = counter + 1;
        }

        // create a closure to convert usize to F vector
        let to_f_vec = |vec: &Vec<usize>| -> Vec<F> {
            vec.iter()
                .map(|x| F::from_u64(*x as u64).unwrap())
                .collect()
        };

        let v_read_write = FiveTuplePoly::from_elf_r1cs(&trace);

        [
            to_f_vec(&a_read_write_usize),
            v_read_write,
            to_f_vec(&read_cts),
        ]
    }

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::validate_bytecode")]
    fn validate_bytecode(bytecode: &Vec<ELFRow>, trace: &Vec<ELFRow>) {
        let mut bytecode_map: HashMap<usize, &ELFRow> = HashMap::new();

        for bytecode_row in bytecode.iter() {
            bytecode_map.insert(bytecode_row.address, bytecode_row);
        }

        for trace_row in trace {
            assert_eq!(
                **bytecode_map
                    .get(&trace_row.address)
                    .expect("couldn't find in bytecode"),
                *trace_row
            );
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::preprocess")]
    fn preprocess(bytecode: &mut Vec<ELFRow>, trace: &mut Vec<ELFRow>) {
        for instruction in bytecode.iter_mut() {
            assert!(instruction.address >= RAM_START_ADDRESS as usize);
            assert!(instruction.address % BYTES_PER_INSTRUCTION == 0);
            instruction.address -= RAM_START_ADDRESS as usize;
            instruction.address /= BYTES_PER_INSTRUCTION;
            // Account for no-op instruction prepended to bytecode
            instruction.address += 1;
        }
        for instruction in trace.iter_mut() {
            assert!(instruction.address >= RAM_START_ADDRESS as usize);
            assert!(instruction.address % BYTES_PER_INSTRUCTION == 0);
            instruction.address -= RAM_START_ADDRESS as usize;
            instruction.address /= BYTES_PER_INSTRUCTION;
            // Account for no-op instruction prepended to bytecode
            instruction.address += 1;
        }

        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, ELFRow::no_op(0));

        // Bytecode: Pad to nearest power of 2
        for _ in bytecode.len()..bytecode.len().next_power_of_two() {
            bytecode.push(ELFRow::no_op(0));
        }

        // Trace: Pad to nearest power of 2
        for _trace_i in trace.len()..trace.len().next_power_of_two() {
            // All padded elements of the trace point at prepended no-op instruction
            trace.push(ELFRow::no_op(0));
        }
    }

    /// Computes the maximum number of group generators needed to commit to bytecode
    /// polynomials using Hyrax, given the maximum bytecode size and maximum trace length.
    pub fn num_generators(max_bytecode_size: usize, max_trace_length: usize) -> usize {
        // Account for no-op prepended to bytecode
        let max_bytecode_size = (max_bytecode_size + 1).next_power_of_two();
        let max_trace_length = max_trace_length.next_power_of_two();

        // a_read_write, t_read, v_read_write (opcode, rs1, rs2, rd, imm)
        let num_read_write_generators =
            matrix_dimensions(max_trace_length.log_2(), NUM_R1CS_POLYS).1;
        // t_final, v_init_final (opcode, rs1, rs2, rd, imm)
        let num_init_final_generators =
            matrix_dimensions((max_bytecode_size * 6).next_power_of_two().log_2(), 1).1;
        std::cmp::max(num_read_write_generators, num_init_final_generators)
    }
}

pub struct BatchedBytecodePolynomials<F: PrimeField> {
    // Contains:
    // - t_final, v_init_final
    combined_init_final: DensePolynomial<F>,
}

pub struct BytecodeCommitment<G: CurveGroup> {
    pub read_write_generators: HyraxGenerators<NUM_R1CS_POLYS, G>,
    pub read_write_commitments: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,

    // Combined commitment for:
    // - t_final, v_init_final
    pub init_final_commitments: ConcatenatedPolynomialCommitment<G>,
}

// impl<G: CurveGroup> BytecodeCommitment<G> {
//     pub fn get_polys_r1cs(&self) -> Vec<HyraxCommitment<NUM_R1CS_POLYS, G>> {
//         self.read_write_commitments
//     } 
// }

impl<F, G> BatchablePolynomials<G> for BytecodePolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type BatchedPolynomials = BatchedBytecodePolynomials<F>;
    type Commitment = BytecodeCommitment<G>;

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::batch")]
    fn batch(&self) -> Self::BatchedPolynomials {
        let combined_init_final = DensePolynomial::merge(&vec![
            &self.t_final,
            &self.v_init_final.opcode,
            &self.v_init_final.rd,
            &self.v_init_final.rs1,
            &self.v_init_final.rs2,
            &self.v_init_final.imm,
        ]);

        Self::BatchedPolynomials {
            combined_init_final,
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::commit")]
    fn commit(
        &self,
        batched_polys: &Self::BatchedPolynomials,
        pedersen_generators: &PedersenGenerators<G>,
    ) -> Self::Commitment {
        let read_write_generators =
            HyraxGenerators::new(self.a_read_write.get_num_vars(), pedersen_generators);
        let read_write_commitments = [
            &self.a_read_write,
            &self.t_read, // t_read isn't used in r1cs, but it's cleaner to commit to it as a rectangular matrix alongside everything else
            &self.v_read_write.opcode,
            &self.v_read_write.rd,
            &self.v_read_write.rs1,
            &self.v_read_write.rs2,
            &self.v_read_write.imm,
        ]
        .par_iter()
        .map(|poly| HyraxCommitment::commit(poly, &read_write_generators))
        .collect::<Vec<_>>();

        let init_final_commitments = batched_polys
            .combined_init_final
            .combined_commit(pedersen_generators);

        Self::Commitment {
            read_write_generators,
            read_write_commitments,
            init_final_commitments,
        }
    }
}

impl<F, G> MemoryCheckingProver<F, G, BytecodePolynomials<F, G>, NoPreprocessing>
    for BytecodeProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type ReadWriteOpenings = BytecodeReadWriteOpenings<F>;
    type InitFinalOpenings = BytecodeInitFinalOpenings<F>;

    // [a, opcode, rd, rs1, rs2, imm, t]
    type MemoryTuple = [F; 7];

    fn fingerprint(inputs: &Self::MemoryTuple, gamma: &F, tau: &F) -> F {
        let mut result = F::zero();
        let mut gamma_term = F::one();
        for input in inputs {
            result += *input * gamma_term;
            gamma_term *= gamma;
        }
        result - tau
    }

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::compute_leaves")]
    fn compute_leaves(
        _: &NoPreprocessing,
        polynomials: &BytecodePolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) {
        let num_ops = polynomials.a_read_write.len();
        let memory_size = polynomials.v_init_final.opcode.len();

        let read_fingerprints = (0..num_ops)
            .into_par_iter()
            .map(|i| {
                Self::fingerprint(
                    &[
                        polynomials.a_read_write[i],
                        polynomials.v_read_write.opcode[i],
                        polynomials.v_read_write.rd[i],
                        polynomials.v_read_write.rs1[i],
                        polynomials.v_read_write.rs2[i],
                        polynomials.v_read_write.imm[i],
                        polynomials.t_read[i],
                    ],
                    gamma,
                    tau,
                )
            })
            .collect();
        let read_leaves = DensePolynomial::new(read_fingerprints);

        let init_fingerprints = (0..memory_size)
            .into_par_iter()
            .map(|i| {
                Self::fingerprint(
                    &[
                        F::from_u64(i as u64).unwrap(),
                        polynomials.v_init_final.opcode[i],
                        polynomials.v_init_final.rd[i],
                        polynomials.v_init_final.rs1[i],
                        polynomials.v_init_final.rs2[i],
                        polynomials.v_init_final.imm[i],
                        F::zero(),
                    ],
                    gamma,
                    tau,
                )
            })
            .collect();
        let init_leaves = DensePolynomial::new(init_fingerprints);

        let write_fingerprints = (0..num_ops)
            .into_par_iter()
            .map(|i| {
                Self::fingerprint(
                    &[
                        polynomials.a_read_write[i],
                        polynomials.v_read_write.opcode[i],
                        polynomials.v_read_write.rd[i],
                        polynomials.v_read_write.rs1[i],
                        polynomials.v_read_write.rs2[i],
                        polynomials.v_read_write.imm[i],
                        polynomials.t_read[i] + F::one(),
                    ],
                    gamma,
                    tau,
                )
            })
            .collect();
        let write_leaves = DensePolynomial::new(write_fingerprints);

        let final_fingerprints = (0..memory_size)
            .into_par_iter()
            .map(|i| {
                Self::fingerprint(
                    &[
                        F::from_u64(i as u64).unwrap(),
                        polynomials.v_init_final.opcode[i],
                        polynomials.v_init_final.rd[i],
                        polynomials.v_init_final.rs1[i],
                        polynomials.v_init_final.rs2[i],
                        polynomials.v_init_final.imm[i],
                        polynomials.t_final[i],
                    ],
                    gamma,
                    tau,
                )
            })
            .collect();
        let final_leaves = DensePolynomial::new(final_fingerprints);

        (
            vec![read_leaves, write_leaves],
            vec![init_leaves, final_leaves],
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"Bytecode memory checking"
    }
}

impl<F, G> MemoryCheckingVerifier<F, G, BytecodePolynomials<F, G>, NoPreprocessing>
    for BytecodeProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    fn read_tuples(
        _: &NoPreprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![[
            openings.a_read_write_opening,
            openings.v_read_write_openings[0], // opcode
            openings.v_read_write_openings[1], // rd
            openings.v_read_write_openings[2], // rs1
            openings.v_read_write_openings[3], // rs2
            openings.v_read_write_openings[4], // imm
            openings.t_read_opening,
        ]]
    }
    fn write_tuples(
        _: &NoPreprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![[
            openings.a_read_write_opening,
            openings.v_read_write_openings[0], // opcode
            openings.v_read_write_openings[1], // rd
            openings.v_read_write_openings[2], // rs1
            openings.v_read_write_openings[3], // rs2
            openings.v_read_write_openings[4], // imm
            openings.t_read_opening + F::one(),
        ]]
    }
    fn init_tuples(
        _: &NoPreprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![[
            openings.a_init_final.unwrap(),
            openings.v_init_final[0], // opcode
            openings.v_init_final[1], // rd
            openings.v_init_final[2], // rs1
            openings.v_init_final[3], // rs2
            openings.v_init_final[4], // imm
            F::zero(),
        ]]
    }
    fn final_tuples(
        _: &NoPreprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![[
            openings.a_init_final.unwrap(),
            openings.v_init_final[0], // opcode
            openings.v_init_final[1], // rd
            openings.v_init_final[2], // rs1
            openings.v_init_final[3], // rs2
            openings.v_init_final[4], // imm
            openings.t_final,
        ]]
    }
}

pub struct BytecodeReadWriteOpenings<F>
where
    F: PrimeField,
{
    /// Evaluation of the a_read_write polynomial at the opening point.
    a_read_write_opening: F,
    /// Evaluation of the v_read_write polynomials at the opening point.
    v_read_write_openings: Vec<F>,
    /// Evaluation of the t_read polynomial at the opening point.
    t_read_opening: F,
}

impl<F, G> StructuredOpeningProof<F, G, BytecodePolynomials<F, G>> for BytecodeReadWriteOpenings<F>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Proof = BatchedHyraxOpeningProof<NUM_R1CS_POLYS, G>;

    #[tracing::instrument(skip_all, name = "BytecodeReadWriteOpenings::open")]
    fn open(polynomials: &BytecodePolynomials<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        Self {
            a_read_write_opening: polynomials.a_read_write.evaluate_at_chi(&chis),
            v_read_write_openings: polynomials.v_read_write.evaluate_at_chi(&chis),
            t_read_opening: polynomials.t_read.evaluate_at_chi(&chis),
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadWriteOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BytecodePolynomials<F, G>,
        _: &BatchedBytecodePolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        let mut combined_openings: Vec<F> = vec![
            openings.a_read_write_opening.clone(),
            openings.t_read_opening.clone(),
        ];
        combined_openings.extend(openings.v_read_write_openings.iter());

        BatchedHyraxOpeningProof::prove(
            &[
                &polynomials.a_read_write,
                &polynomials.t_read,
                &polynomials.v_read_write.opcode,
                &polynomials.v_read_write.rd,
                &polynomials.v_read_write.rs1,
                &polynomials.v_read_write.rs2,
                &polynomials.v_read_write.imm,
            ],
            &opening_point,
            &combined_openings,
            transcript,
        )
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &BytecodeCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let mut combined_openings: Vec<F> = vec![
            self.a_read_write_opening.clone(),
            self.t_read_opening.clone(),
        ];
        combined_openings.extend(self.v_read_write_openings.iter());

        opening_proof.verify(
            &commitment.read_write_generators,
            opening_point,
            &combined_openings,
            &commitment.read_write_commitments.iter().collect::<Vec<_>>(),
            transcript,
        )
    }
}

pub struct BytecodeInitFinalOpenings<F>
where
    F: PrimeField,
{
    /// Evaluation of the a_init_final polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    a_init_final: Option<F>,
    /// Evaluation of the v_init/final polynomials at the opening point.
    v_init_final: Vec<F>,
    /// Evaluation of the t_final polynomial at the opening point.
    t_final: F,
}

impl<F, G> StructuredOpeningProof<F, G, BytecodePolynomials<F, G>> for BytecodeInitFinalOpenings<F>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    #[tracing::instrument(skip_all, name = "BytecodeInitFinalOpenings::open")]
    fn open(polynomials: &BytecodePolynomials<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        Self {
            a_init_final: None,
            v_init_final: polynomials.v_init_final.evaluate_at_chi(&chis),
            t_final: polynomials.t_final.evaluate_at_chi(&chis),
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeInitFinalOpenings::prove_openings")]
    fn prove_openings(
        _: &BytecodePolynomials<F, G>,
        batched_polynomials: &BatchedBytecodePolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        let mut combined_openings: Vec<F> = vec![openings.t_final];
        combined_openings.extend(openings.v_init_final.iter());
        ConcatenatedPolynomialOpeningProof::prove(
            &batched_polynomials.combined_init_final,
            &opening_point,
            &combined_openings,
            transcript,
        )
    }

    fn compute_verifier_openings(&mut self, opening_point: &Vec<F>) {
        self.a_init_final =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &BytecodeCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let mut combined_openings: Vec<F> = vec![self.t_final.clone()];
        combined_openings.extend(self.v_init_final.iter());

        opening_proof.verify(
            opening_point,
            &combined_openings,
            &commitment.init_final_commitments,
            transcript,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_curve25519::{EdwardsProjective, Fr};
    use std::collections::HashSet;

    #[test]
    fn five_tuple_poly() {
        let program = vec![
            ELFRow::new(to_ram_address(0), 2u64, 3u64, 4u64, 5u64, 6u64),
            ELFRow::new(to_ram_address(1), 7u64, 8u64, 9u64, 10u64, 11u64),
            ELFRow::new(to_ram_address(2), 12u64, 13u64, 14u64, 15u64, 16u64),
            ELFRow::new(to_ram_address(3), 17u64, 18u64, 19u64, 20u64, 21u64),
        ];
        let tuple: FiveTuplePoly<Fr> = FiveTuplePoly::from_elf(&program);
        let expected_opcode: DensePolynomial<Fr> = DensePolynomial::from_usize(&vec![2, 7, 12, 17]);
        let expected_rd: DensePolynomial<Fr> = DensePolynomial::from_usize(&vec![3, 8, 13, 18]);
        let expected_rs1: DensePolynomial<Fr> = DensePolynomial::from_usize(&vec![4, 9, 14, 19]);
        let expected_rs2: DensePolynomial<Fr> = DensePolynomial::from_usize(&vec![5, 10, 15, 20]);
        let expected_imm: DensePolynomial<Fr> = DensePolynomial::from_usize(&vec![6, 11, 16, 21]);
        assert_eq!(tuple.opcode, expected_opcode);
        assert_eq!(tuple.rd, expected_rd);
        assert_eq!(tuple.rs1, expected_rs1);
        assert_eq!(tuple.rs2, expected_rs2);
        assert_eq!(tuple.imm, expected_imm);
    }

    fn get_difference<T: Clone + Eq + std::hash::Hash>(vec1: &[T], vec2: &[T]) -> Vec<T> {
        let set1: HashSet<_> = vec1.iter().cloned().collect();
        let set2: HashSet<_> = vec2.iter().cloned().collect();
        set1.symmetric_difference(&set2).cloned().collect()
    }

    #[test]
    fn bytecode_poly_leaf_construction() {
        let program = vec![
            ELFRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2u64),
            ELFRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4u64),
            ELFRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            ELFRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
        ];
        let trace = vec![
            ELFRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            ELFRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
        ];
        let polys: BytecodePolynomials<Fr, EdwardsProjective> =
            BytecodePolynomials::new(program, trace);

        let (gamma, tau) = (&Fr::from(100), &Fr::from(35));
        let (read_write_leaves, init_final_leaves) =
            BytecodeProof::compute_leaves(&NoPreprocessing, &polys, &gamma, &tau);
        let init_leaves = &init_final_leaves[0];
        let read_leaves = &read_write_leaves[0];
        let write_leaves = &read_write_leaves[1];
        let final_leaves = &init_final_leaves[1];

        let read_final_leaves = vec![read_leaves.evals(), final_leaves.evals()].concat();
        let init_write_leaves = vec![init_leaves.evals(), write_leaves.evals()].concat();
        let difference: Vec<Fr> = get_difference(&read_final_leaves, &init_write_leaves);
        assert_eq!(difference.len(), 0);
    }

    #[test]
    fn e2e_memchecking() {
        let program = vec![
            ELFRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2u64),
            ELFRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4u64),
            ELFRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            ELFRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
        ];
        let trace = vec![
            ELFRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            ELFRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
        ];
        let num_generators = BytecodePolynomials::<Fr, EdwardsProjective>::num_generators(
            program.len(),
            trace.len(),
        );

        let polys: BytecodePolynomials<Fr, EdwardsProjective> =
            BytecodePolynomials::new(program, trace);

        let mut transcript = Transcript::new(b"test_transcript");

        let batched_polys = polys.batch();
        let generators = PedersenGenerators::new(num_generators, b"test");
        let commitments = polys.commit(&batched_polys, &generators);
        let proof = BytecodeProof::prove_memory_checking(
            &NoPreprocessing,
            &polys,
            &batched_polys,
            &mut transcript,
        );

        let mut transcript = Transcript::new(b"test_transcript");
        BytecodeProof::verify_memory_checking(
            &NoPreprocessing,
            proof,
            &commitments,
            &mut transcript,
        )
        .expect("proof should verify");
    }

    #[test]
    fn e2e_mem_checking_non_pow_2() {
        let program = vec![
            ELFRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2u64),
            ELFRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4u64),
            ELFRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            ELFRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            ELFRow::new(to_ram_address(4), 32u64, 32u64, 32u64, 32u64, 32u64),
        ];
        let trace = vec![
            ELFRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            ELFRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            ELFRow::new(to_ram_address(4), 32u64, 32u64, 32u64, 32u64, 32u64),
        ];

        let num_generators = BytecodePolynomials::<Fr, EdwardsProjective>::num_generators(
            program.len(),
            trace.len(),
        );
        let polys: BytecodePolynomials<Fr, EdwardsProjective> =
            BytecodePolynomials::new(program, trace);
        let batch = polys.batch();
        let generators = PedersenGenerators::new(num_generators, b"test");
        let commitments = polys.commit(&batch, &generators);

        let mut transcript = Transcript::new(b"test_transcript");

        let proof =
            BytecodeProof::prove_memory_checking(&NoPreprocessing, &polys, &batch, &mut transcript);

        let mut transcript = Transcript::new(b"test_transcript");
        BytecodeProof::verify_memory_checking(
            &NoPreprocessing,
            proof,
            &commitments,
            &mut transcript,
        )
        .expect("should verify");
    }

    #[test]
    #[should_panic]
    fn bytecode_validation_fake_trace() {
        let program = vec![
            ELFRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2u64),
            ELFRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4u64),
            ELFRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            ELFRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            ELFRow::new(to_ram_address(4), 32u64, 32u64, 32u64, 32u64, 32u64),
        ];
        let trace = vec![
            ELFRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            ELFRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            ELFRow::new(to_ram_address(5), 0u64, 0u64, 0u64, 0u64, 0u64), // no_op: shouldn't exist in pgoram
        ];
        let _polys: BytecodePolynomials<Fr, EdwardsProjective> =
            BytecodePolynomials::new(program, trace);
    }

    #[test]
    #[should_panic]
    fn bytecode_validation_bad_prog_increment() {
        let program = vec![
            ELFRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2u64),
            ELFRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4u64),
            ELFRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            ELFRow::new(to_ram_address(4), 16u64, 16u64, 16u64, 16u64, 16u64), // Increment by 2
        ];
        let trace = vec![
            ELFRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            ELFRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
        ];
        let _polys: BytecodePolynomials<Fr, EdwardsProjective> =
            BytecodePolynomials::new(program, trace);
    }
}
