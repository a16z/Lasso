use crate::jolt::instruction::add::ADDInstruction;
use crate::jolt::vm::bytecode::{random_bytecode_trace, BytecodePolynomials, BytecodeRow};
use crate::jolt::vm::instruction_lookups::InstructionPolynomials;
use crate::jolt::vm::read_write_memory::{random_memory_trace, RandomInstruction, ReadWriteMemory};
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, C, M, RV32I};
use crate::jolt::vm::Jolt;
use crate::poly::dense_mlpoly::bench::{init_commit_bench, run_commit_bench};
use ark_bn254::{Fr, G1Projective};
use common::constants::MEMORY_OPS_PER_INSTRUCTION;
use common::rv_trace::{ELFInstruction, JoltDevice, MemoryOp, RVTraceRow};
use criterion::black_box;
use jolt_sdk::host;
use merlin::Transcript;
use rand_core::SeedableRng;
use serde::Serialize;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Poly,
    EverythingExceptR1CS,
    Bytecode,
    ReadWriteMemory,
    InstructionLookups,
    Fibonacci,
    Sha2,
    Sha3,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(
    bench_type: BenchType,
    num_cycles: Option<usize>,
    memory_size: Option<usize>,
    bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::Poly => dense_ml_poly(),
        BenchType::EverythingExceptR1CS => {
            prove_e2e_except_r1cs(num_cycles, memory_size, bytecode_size)
        }
        BenchType::Bytecode => prove_bytecode(num_cycles, bytecode_size),
        BenchType::ReadWriteMemory => prove_memory(num_cycles, memory_size, bytecode_size),
        BenchType::InstructionLookups => prove_instruction_lookups(num_cycles),
        BenchType::Sha2 => sha2(),
        BenchType::Sha3 => sha3(),
        BenchType::Fibonacci => fibonacci(),
        _ => panic!("BenchType does not have a mapping"),
    }
}

fn prove_e2e_except_r1cs(
    num_cycles: Option<usize>,
    memory_size: Option<usize>,
    bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);

    let memory_size = memory_size.unwrap_or(1 << 22); // 4,194,304 = 4 MB
    let bytecode_size = bytecode_size.unwrap_or(1 << 16); // 65,536 = 64 kB
    let num_cycles = num_cycles.unwrap_or(1 << 16); // 65,536

    let ops: Vec<RV32I> = std::iter::repeat_with(|| RV32I::random_instruction(&mut rng))
        .take(num_cycles)
        .collect();

    let bytecode: Vec<ELFInstruction> = (0..bytecode_size)
        .map(|i| ELFInstruction::random(i, &mut rng))
        .collect();
    let memory_trace = random_memory_trace(&bytecode, memory_size, num_cycles, &mut rng);
    let bytecode_rows: Vec<BytecodeRow> = (0..bytecode_size)
        .map(|i| BytecodeRow::random(i, &mut rng))
        .collect();
    let bytecode_trace = random_bytecode_trace(&bytecode_rows, num_cycles, &mut rng);

    let preprocessing = RV32IJoltVM::preprocess(
        bytecode,
        JoltDevice::new(),
        bytecode_size,
        memory_size,
        num_cycles,
    );
    let mut transcript = Transcript::new(b"example");

    let work = Box::new(move || {
        let _: (_, BytecodePolynomials<Fr, G1Projective>, _) = RV32IJoltVM::prove_bytecode(
            &preprocessing.bytecode,
            bytecode_trace,
            &preprocessing.generators,
            &mut transcript,
        );
        let _: (_, ReadWriteMemory<Fr, G1Projective>, _) = RV32IJoltVM::prove_memory(
            &preprocessing.read_write_memory,
            memory_trace,
            &preprocessing.generators,
            &mut transcript,
        );
        let _: (_, InstructionPolynomials<Fr, G1Projective>, _) =
            RV32IJoltVM::prove_instruction_lookups(
                &preprocessing.instruction_lookups,
                ops,
                &preprocessing.generators,
                &mut transcript,
            );
    });
    vec![(
        tracing::info_span!("prove_bytecode + prove_memory + prove_instruction_lookups"),
        work,
    )]
}

fn prove_bytecode(
    num_cycles: Option<usize>,
    bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);

    let bytecode_size = bytecode_size.unwrap_or(1 << 16); // 65,536 = 64 kB
    let num_cycles = num_cycles.unwrap_or(1 << 16); // 65,536

    let bytecode: Vec<ELFInstruction> = (0..bytecode_size)
        .map(|i| ELFInstruction::random(i, &mut rng))
        .collect();

    let bytecode_rows: Vec<BytecodeRow> = bytecode
        .iter()
        .map(|instr| BytecodeRow::from_instruction::<RV32I>(instr))
        .collect();
    let bytecode_trace = random_bytecode_trace(&bytecode_rows, num_cycles, &mut rng);

    let preprocessing =
        RV32IJoltVM::preprocess(bytecode, JoltDevice::new(), bytecode_size, 1, num_cycles);
    let mut transcript = Transcript::new(b"example");

    let work = Box::new(move || {
        let _: (_, BytecodePolynomials<Fr, G1Projective>, _) = RV32IJoltVM::prove_bytecode(
            &preprocessing.bytecode,
            bytecode_trace,
            &preprocessing.generators,
            &mut transcript,
        );
    });
    vec![(tracing::info_span!("prove_bytecode"), work)]
}

fn prove_memory(
    num_cycles: Option<usize>,
    memory_size: Option<usize>,
    bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);

    let memory_size = memory_size.unwrap_or(1 << 22); // 4,194,304 = 4 MB
    let bytecode_size = bytecode_size.unwrap_or(1 << 16); // 65,536 = 64 kB
    let num_cycles = num_cycles.unwrap_or(1 << 16); // 65,536

    let bytecode: Vec<ELFInstruction> = (0..bytecode_size)
        .map(|i| ELFInstruction::random(i, &mut rng))
        .collect();
    let memory_trace = random_memory_trace(&bytecode, memory_size, num_cycles, &mut rng);

    let preprocessing = RV32IJoltVM::preprocess(
        bytecode,
        JoltDevice::new(),
        bytecode_size,
        memory_size,
        num_cycles,
    );

    let work = Box::new(move || {
        let mut transcript = Transcript::new(b"example");
        let _: (_, ReadWriteMemory<Fr, G1Projective>, _) = RV32IJoltVM::prove_memory(
            &preprocessing.read_write_memory,
            memory_trace,
            &preprocessing.generators,
            &mut transcript,
        );
    });
    vec![(tracing::info_span!("prove_memory"), work)]
}

fn prove_instruction_lookups(num_cycles: Option<usize>) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);

    let num_cycles = num_cycles.unwrap_or(1 << 16); // 65,536
    let ops: Vec<RV32I> = std::iter::repeat_with(|| RV32I::random_instruction(&mut rng))
        .take(num_cycles)
        .collect();

    let preprocessing = RV32IJoltVM::preprocess(vec![], JoltDevice::new(), 1, 1, num_cycles);
    let mut transcript = Transcript::new(b"example");

    let work = Box::new(move || {
        let _: (_, InstructionPolynomials<Fr, G1Projective>, _) =
            RV32IJoltVM::prove_instruction_lookups(
                &preprocessing.instruction_lookups,
                ops,
                &preprocessing.generators,
                &mut transcript,
            );
    });
    vec![(tracing::info_span!("prove_instruction_lookups"), work)]
}

fn dense_ml_poly() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let log_sizes = [20];
    let mut tasks = Vec::new();

    // Normal benchmark
    for &log_size in &log_sizes {
        let (gens, poly) = init_commit_bench(log_size);
        let task = move || {
            black_box(run_commit_bench(gens, poly));
        };
        tasks.push((
            tracing::info_span!("DensePoly::commit", log_size = log_size),
            Box::new(task) as Box<dyn FnOnce()>,
        ));
    }
    tasks
}

fn fibonacci() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("fibonacci-guest", &9u32)
}

fn sha2() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("sha2-guest", &vec![5u8; 2048])
}

fn sha3() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("sha3-guest", &vec![5u8; 2048])
}

fn prove_example<T: Serialize>(
    example_name: &str,
    input: &T,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut tasks = Vec::new();
    let program = host::Program::new(example_name).input(input);

    let task = move || {
        let (trace, bytecode, io_device) = program.trace();

        let bytecode_trace: Vec<BytecodeRow> = trace
            .iter()
            .map(|row| BytecodeRow::from_instruction::<RV32I>(&row.instruction))
            .collect();

        let instructions_r1cs: Vec<RV32I> = trace
            .iter()
            .map(|row| {
                if let Ok(jolt_instruction) = RV32I::try_from(row) {
                    jolt_instruction
                } else {
                    ADDInstruction(0_u64, 0_u64).into()
                }
            })
            .collect();

        let memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]> =
            trace.iter().map(|row| row.into()).collect();
        let circuit_flags = trace
            .iter()
            .flat_map(|row| {
                row.instruction
                    .to_circuit_flags()
                    .iter()
                    .map(|&flag| flag.into())
                    .collect::<Vec<Fr>>()
            })
            .collect();

        let preprocessing: crate::jolt::vm::JoltPreprocessing<
            ark_ff::Fp<ark_ff::MontBackend<ark_bn254::FrConfig, 4>, 4>,
            ark_ec::short_weierstrass::Projective<ark_bn254::g1::Config>,
        > = RV32IJoltVM::preprocess(bytecode.clone(), io_device, 1 << 20, 1 << 20, 1 << 22);

        let (jolt_proof, jolt_commitments) = <RV32IJoltVM as Jolt<_, G1Projective, C, M>>::prove(
            bytecode,
            bytecode_trace,
            memory_trace,
            instructions_r1cs,
            circuit_flags,
            preprocessing.clone(),
        );
        let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}
