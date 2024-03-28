use crate::{jolt::vm::{Jolt, JoltCommitments}, poly::{dense_mlpoly::DensePolynomial, hyrax::{HyraxCommitment, HyraxGenerators}}, r1cs::r1cs_shape::R1CSShape, utils::{thread::drop_in_background_thread, transcript::ProofTranscript}};
use crate::utils::transcript::AppendToTranscript;

use super::{constraints::R1CSBuilder, spartan::{SpartanError, UniformShapeBuilder, UniformSpartanKey, UniformSpartanProof}}; 

use ark_ec::CurveGroup;
use common::constants::{NUM_R1CS_POLYS, RAM_START_ADDRESS};
use ark_ff::PrimeField;
use merlin::Transcript;
use rayon::prelude::*;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

/// Reorder and drop first element [[a1, b1, c1], [a2, b2, c2]] => [[a2], [b2], [c2]]
#[tracing::instrument(skip_all)]
fn reassemble_segments<F: PrimeField>(jolt_witnesses: Vec<Vec<F>>) -> Vec<Vec<F>> {
  let trace_len = jolt_witnesses.len();
  let num_variables = jolt_witnesses[0].len();
  let mut result: Vec<Vec<F>> = vec![vec![F::ZERO; trace_len]; num_variables - 1]; // ignore 1 

  result.par_iter_mut().enumerate().for_each(|(variable_idx, variable_segment)| {
    for step in 0..trace_len {
      variable_segment[step] = jolt_witnesses[step][variable_idx]; // NOTE: 1 is at the end!
    }
  });

  result 
}

/// Reorder and drop first element [[a1, b1, c1], [a2, b2, c2]] => [[a2], [b2], [c2]]
#[tracing::instrument(skip_all, name = "reassemble_segments_partial")]
fn reassemble_segments_partial<F: PrimeField>(jolt_witnesses: Vec<Vec<F>>, num_front: usize, num_back: usize) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
  let trace_len = jolt_witnesses.len();
  let total_length = jolt_witnesses[0].len();
  let mut front_result: Vec<Vec<F>> = vec![vec![F::ZERO; trace_len]; num_front]; 
  let mut back_result: Vec<Vec<F>> = vec![vec![F::ZERO; trace_len]; num_back]; 

  // [1 || output_state] starts at the beginning
  front_result.par_iter_mut().enumerate().for_each(|(variable_idx, variable_segment)| {
    for step in 0..trace_len {
      variable_segment[step] = jolt_witnesses[step][variable_idx+1]; // NOTE: 1 is at the beginning!
    }
  });

  // [.. || aux] is the end
  back_result.par_iter_mut().enumerate().for_each(|(variable_idx, variable_segment)| {
    for step in 0..trace_len {
      variable_segment[step] = jolt_witnesses[step][(total_length-num_back) + variable_idx]; 
    }
  });

  drop_in_background_thread(jolt_witnesses);

  (front_result, back_result)
}

#[tracing::instrument(name = "synthesize_state_aux_segments", skip_all)]
fn synthesize_state_aux_segments<F: PrimeField>(inputs: &R1CSInputs<F>, num_state: usize, num_aux: usize) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
  let jolt_witnesses = synthesize_witnesses(inputs);
  // TODO(sragss / arasuarun): Synthsize witnesses should just return (io, aux)
  reassemble_segments_partial(jolt_witnesses, num_state, num_aux)
}

#[tracing::instrument(name = "synthesize_witnesses", skip_all)]
fn synthesize_witnesses<F: PrimeField>(inputs: &R1CSInputs<F>) -> Vec<Vec<F>> {
  let mut step_z = inputs.clone_to_stepwise();

  // Compute the aux
  let span = tracing::span!(tracing::Level::INFO, "calc_aux");
  let _guard = span.enter();
  step_z.par_chunks_mut(1 << 12).for_each(|step| {
    for i in 0..step.len() {
      R1CSBuilder::calculate_aux(&mut step[i]);
    }
  });

  step_z
}

#[derive(Clone, Debug, Default)]
pub struct R1CSInputs<F: PrimeField> {
    padded_trace_len: usize,
    bytecode_a: Vec<F>,
    bytecode_v: Vec<F>,
    packed_flags: Vec<F>,
    memreg_a_rw: Vec<F>,
    memreg_v_reads: Vec<F>,
    memreg_v_writes: Vec<F>,
    chunks_x: Vec<F>,
    chunks_y: Vec<F>,
    chunks_query: Vec<F>,
    lookup_outputs: Vec<F>,
    circuit_flags_bits: Vec<F>,
}

impl<F: PrimeField> R1CSInputs<F> {
  #[tracing::instrument(skip_all, name = "R1CSInputs::new")]
  pub fn new(
    padded_trace_len: usize,
    bytecode_a: Vec<F>,
    bytecode_v: Vec<F>,
    packed_flags: Vec<F>,
    memreg_a_rw: Vec<F>,
    memreg_v_reads: Vec<F>,
    memreg_v_writes: Vec<F>,
    chunks_x: Vec<F>,
    chunks_y: Vec<F>,
    chunks_query: Vec<F>,
    lookup_outputs: Vec<F>,
    circuit_flags_bits: Vec<F>
  ) -> Self {

    assert!(bytecode_a.len() % padded_trace_len == 0);
    assert!(bytecode_v.len() % padded_trace_len == 0);
    assert!(packed_flags.len() % padded_trace_len == 0);
    assert!(memreg_a_rw.len() % padded_trace_len == 0);
    assert!(memreg_v_reads.len() % padded_trace_len == 0);
    assert!(memreg_v_writes.len() % padded_trace_len == 0);
    assert!(chunks_x.len() % padded_trace_len == 0);
    assert!(chunks_y.len() % padded_trace_len == 0);
    assert!(chunks_query.len() % padded_trace_len == 0);
    assert!(lookup_outputs.len() % padded_trace_len == 0);
    assert!(circuit_flags_bits.len() % padded_trace_len == 0);

    Self {
      padded_trace_len,
      bytecode_a,
      bytecode_v,
      packed_flags,
      memreg_a_rw,
      memreg_v_reads,
      memreg_v_writes,
      chunks_x,
      chunks_y,
      chunks_query,
      lookup_outputs,
      circuit_flags_bits,
    }
  }

  #[tracing::instrument(skip_all, name = "R1CSInputs::clone_to_stepwise")]
  pub fn clone_to_stepwise(&self) -> Vec<Vec<F>> {
    const PREFIX_VARS_PER_STEP: usize = 5;

    // AUX_VARS_PER_STEP has to be greater than the number of additional vars pushed by the constraint system
    const AUX_VARS_PER_STEP: usize = 20; 
    let num_inputs_per_step = self.num_vars_per_step() + PREFIX_VARS_PER_STEP;

    let stepwise = (0..self.trace_len()).into_par_iter().map(|step_index| {
      let mut step: Vec<F> = Vec::with_capacity(num_inputs_per_step + AUX_VARS_PER_STEP);
      let program_counter = if step_index > 0 && self.bytecode_a[step_index] == F::ZERO {
        F::ZERO
      } else {
        self.bytecode_a[step_index] * F::from(4u64) + F::from(RAM_START_ADDRESS)
      };

      // 1 is constant, 0s in slots 1, 2 are filled by aux computation
      step.extend([F::from(1u64), F::from(0u64), F::from(0u64), F::from(step_index as u64), program_counter]);

      let push_to_step = |data: &Vec<F>, step: &mut Vec<F>| {
        let num_vals = data.len() / self.padded_trace_len;
        for var_index in 0..num_vals {
          step.push(data[var_index * self.padded_trace_len + step_index]);
        }
      };

      push_to_step(&self.bytecode_a, &mut step);
      push_to_step(&self.bytecode_v, &mut step);
      push_to_step(&self.packed_flags, &mut step);
      push_to_step(&self.memreg_a_rw, &mut step);
      push_to_step(&self.memreg_v_reads, &mut step);
      push_to_step(&self.memreg_v_writes, &mut step);
      push_to_step(&self.chunks_x, &mut step);
      push_to_step(&self.chunks_y, &mut step);
      push_to_step(&self.chunks_query, &mut step);
      push_to_step(&self.lookup_outputs, &mut step);
      push_to_step(&self.circuit_flags_bits, &mut step);

      assert_eq!(num_inputs_per_step, step.len());

      step
    }).collect();

    stepwise
  }


  pub fn trace_len(&self) -> usize {
    self.padded_trace_len
  }

  pub fn num_vars_per_step(&self) -> usize {
    let trace_len = self.trace_len();
    self.bytecode_a.len() / trace_len
      + self.bytecode_v.len() / trace_len
      + self.packed_flags.len() / trace_len
      + self.memreg_a_rw.len() / trace_len
      + self.memreg_v_reads.len() / trace_len
      + self.memreg_v_writes.len() / trace_len
      + self.chunks_x.len() / trace_len
      + self.chunks_y.len() / trace_len
      + self.chunks_query.len() / trace_len
      + self.lookup_outputs.len() / trace_len
      + self.circuit_flags_bits.len() / trace_len
  }

  #[tracing::instrument(skip_all, name = "R1CSInputs::trace_len_chunks")]
  pub fn clone_to_trace_len_chunks(&self, padded_trace_len: usize) -> Vec<Vec<F>> {
    // TODO(sragss / arasuarun): Explain why non-trace-len relevant stuff (ex: bytecode) gets chunked to padded_trace_len
    let mut chunks: Vec<Vec<F>> = Vec::new();
    chunks.par_extend(self.bytecode_a.par_chunks(padded_trace_len).map(|chunk| chunk.to_vec()));
    chunks.par_extend(self.bytecode_v.par_chunks(padded_trace_len).map(|chunk| chunk.to_vec()));
    chunks.par_extend(self.packed_flags.par_chunks(padded_trace_len).map(|chunk| chunk.to_vec()));
    chunks.par_extend(self.memreg_a_rw.par_chunks(padded_trace_len).map(|chunk| chunk.to_vec()));
    chunks.par_extend(self.memreg_v_reads.par_chunks(padded_trace_len).map(|chunk| chunk.to_vec()));
    chunks.par_extend(self.memreg_v_writes.par_chunks(padded_trace_len).map(|chunk| chunk.to_vec()));
    chunks.par_extend(self.chunks_x.par_chunks(padded_trace_len).map(|chunk| chunk.to_vec()));
    chunks.par_extend(self.chunks_y.par_chunks(padded_trace_len).map(|chunk| chunk.to_vec()));
    chunks.par_extend(self.chunks_query.par_chunks(padded_trace_len).map(|chunk| chunk.to_vec()));
    chunks.par_extend(self.lookup_outputs.par_chunks(padded_trace_len).map(|chunk| chunk.to_vec()));
    chunks.par_extend(self.circuit_flags_bits.par_chunks(padded_trace_len).map(|chunk| chunk.to_vec()));
    chunks
  }
}

/// Derived elements exclusive to the R1CS circuit.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSInternalCommitments<G: CurveGroup> {
  io: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,
  aux: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,
}

/// Commitments unique to R1CS.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSUniqueCommitments<G: CurveGroup> {
  internal_commitments: R1CSInternalCommitments<G>,

  chunks_x: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,
  chunks_y: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,
  lookup_outputs: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,
  packed_flags: HyraxCommitment<NUM_R1CS_POLYS, G>,
  circuit_flags: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,

  generators: HyraxGenerators<NUM_R1CS_POLYS, G>
}

impl<G: CurveGroup> R1CSUniqueCommitments<G> {
    pub fn new(
        internal_commitments: R1CSInternalCommitments<G>,
        chunks_x: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,
        chunks_y: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,
        lookup_outputs: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,
        packed_flags: HyraxCommitment<NUM_R1CS_POLYS, G>,
        circuit_flags: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,
        generators: HyraxGenerators<NUM_R1CS_POLYS, G>,
    ) -> Self {
      // TODO(sragss): Assert the sizes make sense.
        Self {
            internal_commitments,
            chunks_x,
            chunks_y,
            lookup_outputs,
            packed_flags,
            circuit_flags,
            generators,
        }
    }

    #[tracing::instrument(skip_all, name = "R1CSUniqueCommitments::append_to_transcript")]
    pub fn append_to_transcript(&self, transcript: &mut Transcript) {
      self.internal_commitments.io.iter().for_each(|comm| comm.append_to_transcript(b"io", transcript));
      self.internal_commitments.aux.iter().for_each(|comm| comm.append_to_transcript(b"aux", transcript));
      self.chunks_x.iter().for_each(|comm| comm.append_to_transcript(b"chunk_x", transcript));
      self.chunks_y.iter().for_each(|comm| comm.append_to_transcript(b"chunk_y", transcript));
      self.lookup_outputs.iter().for_each(|comm| comm.append_to_transcript(b"lookup_outputs", transcript));
      self.packed_flags.append_to_transcript(b"packed_flags", transcript);
      self.circuit_flags.iter().for_each(|comm| comm.append_to_transcript(b"circuit_flags", transcript));
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSProof<F: PrimeField, G: CurveGroup<ScalarField = F>>  {
  pub key: UniformSpartanKey<F>,
  proof: UniformSpartanProof<F, G>,
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> R1CSProof<F, G> {
  /// Computes the full witness in segments of len `padded_trace_len`, commits to new required intermediary variables.
  #[tracing::instrument(skip_all, name = "R1CSProof::compute_witness_commit")]
  pub fn compute_witness_commit(
      _W: usize, 
      _C: usize, 
      padded_trace_len: usize, 
      inputs: R1CSInputs<F>,
      generators: &HyraxGenerators<NUM_R1CS_POLYS, G>,
  ) -> Result<(UniformSpartanKey<F>, Vec<Vec<F>>, R1CSInternalCommitments<G>), SpartanError> {
      let span = tracing::span!(tracing::Level::TRACE, "shape_stuff");
      let _enter = span.enter();
      let mut jolt_shape = R1CSBuilder::default(); 
      R1CSBuilder::get_matrices(&mut jolt_shape); 
      let key = UniformSpartanProof::<F,G>::setup_precommitted(&jolt_shape, padded_trace_len)?;
      drop(_enter);
      drop(span);

      let (io_segments, aux_segments) = synthesize_state_aux_segments(&inputs, 4, jolt_shape.num_internal);

      // Commit to segments
      let commit_segments = |segments: &Vec<Vec<F>>| -> Vec<HyraxCommitment<NUM_R1CS_POLYS, G>> {
        let span = tracing::span!(tracing::Level::TRACE, "commit_segments");
        let _g = span.enter();
        segments.iter().map(|segment| {
          HyraxCommitment::commit_slice(segment, &generators)
        }).collect()
      };

      let io_comms = commit_segments(&io_segments);
      let aux_comms = commit_segments(&aux_segments);

      let r1cs_commitments = R1CSInternalCommitments::<G> {
        io: io_comms,
        aux: aux_comms,
      };

      let cloning_stuff_span = tracing::span!(tracing::Level::TRACE, "cloning_to_witness_segments");
      let _enter = cloning_stuff_span.enter();
      let inputs_segments = inputs.clone_to_trace_len_chunks(padded_trace_len);

      let mut w_segments: Vec<Vec<F>> = Vec::with_capacity(io_segments.len() + inputs_segments.len() + aux_segments.len());
      // TODO(sragss / arasuarun): rm clones in favor of references -- can be removed when HyraxCommitment can take Vec<Vec<F>>.
      w_segments.par_extend(io_segments.into_par_iter());
      w_segments.par_extend(inputs_segments.into_par_iter());
      w_segments.par_extend(aux_segments.into_par_iter());

      drop(_enter);
      drop(cloning_stuff_span);

      Ok((key, w_segments, r1cs_commitments))
  }

  #[tracing::instrument(skip_all, name = "R1CSProof::prove")]
  pub fn prove(
      key: UniformSpartanKey<F>,
      witness_segments: Vec<Vec<F>>,
      transcript: &mut Transcript
  ) -> Result<Self, SpartanError> {
    // TODO(sragss): Fiat shamir (relevant) commitments
      let proof = UniformSpartanProof::prove_precommitted(&key, witness_segments, transcript)?;
      Ok(R1CSProof::<F, G> {
        proof,
        key,
      })
  }

  fn format_commitments(
    jolt_commitments: &JoltCommitments<G>,
    C: usize
  ) -> Vec<&HyraxCommitment<NUM_R1CS_POLYS, G>>{
      let r1cs_commitments = &jolt_commitments.r1cs;
      let bytecode_read_write_commitments = &jolt_commitments.bytecode.read_write_commitments;
      let ram_a_v_commitments = &jolt_commitments.read_write_memory.a_v_read_write_commitments;
      let instruction_lookup_indices_commitments = &jolt_commitments.instruction_lookups.dim_read_commitment[0..C];

      let mut combined_commitments: Vec<&HyraxCommitment<NUM_R1CS_POLYS, G>> = Vec::new();
      combined_commitments.extend(r1cs_commitments.internal_commitments.io.iter());

      combined_commitments.push(&bytecode_read_write_commitments[0]); // a
      combined_commitments.push(&bytecode_read_write_commitments[2]); // opcode
      combined_commitments.push(&bytecode_read_write_commitments[3]); // rd
      combined_commitments.push(&bytecode_read_write_commitments[4]); // rs1
      combined_commitments.push(&bytecode_read_write_commitments[5]); // rs2
      combined_commitments.push(&bytecode_read_write_commitments[6]); // imm

      combined_commitments.push(&r1cs_commitments.packed_flags);

      combined_commitments.extend(ram_a_v_commitments.iter());

      combined_commitments.extend(r1cs_commitments.chunks_x.iter());
      combined_commitments.extend(r1cs_commitments.chunks_y.iter());

      combined_commitments.extend(instruction_lookup_indices_commitments.iter());

      combined_commitments.extend(r1cs_commitments.lookup_outputs.iter());

      combined_commitments.extend(r1cs_commitments.circuit_flags.iter());

      combined_commitments.extend(r1cs_commitments.internal_commitments.aux.iter());

      combined_commitments
  }

  pub fn verify(
    &self, 
    jolt_commitments: JoltCommitments<G>,
    C: usize,
    transcript: &mut Transcript) -> Result<(), SpartanError> {
    // TODO(sragss): Fiat shamir (relevant) commitments
    let witness_segment_commitments = Self::format_commitments(&jolt_commitments, C);
    self.proof.verify_precommitted(witness_segment_commitments, &self.key, &[], &jolt_commitments.r1cs.generators, transcript)
  }
}

impl<F: PrimeField> UniformShapeBuilder<F> for R1CSBuilder {
  fn single_step_shape(&self) -> R1CSShape<F> {
    let mut jolt_shape = R1CSBuilder::default(); 
    R1CSBuilder::get_matrices(&mut jolt_shape); 
    let constraints_F = jolt_shape.convert_to_field(); 
    let shape_single = R1CSShape::<F> {
        A: constraints_F.0,
        B: constraints_F.1,
        C: constraints_F.2,
        num_cons: jolt_shape.num_constraints,
        num_vars: jolt_shape.num_aux, // shouldn't include 1 or IO 
        num_io: jolt_shape.num_inputs,
    };

    shape_single.pad_vars()
  }
}
