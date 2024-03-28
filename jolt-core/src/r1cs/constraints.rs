/// Handwritten circuit 

use ark_ff::PrimeField; 
use smallvec::{smallvec, SmallVec};
use rayon::prelude::*;
use common::{constants::{RAM_START_ADDRESS, RAM_WITNESS_OFFSET}, rv_trace::NUM_CIRCUIT_FLAGS};

/* Compiler Variables */
const C: usize = 4; 
const W: usize = 32;
const LOG_M: usize = 16; 
const MEMORY_ADDRESS_OFFSET: usize = (RAM_START_ADDRESS - RAM_WITNESS_OFFSET) as usize; 
// "memreg ops per step" 
const MOPS: usize = 7;
/* End of Compiler Variables */

const L_CHUNK: usize = LOG_M/2;  
const STEP_NUM_IDX: usize = 0; 
const PC_IDX: usize = 1; 

const ALL_ONES: i64 = 0xffffffff;

const ONE: (usize, i64) = (0, 1);

const STATE_LENGTH: usize = 2;

#[derive(Debug, PartialEq, Eq, Hash)]
enum InputType {
    Constant       = 0,
    OutputState    = 1,
    InputState     = 2,
    ProgARW        = 3,
    ProgVRW        = 4,
    MemregARW      = 5,
    MemregVReads   = 6,
    MemregVWrites  = 7,
    ChunksX        = 8,
    ChunksY        = 9,
    ChunksQuery    = 10,
    LookupOutput   = 11,
    OpFlags        = 12,
}

const INPUT_SIZES: &[(InputType, usize)] = &[
    (InputType::Constant,        1), 
    (InputType::OutputState,     STATE_LENGTH),
    (InputType::InputState,      STATE_LENGTH),
    (InputType::ProgARW,         1),
    (InputType::ProgVRW,         6),
    (InputType::MemregARW,       7),
    (InputType::MemregVReads,    7),
    (InputType::MemregVWrites,   7),
    (InputType::ChunksX,         C),
    (InputType::ChunksY,         C),
    (InputType::ChunksQuery,     C),
    (InputType::LookupOutput,    1),
    (InputType::OpFlags,         NUM_CIRCUIT_FLAGS),
];

const INPUT_OFFSETS: [usize; INPUT_SIZES.len()] = {
    let mut arr = [0; INPUT_SIZES.len()];
    let mut sum = 0;
    let mut i = 0;
    while i < INPUT_SIZES.len() {
        arr[i] = sum;
        sum += INPUT_SIZES[i].1;
        i += 1;
    }
    arr
};

fn GET_TOTAL_LEN() -> usize {
    INPUT_SIZES.iter().map(|(_, value)| value).sum()
}

fn GET_INDEX(input_type: InputType, offset: usize) -> usize {
    INPUT_OFFSETS[input_type as usize] + offset
}

const SMALLVEC_SIZE: usize = 4;

#[derive(Debug)]
pub struct R1CSBuilder {
    pub A: Vec<(usize, usize, i64)>,
    pub B: Vec<(usize, usize, i64)>,
    pub C: Vec<(usize, usize, i64)>,
    pub num_constraints: usize,
    pub num_variables: usize,
    pub num_inputs: usize, 
    pub num_aux: usize, 
    pub num_internal: usize, 
}

fn subtract_vectors(mut x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) -> SmallVec<[(usize, i64); SMALLVEC_SIZE]> {
    for (y_idx, y_coeff) in y {
        if let Some((_, x_coeff)) = x.iter_mut().find(|(x_idx, _)| *x_idx == y_idx) {
            *x_coeff -= y_coeff;
        } else {
            x.push((y_idx, -y_coeff));
        }
    }
    x
}

// big-endian
fn combine_chunks_vec(start_idx: usize, L: usize, N: usize) -> SmallVec<[(usize, i64); SMALLVEC_SIZE]> {
    let mut result = SmallVec::with_capacity(N);
    for i in 0..N {
        result.push((start_idx + i, 1 << ((N-1-i)*L)));
    }
    result
}

fn concat_constraint_vecs(mut x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) -> SmallVec<[(usize, i64); SMALLVEC_SIZE]> {
    for (y_idx, y_coeff) in y {
        if let Some((_, x_coeff)) = x.iter_mut().find(|(x_idx, _)| *x_idx == y_idx) {
            *x_coeff += y_coeff;
        } else {
            x.push((y_idx, y_coeff));
        }
    }
    x
}

fn i64_to_f<F: PrimeField>(num: i64) -> F {
    // TODO(sragss): Make from_u64
    if num < 0 {
        F::zero() - F::from_u64((-num) as u64).unwrap()
    } else {
        F::from_u64(num as u64).unwrap()
    }
}

impl R1CSBuilder {
    fn new_constraint(&mut self, a: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, b: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, c: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) {
        let row: usize = self.num_constraints; 
        let prepend_row = |vec: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, matrix: &mut Vec<(usize, usize, i64)>| {
            for (idx, val) in vec {
                    matrix.push((row, idx, val));
            }
        };
        prepend_row(a, &mut self.A); 
        prepend_row(b, &mut self.B); 
        prepend_row(c, &mut self.C); 
        self.num_constraints += 1;
    }

    fn assign_aux(&mut self) -> usize {
        let idx = self.num_variables; 
        self.num_aux += 1;
        self.num_variables += 1;
        self.num_internal += 1; 
        idx
    }   

    fn constr_abc(&mut self, a: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, b: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, c: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) {
        self.new_constraint(a, b, c); 
    }

    // Combines the L-bit wire values of [start_idx, ..., start_idx + N - 1] into a single value 
    // and constraints it to be equal to the wire value at result_idx.
    // The combination is big-endian with the most significant bit at start_idx.
    fn combine_constraint(&mut self, start_idx: usize, L: usize, N: usize, result_idx: usize) {
        let mut constraint_A = SmallVec::with_capacity(N);
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(result_idx, 1)];

        for i in 0..N {
            constraint_A.push((start_idx + i, 1 << ((N-1-i)*L)));
        }

        // Here the result_idx is assumed to already be assigned

        self.new_constraint(constraint_A, constraint_B, constraint_C); 
    }

    fn combine_le(&mut self, start_idx: usize, L: usize, N: usize) -> usize {
        let result_idx = Self::assign_aux(self);

        let mut constraint_A = SmallVec::with_capacity(N);
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(result_idx, 1)];

        for i in 0..N {
            constraint_A.push((start_idx + i, 1 << (i*L)));
        }

        self.new_constraint(constraint_A, constraint_B, constraint_C); 
        result_idx
    }

    fn combine_be(&mut self, start_idx: usize, L: usize, N: usize) -> usize {
        let result_idx = Self::assign_aux(self);

        let mut constraint_A = SmallVec::with_capacity(N);
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(result_idx, 1)];

        for i in 0..N {
            constraint_A.push((start_idx + i, 1 << ((N-1-i)*L)));
        }

        self.new_constraint(constraint_A, constraint_B, constraint_C); 
        result_idx
    }

    fn multiply(&mut self, x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) -> usize {
        let xy_idx = Self::assign_aux(self); 

        R1CSBuilder::constr_abc(self, 
            x, 
            y, 
            smallvec![(xy_idx, 1)], 
        ); 

        xy_idx 
    }

    fn if_else(&mut self, choice: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) -> usize {
        let z_idx = Self::assign_aux(self);

        let y_minus_x = subtract_vectors(y, x.clone()); 
        let z_minus_x = subtract_vectors(smallvec![(z_idx, 1)], x);

        self.new_constraint(choice, y_minus_x, z_minus_x); 
        z_idx 
    }

    fn if_else_simple(&mut self, choice: usize, x: usize, y: usize) -> usize {
        Self::if_else(self, smallvec![(choice, 1)], smallvec![(x, 1)], smallvec![(y, 1)])  
    }

    // The left side is an lc but the right is a single index
    fn eq_simple(&mut self, left_idx: usize, right_idx: usize) {
        let constraint_A = smallvec![(left_idx, 1)];
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(right_idx, 1)];

        self.new_constraint(constraint_A, constraint_B, constraint_C); 
    }

    // The left side is an lc but the right is a single index
    fn eq(&mut self, left: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, right_idx: usize, assign: bool) {
        let constraint_A = left;
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(right_idx, 1)];

        self.new_constraint(constraint_A, constraint_B, constraint_C); 
    }

    fn constr_prod_0(&mut self, x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, z: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) {
        let xy_idx = Self::assign_aux(self); 

        R1CSBuilder::constr_abc(self, 
            x, 
            y, 
            smallvec![(xy_idx, 1)], 
        ); 

        R1CSBuilder::constr_abc(self, 
            smallvec![(xy_idx, 1)], 
            z, 
            smallvec![], 
        ); 
    }

    pub fn default() -> Self {
        R1CSBuilder {
            A: Vec::with_capacity(100),
            B: Vec::with_capacity(100),
            C: Vec::with_capacity(100),
            num_constraints: 0,
            num_variables: GET_TOTAL_LEN(), // includes ("constant", 1) and ("output_state", ..)
            num_inputs: 0, // technically inputs are also aux, so keep this 0
            num_aux: GET_TOTAL_LEN()-1, // dont' include the constant  
            num_internal: 0, 
        }
    }

    pub fn get_matrices(instance: &mut R1CSBuilder) {
        // Parse the input indices 
        let opcode = GET_INDEX(InputType::ProgVRW, 0);
        let rd = GET_INDEX(InputType::ProgVRW, 1);
        let rs1 = GET_INDEX(InputType::ProgVRW, 2);
        let rs2 = GET_INDEX(InputType::ProgVRW, 3);
        let immediate_before_processing = GET_INDEX(InputType::ProgVRW, 4);
        let op_flags_packed = GET_INDEX(InputType::ProgVRW, 5);

        let is_load_instr: usize = GET_INDEX(InputType::OpFlags, 2);
        let is_store_instr: usize = GET_INDEX(InputType::OpFlags, 3);
        let is_jump_instr: usize = GET_INDEX(InputType::OpFlags, 4);
        let is_branch_instr: usize = GET_INDEX(InputType::OpFlags, 5);
        let if_update_rd_with_lookup_output: usize = GET_INDEX(InputType::OpFlags, 6);
        let is_add_instr: usize = GET_INDEX(InputType::OpFlags, 7);
        let is_sub_instr: usize = GET_INDEX(InputType::OpFlags, 8);
        let is_mul_instr: usize = GET_INDEX(InputType::OpFlags, 9);
        let sign_imm_flag: usize = GET_INDEX(InputType::OpFlags, 10);
        let is_concat: usize = GET_INDEX(InputType::OpFlags, 11);
        let is_lui_auipc: usize = GET_INDEX(InputType::OpFlags, 12);
        let is_shift: usize = GET_INDEX(InputType::OpFlags, 13);

        let PC = GET_INDEX(InputType::InputState, PC_IDX); 

        // Constraint 1: relation between PC and prog_a_rw
        // TODO(arasuarun): this should be done after fixing the padding issue for prog_a_rw

        // Combine flag_bits and check that they equal op_flags_packed. 
        R1CSBuilder::combine_constraint(instance, GET_INDEX(InputType::OpFlags, 0), 1, NUM_CIRCUIT_FLAGS, op_flags_packed);

        // Constriant: signal immediate <== if_else()([is_lui_auipc, immediate_before_processing, immediate_before_processing * (2**12)]);
        let immediate: usize = R1CSBuilder::if_else(instance, smallvec![(is_lui_auipc, 1)], smallvec![(immediate_before_processing, 1)], smallvec![(immediate_before_processing, 1<<12)]); 

        // Constraint: rs1 === memreg_a_rw[0];
        R1CSBuilder::eq_simple(instance, rs1, GET_INDEX(InputType::MemregARW, 0)); 
        // Constraint: rs2 === memreg_a_rw[1];
        R1CSBuilder::eq_simple(instance, rs2, GET_INDEX(InputType::MemregARW, 1));
        // Constraint: rd === memreg_a_rw[2];
        R1CSBuilder::eq_simple(instance, rd, GET_INDEX(InputType::MemregARW, 2));
        // Constraint: memreg_v_reads[0] === memreg_v_writes[0];
        R1CSBuilder::eq_simple(instance, GET_INDEX(InputType::MemregVReads, 0), GET_INDEX(InputType::MemregVWrites, 0)); 
        // Constraint: memreg_v_reads[1] === memreg_v_writes[1];
        R1CSBuilder::eq_simple(instance, GET_INDEX(InputType::MemregVReads, 1), GET_INDEX(InputType::MemregVWrites, 1));

        let rs1_val = GET_INDEX(InputType::MemregVReads, 0);
        let rs2_val = GET_INDEX(InputType::MemregVReads, 1);

        /*
            signal mem_v_bytes[MOPS()-3] <== subarray(3, MOPS()-3, MOPS())(memreg_v_writes);
            signal load_or_store_value <== combine_chunks_le(MOPS()-3, 8)(mem_v_bytes); 
        */
        let load_or_store_value = R1CSBuilder::combine_le(instance, GET_INDEX(InputType::MemregVWrites, 3), 8, MOPS-3); 

        /*
            signal _x <== if_else()([op_flags[0], rs1_val, PC]); // TODO: change this for virtual instructions
            signal x <== if_else()([is_load_inst + is_store_instr, _x, 0]);
            signal _y <== if_else()([op_flags[1], rs2_val, immediate]);
            signal y <== if_else()([is_store_instr, _y, load_or_store_value]);
         */
        let _x = R1CSBuilder::if_else_simple(instance, GET_INDEX(InputType::OpFlags, 0), rs1_val, PC);
        let x = R1CSBuilder::if_else(instance, smallvec![(is_load_instr, 1), (is_store_instr, 1)], smallvec![(_x, 1)], smallvec![]);
        let _y = R1CSBuilder::if_else_simple(instance, GET_INDEX(InputType::OpFlags, 1), rs2_val, immediate);
        let y = R1CSBuilder::if_else_simple(instance, is_load_instr, _y, load_or_store_value);

        /* 
            signal immediate_signed <== if_else()([sign_imm_flag, immediate, -ALL_ONES() + immediate - 1]);
            (is_load_instr + is_store_instr) * ((rs1_val + immediate_signed) - (memreg_a_rw[3] + MEMORY_ADDRESS_OFFSET())) === 0;
        */
        let immediate_signed = R1CSBuilder::if_else(instance, smallvec![(sign_imm_flag, 1)], smallvec![(immediate, 1)], smallvec![(immediate, 1), (0, -ALL_ONES - 1)]);
        R1CSBuilder::constr_abc(instance, smallvec![(is_load_instr, 1), (is_store_instr, 1)], smallvec![(rs1_val, 1), (immediate_signed, 1), (GET_INDEX(InputType::MemregARW, 3), -1), (0, -1 * MEMORY_ADDRESS_OFFSET as i64)], smallvec![]); 

        /*
            for (var i=1; i<MOPS()-3; i++) {
                // the first three are rs1, rs2, rd so memory starts are index 3
                (memreg_a_rw[3+i] - (memreg_a_rw[3] + i)) *  memreg_a_rw[3+i] === 0; 
            }
        */
        for i in 1..MOPS-3 {
            R1CSBuilder::constr_abc(instance, smallvec![(GET_INDEX(InputType::MemregARW, 3+i), 1), (GET_INDEX(InputType::MemregARW, 3), -1), (0, i as i64 * -1)], smallvec![(GET_INDEX(InputType::MemregARW, 3+i), 1)], smallvec![]);
        }

        /*
            for (var i=0; i<MOPS()-3; i++) {
                (memreg_v_reads[3+i] - memreg_v_writes[3+i]) * is_load_instr === 0;
            }
        */
        for i in 0..MOPS-3 {
            R1CSBuilder::constr_abc(instance, smallvec![(is_load_instr, 1)], smallvec![(GET_INDEX(InputType::MemregVReads, 3+i), 1), (GET_INDEX(InputType::MemregVWrites, 3+i), -1)], smallvec![]);
        }

        // is_store_instr * (load_or_store_value - rs2_val) === 0;
        R1CSBuilder::constr_abc(instance, smallvec![(is_store_instr, 1)], smallvec![(load_or_store_value, 1), (GET_INDEX(InputType::LookupOutput, 0), -1)], smallvec![]);


        /* Create the lookup query 
        - First, obtain combined_z_chunks (which should be the query)
        - Verify that the query is structured correctly, based on the instruction.
        - Then verify that the chunks of x, y, z are correct. 

        Constraints to check correctness of chunks_query 
            If NOT a concat query: chunks_query === chunks_z 
            If its a concat query: then chunks_query === zip(chunks_x, chunks_y)
        For concat, queries the tests are done later. 

            signal combined_z_chunks <== combine_chunks(C(), LOG_M())(chunks_query);
            is_add_instr * (combined_z_chunks - (x + y)) === 0; 
            is_sub_instr * (combined_z_chunks - (x + (ALL_ONES() - y + 1))) === 0; 

            // This creates a big aux witness value only for mul instructions. 
            signal is_mul_x <== is_mul_instr * x; 
            signal is_mul_xy <== is_mul_instr * y;
            is_mul_instr * (combined_z_chunks - is_mul_xy) === 0;
        */

        let combined_z_chunks = R1CSBuilder::combine_be(instance, GET_INDEX(InputType::ChunksQuery, 0), LOG_M, C);
        R1CSBuilder::constr_abc(instance,
            smallvec![(is_add_instr, 1)], 
            smallvec![(combined_z_chunks, 1), (x, -1), (y, -1)],
            smallvec![]
        ); 
        R1CSBuilder::constr_abc(instance,
            smallvec![(is_sub_instr, 1)], 
            smallvec![(combined_z_chunks, 1), (x, -1), (y, 1), (0, -1 * (ALL_ONES + 1))],
            smallvec![]
        ); 

        let is_mul_x = R1CSBuilder::multiply(instance, smallvec![(is_mul_instr, 1)], smallvec![(x, 1)]);
        let is_mul_xy = R1CSBuilder::multiply(instance, smallvec![(is_mul_x, 1)], smallvec![(y, 1)]);
        R1CSBuilder::constr_abc(instance,
            smallvec![(is_mul_instr, 1)], 
            smallvec![(combined_z_chunks, 1), (is_mul_xy, -1)],
            smallvec![]
        );

        /* Verify the chunks of x and y for concat instructions. 
            signal combined_x_chunks <== combine_chunks(C(), L_CHUNK())(chunks_x);
            (combined_x_chunks - x) * is_concat === 0; 

            Note that a wire value need not be assigned for combined x_chunks. 
            Repeat for y. 
        */
        R1CSBuilder::constr_abc(
            instance,  
            concat_constraint_vecs(
                combine_chunks_vec(GET_INDEX(InputType::ChunksX, 0), L_CHUNK, C), 
                smallvec![(x, -1)]
            ), 
            smallvec![(is_concat, 1)],
            smallvec![], 
        ); 
        R1CSBuilder::constr_abc(
            instance,  
            concat_constraint_vecs(
                combine_chunks_vec(GET_INDEX(InputType::ChunksY, 0), L_CHUNK, C),
                smallvec![(y, -1)]
            ), 
            smallvec![(is_concat, 1)],
            smallvec![], 
        ); 

        /* Concat query construction: 
            signal chunk_y_used[C()]; 
            for (var i=0; i<C(); i++) {
                chunk_y_used[i] <== if_else()([is_shift, chunks_y[i], chunks_y[C()-1]]);
                (chunks_query[i] - (chunk_y_used[i] + chunks_x[i] * 2**L_CHUNK())) * is_concat === 0;
            }  
        */
        for i in 0..C {
            let chunk_y_used_i = R1CSBuilder::if_else_simple(instance, is_shift, GET_INDEX(InputType::ChunksY, i), GET_INDEX(InputType::ChunksY, C-1));
            R1CSBuilder::constr_abc(
                instance,  
                smallvec![(GET_INDEX(InputType::ChunksQuery, i), 1), (chunk_y_used_i, -1), (GET_INDEX(InputType::ChunksX, i), (1<<L_CHUNK)*-1)],
                smallvec![(is_concat, 1)],
                smallvec![], 
            ); 
        }

        // TODO: handle case when C() doesn't divide W() 
        // Maybe like this: var idx_ms_chunk = C()-1; (chunks_query[idx_ms_chunk] - (chunks_x[idx_ms_chunk] + chunks_y[idx_ms_chunk] * 2**(L_MS_CHUNK()))) * is_concat === 0;

        /* Constraints for storing value in register rd.
            // lui doesn't need a lookup and simply requires the lookup_output to be set to immediate 
            // so it can be stored in the destination register. 

            signal rd_val <== memreg_v_writes[2];
            is_load_instr * (rd_val - load_or_store_value) === 0;

            component rd_test_lookup = prodZeroTest(3);
            rd_test_lookup.in <== [rd, if_update_rd_with_lookup_output, (rd_val - lookup_output)]; 
            component rd_test_jump = prodZeroTest(3); 
            rd_test_jump.in <== [rd, is_jump_instr, (rd_val - (prog_a_rw + 4))]; 
        */
        let rd_val = GET_INDEX(InputType::MemregVWrites, 2);
        R1CSBuilder::constr_abc(instance, smallvec![(is_load_instr, 1)], smallvec![(rd_val, 1), (GET_INDEX(InputType::LookupOutput, 0), -1)], smallvec![]);
        R1CSBuilder::constr_prod_0(
            instance, 
            smallvec![(rd, 1)], 
            smallvec![(if_update_rd_with_lookup_output, 1)], 
            smallvec![(rd_val, 1), (GET_INDEX(InputType::LookupOutput, 0), -1)], 
        );
        R1CSBuilder::constr_prod_0(
            instance, 
            smallvec![(rd, 1)], 
            smallvec![(is_jump_instr, 1)], 
            smallvec![(rd_val, 1), (PC, -1), (0, -4)], 
        ); 
        
        /* output_state[STEP_NUM_IDX()] <== input_state[STEP_NUM_IDX()]+1; */
        R1CSBuilder::constr_abc(instance, 
            smallvec![(GET_INDEX(InputType::OutputState, STEP_NUM_IDX), 1)], 
            smallvec![ONE], 
            smallvec![(GET_INDEX(InputType::InputState, STEP_NUM_IDX), 1), (0, 1)], 
        );
        R1CSBuilder::eq(instance, 
            smallvec![(GET_INDEX(InputType::InputState, STEP_NUM_IDX), 1), (0, 1)], 
            GET_INDEX(InputType::OutputState, STEP_NUM_IDX),
            true, 
        );


        /*  Set next PC 
            signal next_pc_j <== if_else()([
                is_jump_instr,  
                input_state[PC_IDX()] + 4, 
                lookup_output
            ]);
            signal next_pc_j_b <== if_else()([
                is_branch_instr * lookup_output, 
                next_pc_j, 
                input_state[PC_IDX()] + immediate_signed
            ]);
            output_state[PC_IDX()] <== next_pc_j_b;
        */
        let is_branch_times_lookup_output = R1CSBuilder::multiply(instance, smallvec![(is_branch_instr, 1)], smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1)]); 
        let next_pc_j = R1CSBuilder::if_else(instance, smallvec![(is_jump_instr, 1)], smallvec![(PC, 1), (0, 4)], smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1)]);
        let next_pc_j_b = R1CSBuilder::if_else(instance, smallvec![(is_branch_times_lookup_output, 1)], smallvec![(next_pc_j, 1)], smallvec![(PC, 1), (immediate_signed, 1)]);

        R1CSBuilder::eq(instance, 
            smallvec![(next_pc_j_b, 1)], 
            GET_INDEX(InputType::OutputState, PC_IDX), 
            true, 
        );

        R1CSBuilder::move_constant_to_end(instance);
    }

    pub fn calculate_aux<F: PrimeField>(inputs: &mut Vec<F>) {
        // Parse the input indices 
        let opcode = GET_INDEX(InputType::ProgVRW, 0);
        let rd = GET_INDEX(InputType::ProgVRW, 1);
        let rs1 = GET_INDEX(InputType::ProgVRW, 2);
        let rs2 = GET_INDEX(InputType::ProgVRW, 3);
        let immediate_before_processing = GET_INDEX(InputType::ProgVRW, 4);
        let op_flags_packed = GET_INDEX(InputType::ProgVRW, 5);

        let is_load_instr: usize = GET_INDEX(InputType::OpFlags, 2);
        let is_store_instr: usize = GET_INDEX(InputType::OpFlags, 3);
        let is_jump_instr: usize = GET_INDEX(InputType::OpFlags, 4);
        let is_branch_instr: usize = GET_INDEX(InputType::OpFlags, 5);
        let if_update_rd_with_lookup_output: usize = GET_INDEX(InputType::OpFlags, 6);
        let is_add_instr: usize = GET_INDEX(InputType::OpFlags, 7);
        let is_sub_instr: usize = GET_INDEX(InputType::OpFlags, 8);
        let is_mul_instr: usize = GET_INDEX(InputType::OpFlags, 9);
        let sign_imm_flag: usize = GET_INDEX(InputType::OpFlags, 10);
        let is_concat: usize = GET_INDEX(InputType::OpFlags, 11);
        let is_lui_auipc: usize = GET_INDEX(InputType::OpFlags, 12);
        let is_shift: usize = GET_INDEX(InputType::OpFlags, 13);

        let PC = GET_INDEX(InputType::InputState, PC_IDX); 


        // 1. let immediate: usize = R1CSBuilder::if_else(instance, smallvec![(is_lui_auipc, 1)], smallvec![(immediate_before_processing, 1)], smallvec![(immediate_before_processing, 1<<12)]); 
        let immediate = inputs.len(); 
        inputs.push(
            if inputs[is_lui_auipc].is_zero() {
                inputs[immediate_before_processing]
            } else {
                inputs[immediate_before_processing] * F::from_u64(1u64<<12).unwrap()
            }
        );

        let rs1_val = GET_INDEX(InputType::MemregVReads, 0);
        let rs2_val = GET_INDEX(InputType::MemregVReads, 1);

        // 2. let load_or_store_value = R1CSBuilder::combine_le(instance, GET_INDEX(InputType::MemregVWrites, 3), 8, MOPS-3);
        let load_or_store_value = inputs.len();
        inputs.push({
            let mut val = F::zero(); 
            let (L, N) = (8, MOPS-3);
            for i in 0..N {
                val += inputs[GET_INDEX(InputType::MemregVWrites, 3) + i] * F::from_u64(1u64<<(i*L)).unwrap();
            }
            val 
        });

        // 3. let _x = R1CSBuilder::if_else_simple(instance, GET_INDEX(InputType::OpFlags, 0), rs1_val, PC);
        let _x = inputs.len(); 
        inputs.push(
            if inputs[GET_INDEX(InputType::OpFlags, 0)].is_zero() {
                inputs[rs1_val]
            } else {
                inputs[PC]
            }
        );

        // 4.let x = R1CSBuilder::if_else(instance, smallvec![(is_load_instr, 1), (is_store_instr, 1)], smallvec![(_x, 1)], smallvec![]);
        let x = inputs.len(); 
        inputs.push(
            if (inputs[is_store_instr] + inputs[is_load_instr]).is_zero() {
                inputs[_x]
            } else {
                F::zero()
            }
        );


        // 5. let _y = R1CSBuilder::if_else_simple(instance, GET_INDEX(InputType::OpFlags, 1), rs2_val, immediate);
        let _y = inputs.len();
        inputs.push(
            if inputs[GET_INDEX(InputType::OpFlags, 1)].is_zero() {
                inputs[rs2_val]
            } else {
                inputs[immediate]
            }
        );


        // 6. let y = R1CSBuilder::if_else_simple(instance, is_load_instr, _y, load_or_store_value);
        let y = inputs.len();
        inputs.push(
            if inputs[is_load_instr].is_zero() {
                inputs[_y]
            } else {
                inputs[load_or_store_value]
            }
        );


        // 7. let immediate_signed = R1CSBuilder::if_else(instance, smallvec![(sign_imm_flag, 1)], smallvec![(immediate, 1)], smallvec![(immediate, 1), (0, -ALL_ONES - 1)]);
        let immediate_signed = inputs.len();
        inputs.push(
            if inputs[sign_imm_flag].is_zero() {
                inputs[immediate]
            } else {
                inputs[immediate] - F::from_u64((ALL_ONES+1) as u64).unwrap()
            }
        );

        // 8. let combined_z_chunks = R1CSBuilder::combine_be(instance, GET_INDEX(InputType::ChunksQuery, 0), LOG_M, C);
        let combined_z_chunks = inputs.len();
        inputs.push({
            let mut val = F::zero();
            let (L, N) = (LOG_M, C);
            for i in 0..N {
                val += inputs[GET_INDEX(InputType::ChunksQuery, 0) + i] * F::from_u64(1u64<<((N-1-i)*L)).unwrap();
            }
            val 
        });

        // 9. let is_mul_x = R1CSBuilder::multiply(instance, smallvec![(is_mul_instr, 1)], smallvec![(x, 1)]);
        let is_mul_x = inputs.len();
        inputs.push(
            inputs[is_mul_instr] * inputs[x]
        );


        // 10. let is_mul_xy = R1CSBuilder::multiply(instance, smallvec![(is_mul_x, 1)], smallvec![(y, 1)]);
        let is_mul_xy = inputs.len();
        inputs.push(
            inputs[is_mul_x] * inputs[y]
        );


        // 11-14. let chunk_y_used_i = R1CSBuilder::if_else_simple(&mut instance, is_shift, GET_INDEX(InputType::ChunksY, i), GET_INDEX(InputType::ChunksY, C-1));
        let mut chunk_y_used = [0; C];
        for i in 0..C {
            chunk_y_used[i] = inputs.len(); 
            inputs.push(
                if inputs[is_shift].is_zero() {
                    inputs[GET_INDEX(InputType::ChunksY, i)]
                } else {
                    inputs[GET_INDEX(InputType::ChunksY, C-1)]
                }
            );
        }

        let rd_val = GET_INDEX(InputType::MemregVWrites, 2);

        // 15. R1CSBuilder::constr_prod_0(smallvec![(rd, 1)], smallvec![(if_update_rd_with_lookup_output, 1)], smallvec![(rd_val, 1), (GET_INDEX(InputType::LookupOutput, 0), -1)], );
        let _ = inputs.len();
        inputs.push(inputs[rd] * inputs[if_update_rd_with_lookup_output]); 

        // 16. constr_prod_0[is_jump_instr, rd, rd_val, prog_a_rw, 4]
        let _ = inputs.len();
        inputs.push(inputs[rd] * inputs[is_jump_instr]); 

        // 17. let is_branch_times_lookup_output = R1CSBuilder::multiply(instance, smallvec![(is_branch_instr, 1)], smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1)]); 
        let is_branch_times_lookup_output = inputs.len();
        inputs.push(inputs[is_branch_instr] * inputs[GET_INDEX(InputType::LookupOutput, 0)]);

        // 18. let next_pc_j = R1CSBuilder::if_else(instance, smallvec![(is_jump_instr, 1)], smallvec![(PC, 1), (0, 4)], smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1)]);
        let next_pc_j = inputs.len();
        inputs.push(
            if inputs[is_jump_instr].is_zero() {
                inputs[PC] + F::from_u64(4).unwrap()
            } else {
                inputs[GET_INDEX(InputType::LookupOutput, 0)]
            }
        );

        // 19. let next_pc_j_b = R1CSBuilder::if_else(instance, smallvec![(is_branch_times_lookup_output, 1)], smallvec![(next_pc_j, 1)], smallvec![(PC, 1), (immediate_signed, 1)]);
        let next_pc_j_b = inputs.len();
        inputs.push(
            if inputs[is_branch_times_lookup_output].is_zero() {
                inputs[next_pc_j]
            } else {
                inputs[PC] + inputs[immediate_signed]
            }
        );


        // Assign outputs: (step counter, PC)
        inputs[1] = inputs[3]+F::ONE;  
        inputs[2] = inputs[next_pc_j_b];
    }

    fn move_constant_to_end(&mut self) {
        let modify_matrix= |mat: &mut Vec<(usize, usize, i64)>| {
            for &mut (_, ref mut value, _) in mat {
                if *value != 0 {
                    *value -= 1;
                } else {
                    *value = self.num_variables-1;
                }
            }
        };
        
        let _ = rayon::join(|| modify_matrix(&mut self.A),
        || rayon::join(|| modify_matrix(&mut self.B),
                       || modify_matrix(&mut self.C)));
    }

    #[tracing::instrument(skip_all, name = "Shape::convert_to_field")]
    pub fn convert_to_field<F: PrimeField>(&self) -> (Vec<(usize, usize, F)>, Vec<(usize, usize, F)>, Vec<(usize, usize, F)>) {
        (
            self.A.par_iter().map(|(row, idx, val)| (*row, *idx, i64_to_f::<F>(*val))).collect(),
            self.B.par_iter().map(|(row, idx, val)| (*row, *idx, i64_to_f::<F>(*val))).collect(),
            self.C.par_iter().map(|(row, idx, val)| (*row, *idx, i64_to_f::<F>(*val))).collect(),
        )
    }
}


    #[cfg(test)]
    mod tests {
        use super::*;

        // #[test]
        // fn test_get_matrices() {
        //     let instance = R1CSBuilder::<F>::get_matrices(None);
        //     println!("{:?}", instance.unwrap());

        // }

        // #[test]
        // fn test_get_matrices_all_zeros() {
        //     let instance = R1CSBuilder::<F>::get_matrices(Some(vec![F::from(0); GET_TOTAL_LEN()])).unwrap();
        //     // println!("{:?}", instance);
        //     println!("z vector is {:?}", instance.z.unwrap());
        //     // println!("z[2] vector is {:?}", instance.z.unwrap()[2] * F::from(-1));
        // }
    }