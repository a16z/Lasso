use crate::constants::{MEMORY_OPS_PER_INSTRUCTION, PANIC_ADDRESS, INPUT_START_ADDRESS, OUTPUT_START_ADDRESS};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};
use strum_macros::FromRepr;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct RVTraceRow {
    pub instruction: ELFInstruction,
    pub register_state: RegisterState,
    pub memory_state: Option<MemoryState>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum MemoryOp {
    Read(u64, u64),  // (address, value)
    Write(u64, u64), // (address, new_value)
}

impl MemoryOp {
    pub fn no_op() -> Self {
        Self::Read(0, 0)
    }
}

fn sum_u64_i32(a: u64, b: i32) -> u64 {
    if b.is_negative() {
        let abs_b = b.abs() as u64;
        if a < abs_b {
            panic!("overflow")
        }
        a - abs_b
    } else {
        let b_u64: u64 = b.try_into().expect("failed u64 convesion");
        a + b_u64
    }
}

impl Into<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]> for &RVTraceRow {
    fn into(self) -> [MemoryOp; MEMORY_OPS_PER_INSTRUCTION] {
        let instruction_type = self.instruction.opcode.instruction_type();

        let rs1_read = || {
            MemoryOp::Read(
                self.instruction.rs1.unwrap(),
                self.register_state.rs1_val.unwrap(),
            )
        };
        let rs2_read = || {
            MemoryOp::Read(
                self.instruction.rs2.unwrap(),
                self.register_state.rs2_val.unwrap(),
            )
        };
        let rd_write = || {
            MemoryOp::Write(
                self.instruction.rd.unwrap(),
                self.register_state.rd_post_val.unwrap(),
            )
        };

        let ram_byte_read = |index: usize| match self.memory_state {
            Some(MemoryState::Read { address, value }) => (self.register_state.rd_post_val.unwrap() >> (index * 8)) as u8,
            Some(MemoryState::Write {
                address,
                pre_value,
                post_value,
            }) => (pre_value >> (index * 8)) as u8,
            None => panic!("Memory state not found"),
        };
        let ram_byte_written = |index: usize| match self.memory_state {
            Some(MemoryState::Read { address, value }) => panic!("Unexpected MemoryState::Read"),
            Some(MemoryState::Write {
                address,
                pre_value,
                post_value,
            }) => (post_value >> (index * 8)) as u8,
            None => panic!("Memory state not found"),
        };

        let rs1_offset = || -> u64 {
            let rs1_val = self.register_state.rs1_val.unwrap();
            let imm = self.instruction.imm.unwrap();
            sum_u64_i32(rs1_val, imm as i32)
        };

        // Canonical ordering for memory instructions
        // 0: rs1
        // 1: rs2
        // 2: rd
        // 3: byte_0
        // 4: byte_1
        // 5: byte_2
        // 6: byte_3
        // If any are empty a no_op is inserted.

        // Validation: Number of ops should be a multiple of 7
        match instruction_type {
            RV32InstructionFormat::R => [
                rs1_read(),
                rs2_read(),
                rd_write(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
            ],
            RV32InstructionFormat::U => [
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                rd_write(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
            ],
            RV32InstructionFormat::I => match self.instruction.opcode {
                RV32IM::ADDI
                | RV32IM::SLLI
                | RV32IM::SRLI
                | RV32IM::SRAI
                | RV32IM::ANDI
                | RV32IM::ORI
                | RV32IM::XORI
                | RV32IM::SLTI
                | RV32IM::SLTIU => [
                    rs1_read(),
                    MemoryOp::no_op(),
                    rd_write(),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                ],
                RV32IM::LB | RV32IM::LBU => [
                    rs1_read(),
                    MemoryOp::no_op(),
                    rd_write(),
                    MemoryOp::Read(rs1_offset(), ram_byte_read(0) as u64),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                ],
                RV32IM::LH | RV32IM::LHU => [
                    rs1_read(),
                    MemoryOp::no_op(),
                    rd_write(),
                    MemoryOp::Read(rs1_offset(), ram_byte_read(0) as u64),
                    MemoryOp::Read(rs1_offset() + 1, ram_byte_read(1) as u64),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                ],
                RV32IM::LW => [
                    rs1_read(),
                    MemoryOp::no_op(),
                    rd_write(),
                    MemoryOp::Read(rs1_offset(), ram_byte_read(0) as u64),
                    MemoryOp::Read(rs1_offset() + 1, ram_byte_read(1) as u64),
                    MemoryOp::Read(rs1_offset() + 2, ram_byte_read(2) as u64),
                    MemoryOp::Read(rs1_offset() + 3, ram_byte_read(3) as u64),
                ],
                RV32IM::JALR => [
                    rs1_read(),
                    MemoryOp::no_op(),
                    rd_write(),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                ],
                _ => unreachable!("{self:?}"),
            },
            RV32InstructionFormat::S => match self.instruction.opcode {
                RV32IM::SB => [
                    rs1_read(),
                    rs2_read(),
                    MemoryOp::no_op(),
                    MemoryOp::Write(rs1_offset(), ram_byte_written(0) as u64),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                ],
                RV32IM::SH => [
                    rs1_read(),
                    rs2_read(),
                    MemoryOp::no_op(),
                    MemoryOp::Write(rs1_offset(), ram_byte_written(0) as u64),
                    MemoryOp::Write(rs1_offset() + 1, ram_byte_written(1) as u64),
                    MemoryOp::no_op(),
                    MemoryOp::no_op(),
                ],
                RV32IM::SW => [
                    rs1_read(),
                    rs2_read(),
                    MemoryOp::no_op(),
                    MemoryOp::Write(rs1_offset(), ram_byte_written(0) as u64),
                    MemoryOp::Write(rs1_offset() + 1, ram_byte_written(1) as u64),
                    MemoryOp::Write(rs1_offset() + 2, ram_byte_written(2) as u64),
                    MemoryOp::Write(rs1_offset() + 3, ram_byte_written(3) as u64),
                ],
                _ => unreachable!(),
            },
            RV32InstructionFormat::UJ => [
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                rd_write(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
            ],
            RV32InstructionFormat::SB => [
                rs1_read(),
                rs2_read(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
                MemoryOp::no_op(),
            ],
            _ => unreachable!("{self:?}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ELFInstruction {
    pub address: u64,
    pub opcode: RV32IM,
    pub raw: u32,
    pub rs1: Option<u64>,
    pub rs2: Option<u64>,
    pub rd: Option<u64>,
    pub imm: Option<u32>,
}

pub const NUM_CIRCUIT_FLAGS: usize = 17;

impl ELFInstruction {
    #[rustfmt::skip]
    pub fn to_circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        // Jolt Appendix A.1
        // 0: first_operand == rs1 (1 if PC)
        // 1: second_operand == rs2 (1 if imm)
        // 2: Load instruction
        // 3: Store instruciton
        // 4: Jump instruction
        // 5: Branch instruciton
        // 6: Instruction writes lookup output to rd
        // 7: Instruction adds operands (ie, and uses the ADD lookup table)
        // 8: Instruction subtracts operands
        // 9: Instruction multiplies operands
        // 10: Instruction involves non-deterministic advice?
        // 11: Instruction asserts lookup output as false
        // 12: Instruction asserts lookup output as true
        // 13: Sign-bit of imm
        // 14: Is concat (Note: used to be is_lui)
        // Arasu: Extra to get things working
        // 15: is lui or auipc
        // 16: is jal

        let mut flags = [false; NUM_CIRCUIT_FLAGS];

        flags[0] = match self.opcode {
            RV32IM::JAL | RV32IM::LUI | RV32IM::AUIPC => true,
            _ => false,
        };

        flags[1] = match self.opcode {
            RV32IM::ADDI
            | RV32IM::XORI
            | RV32IM::ORI
            | RV32IM::ANDI
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::SLTI
            | RV32IM::SLTIU
            | RV32IM::AUIPC
            | RV32IM::JAL
            | RV32IM::JALR => true,
            _ => false,
        };

        flags[2] = match self.opcode {
            RV32IM::LB | RV32IM::LH | RV32IM::LW | RV32IM::LBU | RV32IM::LHU => true,
            _ => false,
        };

        flags[3] = match self.opcode {
            RV32IM::SB | RV32IM::SH | RV32IM::SW => true,
            _ => false,
        };

        flags[4] = match self.opcode {
            RV32IM::JAL | RV32IM::JALR => true,
            _ => false,
        };

        flags[5] = match self.opcode {
            RV32IM::BEQ | RV32IM::BNE | RV32IM::BLT | RV32IM::BGE | RV32IM::BLTU | RV32IM::BGEU => {
                true
            }
            _ => false,
        };

        // loads, stores, branches, jumps do not store the lookup output to rd (they may update rd in other ways)
        flags[6] = match self.opcode {
            RV32IM::LB
            | RV32IM::LH
            | RV32IM::LW
            | RV32IM::LBU
            | RV32IM::LHU
            | RV32IM::SB
            | RV32IM::SH
            | RV32IM::SW
            | RV32IM::BEQ
            | RV32IM::BNE
            | RV32IM::BLT
            | RV32IM::BGE
            | RV32IM::BLTU
            | RV32IM::BGEU
            | RV32IM::JAL
            | RV32IM::JALR
            | RV32IM::LUI => false,
            _ => true,
        };

        flags[7] = match self.opcode {
            RV32IM::ADD 
            | RV32IM::ADDI 
            // Store and load instructions only have one lookup operand (rs2 and the RAM word, respectively)
            // so we add the operand to a dummy operand of 0 in the circuit to obtain the lookup query.
            | RV32IM::SB
            | RV32IM::SH
            | RV32IM::SW
            | RV32IM::LB
            | RV32IM::LH
            | RV32IM::LW
            | RV32IM::LBU
            | RV32IM::LHU
            | RV32IM::JAL 
            | RV32IM::JALR 
            | RV32IM::AUIPC => true,
            _ => false,
        };

        flags[8] = match self.opcode {
            RV32IM::SUB => true,
            _ => false,
        };

        flags[9] = match self.opcode {
            RV32IM::MUL | RV32IM::MULU | RV32IM::MULH | RV32IM::MULSU => true,
            _ => false,
        };

        // TODO(JOLT-29): Used in the 'M' extension
        flags[10] = match self.opcode {
            _ => false,
        };

        // TODO(JOLT-29): Used in the 'M' extension
        flags[11] = match self.opcode {
            _ => false,
        };

        // TODO(JOLT-29): Used in the 'M' extension
        flags[12] = match self.opcode {
            _ => false,
        };

        let mask = 1u32 << 31;
        flags[13] = match self.imm {
            Some(imm) if imm & mask == mask => true,
            _ => false,
        };

        flags[14] = match self.opcode {
            RV32IM::XOR
            | RV32IM::XORI
            | RV32IM::OR
            | RV32IM::ORI
            | RV32IM::AND
            | RV32IM::ANDI
            | RV32IM::SLL
            | RV32IM::SRL
            | RV32IM::SRA
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::SLT
            | RV32IM::SLTU
            | RV32IM::SLTI
            | RV32IM::SLTIU
            | RV32IM::BEQ
            | RV32IM::BNE
            | RV32IM::BLT
            | RV32IM::BGE
            | RV32IM::BLTU
            | RV32IM::BGEU => true,
            _ => false,
        };

        flags[15] = match self.opcode {
            RV32IM::LUI | RV32IM::AUIPC => true,
            _ => false,
        };

        flags[16] = match self.opcode {
            RV32IM::SLL
            | RV32IM::SRL
            | RV32IM::SRA
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI => true,
            _ => false,
        };

        flags
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RegisterState {
    pub rs1_val: Option<u64>,
    pub rs2_val: Option<u64>,
    pub rd_pre_val: Option<u64>,
    pub rd_post_val: Option<u64>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum MemoryState {
    Read {
        address: u64,
        value: u64,
    },
    Write {
        address: u64,
        pre_value: u64,
        post_value: u64,
    },
}

impl RVTraceRow {
    pub fn imm_u64(&self) -> u64 {
        match self.instruction.opcode.instruction_type() {
            RV32InstructionFormat::R => unimplemented!("R type does not use imm u64"),
            RV32InstructionFormat::I => self.instruction.imm.unwrap() as u64,
            RV32InstructionFormat::U => ((self.instruction.imm.unwrap() as u32) << 12u32) as u64,
            RV32InstructionFormat::S => unimplemented!("S type does not use imm u64"),
            // UJ-type instructions point to address offsets: even numbers.
            // TODO(JOLT-88): De-normalizing was already done elsewhere. Should make this is consistent.
            RV32InstructionFormat::UJ => (self.instruction.imm.unwrap() as u64) << 0u64,
            _ => unimplemented!(),
        }
    }
}

// Reference: https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/notebooks/RISCV/RISCV_CARD.pdf
#[derive(Debug, PartialEq, Eq, Clone, Copy, FromRepr, Serialize, Deserialize, Hash)]
#[repr(u8)]
pub enum RV32IM {
    ADD,
    SUB,
    XOR,
    OR,
    AND,
    SLL,
    SRL,
    SRA,
    SLT,
    SLTU,
    ADDI,
    XORI,
    ORI,
    ANDI,
    SLLI,
    SRLI,
    SRAI,
    SLTI,
    SLTIU,
    LB,
    LH,
    LW,
    LBU,
    LHU,
    SB,
    SH,
    SW,
    BEQ,
    BNE,
    BLT,
    BGE,
    BLTU,
    BGEU,
    JAL,
    JALR,
    LUI,
    AUIPC,
    ECALL,
    EBREAK,
    MUL,
    MULH,
    MULSU,
    MULU,
    DIV,
    DIVU,
    REM,
    REMU,
    UNIMPL,
}

impl RV32IM {
    pub fn from_str(s: &str) -> Self {
        match s {
            "ADD" => Self::ADD,
            "SUB" => Self::SUB,
            "XOR" => Self::XOR,
            "OR" => Self::OR,
            "AND" => Self::AND,
            "SLL" => Self::SLL,
            "SRL" => Self::SRL,
            "SRA" => Self::SRA,
            "SLT" => Self::SLT,
            "SLTU" => Self::SLTU,
            "ADDI" => Self::ADDI,
            "XORI" => Self::XORI,
            "ORI" => Self::ORI,
            "ANDI" => Self::ANDI,
            "SLLI" => Self::SLLI,
            "SRLI" => Self::SRLI,
            "SRAI" => Self::SRAI,
            "SLTI" => Self::SLTI,
            "SLTIU" => Self::SLTIU,
            "LB" => Self::LB,
            "LH" => Self::LH,
            "LW" => Self::LW,
            "LBU" => Self::LBU,
            "LHU" => Self::LHU,
            "SB" => Self::SB,
            "SH" => Self::SH,
            "SW" => Self::SW,
            "BEQ" => Self::BEQ,
            "BNE" => Self::BNE,
            "BLT" => Self::BLT,
            "BGE" => Self::BGE,
            "BLTU" => Self::BLTU,
            "BGEU" => Self::BGEU,
            "JAL" => Self::JAL,
            "JALR" => Self::JALR,
            "LUI" => Self::LUI,
            "AUIPC" => Self::AUIPC,
            "ECALL" => Self::ECALL,
            "EBREAK" => Self::EBREAK,
            "MUL" => Self::MUL,
            "MULH" => Self::MULH,
            "MULSU" => Self::MULSU,
            "MULU" => Self::MULU,
            "DIV" => Self::DIV,
            "DIVU" => Self::DIVU,
            "REM" => Self::REM,
            "REMU" => Self::REMU,
            "UNIMPL" => Self::UNIMPL,
            _ => panic!("Could not match instruction to RV32IM set."),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum RV32InstructionFormat {
    R,
    I,
    S,
    SB,
    U,
    UJ,
}

impl RV32IM {
    #[rustfmt::skip] // keep matches pretty
    pub fn instruction_type(&self) -> RV32InstructionFormat {
        match self {
            RV32IM::ADD
            | RV32IM::SUB
            | RV32IM::XOR
            | RV32IM::OR
            | RV32IM::AND
            | RV32IM::SLL
            | RV32IM::SRL
            | RV32IM::SRA
            | RV32IM::SLT
            | RV32IM::SLTU
            | RV32IM::MUL
            | RV32IM::MULH
            | RV32IM::MULSU
            | RV32IM::MULU
            | RV32IM::DIV
            | RV32IM::DIVU
            | RV32IM::REM
            | RV32IM::REMU => RV32InstructionFormat::R,

            RV32IM::ADDI
            | RV32IM::XORI
            | RV32IM::ORI
            | RV32IM::ANDI
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::SLTI
            | RV32IM::SLTIU => RV32InstructionFormat::I,

            RV32IM::LB 
            | RV32IM::LH 
            | RV32IM::LW 
            | RV32IM::LBU 
            | RV32IM::LHU 
            | RV32IM::JALR => {
                RV32InstructionFormat::I
            }

            RV32IM::SB 
            | RV32IM::SH 
            | RV32IM::SW => RV32InstructionFormat::S,

            RV32IM::BEQ 
            | RV32IM::BNE 
            | RV32IM::BLT 
            | RV32IM::BGE 
            | RV32IM::BLTU 
            | RV32IM::BGEU => {
                RV32InstructionFormat::SB
            }

            RV32IM::LUI 
            | RV32IM::AUIPC => RV32InstructionFormat::U,

            RV32IM::JAL => RV32InstructionFormat::UJ,

            RV32IM::ECALL 
            | RV32IM::EBREAK 
            | RV32IM::UNIMPL => unimplemented!(),
        }
    }
}

/// Represented as a "peripheral device" in the RISC-V emulator, this captures
/// all reads from the reserved memory address space for program inputs and all writes
/// to the reserved memory address space for program outputs.
/// The inputs and outputs are part of the public inputs to the proof.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltDevice {
    pub inputs: Vec<u8>,
    pub outputs: Vec<u8>,
    pub panic: bool,
}

impl JoltDevice {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            panic: false,
        }
    }

    pub fn load(&self, address: u64) -> u8 {
        let internal_address = convert_read_address(address);
        if self.inputs.len() <= internal_address {
            0
        } else {
            self.inputs[internal_address]
        }
    }

    pub fn store(&mut self, address: u64, value: u8) {
        if address == PANIC_ADDRESS {
            self.panic = true;
            return;
        }

        let internal_address = convert_write_address(address);
        if self.outputs.len() <= internal_address {
            self.outputs.resize(internal_address + 1, 0);
        }

        self.outputs[internal_address] = value;
    }

    pub fn size(&self) -> usize {
        self.inputs.len() + self.outputs.len()
    }
}

fn convert_read_address(address: u64) -> usize {
    (address - INPUT_START_ADDRESS) as usize
}

fn convert_write_address(address: u64) -> usize {
    (address - OUTPUT_START_ADDRESS) as usize
}
