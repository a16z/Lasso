pub const XLEN: usize = 32;
pub const REGISTER_COUNT: u64 = 32;
pub const REGISTER_START_ADDRESS: usize = 0;
pub const RAM_START_ADDRESS: u64 = 0x80000000;
pub const BYTES_PER_INSTRUCTION: usize = 4;
pub const MEMORY_OPS_PER_INSTRUCTION: usize = 7;
pub const NUM_R1CS_POLYS: usize = 82;

pub const INPUT_START_ADDRESS: usize = 0x20000000;
pub const INPUT_END_ADDRESS: usize = 0x2FFFFFFF;
pub const OUTPUT_START_ADDRESS: usize = 0x30000000;
pub const OUTPUT_END_ADDRESS: usize = 0x3FFFFFFF;
pub const PANIC_ADDRESS: usize = 0x40000000;
