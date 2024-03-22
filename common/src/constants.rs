pub const XLEN: usize = 32;
pub const REGISTER_COUNT: u64 = 32;
pub const BYTES_PER_INSTRUCTION: usize = 4;
pub const MEMORY_OPS_PER_INSTRUCTION: usize = 7;
pub const NUM_R1CS_POLYS: usize = 1;

pub const RAM_START_ADDRESS: u64 = 0x80000000;
pub const MAX_INPUT_SIZE: u64 = 4096;
pub const MAX_OUTPUT_SIZE: u64 = 4096;
pub const RAM_WITNESS_OFFSET: u64 =
    (REGISTER_COUNT + MAX_INPUT_SIZE + MAX_OUTPUT_SIZE + 1).next_power_of_two();
pub const INPUT_START_ADDRESS: u64 = RAM_START_ADDRESS - RAM_WITNESS_OFFSET + REGISTER_COUNT;
pub const INPUT_END_ADDRESS: u64 = INPUT_START_ADDRESS + MAX_INPUT_SIZE;
pub const OUTPUT_START_ADDRESS: u64 = INPUT_END_ADDRESS + 1;
pub const OUTPUT_END_ADDRESS: u64 = OUTPUT_START_ADDRESS + MAX_OUTPUT_SIZE;
pub const PANIC_ADDRESS: u64 = OUTPUT_END_ADDRESS + 1;

pub const fn memory_address_to_witness_index(address: u64) -> usize {
    (address + RAM_WITNESS_OFFSET - RAM_START_ADDRESS) as usize
}
pub const fn witness_index_to_memory_address(index: usize) -> u64 {
    index as u64 + RAM_START_ADDRESS - RAM_WITNESS_OFFSET
}
// Layout of the witness (where || denotes concatenation):
//     registers || inputs || outputs || panic || padding || RAM
// Layout of VM memory:
//     peripheral devices || inputs || outputs || panic || padding || RAM
// Notably, we want to be able to map the VM memory address space to witness indices
// using a constant shift, namely (RAM_WITNESS_OFFSET + RAM_START_ADDRESS)
