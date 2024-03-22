#![no_std]
#![no_main]

use sha3::{Keccak256, Digest};

#[jolt_sdk::main]
fn hash(input: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(input);
    let result = hasher.finalize();
    Into::<[u8; 32]>::into(result)
}


