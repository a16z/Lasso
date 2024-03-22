#![allow(dead_code)]

use std::{fs::File, io::Read, path::PathBuf};

use common::{self, constants::RAM_START_ADDRESS, serializable::Serializable};
use emulator::{
    cpu::{self, Xlen},
    default_terminal::DefaultTerminal,
    Emulator,
};

use object::{Object, ObjectSection};

mod decode;
mod emulator;
mod trace;

pub use common::rv_trace::{
    ELFInstruction, JoltDevice, MemoryState, RVTraceRow, RegisterState, RV32IM,
};

use crate::decode::decode_raw;

pub fn trace(elf: &PathBuf, inputs: Vec<u8>) -> (Vec<RVTraceRow>, JoltDevice) {
    let term = DefaultTerminal::new();
    let mut emulator = Emulator::new(Box::new(term));
    emulator.update_xlen(get_xlen());

    let mut jolt_device = JoltDevice::new();
    jolt_device.inputs = inputs;
    emulator.get_mut_cpu().get_mut_mmu().jolt_device = jolt_device;

    let mut elf_file = File::open(elf).unwrap();

    let mut elf_contents = Vec::new();
    elf_file.read_to_end(&mut elf_contents).unwrap();

    emulator.setup_program(elf_contents);

    let mut prev_pc = 0;
    loop {
        let pc = emulator.get_cpu().read_pc();
        emulator.tick();

        // This is a trick to see if the program has terminated by throwing itself
        // into an infinite loop. It seems to be a good heuristic for now but we
        // should eventually migrate to an explicit shutdown signal.
        if prev_pc == pc {
            break;
        }

        prev_pc = pc;
    }

    let mut rows = emulator.get_mut_cpu().tracer.rows.try_borrow_mut().unwrap();
    let mut output = Vec::new();
    output.append(&mut rows);
    drop(rows);

    let device = emulator.get_mut_cpu().get_mut_mmu().jolt_device.clone();

    (output, device)
}

pub fn decode(elf: &PathBuf) -> Vec<ELFInstruction> {
    let mut elf_file = File::open(elf).unwrap();
    let mut elf_contents = Vec::new();
    elf_file.read_to_end(&mut elf_contents).unwrap();

    let obj = object::File::parse(&*elf_contents).unwrap();

    let sections = obj
        .sections()
        .filter(|s| s.address() >= RAM_START_ADDRESS)
        .collect::<Vec<_>>();

    let mut instructions = Vec::new();
    for section in sections {
        let data = section.data().unwrap();

        for (chunk, word) in data.chunks(4).enumerate() {
            let word = u32::from_le_bytes(word.try_into().unwrap());
            let address = chunk as u64 * 4 + section.address();

            if let Ok(inst) = decode_raw(word) {
                if let Some(trace) = inst.trace {
                    let inst = trace(&inst, &get_xlen(), word, address);
                    instructions.push(inst);
                    continue;
                }
            }
            // Unrecognized instruction, or from a ReadOnlyData section
            instructions.push(ELFInstruction {
                address,
                opcode: RV32IM::from_str("UNIMPL"),
                raw: word,
                rs1: None,
                rs2: None,
                rd: None,
                imm: None,
            });
        }
    }

    instructions
}

fn get_xlen() -> Xlen {
    match common::constants::XLEN {
        32 => cpu::Xlen::Bit32,
        64 => cpu::Xlen::Bit64,
        _ => panic!("Emulator only supports 32 / 64 bit registers."),
    }
}
