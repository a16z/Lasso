#![cfg_attr(not(feature = "std"), no_std)]

extern crate jolt_sdk_macros;

pub use jolt_sdk_macros::main;
pub use postcard;

#[cfg(feature = "std")]
pub mod host {
    use core::{str::FromStr, u8};
    use std::{
        collections::HashMap,
        io::{self, Write},
        path::PathBuf,
        process::Command,
    };

    use common::rv_trace::{JoltDevice, RVTraceRow, RV32IM};
    use serde::Serialize;
    use tracer::ELFInstruction;

    pub struct Program {
        guest: String,
        input: Vec<u8>,
    }

    impl Program {
        pub fn new(guest: &str) -> Self {
            Self {
                guest: guest.to_string(),
                input: Vec::new(),
            }
        }

        pub fn input<T: Serialize>(mut self, input: &T) -> Self {
            let mut serialized = postcard::to_stdvec(input).unwrap();
            self.input.append(&mut serialized);

            self
        }

        pub fn trace(self) -> (Vec<RVTraceRow>, Vec<ELFInstruction>, JoltDevice) {
            let output = Command::new("cargo")
                .args(&[
                    "build",
                    "--release",
                    "-p",
                    &self.guest,
                    "--target-dir",
                    "/tmp/jolt-guest-target",
                ])
                .output()
                .expect("failed to build guest");

            io::stdout().write(&output.stdout).unwrap();
            io::stderr().write(&output.stderr).unwrap();

            let elf = format!(
                "/tmp/jolt-guest-target/riscv32i-unknown-none-elf/release/{}",
                self.guest
            );
            let elf = PathBuf::from_str(&elf).unwrap();

            let instructions = tracer::decode(&elf);
            let (trace, io_device) = tracer::trace(&elf, self.input);

            (trace, instructions, io_device)
        }

        pub fn trace_analyze(self) {
            let (rows, _, _) = self.trace();

            let mut counts = HashMap::<RV32IM, u64>::new();
            for row in rows {
                let op = row.instruction.opcode;
                if let Some(count) = counts.get(&op) {
                    counts.insert(op, count + 1);
                } else {
                    counts.insert(op, 1);
                }
            }

            let mut counts: Vec<_> = counts.into_iter().collect();
            counts.sort_by_key(|v| v.1);
            counts.reverse();

            for (op, count) in counts {
                println!("{:?}: {}", op, count)
            }
        }
    }
}
