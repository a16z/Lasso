use jolt_sdk::host::Program;

pub fn main() {
    let input: &[u8] = &[5u8; 32];
    Program::new("sha2-guest").input(&input).trace_analyze();

    // let (_, device) = Program::new("hash-guest").input(&input).trace();
    // println!("{}", hex::encode(device.outputs));
}
