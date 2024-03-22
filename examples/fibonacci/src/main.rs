use jolt_sdk::host::Program;

pub fn main() {
    let input = 9u32;
    Program::new("fibonacci-guest").input(&input).trace_analyze();

    // let device = Program::new("fibonacci-guest").input(&input).trace();
    // let result: u32 = jolt_sdk::postcard::from_bytes(&device.outputs).unwrap();
    // println!("{:?}", result);
}
