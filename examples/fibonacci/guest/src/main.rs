#![no_std]
#![no_main]

#[jolt_sdk::main]
fn fib(n: u32) -> u32 {
    fib_rec(n)
}

fn fib_rec(n: u32) -> u32 {
    if n <= 1 {
        n
    } else {
        fib_rec(n - 1) + fib_rec(n - 2)
    }
}


