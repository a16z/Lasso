fn main() {
    println!("cargo:rustc-link-arg-bin=fibonacci-guest=-Texamples/fibonacci/guest/linker.ld");
}
