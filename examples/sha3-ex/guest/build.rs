fn main() {
    println!("cargo:rustc-link-arg-bin=sha3-guest=-Texamples/sha3-ex/guest/linker.ld");
}
