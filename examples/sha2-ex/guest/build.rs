fn main() {
    println!("cargo:rustc-link-arg-bin=sha2-guest=-Texamples/sha2-ex/guest/linker.ld");
}
