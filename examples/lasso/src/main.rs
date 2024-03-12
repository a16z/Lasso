use liblasso::{jolt::instruction::and::ANDInstruction, lasso::surge::Surge};
use ark_bn254::{Fr, G1Projective};
use merlin::Transcript;

fn main() {
    let ops = vec![
        ANDInstruction(12, 12),
        ANDInstruction(12, 82),
        ANDInstruction(12, 12),
        ANDInstruction(25, 12),
    ];
    const C: usize = 4;
    const M: usize = 1 << 16;

    let mut transcript = Transcript::new(b"test_transcript");
    let surge = <Surge<Fr, G1Projective, ANDInstruction, C>>::new(ops, M);
    let proof = surge.prove(&mut transcript);

    let mut transcript = Transcript::new(b"test_transcript");
    <Surge<Fr, G1Projective, ANDInstruction, C>>::verify(proof, &mut transcript, M)
        .expect("should work");
}
