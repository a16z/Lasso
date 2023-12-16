use super::sumcheck::{CubicSumcheckParams, SumcheckInstanceProof};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::subprotocols::sumcheck::CubicSumcheckType;
use crate::utils::math::Math;
use crate::utils::{mul_0_1_optimized, is_power_of_two};
use crate::utils::transcript::ProofTranscript;
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::*;
use merlin::Transcript;

#[derive(Debug, Clone, PartialEq)]
pub struct GrandProductCircuit<F> {
    left_vec: Vec<DensePolynomial<F>>,
    right_vec: Vec<DensePolynomial<F>>,
}

impl<F: PrimeField> GrandProductCircuit<F> {
    fn compute_layer(
        inp_left: &DensePolynomial<F>,
        inp_right: &DensePolynomial<F>,
    ) -> (DensePolynomial<F>, DensePolynomial<F>) {
        // TODO(sragss): Write a cache checker for how many cache hits I'd have for a cacher.
        let len = inp_left.len() + inp_right.len();
        let outp_left = (0..len / 4)
            .map(|i| 
                // inp_left[i] * inp_right[i]
                mul_0_1_optimized(&inp_left[i], &inp_right[i])
            )
            .collect::<Vec<F>>();
        let outp_right = (len / 4..len / 2)
            .map(|i| 
                // inp_left[i] * inp_right[i]
                mul_0_1_optimized(&inp_left[i], &inp_right[i])
            )
            .collect::<Vec<F>>();

        (
            DensePolynomial::new(outp_left),
            DensePolynomial::new(outp_right),
        )
    }

    pub fn new(leaves: &DensePolynomial<F>) -> Self {
        let mut left_vec: Vec<DensePolynomial<F>> = Vec::new();
        let mut right_vec: Vec<DensePolynomial<F>> = Vec::new();

        let num_layers = leaves.len().log_2();
        let (outp_left, outp_right) = leaves.split(leaves.len() / 2);

        left_vec.push(outp_left);
        right_vec.push(outp_right);

        for i in 0..num_layers - 1 {
            let (outp_left, outp_right) =
                GrandProductCircuit::compute_layer(&left_vec[i], &right_vec[i]);
            left_vec.push(outp_left);
            right_vec.push(outp_right);
        }

        GrandProductCircuit {
            left_vec,
            right_vec,
        }
    }

    pub fn new_split(left_leaves: DensePolynomial<F>, right_leaves: DensePolynomial<F>) -> Self {
        let num_layers = left_leaves.len().log_2() + 1; 
        let mut left_vec: Vec<DensePolynomial<F>> = Vec::with_capacity(num_layers);
        let mut right_vec: Vec<DensePolynomial<F>> = Vec::with_capacity(num_layers);

        left_vec.push(left_leaves);
        right_vec.push(right_leaves);

        for i in 0..num_layers - 1 {
            let (outp_left, outp_right) =
                GrandProductCircuit::compute_layer(&left_vec[i], &right_vec[i]);
            left_vec.push(outp_left);
            right_vec.push(outp_right);
        }

        GrandProductCircuit {
            left_vec,
            right_vec,
        }
    }

    // TODO(sragss): Attempt sparse repr
    /// Idea here is that we can interleave the polys to have a representation closer to that
    /// of a binary tree of multiplication gates (they alternate 0/1). This can then be reused
    /// in sumcheck itself.
    pub fn new_interleaves(leaves: Vec<F>) {
        assert!(is_power_of_two(leaves.len()));

        let num_leaves = leaves.len();
        let num_layers = num_leaves.log_2(); 

        let mut tree: Vec<Vec<F>> = Vec::with_capacity(num_layers);

        tree.push(leaves);

        // Allocate all the vectors
        let mut layer_size = num_leaves / 2;
        let span_layer = tracing::info_span!("Layer Allocation");
        let _enter = span_layer.enter();
        for layer_index in 1..num_layers {
            // let layer = vec![F::one(); layer_size];
            let layer = Vec::with_capacity(layer_size);
            tree.push(layer);
            println!("allocating {layer_size}");
            layer_size = layer_size / 2;
        }
        drop(_enter);
        drop(span_layer);

        let mut layer_leaves = num_leaves / 2;
        let span_interleave = tracing::info_span!("Interleave Loop");
        {
            let _enter = span_interleave.enter();
            for layer_index in 1..num_layers {
                let span_layer = tracing::info_span!("Layer", layer_index = layer_index);
                {
                    let _enter = span_layer.enter();
                    for leaf_index in 0..layer_leaves {
                        let l = tree[layer_index - 1][2*leaf_index];
                        let r = tree[layer_index - 1][2*leaf_index+1];
                        // if !(l.is_one() && r.is_one()) {
                            // tree[layer_index][leaf_index] = mul_0_1_optimized(&l, &r);
                        // }
                        tree[layer_index].push(mul_0_1_optimized(&l, &r));
                    }
                }
                drop(span_layer);

                layer_leaves /= 2;
            }
        }
        drop(span_interleave);

        // TODO(sragss): Have to convert this to a new format of GrandProductCircuit
    }

    /// Fast version of new_split intended to handle GKR leaf construction vertically (depth-first) rather than horizontally to aid with caching.
    // pub fn new_fast(left_leaves: Vec<F>, right_leaves: Vec<F>) -> Self {
    //     assert_eq!(left_leaves.len(), right_leaves.len());
    //     assert!(is_power_of_two(left_leaves.len()));

    //     let num_leaves = left_leaves.len() * 2;
    //     let num_layers = num_leaves.log_2(); 

    //     let mut left_vec: Vec<Vec<F>> = Vec::with_capacity(num_layers);
    //     let mut right_vec: Vec<Vec<F>> = Vec::with_capacity(num_layers);

    //     // TODO(sragss): For now we're assuming l * r, but could do vertical trees with each as well.
    //     left_vec.push(left_leaves);
    //     right_vec.push(right_leaves);

    //     // Allocate all the vectors
    //     let mut layer_size = num_leaves / 4;
    //     for layer_index in 1..num_layers {
    //         let left = vec![F::one(); layer_size];
    //         let right = left.clone();
    //         left_vec.push(left);
    //         right_vec.push(right);
    //         println!("allocating l, r {layer_size}");
    //         layer_size = layer_size / 2;
    //     }

    //     // Tunable
    //     let num_subtree_leaves = 4; 
    //     let num_subtree_muls = num_subtree_leaves / 2; // Number of multiplications performed during this subtree

    //     assert!(num_subtree_leaves < num_leaves); // TODO(sragss): Just fall back to new_split
    //     let subtree_height = num_subtree_leaves.log_2();

    //     // let num_subtrees = num_layers / subtree_height;
    //     // for subtree_index in 0..num_subtrees {

    //     //     for leaf_index in (subtree_index * num_subtree_leaves)..((subtree_index + 1) * num_subtree_leaves) {
    //     //         assert!(leaf_index < num_leaves);

    //     //         for height_index in (subtree_index * subtree_height + 1)..((subtree_index + 1) * subtree_height) {
    //     //             assert!(height_index < num_layers);

    //     //             if leaf_index < num_leaves / 2 { // left
    //     //                 left_vec[height_index + 1][leaf_index] = left_vec[height_index][leaf_index] * right_vec[height_index][leaf_index];
    //     //             } else { // right
    //     //                 right_vec[height_index + 1][leaf_index - num_leaves / 2] = left_vec[height_index][leaf_index] * right_vec[height_index][leaf_index];
    //     //             }
    //     //         }
    //     //     }
    //     // }

    //     // let num_subtrees = num_leaves / num_subtree_leaves; // Should be a power of 2!
    //     // for subtree_index in 0..num_subtrees {
    //     //     let mut num_muls = num_subtree_muls;
    //     //     println!("\nSubtree {subtree_index}");

    //     //     for height_index in 0..subtree_height {
    //     //         println!("Height {height_index}");
    //     //         for leaf_index in (subtree_index * num_muls)..((subtree_index + 1) * num_muls) {
    //     //             println!("Leaf {leaf_index}");
    //     //             if leaf_index < num_muls { // left
    //     //                 println!("Loading L[{}][{}] = L[{}][{}] * R[{}][{}]", height_index + 1, leaf_index, height_index, leaf_index, height_index, leaf_index);
    //     //                 left_vec[height_index + 1][leaf_index] = left_vec[height_index][leaf_index] * right_vec[height_index][leaf_index];
    //     //             } else { // right
    //     //                 let update_leaf_index = leaf_index - num_muls;
    //     //                 println!("Loading R[{}][{}] = L[{}][{}] * R[{}][{}]", height_index + 1, update_leaf_index, height_index, leaf_index, height_index, leaf_index);
    //     //                 right_vec[height_index + 1][update_leaf_index] = 
    //     //                     left_vec[height_index][leaf_index] 
    //     //                     * right_vec[height_index][leaf_index];
    //     //             }
    //     //         }
    //     //         num_muls = num_muls / 2;
    //     //     }

    //     // }

    //     let num_subtrees = num_leaves / num_subtree_leaves; // Should be a power of 2!
    //     for subtree_index in 0..num_subtrees {
    //         let mut num_muls = num_subtree_muls;

    //         for height_index in 0..subtree_height {
    //             for leaf_index in (subtree_index * num_muls)..((subtree_index + 1) * num_muls) {
    //                 if leaf_index < num_muls { // left
    //                     println!("Loading L[{}][{}] = L[{}][{}] * L[{}][{}]", height_index + 1, leaf_index, height_index, leaf_index, height_index, leaf_index);
    //                     left_vec[height_index + 1][leaf_index] = left_vec[height_index][leaf_index] * right_vec[height_index][leaf_index];
    //                 } else { // right
    //                     let update_leaf_index = leaf_index - num_muls;
    //                     println!("Loading R[{}][{}] = L[{}][{}] * R[{}][{}]", height_index + 1, update_leaf_index, height_index, leaf_index, height_index, leaf_index);
    //                     right_vec[height_index + 1][update_leaf_index] = 
    //                         left_vec[height_index][leaf_index] 
    //                         * right_vec[height_index][leaf_index];
    //                 }
    //             }
    //             num_muls = num_muls / 2;
    //         }

    //     }

    //     // TODO(sragss): What to do about top of tree??

    //     let left_vec = left_vec.into_iter().map(|v| DensePolynomial::new(v)).collect();
    //     let right_vec = right_vec.into_iter().map(|v| DensePolynomial::new(v)).collect();

    //     GrandProductCircuit {
    //         left_vec,
    //         right_vec
    //     }
    // }

    pub fn evaluate(&self) -> F {
        let len = self.left_vec.len();
        assert_eq!(self.left_vec[len - 1].get_num_vars(), 0);
        assert_eq!(self.right_vec[len - 1].get_num_vars(), 0);
        self.left_vec[len - 1][0] * self.right_vec[len - 1][0]
    }

    pub fn take_layer(&mut self, layer_id: usize) -> (DensePolynomial<F>, DensePolynomial<F>) {
        let left = std::mem::replace(
            &mut self.left_vec[layer_id],
            DensePolynomial::new(vec![F::zero()]),
        );
        let right = std::mem::replace(
            &mut self.right_vec[layer_id],
            DensePolynomial::new(vec![F::zero()]),
        );
        (left, right)
    }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerProofBatched<F: PrimeField> {
    pub proof: SumcheckInstanceProof<F>,
    pub claims_poly_A: Vec<F>,
    pub claims_poly_B: Vec<F>,
    pub combine_prod: bool, // TODO(sragss): Use enum. Sumcheck.rs/CubicType
}

#[allow(dead_code)]
impl<F: PrimeField> LayerProofBatched<F> {
    pub fn verify<G, T: ProofTranscript<G>>(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut T,
    ) -> (F, Vec<F>)
    where
        G: CurveGroup<ScalarField = F>,
    {
        self.proof
            .verify::<G, T>(claim, num_rounds, degree_bound, transcript)
            .unwrap()
    }
}

pub struct BatchedGrandProductCircuit<F: PrimeField> {
    pub circuits: Vec<GrandProductCircuit<F>>,

    flags_present: bool,
    flags: Option<Vec<DensePolynomial<F>>>,
    flag_map: Option<Vec<usize>>,
    fingerprint_polys: Option<Vec<DensePolynomial<F>>>,
}

impl<F: PrimeField> BatchedGrandProductCircuit<F> {
    pub fn new_batch(circuits: Vec<GrandProductCircuit<F>>) -> Self {
        Self {
            circuits,

            flags_present: false,
            flags: None,
            flag_map: None,
            fingerprint_polys: None,
        }
    }

    pub fn new_batch_flags(
        circuits: Vec<GrandProductCircuit<F>>,
        flags: Vec<DensePolynomial<F>>,
        flag_map: Vec<usize>,
        fingerprint_polys: Vec<DensePolynomial<F>>,
    ) -> Self {
        assert_eq!(circuits.len(), flag_map.len());
        assert_eq!(circuits.len(), fingerprint_polys.len());
        flag_map.iter().for_each(|i| assert!(*i < flags.len()));

        Self {
            circuits,

            flags_present: true,
            flags: Some(flags),
            flag_map: Some(flag_map),
            fingerprint_polys: Some(fingerprint_polys),
        }
    }

    fn num_layers(&self) -> usize {
        let prod_layers = self.circuits[0].left_vec.len();

        if self.flags.is_some() {
            prod_layers + 1
        } else {
            prod_layers
        }
    }

    #[tracing::instrument(skip_all, name = "GrandProduct.sumcheck_layer_params")]
    fn sumcheck_layer_params(
        &mut self,
        layer_id: usize,
        eq: DensePolynomial<F>,
    ) -> CubicSumcheckParams<F> {
        if self.flags_present && layer_id == 0 {
            let flags = self.flags.as_ref().unwrap();
            debug_assert_eq!(flags[0].len(), eq.len());

            let num_rounds = eq.get_num_vars();

            // Each of these is needed exactly once, transfer ownership rather than clone.
            let fingerprint_polys = self.fingerprint_polys.take().unwrap();
            let flags = self.flags.take().unwrap();
            let flag_map = self.flag_map.take().unwrap();
            CubicSumcheckParams::new_flags(fingerprint_polys, flags, eq, flag_map, num_rounds)
        } else {
            // If flags is present layer_id 1 corresponds to circuits.left_vec/right_vec[0]
            let layer_id = if self.flags_present {
                layer_id - 1
            } else {
                layer_id
            };

            let num_rounds = self.circuits[0].left_vec[layer_id].get_num_vars();

            let (lefts, rights): (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) = self
                .circuits
                .iter_mut()
                .map(|circuit| circuit.take_layer(layer_id))
                .unzip();
            if self.flags_present {
                CubicSumcheckParams::new_prod_ones(lefts, rights, eq, num_rounds)
            } else {
                CubicSumcheckParams::new_prod(lefts, rights, eq, num_rounds)
            }
        }
    }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductArgument<F: PrimeField> {
    proof: Vec<LayerProofBatched<F>>,
}

impl<F: PrimeField> BatchedGrandProductArgument<F> {
    #[tracing::instrument(skip_all, name = "BatchedGrandProductArgument.prove")]
    pub fn prove<G>(
        mut batch: BatchedGrandProductCircuit<F>,
        transcript: &mut Transcript,
    ) -> (Self, Vec<F>)
    where
        G: CurveGroup<ScalarField = F>,
    {
        let mut proof_layers: Vec<LayerProofBatched<F>> = Vec::new();
        let mut claims_to_verify = (0..batch.circuits.len())
            .map(|i| batch.circuits[i].evaluate())
            .collect::<Vec<F>>();

        let mut rand = Vec::new();
        for layer_id in (0..batch.num_layers()).rev() {
            let span = tracing::span!(tracing::Level::TRACE, "grand_product_layer", layer_id);
            let _enter = span.enter();

            // produce a fresh set of coeffs and a joint claim
            let coeff_vec: Vec<F> = <Transcript as ProofTranscript<G>>::challenge_vector(
                transcript,
                b"rand_coeffs_next_layer",
                claims_to_verify.len(),
            );
            let claim = (0..claims_to_verify.len())
                .map(|i| claims_to_verify[i] * coeff_vec[i])
                .sum();

            let eq = DensePolynomial::new(EqPolynomial::<F>::new(rand.clone()).evals());
            let params = batch.sumcheck_layer_params(layer_id, eq);
            let sumcheck_type = params.sumcheck_type.clone();
            let (proof, rand_prod, claims_prod) =
                SumcheckInstanceProof::prove_cubic_batched::<G>(
                    &claim, params, &coeff_vec, transcript,
                );

            let (claims_poly_A, claims_poly_B, _claim_eq) = claims_prod;
            for i in 0..batch.circuits.len() {
                <Transcript as ProofTranscript<G>>::append_scalar(
                    transcript,
                    b"claim_prod_left",
                    &claims_poly_A[i],
                );

                <Transcript as ProofTranscript<G>>::append_scalar(
                    transcript,
                    b"claim_prod_right",
                    &claims_poly_B[i],
                );
            }

            if sumcheck_type == CubicSumcheckType::Prod || sumcheck_type == CubicSumcheckType::ProdOnes {
                // Prod layers must generate an additional random coefficient. The sumcheck randomness indexes into the current layer,
                // but the resulting randomness and claims are about the next layer. The next layer is indexed by an additional variable
                // in the MSB. We use the evaluations V_i(r,0), V_i(r,1) to compute V_i(r, r').

                // produce a random challenge to condense two claims into a single claim
                let r_layer = <Transcript as ProofTranscript<G>>::challenge_scalar(
                    transcript,
                    b"challenge_r_layer",
                );

                claims_to_verify = (0..batch.circuits.len())
                    .map(|i| claims_poly_A[i] + r_layer * (claims_poly_B[i] - claims_poly_A[i]))
                    .collect::<Vec<F>>();

                let mut ext = vec![r_layer];
                ext.extend(rand_prod);
                rand = ext;

                proof_layers.push(LayerProofBatched {
                    proof,
                    claims_poly_A,
                    claims_poly_B,
                    combine_prod: true,
                });
            } else {
                // CubicSumcheckType::Flags
                // Flag layers do not need the additional bit as the randomness from the previous layers have already fully determined
                assert_eq!(layer_id, 0);
                rand = rand_prod;

                proof_layers.push(LayerProofBatched {
                    proof,
                    claims_poly_A,
                    claims_poly_B,
                    combine_prod: false,
                });
            }
            drop(_enter);
        }

        (
            BatchedGrandProductArgument {
                proof: proof_layers,
            },
            rand,
        )
    }

    pub fn verify<G, T: ProofTranscript<G>>(
        &self,
        claims_prod_vec: &Vec<F>,
        transcript: &mut T,
    ) -> (Vec<F>, Vec<F>)
    where
        G: CurveGroup<ScalarField = F>,
    {
        let mut rand: Vec<F> = Vec::new();
        let num_layers = self.proof.len();

        let mut claims_to_verify = claims_prod_vec.to_owned();
        for (num_rounds, i) in (0..num_layers).enumerate() {
            // produce random coefficients, one for each instance
            let coeff_vec =
                transcript.challenge_vector(b"rand_coeffs_next_layer", claims_to_verify.len());

            // produce a joint claim
            let claim = (0..claims_to_verify.len())
                .map(|i| claims_to_verify[i] * coeff_vec[i])
                .sum();

            let (claim_last, rand_prod) =
                self.proof[i].verify::<G, T>(claim, num_rounds, 3, transcript);

            let claims_prod_left = &self.proof[i].claims_poly_A;
            let claims_prod_right = &self.proof[i].claims_poly_B;
            assert_eq!(claims_prod_left.len(), claims_prod_vec.len());
            assert_eq!(claims_prod_right.len(), claims_prod_vec.len());

            for i in 0..claims_prod_vec.len() {
                transcript.append_scalar(b"claim_prod_left", &claims_prod_left[i]);
                transcript.append_scalar(b"claim_prod_right", &claims_prod_right[i]);
            }

            assert_eq!(rand.len(), rand_prod.len());
            let eq: F = (0..rand.len())
                .map(|i| rand[i] * rand_prod[i] + (F::one() - rand[i]) * (F::one() - rand_prod[i]))
                .product();

            // Compute the claim_expected which is a random linear combination of the batched evaluations.
            // The evaluation is the combination of eq / A / B depending on the cubic layer type (flags / prod).
            // We also compute claims_to_verify which computes sumcheck_cubic_poly(r, r') from
            // sumcheck_cubic_poly(r, 0), sumcheck_subic_poly(r, 1)
            let claim_expected = if self.proof[i].combine_prod {
                let claim_expected: F = (0..claims_prod_vec.len())
                    .map(|i| {
                        coeff_vec[i]
                            * CubicSumcheckParams::combine_prod(
                                &claims_prod_left[i],
                                &claims_prod_right[i],
                                &eq,
                            )
                    })
                    .sum();

                // produce a random challenge
                let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

                claims_to_verify = (0..claims_prod_left.len())
                    .map(|i| {
                        claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i])
                    })
                    .collect::<Vec<F>>();

                let mut ext = vec![r_layer];
                ext.extend(rand_prod);
                rand = ext;

                claim_expected
            } else {
                let claim_expected: F = (0..claims_prod_vec.len())
                    .map(|i| {
                        coeff_vec[i]
                            * CubicSumcheckParams::combine_flags(
                                &claims_prod_left[i],
                                &claims_prod_right[i],
                                &eq,
                            )
                    })
                    .sum();

                rand = rand_prod;

                claims_to_verify = (0..claims_prod_left.len())
                    .map(|i| {
                        claims_prod_left[i] * claims_prod_right[i]
                            + (F::one() - claims_prod_right[i])
                    })
                    .collect::<Vec<F>>();

                claim_expected
            };

            assert_eq!(claim_expected, claim_last);
        }
        (claims_to_verify, rand)
    }
}

#[cfg(test)]
mod grand_product_circuit_tests {
    use super::*;
    use ark_curve25519::{EdwardsProjective as G1Projective, Fr};
    use ark_std::{One, Zero, rand::thread_rng, UniformRand};

    #[test]
    fn prove_verify() {
        let factorial =
            DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]);
        let factorial_circuit = GrandProductCircuit::new(&factorial);
        let expected_eval = vec![Fr::from(24)];
        assert_eq!(factorial_circuit.evaluate(), Fr::from(24));

        let mut transcript = Transcript::new(b"test_transcript");
        let circuits_vec = vec![factorial_circuit];
        let batch = BatchedGrandProductCircuit::new_batch(circuits_vec);
        let (proof, _) = BatchedGrandProductArgument::prove::<G1Projective>(batch, &mut transcript);

        let mut transcript = Transcript::new(b"test_transcript");
        proof.verify::<G1Projective, _>(&expected_eval, &mut transcript);
    }

    #[test]
    fn gp_unflagged() {
        // Fundamentally grand products performs a multi-set check, so skip fingerprinting and all that, construct GP circuits directly
        let read_leaves = vec![Fr::from(10), Fr::from(20)];
        let write_leaves = vec![Fr::from(100), Fr::from(200)];

        let read_poly = DensePolynomial::new(read_leaves);
        let write_poly = DensePolynomial::new(write_leaves);

        let (read_gpc, write_gpc) = (
            GrandProductCircuit::new(&read_poly),
            GrandProductCircuit::new(&write_poly),
        );
        let batch = BatchedGrandProductCircuit::new_batch(vec![read_gpc, write_gpc]);

        let mut transcript = Transcript::new(b"test_transcript");
        let (proof, prove_rand) =
            BatchedGrandProductArgument::<Fr>::prove::<G1Projective>(batch, &mut transcript);

        let expected_eval_read = Fr::from(10) * Fr::from(20);
        let expected_eval_write = Fr::from(100) * Fr::from(200);
        let mut transcript = Transcript::new(b"test_transcript");
        let (verify_claims, verify_rand) = proof.verify::<G1Projective, _>(
            &vec![expected_eval_read, expected_eval_write],
            &mut transcript,
        );

        assert_eq!(prove_rand, verify_rand);
        assert_eq!(verify_claims.len(), 2);
        assert_eq!(verify_claims[0], read_poly.evaluate(&verify_rand));
        assert_eq!(verify_claims[1], write_poly.evaluate(&verify_rand));
    }

    #[test]
    fn gp_flagged() {
        let read_fingerprints = vec![Fr::from(10), Fr::from(20), Fr::from(30), Fr::from(40)];
        let write_fingerprints = vec![Fr::from(100), Fr::from(200), Fr::from(300), Fr::from(400)];

        // toggle off index '2'
        let flag_poly = DensePolynomial::new(vec![Fr::one(), Fr::one(), Fr::zero(), Fr::one()]);

        // Grand Product Circuit leaves are those that are toggled
        let mut read_leaves = read_fingerprints.clone();
        read_leaves[2] = Fr::one();
        let mut write_leaves = write_fingerprints.clone();
        write_leaves[2] = Fr::one();

        let read_leaf_poly = DensePolynomial::new(read_leaves);
        let write_leaf_poly = DensePolynomial::new(write_leaves);

        let flag_map = vec![0, 0];

        let fingerprint_polys = vec![
            DensePolynomial::new(read_fingerprints),
            DensePolynomial::new(write_fingerprints),
        ];

        // Construct the GPCs not from the raw fingerprints, but from the flagged fingerprints!
        let (read_gpc, write_gpc) = (
            GrandProductCircuit::new(&read_leaf_poly),
            GrandProductCircuit::new(&write_leaf_poly),
        );

        // Batch takes reference to the "untoggled" fingerprint_polys for the final flag layer that feeds into the leaves, which have been flagged (set to 1 if the flag is not 1)
        let batch = BatchedGrandProductCircuit::new_batch_flags(
            vec![read_gpc, write_gpc],
            vec![flag_poly.clone()],
            flag_map,
            fingerprint_polys.clone(),
        );

        let mut transcript = Transcript::new(b"test_transcript");
        let (proof, prove_rand) =
            BatchedGrandProductArgument::<Fr>::prove::<G1Projective>(batch, &mut transcript);

        let expected_eval_read: Fr = Fr::from(10) * Fr::from(20) * Fr::from(40);
        let expected_eval_write: Fr = Fr::from(100) * Fr::from(200) * Fr::from(400);
        let expected_evals = vec![expected_eval_read, expected_eval_write];

        let mut transcript = Transcript::new(b"test_transcript");
        let (verify_claims, verify_rand) =
            proof.verify::<G1Projective, _>(&expected_evals, &mut transcript);

        assert_eq!(prove_rand, verify_rand);
        assert_eq!(verify_claims.len(), 2);

        assert_eq!(proof.proof.len(), 3);
        // Claims about raw fingerprints bound to r_z
        assert_eq!(
            proof.proof[2].claims_poly_A[0],
            fingerprint_polys[0].evaluate(&verify_rand)
        );
        assert_eq!(
            proof.proof[2].claims_poly_A[1],
            fingerprint_polys[1].evaluate(&verify_rand)
        );

        // Claims about flags bound to r_z
        assert_eq!(
            proof.proof[2].claims_poly_B[0],
            flag_poly.evaluate(&verify_rand)
        );
        assert_eq!(
            proof.proof[2].claims_poly_B[1],
            flag_poly.evaluate(&verify_rand)
        );

        let verifier_flag_eval = flag_poly.evaluate(&verify_rand);
        let verifier_read_eval = verifier_flag_eval * fingerprint_polys[0].evaluate(&verify_rand)
            + Fr::one()
            - verifier_flag_eval;
        let verifier_write_eval = verifier_flag_eval * fingerprint_polys[1].evaluate(&verify_rand)
            + Fr::one()
            - verifier_flag_eval;
        assert_eq!(verify_claims[0], verifier_read_eval);
        assert_eq!(verify_claims[1], verifier_write_eval);
    }

    #[test]
    fn fast_gp_construct_parity() {
        let num_leaves = 8;
        let mut rng = thread_rng();

        let left_leaves = vec![Fr::rand(&mut rng); num_leaves / 2];
        let right_leaves = vec![Fr::rand(&mut rng); num_leaves / 2];

        let other_gp = GrandProductCircuit::new_fast(left_leaves.clone(), right_leaves.clone());
        let gp = GrandProductCircuit::new_split(DensePolynomial::new(left_leaves.clone()), DensePolynomial::new(right_leaves.clone()));
        assert_eq!(gp, other_gp);
    }
}
