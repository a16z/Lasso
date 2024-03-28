use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use merlin::Transcript;

use crate::{
    poly::dense_mlpoly::DensePolynomial,
    poly::hyrax::{HyraxCommitment, HyraxGenerators, HyraxOpeningProof},
    utils::{
        errors::ProofVerifyError,
        math::Math,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};


#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct ConcatenatedPolynomialCommitment<G: CurveGroup> {
    pub generators: HyraxGenerators<1, G>,
    pub joint_commitment: HyraxCommitment<1, G>,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ConcatenatedPolynomialOpeningProof<G: CurveGroup> {
    joint_proof: HyraxOpeningProof<1, G>,
}

impl<G: CurveGroup> ConcatenatedPolynomialOpeningProof<G> {
    /// evaluates both polynomials at r and produces a joint proof of opening
    #[tracing::instrument(skip_all, name = "ConcatenatedPolynomialOpeningProof::prove")]
    pub fn prove(
        concatenated_poly: &DensePolynomial<G::ScalarField>,
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        transcript: &mut Transcript,
    ) -> Self {
        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        let evals = {
            let mut evals: Vec<G::ScalarField> = openings.to_vec();
            evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());
            evals.to_vec()
        };

        assert_eq!(
            concatenated_poly.get_num_vars(),
            opening_point.len() + evals.len().log_2()
        );

        // append the claimed evaluations to transcript
        <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", &evals);

        // Reduce openings p_1(r), p_2(r), ..., p_n(r) to a single opening
        let (r_joint, eval_joint) = {
            let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
                transcript,
                b"challenge_combine_n_to_one",
                evals.len().log_2(),
            );

            let mut poly_evals = DensePolynomial::new(evals.to_vec());
            for i in (0..challenges.len()).rev() {
                poly_evals.bound_poly_var_bot(&challenges[i]);
            }
            assert_eq!(poly_evals.len(), 1);
            let joint_claim_eval = poly_evals[0];
            let mut r_joint = challenges;
            r_joint.extend(opening_point);

            debug_assert_eq!(concatenated_poly.evaluate(&r_joint), joint_claim_eval);
            (r_joint, joint_claim_eval)
        };
        // decommit the joint polynomial at r_joint
        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"joint_claim_eval",
            &eval_joint,
        );

        let joint_proof = HyraxOpeningProof::prove(
            concatenated_poly,
            &r_joint,
            transcript,
        );

        ConcatenatedPolynomialOpeningProof { joint_proof }
    }

    // verify evaluations of both polynomials at r
    pub fn verify(
        &self,
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        commitment: &ConcatenatedPolynomialCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());
        
        let mut evals = openings.to_owned();
        evals.resize(evals.len().next_power_of_two(), G::ScalarField::zero());

        // append the claimed evaluations to transcript
        <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", &evals);

        // Reduce openings p_1(r), p_2(r), ..., p_n(r) to a single opening
        let challenges = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"challenge_combine_n_to_one",
            evals.len().log_2(),
        );
        let mut poly_evals = DensePolynomial::new(evals.to_vec());
        for i in (0..challenges.len()).rev() {
            poly_evals.bound_poly_var_bot(&challenges[i]);
        }
        assert_eq!(poly_evals.len(), 1);
        let joint_claim_eval = poly_evals[0];
        let mut r_joint = challenges;
        r_joint.extend(opening_point);

        // decommit the joint polynomial at r_joint
        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"joint_claim_eval",
            &joint_claim_eval,
        );

        self.joint_proof.verify(
            &commitment.generators,
            transcript,
            &r_joint,
            &joint_claim_eval,
            &commitment.joint_commitment,
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"Lasso ConcatenatedPolynomialOpeningProof"
    }
}

impl<G: CurveGroup> AppendToTranscript<G> for ConcatenatedPolynomialCommitment<G> {
    fn append_to_transcript<T: ProofTranscript<G>>(
        &self,
        label: &'static [u8],
        transcript: &mut T,
    ) {
        transcript.append_message(
            b"subtable_evals_commitment",
            b"begin_subtable_evals_commitment",
        );
        self.joint_commitment
            .append_to_transcript(label, transcript);
        transcript.append_message(
            b"subtable_evals_commitment",
            b"end_subtable_evals_commitment",
        );
    }
}
