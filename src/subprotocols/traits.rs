use merlin::Transcript;

pub trait PolynomialCommitmentScheme {
  // Abstracting over Polynomial allows us to have batched and non-batched PCS
  type Polynomial;
  type Commitment;
  type Evaluation;
  type Challenge;
  type Proof;
  type Error;

  type ProverKey<'p>;
  type CommitmentKey<'c>;
  type VerifierKey;

  //TODO: convert to impl IntoIterator<Item = Self::Polynomial>
  fn commit<'a, 'c>(
    poly: &'a Self::Polynomial,
    ck: Self::CommitmentKey<'c>,
  ) -> Result<Self::Commitment, Self::Error>;

  fn prove<'a, 'p>(
    poly: &'a Self::Polynomial,
    evals: &'a Self::Evaluation,
    challenges: &'a Self::Challenge,
    pk: Self::ProverKey<'p>,
    transcript: &mut Transcript,
  ) -> Result<Self::Proof, Self::Error>
    where
      Self::Challenge: 'a;

  fn verify<'a>(
    commitments: &'a Self::Commitment,
    evals: &'a Self::Evaluation,
    challenges: Self::Challenge,
    vk: &'a Self::VerifierKey,
    transcript: &mut Transcript,
    proof: Self::Proof,
  ) -> Result<(), Self::Error>
    where
      Self::Commitment: 'a,
      Self::VerifierKey: 'a;
}


/*
/// Describes the interface for a polynomial commitment scheme that allows
/// a sender to commit to multiple polynomials and later provide a succinct proof
/// of evaluation for the corresponding commitments at a query set `Q`, while
/// enforcing per-polynomial degree bounds.
pub trait PolynomialCommitment<F: PrimeField, P: Polynomial<F>, S: CryptographicSponge>:
    Sized
{
    /// The universal parameters for the commitment scheme. These are "trimmed"
    /// down to `Self::CommitterKey` and `Self::VerifierKey` by `Self::trim`.
    type UniversalParams: PCUniversalParams;
    /// The committer key for the scheme; used to commit to a polynomial and then
    /// open the commitment to produce an evaluation proof.
    type CommitterKey: PCCommitterKey;
    /// The verifier key for the scheme; used to check an evaluation proof.
    type VerifierKey: PCVerifierKey;
    /// The commitment to a polynomial.
    type Commitment: PCCommitment + Default;
    /// Auxiliary state of the commitment, output by the `commit` phase.
    /// It contains information that can be reused by the committer
    /// during the `open` phase, such as the commitment randomness.
    /// Not to be shared with the verifier.
    type CommitmentState: PCCommitmentState;
    /// The evaluation proof for a single point.
    type Proof: Clone;
    /// The evaluation proof for a query set.
    type BatchProof: Clone
        + From<Vec<Self::Proof>>
        + Into<Vec<Self::Proof>>
        + CanonicalSerialize
        + CanonicalDeserialize;
    /// The error type for the scheme.
    type Error: ark_std::error::Error + From<Error>;

    /// Constructs public parameters when given as input the maximum degree `degree`
    /// for the polynomial commitment scheme. `num_vars` specifies the number of
    /// variables for multivariate setup
    fn setup<R: RngCore>(
        max_degree: usize,
        num_vars: Option<usize>,
        rng: &mut R,
    ) -> Result<Self::UniversalParams, Self::Error>;

    /// Specializes the public parameters for polynomials up to the given `supported_degree`
    /// and for enforcing degree bounds in the range `1..=supported_degree`.
    fn trim(
        pp: &Self::UniversalParams,
        supported_degree: usize,
        supported_hiding_bound: usize,
        enforced_degree_bounds: Option<&[usize]>,
    ) -> Result<(Self::CommitterKey, Self::VerifierKey), Self::Error>;

    /// Outputs a list of commitments to `polynomials`. If `polynomials[i].is_hiding()`,
    /// then the `i`-th commitment is hiding up to `polynomials.hiding_bound()` queries.
    /// `rng` should not be `None` if `polynomials[i].is_hiding() == true` for any `i`.
    ///
    /// If for some `i`, `polynomials[i].is_hiding() == false`, then the
    /// corresponding randomness is `Self::Randomness::empty()`.
    ///
    /// If for some `i`, `polynomials[i].degree_bound().is_some()`, then that
    /// polynomial will have the corresponding degree bound enforced.
    fn commit<'a>(
        ck: &Self::CommitterKey,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<F, P>>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<
        (
            Vec<LabeledCommitment<Self::Commitment>>,
            Vec<Self::CommitmentState>,
        ),
        Self::Error,
    >
    where
        P: 'a;

    /// open but with individual challenges
    fn open<'a>(
        ck: &Self::CommitterKey,
        labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<F, P>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        sponge: &mut S,
        states: impl IntoIterator<Item = &'a Self::CommitmentState>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::Proof, Self::Error>
    where
        P: 'a,
        Self::CommitmentState: 'a,
        Self::Commitment: 'a;

    /// check but with individual challenges
    fn check<'a>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = F>,
        proof: &Self::Proof,
        sponge: &mut S,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a;

    /// Open several polynomials at one or more points each (possibly different
    /// for each polynomial). Each entry in the in the query set of points
    /// contains the label of the polynomial which should be queried at that
    /// point.
    ///
    /// Behaviour is undefined if `query_set` contains the entries with the
    /// same point label but different actual points.
    ///
    /// The opening challenges are independent for each batch of polynomials.
    fn batch_open<'a>(
        ck: &Self::CommitterKey,
        labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<F, P>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<P::Point>,
        sponge: &mut S,
        states: impl IntoIterator<Item = &'a Self::CommitmentState>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::BatchProof, Self::Error>
    where
        P: 'a,
        Self::CommitmentState: 'a,
        Self::Commitment: 'a,
    {
        // The default implementation achieves proceeds by rearranging the queries in
        // order to gather (i.e. batch) all polynomials that should be queried at
        // the same point, then opening their commitments simultaneously with a
        // single call to `open` (per point)
        let rng = &mut crate::optional_rng::OptionalRng(rng);
        let poly_st_comm: BTreeMap<_, _> = labeled_polynomials
            .into_iter()
            .zip(states)
            .zip(commitments.into_iter())
            .map(|((poly, st), comm)| (poly.label(), (poly, st, comm)))
            .collect();

        let open_time = start_timer!(|| format!(
            "Opening {} polynomials at query set of size {}",
            poly_st_comm.len(),
            query_set.len(),
        ));

        let mut query_to_labels_map = BTreeMap::new();

        // `label` is the label of the polynomial the query refers to
        // `point_label` is the label of the point being queried
        // `point` is the actual point
        for (label, (point_label, point)) in query_set.iter() {
            // For each point label in `query_set`, we define an entry in
            // `query_to_labels_map` containing a pair whose first element is
            // the actual point and the second one is the set of labels of the
            // polynomials being queried at that point
            let labels = query_to_labels_map
                .entry(point_label)
                .or_insert((point, BTreeSet::new()));
            labels.1.insert(label);
        }

        let mut proofs = Vec::new();
        for (_point_label, (point, labels)) in query_to_labels_map.into_iter() {
            let mut query_polys: Vec<&'a LabeledPolynomial<_, _>> = Vec::new();
            let mut query_states: Vec<&'a Self::CommitmentState> = Vec::new();
            let mut query_comms: Vec<&'a LabeledCommitment<Self::Commitment>> = Vec::new();

            // Constructing matching vectors with the polynomial, commitment
            // randomness and actual commitment for each polynomial being
            // queried at `point`
            for label in labels {
                let (polynomial, state, comm) =
                    poly_st_comm.get(label).ok_or(Error::MissingPolynomial {
                        label: label.to_string(),
                    })?;

                query_polys.push(polynomial);
                query_states.push(state);
                query_comms.push(comm);
            }

            let proof_time = start_timer!(|| "Creating proof");

            // Simultaneously opening the commitments of all polynomials that
            // refer to the the current point using the plain `open` function
            let proof = Self::open(
                ck,
                query_polys,
                query_comms,
                &point,
                sponge,
                query_states,
                Some(rng),
            )?;

            end_timer!(proof_time);

            proofs.push(proof);
        }
        end_timer!(open_time);

        Ok(proofs.into())
    }

    /// Verify opening proofs for several polynomials at one or more points
    /// each (possibly different for each polynomial). Each entry in
    /// the query set of points contains the label of the polynomial which
    /// was queried at that point.
    ///
    /// Behaviour is undefined if `query_set` contains the entries with the
    /// same point label but different points.
    ///
    /// Behaviour is also undefined if proofs are not ordered the same way as
    /// queries in `query_to_labels_map` (this is the outcome of calling
    /// `batch_open` for the same commitment list and query set).H
    ///
    /// The opening challenges are independent for each batch of polynomials.
    fn batch_check<'a, R: RngCore>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<P::Point>,
        evaluations: &Evaluations<P::Point, F>,
        proof: &Self::BatchProof,
        sponge: &mut S,
        rng: &mut R,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        // The default implementation proceeds by rearranging the queries in
        // order to gather (i.e. batch) the proofs of all polynomials that should
        // have been opened at the same point, then verifying those proofs
        // simultaneously with a single call to `check` (per point).
        let commitments: BTreeMap<_, _> = commitments.into_iter().map(|c| (c.label(), c)).collect();
        let mut query_to_labels_map = BTreeMap::new();

        // `label` is the label of the polynomial the query refers to
        // `point_label` is the label of the point being queried
        // `point` is the actual point
        for (label, (point_label, point)) in query_set.iter() {
            // For each point label in `query_set`, we define an entry in
            // `query_to_labels_map` containing a pair whose first element is
            // the actual point and the second one is the set of labels of the
            // polynomials being queried at that point
            let labels = query_to_labels_map
                .entry(point_label)
                .or_insert((point, BTreeSet::new()));
            labels.1.insert(label);
        }

        // Implicit assumption: proofs are ordered in same manner as queries in
        // `query_to_labels_map`.
        let proofs: Vec<_> = proof.clone().into();
        assert_eq!(proofs.len(), query_to_labels_map.len());

        let mut result = true;
        for ((_point_label, (point, labels)), proof) in query_to_labels_map.into_iter().zip(proofs)
        {
            // Constructing matching vectors with the commitment and claimed
            // value of each polynomial being queried at `point`
            let mut comms: Vec<&'_ LabeledCommitment<_>> = Vec::new();
            let mut values = Vec::new();
            for label in labels.into_iter() {
                let commitment = commitments.get(label).ok_or(Error::MissingPolynomial {
                    label: label.to_string(),
                })?;

                let v_i = evaluations.get(&(label.clone(), point.clone())).ok_or(
                    Error::MissingEvaluation {
                        label: label.to_string(),
                    },
                )?;

                comms.push(commitment);
                values.push(*v_i);
            }

            let proof_time = start_timer!(|| "Checking per-query proof");

            // Verify all proofs referring to the current point simultaneously
            // with a single call to `check`
            result &= Self::check(vk, comms, &point, values, &proof, sponge, Some(rng))?;
            end_timer!(proof_time);
        }
        Ok(result)
    }

    /// Open commitments to all polynomials involved in a number of linear
    /// combinations (LC) simultaneously.
    fn open_combinations<'a>(
        ck: &Self::CommitterKey,
        linear_combinations: impl IntoIterator<Item = &'a LinearCombination<F>>,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<F, P>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        query_set: &QuerySet<P::Point>,
        sponge: &mut S,
        states: impl IntoIterator<Item = &'a Self::CommitmentState>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<BatchLCProof<F, Self::BatchProof>, Self::Error>
    where
        Self::CommitmentState: 'a,
        Self::Commitment: 'a,
        P: 'a,
    {
        // The default implementation proceeds by batch-opening all polynomials
        // appearing in those LC that are queried at the same point.
        let linear_combinations: Vec<_> = linear_combinations.into_iter().collect();
        let polynomials: Vec<_> = polynomials.into_iter().collect();

        // Rearrange the information about queries on linear combinations into
        // information about queries on individual polynomials.
        let poly_query_set =
            lc_query_set_to_poly_query_set(linear_combinations.iter().copied(), query_set);
        let poly_evals = evaluate_query_set(polynomials.iter().copied(), &poly_query_set);

        // Batch-open all polynomials that refer to each individual point in `query_set`
        let proof = Self::batch_open(
            ck,
            polynomials,
            commitments,
            &poly_query_set,
            sponge,
            states,
            rng,
        )?;
        Ok(BatchLCProof {
            proof,
            evals: Some(poly_evals.values().copied().collect()),
        })
    }

    /// Verify opening proofs for all polynomials involved in a number of
    /// linear combinations (LC) simultaneously.
    fn check_combinations<'a, R: RngCore>(
        vk: &Self::VerifierKey,
        linear_combinations: impl IntoIterator<Item = &'a LinearCombination<F>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        eqn_query_set: &QuerySet<P::Point>,
        eqn_evaluations: &Evaluations<P::Point, F>,
        proof: &BatchLCProof<F, Self::BatchProof>,
        sponge: &mut S,
        rng: &mut R,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        // The default implementation does this by batch-checking each
        // batch-opening proof of polynomials appearing in those LC that were
        // queried at the same point, then computing the evaluations of each LC
        // using the proved polynomial evaluations.
        let BatchLCProof { proof, evals } = proof;
        let lc_s = BTreeMap::from_iter(linear_combinations.into_iter().map(|lc| (lc.label(), lc)));

        // Rearrange the information about queries on linear combinations into
        // information about queries on individual polynomials.
        let poly_query_set = lc_query_set_to_poly_query_set(lc_s.values().copied(), eqn_query_set);
        let sorted_by_poly_and_query_label: BTreeSet<_> = poly_query_set
            .clone()
            .into_iter()
            .map(|(poly_label, v)| ((poly_label.clone(), v.1), v.0))
            .collect();

        let poly_evals = Evaluations::from_iter(
            sorted_by_poly_and_query_label
                .into_iter()
                .zip(evals.clone().unwrap())
                .map(|(((poly_label, point), _query_label), eval)| ((poly_label, point), eval)),
        );

        for &(ref lc_label, (_, ref point)) in eqn_query_set {
            if let Some(lc) = lc_s.get(lc_label) {
                let claimed_rhs = *eqn_evaluations
                    .get(&(lc_label.clone(), point.clone()))
                    .ok_or(Error::MissingEvaluation {
                        label: lc_label.to_string(),
                    })?;

                let mut actual_rhs = F::zero();

                // Compute the value of the linear combination by adding the
                // claimed value for each polynomial in it (to be proved later)
                // scaled by the corresponding coefficient.
                for (coeff, label) in lc.iter() {
                    let eval = match label {
                        LCTerm::One => F::one(),
                        LCTerm::PolyLabel(l) => *poly_evals
                            .get(&(l.clone().into(), point.clone()))
                            .ok_or(Error::MissingEvaluation {
                                label: format!("{}-{:?}", l.clone(), point.clone()),
                            })?,
                    };

                    actual_rhs += &(*coeff * eval);
                }

                // Checking the computed evaluation matches the claimed one
                if claimed_rhs != actual_rhs {
                    eprintln!("Claimed evaluation of {} is incorrect", lc.label());
                    return Ok(false);
                }
            }
        }

        // Verify the claimed evaluation for each polynomial appearing in the
        // linear combinations, batched by point
        let pc_result = Self::batch_check(
            vk,
            commitments,
            &poly_query_set,
            &poly_evals,
            proof,
            sponge,
            rng,
        )?;
        if !pc_result {
            eprintln!("Evaluation proofs failed to verify");
            return Ok(false);
        }

        Ok(true)
    }
}
*/