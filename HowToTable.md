# Custom Subtable Strategies

This document explains how to fill out custom Subtable Strategies that implement the `SubtableStrategy` trait. The trait provides a set of methods and associated constants that need to be implemented to define a custom subtable strategy. The trait is defined as follows:

```rust
pub trait SubtableStrategy<F: PrimeField, const C: usize, const M: usize> {
    const NUM_SUBTABLES: usize;
    const NUM_MEMORIES: usize;

    fn materialize_subtables() -> [Vec<F>; Self::NUM_SUBTABLES];

    fn evaluate_subtable_mle(subtable_index: usize, point: &Vec<F>) -> F;

    fn combine_lookups(vals: &[F; Self::NUM_MEMORIES]) -> F;

    fn g_poly_degree() -> usize;

    // Default impl
    fn combine_lookups_eq(vals: &[F; Self::NUM_MEMORIES + 1]) -> F;
    fn sumcheck_poly_degree() -> usize;
    fn memory_to_subtable_index(memory_index: usize) -> usize;
    fn memory_to_dimension_index(memory_index: usize) -> usize;
    fn to_lookup_polys(
        subtable_entries: &[Vec<F>; Self::NUM_SUBTABLES],
        nz: &[Vec<usize>; C],
        s: usize,
    ) -> [DensePolynomial<F>; Self::NUM_MEMORIES];
}
```

## Implementing the SubtableStrategy Trait

To implement the `SubtableStrategy` trait, you need to define the following methods and associated constants:

### 1. `NUM_SUBTABLES` and `NUM_MEMORIES`

These associated constants specify the number of subtables and memories used in the strategy. Replace `NUM_SUBTABLES` and `NUM_MEMORIES` with the desired values. `NUM_MEMORIES` is typically a multiple of `C`.

```rust
const NUM_SUBTABLES: usize = <desired_value>;
const NUM_MEMORIES: usize = <desired_value>;
```

### 2. `materialize_subtables`

This method materializes the subtables.

```rust
fn materialize_subtables() -> [Vec<F>; Self::NUM_SUBTABLES] {
    // Implementation goes here
}
```

Replace the implementation with your logic to generate and return the materialized subtables.

### 3. `evaluate_subtable_mle`

This method evaluates the mutlilinear extension (MLE) of a subtable at the given `point`. The `subtable_index` parameter specifies which subtable to evaluate.

```rust
fn evaluate_subtable_mle(subtable_index: usize, point: &Vec<F>) -> F {
    // Implementation goes here
}
```

Replace the implementation with your logic to evaluate and return the MLE of the specified subtable at the given point.

### 4. `combine_lookups`
This method defines the $g$ polynomial that computes the combined lookup value $T[r]$ based on the lookup values of different subtables/memories.

```rust
fn combine_lookups(vals: &[F; Self::NUM_MEMORIES]) -> F {
    // Implementation goes here
}
```

Replace the implementation with your logic to combine the lookup values and return the combined result.

### 5. `g_poly_degree`

This method specifies the total degree of the g function. The total degree determines the number of evaluation points sent in each sumcheck round (N+1 points define a degree-N polynomial).

To simplify: Each term in `combine_lookups` can be thought of as a degree-1 polynomial, what is the degree after applying the collation function `g`?

```rust
fn g_poly_degree() -> usize {
    // Specify the desired sumcheck polynomial degree
}
```

Replace the implementation with the desired value for the sumcheck polynomial degree.

### Remaining
The remaining trait functions should be implemented by default.

## Testing
### `materialize_subtables` / `evaluate_subtable_mle`
The parity between these tables can be tested with the `materialization_mle_parity_test`. This macro materializes the required subtables and evaluates the multilinear extension over the boolean hypercube to ensure consistency.

```rust
#[cfg(test)]
mod test {
    use super::*;
    use crate::materialization_mle_parity_test;
    use ark_bn254::Fr;

      materialization_mle_parity_test!(lt_materialization_parity_test, LTSubtableStrategy, Fr, /* m= */ 16, /* NUM_SUBTABLES= */ 2);
}
```

## End-to-End Testing
To ensure the `SubtableStrategy` has been written correctly as a whole we can fall back to the end-to-end proof system of Lasso. A test can be added in `e2e_test.rs` using the `e2e_test!` macro. 

```rust
    e2e_test!(prove_4d_lt, LTSubtableStrategy,  G1Projective, Fr, /* C= */ 4, /* M= */ 16, /* sparsity= */ 16);
```