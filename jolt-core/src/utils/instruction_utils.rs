use ark_ff::PrimeField;

pub fn assert_valid_parameters(word_size: usize, C: usize, log_M: usize) {
    assert!(C * log_M >= word_size);
}

/// Concatenates `C` `vals` field elements each of max size 2^`operand_bits`-1
/// into a single field element. `operand_bits` is the number of bits required to represent
/// each element in `vals`. If an element of `vals` is larger it will not be truncated, which
/// is commonly used by the collation functions of instructions.
pub fn concatenate_lookups<F: PrimeField>(vals: &[F], C: usize, operand_bits: usize) -> F {
    assert_eq!(vals.len(), C);

    let mut sum = F::zero();
    let mut weight = F::one();
    let shift = F::from_u64(1u64 << operand_bits).unwrap();
    for i in 0..C {
        sum += weight * vals[C - i - 1];
        weight *= shift;
    }
    sum
}

/// Returns the chunks of an operand passed as input
pub fn chunk_operand(x: u64, C: usize, chunk_len: usize) -> Vec<u64> {
    let bit_mask = (1 << chunk_len) - 1;
    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * chunk_len) as u32;
            x.checked_shr(shift).unwrap_or(0) & bit_mask
        })
        .collect()
}

/// Returns the chunks of an operand passed as input
pub fn chunk_operand_usize(x: u64, C: usize, chunk_len: usize) -> Vec<usize> {
    let bit_mask = (1 << chunk_len) - 1;
    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * chunk_len) as u32;
            (x.checked_shr(shift).unwrap_or(0) & bit_mask) as usize
        })
        .collect()
}

/// Chunks `x` || `y` into `C` chunks bitwise.
/// `log_M` is the number of bits of each of the `C` expected results.
/// `log_M = num_bits(x || y) / C`
///
/// Given the operation x_0, x_1, x_2, x_3 || y_0, y_1, y_2, y_3 with C=2, log_M =4
/// chunks to `vec![x_0||x_1||y_0||y_1,   x_2||x_3||y_2||y_3]`.
pub fn chunk_and_concatenate_operands(x: u64, y: u64, C: usize, log_M: usize) -> Vec<usize> {
    let operand_bits: usize = log_M / 2;

    #[cfg(test)]
    {
        let max_operand_bits = C * log_M / 2;
        if max_operand_bits != 64 {
            // if 64, handled by normal overflow checking
            let max_operand: u64 = (1 << max_operand_bits) - 1;
            assert!(x <= max_operand);
            assert!(y <= max_operand);
        }
    }

    let operand_bit_mask: usize = (1 << operand_bits) - 1;
    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * operand_bits) as u32;
            let left = x.checked_shr(shift).unwrap_or(0) as usize & operand_bit_mask;
            let right = y.checked_shr(shift).unwrap_or(0) as usize & operand_bit_mask;
            (left << operand_bits) | right
        })
        .collect()
}

/// Chunks `z` into `C` chunks bitwise where `z = x + y`.
/// `log_M` is the number of bits for each of the `C` chunks of `z`.
pub fn add_and_chunk_operands(x: u128, y: u128, C: usize, log_M: usize) -> Vec<usize> {
    let sum_chunk_bits: usize = log_M;
    let sum_chunk_bit_mask: usize = (1 << sum_chunk_bits) - 1;
    let z: u128 = x + y;
    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * sum_chunk_bits) as u32;
            let chunk = z.checked_shr(shift).unwrap_or(0) as usize & sum_chunk_bit_mask;
            chunk
        })
        .collect()
}

/// Splits `x`, `y` into `C` chunks and writes [ x_{C-1} || y_0, ..., x_0 || y_0 ]
/// where `x_{C-1}`` is the the big end of `x``, and `y_0`` is the small end of `y`.
pub fn chunk_and_concatenate_for_shift(x: u64, y: u64, C: usize, log_M: usize) -> Vec<usize> {
    let operand_bits: usize = log_M / 2;
    let operand_bit_mask: usize = (1 << operand_bits) - 1;

    let y_lowest_chunk: usize = y as usize & operand_bit_mask;

    (0..C)
        .map(|i| {
            let shift = ((C - i - 1) * operand_bits) as u32;
            let left = x.checked_shr(shift).unwrap_or(0) as usize & operand_bit_mask;
            (left << operand_bits) | y_lowest_chunk
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_curve25519::Fr;

    #[test]
    fn concatenate_lookups_test() {
        let vals = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
        let concat = concatenate_lookups(&vals, 3, 2);
        assert_eq!(concat, Fr::from(0b01_10_11));

        let vals = vec![Fr::from(7), Fr::from(1), Fr::from(2), Fr::from(3)];
        let concat = concatenate_lookups(&vals, 4, 3);
        assert_eq!(concat, Fr::from(0b111_001_010_011));
    }

    #[test]
    fn chunk_and_concatenate_operands_test() {
        let chunks = chunk_and_concatenate_operands(0b11, 0b10, 2, 2);
        assert_eq!(chunks, vec![0b11, 0b10]);

        let chunks = chunk_and_concatenate_operands(0b11_00_11, 0b10_01_10, 3, 4);
        assert_eq!(chunks, vec![0b11_10, 0b00_01, 0b_11_10]);
    }

    #[test]
    #[should_panic]
    fn chunk_and_concatenate_operands_too_large() {
        // Fail to split 2 integers of length 3-bits in to 2 chunks of length 2-bits
        chunk_and_concatenate_operands(0b111, 0b101, 2, 2);
    }

    #[test]
    #[should_panic]
    fn chunk_and_concatenate_operands_too_large_2() {
        // Fail to split 2 integers of length 6-bits into 3 chunks of length 3-bits
        chunk_and_concatenate_operands(0b11_11_11, 0b10_10_10, 3, 3);
    }

    #[test]
    fn add_and_chunk_operands_test() {
        // x = 0b0011
        // y = 0b1100
        // z = 0b1111
        let chunks = add_and_chunk_operands(0b0011, 0b1100, 2, 2);
        assert_eq!(chunks, vec![0b11, 0b11]);

        // x = 20
        // y = 30
        // z = 50 = 0b11_00_10
        let chunks = add_and_chunk_operands(20u128, 30u128, 3, 2);
        assert_eq!(chunks, vec![0b11, 0b00, 0b10]);
    }

    #[test]
    fn chunk_and_concatenate_for_shift_test() {
        let x = 0b1001;
        let y = 2;
        let C = 2;
        let log_M = 4;
        let chunks = chunk_and_concatenate_for_shift(x, y, C, log_M);
        assert_eq!(chunks.len(), 2);

        // 2-bit operands
        // x_0 | y = 0b10 | 0b10
        assert_eq!(chunks[0], 0b10_10);
        // x_1 | y = 0b01 | 0b10
        assert_eq!(chunks[1], 0b01_10);
    }

    #[test]
    fn chunk_and_concatenate_for_shift_test_larger() {
        let x = 0b10_01_11;
        let y = 0b00_00_01;
        let C = 3;
        let log_M = 4;
        let chunks = chunk_and_concatenate_for_shift(x, y, C, log_M);
        assert_eq!(chunks.len(), 3);

        // 2-bit operands
        // x_0 | y = 0b10 | 0b01
        assert_eq!(chunks[0], 0b10_01);
        // x_1 | y = 0b01 | 0b01
        assert_eq!(chunks[1], 0b01_01);
        // x_2 | y = 0b11 | 0b01
        assert_eq!(chunks[2], 0b11_01);
    }
}
