# Jolt
*Note: This section is still under construction. Expect more details in the future.*

`TODO`
- 3 phases of memory checking + primary sumcheck + r1cs
- subtable flags -> instruction flags
- R1CS checks

## Instruction Collation
The "primary sumcheck" collation when generalized from Lasso -> Jolt looks as follows for a trace of length $m$ and a VM with $f$ unique instructions.

$$
\sum_{x \in \{0,1\}^{log(m)}} [\widetilde{eq}(r,x) \cdot \sum_{f \in \{0,1\}^{log(F)}} {\widetilde{flags_f}(x) \cdot g_f(\text{terms}_f(x))]}
$$

$\widetilde{flags_f}(x) = 1$ if the $f$-th instruction is used during the $x$-th step of the trace when. $x \in \{0,1\}^{log(m)}$ 

$g_f(...)$ is the collation function used by the $f$-th instruction.

$terms_f(x) = [E_1(x), ... E_\alpha(x)]$ where $\alpha$ is the number of independent memories used by an instruction. For simple instructions like the EQ instruction, $\alpha = C$, $terms_f(x) = [E_1(x), ... E_C(x)]$. More complicated instructions such LT might have $terms_f(x) = [E_{eq}(x), E_{lt1}(x), E_{lt2}(x)]$. The exact layout is dependent on the number of subtables required by the decomposition of the instruction. The mappings can be found in the `JoltInstruction::subtable` method implementations.

### Metal Model
For a given $r = x \in \{0,1\}^{log(m)}$ (think integer index of the instruction within the trace), $\widetilde{eq} = 0$ for all but one term of the outer sum. Similarly all $\widetilde{flags_f}(x) = 0$ for all but one term of the inner sum. Leaving just the collation function of a single instruction, evaluting to the collated lookup output of the single instruction. In reality $r$ is a random point $r \in \mathbb{F}^{log(m)}$ selected by the verifier over the course of the protocol. The evaluation point provides a distance amplified encoding of the entire trace of instructions.


To illustrate more concretely imagine a 2 instruction VM for LT and EQ instructions with $C=1$.

$$
\sum_{x \in \{0,1\}^{\log_2(m)}}{\widetilde{eq}(r,x) \cdot [ \widetilde{flags}_{LT}(x) \cdot g_{LT}(E_{LT}(x)) + \widetilde{flags}_{EQ}(x) \cdot g_{EQ}(E_{EQ}(x))]}
$$


## Subtable Flags
`TODO`
- We then use memory checking to determine that each of the memories $E_i$ is well formed
- At a given step of the CPU only a single instruction will be used, that means that only that instruction's subtables will be used. For the rest of the memories we insert a no_op with (a, v) = 0. But we still have to increment the timestamp associated with (a, v) = 0 to ensure that offline memory checking works. In order to make the GKR trees cheaper to comptue and sumcheck we'll add a single additional layer to the GKR tree. During this layer we'll "toggle" each of the GKR leaves to "1" in the case that it is an unused step of the CPU. This will make the binary tree of multiplication gates cheaper. We'll toggle based on a new flags polynomial called $subtable-flags_f$ which is the sum of all of the $instruction-flags_f$ used in the instruction collation above.
- The function to compute each of the leaves becomes $leaf[i] = \text{subtable-flags}[i] \cdot \text{fingerprint}[i] + (1 - \text{subtable-flags}[i])$


## Read Write Memory (VM RAM)

In contrast to our standard procedures for offline memory checking, the registers and RAM within this context are considered *writable* memory. This distinction introduces additional verification requirements:

- The multiset equality typically expressed as $I \cdot W = R \cdot F$ is not adequate for ensuring the accuracy of read values. It is essential to also verify that each read operation retrieves a value that was written in a previous step.

- To formalize this, we assert that the timestamp of each read operation, denoted as $\text{read\_timestamp}$, must not exceed the global timestamp at that particular step. The global timestamp is a monotonically increasing sequence starting from 0 and ending at $\text{TRACE\_LENGTH}$.

- The verification of $\text{read\_timestamp} \leq \text{global\_timestamp}$ is equivalent to confirming that $\text{read\_timestamp}$ falls within the range $[0, \text{TRACE\_LENGTH}]$ and that the difference $(\text{global\_timestamp} - \text{read\_timestamp})$ is also within the same range.

- The process of ensuring that both $\text{read\_timestamp}$ and $(\text{global\_timestamp} - \text{read\_timestamp})$ lie within the specified range is known as range-checking. This is the procedure implemented in `timestamp_range_check.rs`.
