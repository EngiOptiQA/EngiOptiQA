
# Power Flow

**Contributors:**  
Zeynab Kaseb (z.kaseb@tudelft.nl)  
Matthias Moller (m.moller@tudelft.nl)

This module solves the AC power flow (AQPF) equations in rectangular voltage coordinates. The power-flow residual is formulated as a polynomial unconstrained binary optimization (PUBO) problem and solved using OpenJij simulated annealing.

Two implementations are provided:

- **One-shot AQPF:** Solves a single PUBO using configurable binary discretization (default: 6 bits per voltage component).
- **Iterative AQPF:** Repeatedly solves smaller PUBOs using 2 binary variables per voltage component and adaptive voltage updates.

## 1. Background

For a network of `n` buses with complex bus-admittance matrix `Y = G + jB`, the bus voltages are represented in rectangular coordinates:

    V_i = mu_i + j * omega_i

The nodal power-balance equations are:

    P_i + jQ_i = V_i * conj(sum_j Y_ij * V_j)

The bus types are:

| Bus type | Given | Unknown |
|----------|-------|---------|
| Slack | mu, omega | P, Q |
| PV | P, \|V\| | Q, theta |
| PQ | P, Q | mu, omega |

The data convention is `0 = slack`, `1 = PV`, and `2 = PQ`.

Expanding the complex power injections gives:

    P_i = sum_j G_ij (mu_i mu_j + omega_i omega_j)
                + B_ij (omega_i mu_j - mu_i omega_j)

    Q_i = sum_j G_ij (omega_i mu_j - mu_i omega_j)
                - B_ij (mu_i mu_j + omega_i omega_j)

Rectangular coordinates keep the formulation polynomial, avoiding trigonometric expressions in the optimization objective.

## 2. Objective

The normalized squared power-mismatch objective is:

    L(mu, omega) = 1/(2(n-1)) * (
        sum_{i in PQ ∪ PV} (P_i + Pd_i - Pg_i)^2
        + sum_{i in PQ} (Q_i + Qd_i - Qg_i)^2
    )

The slack bus is fixed at `mu = 1` and `omega = 0`.

A zero objective corresponds to zero mismatch for all enforced equations.

Currently, the active-power mismatch is enforced at PV buses, while the PV voltage-magnitude constraint is not independently included.

## 3. Discretization and Optimization

Each non-slack bus has two continuous voltage components: `mu` (real) and `omega` (imaginary).

### 3.1 One-Shot AQPF

**File:** `examples/powerflow/powerflow_4bus.py`

The one-shot approach discretizes the absolute voltage components using EngiOptiQA's `RealNumber` representation.

For a uniform binary range encoding:

    mu_i = a_min + (a_max - a_min)/(2^N - 1)
           * sum_{l=0}^{N-1} 2^l q_l

The total number of binary variables is:

    n_binary = 2 * (n - 1) * N

where `N` is the number of bits per voltage component.

For the 4-bus case with 6 bits:

    n_binary = 2 * 3 * 6 = 36

The workflow is:

1. Load the selected bus case.
2. Compute the continuous reference solution.
3. Generate the absolute-voltage discretization.
4. Construct the power-flow PUBO.
5. Solve the PUBO using OpenJij.
6. Decode the voltage components and calculate the AC residual.

Run from the repository root:

```bash
python examples/powerflow/powerflow_4bus.py \
  --bus 4 \
  --bits 6 \
  --num-reads 200
```

The default voltage-component bounds are `[-0.8, 1.2]`, configurable using `--lower-lim` and `--upper-lim`.

### 3.2 Iterative AQPF

**File:** `examples/powerflow/iterative_aqpf.py`

The iterative approach reduces the binary problem size by using only **2 binary variables per voltage component** at each iteration.

Instead of discretizing absolute voltages, each iteration encodes local corrections around the current voltage estimates:

    mu_i(next) = mu_i(current)
                 + delta_mu * (q_mu_plus - q_mu_minus)

    omega_i(next) = omega_i(current)
                    + delta_omega * (q_omega_plus - q_omega_minus)

Each non-slack bus requires:

- 2 binary variables for `mu`.
- 2 binary variables for `omega`.

The total binary-variable count is:

    n_binary = 4 * (n - 1)

For the 4-bus case:

    n_binary = 4 * 3 = 12

The iterative workflow is:

1. Initialize the voltage estimates with a flat start (`mu = 1`, `omega = 0`).
2. Generate a two-bit correction discretization around the current solution.
3. Construct the corresponding PUBO using EngiOptiQA.
4. Solve the PUBO using OpenJij.
5. Decode candidate voltages and evaluate their AC residuals.
6. Retain the best solution and adjust the correction increments.
7. Repeat until the convergence tolerance or maximum iteration count is reached.

The adaptive strategy reduces the correction increments after successful improvements and increases them when no improvement is obtained.

Run from the repository root:

```bash
python examples/powerflow/iterative_aqpf.py \
  --bus 4 \
  --num-reads 200 \
  --max-iter 100 \
  --tolerance 1e-4
```

The iterative controller is experimental and intended for future integration into OuterOpt.

### 3.3 Comparison

For the 4-bus system with one slack bus:

| Property | One-Shot AQPF | Iterative AQPF |
|----------|---------------|----------------|
| Bits per voltage component | 6 | 2 |
| Binary variables per PUBO | 36 | 12 |
| Optimization calls | 1 | Multiple |
| Solver | OpenJij | OpenJij |
| Voltage representation | Absolute | Incremental |
| Search strategy | Fixed range | Adaptive local corrections |

The iterative approach reduces the number of binary variables per solver call but requires multiple optimization rounds. Its total runtime and accuracy depend on the convergence behavior and annealing configuration.

## 4. PUBO Formulation and Solver

The rectangular power-flow equations are quadratic in voltage components. Squaring the power mismatches produces polynomial terms up to degree 4.

Both approaches use the existing EngiOptiQA formulation:

```python
problem.generate_problem_formulation()

solver = AnnealingSolverOpenJij()
solver.setup_solver()
solver.solve_problem(problem, num_reads=200, seed=0)
```

OpenJij supports direct higher-order binary polynomial optimization through its PUBO-capable sampler.

Therefore, explicit QUBO quadratization is not required for these examples.

QUBO-based approaches remain possible through the existing transformation interfaces, but they may introduce many auxiliary binary variables.

## 5. Validation

A continuous reference solution is available through:

```python
PowerFlow.compute_reference_solution()
```

This uses `scipy.optimize.minimize` with L-BFGS-B to minimize the continuous power-flow residual.

The reference solution provides a baseline for evaluating:

- Voltage magnitudes.
- Voltage phase angles.
- AC power-flow residuals.
- Deviations from the continuous solution.

Previously obtained 4-bus reference results are:

| Bus | Voltage magnitude (p.u.) | Angle (°) |
|-----|--------------------------|-----------|
| 0 | 1.0000 | 0.000 |
| 1 | 0.9070 | -5.850 |
| 2 | 0.9196 | -5.005 |
| 3 | 0.8964 | -6.716 |

The reported continuous reference residual was approximately `4e-13`.

For iterative AQPF, the default convergence tolerance is `1e-4`. Reaching this tolerance depends on discretization, sampling, and adaptive update settings.

## 6. Running the Examples

Run the commands from the EngiOptiQA repository root.

**One-shot, 4-bus:**

```bash
python examples/powerflow/powerflow_4bus.py \
  --bus 4 --bits 6 --num-reads 200
```

**Iterative, 4-bus:**

```bash
python examples/powerflow/iterative_aqpf.py \
  --bus 4 --num-reads 200 --max-iter 100 --tolerance 1e-4
```

**Short iterative smoke test:**

```bash
python examples/powerflow/iterative_aqpf.py \
  --bus 4 --num-reads 10 --max-iter 2
```

The input datasets support 4-, 9-, and 14-bus cases. Larger networks may require additional computational resources.

The one-shot example creates a timestamped results directory under:

    examples/powerflow/results/powerflow_<bus>bus_openjij_one_shot/

The iterative example prints its iteration history, final voltage values, residual, and runtime to the terminal.

## 7. Known Limitations and Future Work

- **PV constraints:** The PV voltage-magnitude equality is not independently enforced in the current objective.
- **One-shot resolution:** Numerical precision depends on the number of binary variables and selected voltage bounds.
- **Iterative convergence:** The adaptive correction strategy does not guarantee convergence.
- **Computational cost:** Iterative AQPF uses fewer binary variables per optimization call but requires repeated PUBO generation and solving.
- **Larger networks:** The 9-bus and 14-bus cases require additional benchmarking and convergence testing.
- **OuterOpt integration:** The iterative controller is an experimental example intended for integration into OuterOpt.

## 8. Files

- [`engioptiqa/problems/powerflow/powerflow_analysis.py`](../../engioptiqa/problems/powerflow/powerflow_analysis.py) — Power-flow formulation and discretization.
- [`examples/powerflow/powerflow_4bus.py`](powerflow_4bus.py) — One-shot OpenJij example.
- [`examples/powerflow/iterative_aqpf.py`](iterative_aqpf.py) — Iterative two-bit OpenJij example.
- [`examples/powerflow/data/`](data/) — Input data for the 4-, 9-, and 14-bus systems.
- [`engioptiqa/solvers/annealing_solvers/annealing_solver_openjij.py`](../../engioptiqa/solvers/annealing_solvers/annealing_solver_openjij.py) — OpenJij PUBO solver.
