[Contributors:
Zeynab Kaseb (z.kaseb@tudelft.nl) 
and Matthias Moller (m.moller@tudelft.nl)
]

# Power Flow

This problem solves the AC power-flow equations in rectangular voltage
coordinates and encodes the residual as a polynomial objective so that
EngiOptiQA can dispatch it to an annealing back-end (D-Wave Ocean,
OpenJij, Amplify).

## 1. Background

For a network of `n` buses with complex bus-admittance matrix
`Y = G + jB`, the bus voltages `V_i = mu_i + j*omega_i` must satisfy the
nodal power balance

```
P_i + j Q_i = V_i * conj( sum_j Y_ij V_j ),     for i = 0 ... n-1
```

with the per-bus boundary conditions

| bus type | given         | unknown            |
|----------|---------------|--------------------|
| slack    | `mu, omega`   | `P, Q`             |
| PV       | `P`, `|V|`    | `Q`, `theta`       |
| PQ       | `P`, `Q`      | `mu`, `omega`      |

The data files use the convention `0 = slack, 1 = PV, 2 = PQ`.

Expanding the complex injection in rectangular form gives a pair of
bilinear expressions

```
P_i = sum_j G_ij ( mu_i mu_j + omega_i omega_j )
            + B_ij ( omega_i mu_j - mu_i omega_j )

Q_i = sum_j G_ij ( omega_i mu_j - mu_i omega_j )
            - B_ij ( mu_i mu_j + omega_i omega_j )
```

which is the formulation used throughout this module. Polar coordinates
would introduce transcendental functions that QUBO cannot express
directly; rectangular form keeps every monomial polynomial.

## 2. Objective

We minimise the per-equation mean squared power mismatch

```
L(mu, omega) = 1/(2(n-1)) * (
        sum_{i in PQ ∪ PV} ( P_i + Pd_i - Pg_i )^2
      + sum_{i in PQ}      ( Q_i + Qd_i - Qg_i )^2 )
```

with the slack bus pinned at `mu = 1`, `omega = 0`. A converged power
flow corresponds to `L = 0`.

Two design choices are worth noting:

1. *Only the active-power mismatch is enforced at PV buses.* The
   classical PV condition `|V_i|^2 = V_set^2` is a separate equality
   constraint and is intentionally not added to the objective in this
   release. Adding it as a quadratic penalty is straightforward but
   raises the polynomial degree once more — see §6.
2. *The factor `1/(2(n-1))` normalises the objective so that the
   gradient does not scale with the case size.* It does not change the
   minimiser.

## 3. Discretisation

Each non-reference bus contributes two continuous variables, `mu_i` and
`omega_i`. They are encoded with the existing
`engioptiqa.variables.real_number.RealNumber` factory:

```
mu_i    = real_number.evaluate(q_mu_i)
omega_i = real_number.evaluate(q_omega_i)
```

where each `q_*_i` is an Amplify `BinarySymbol` array of length
`n_qubits_per_var`. For `binary_representation='range'` the encoding is

```
mu_i = a_min + (a_max - a_min)/(2^N - 1) * sum_{l=0}^{N-1} 2^l q_l
```

so `N` bits produce `2^N` equally spaced grid points across
`[a_min, a_max]`, with resolution `(a_max - a_min) / (2^N - 1)`. The
`adaptive_range` variant adds a `set_range()` hook so the bounds can be
refined between annealing rounds when the user implements an outer
range-update loop.

The total binary-variable count is therefore

```
n_binary = 2 * (n - 1) * n_qubits_per_var
```

(the factor 2 covers `mu` and `omega`; the reference bus is pinned and
contributes no bits). For the shipped 4-bus case with 6 bits this gives
36 binaries before quadratisation.

### Quadratisation footprint

Squaring the bilinear injections turns the objective into a polynomial
of degree 4 in the binaries. Amplify's `IshikawaKZFD` quadratisation
reduces this to degree 2 by introducing auxiliary binaries; in
practice the count grows quickly:

| case   | bits/var | original binaries | after quadratisation |
|--------|----------|-------------------|----------------------|
| 4-bus  | 6        | 36                | ≈ 22 k               |
| 9-bus  | 6        | 96                | ≈ 120 k              |
| 14-bus | 6        | 156               | ≈ 250 k              |

Memory is therefore the binding constraint when extracting the dense
QUBO matrix. The 4-bus case fits comfortably; the 9-bus and 14-bus
cases need either a sparse QUBO pipeline or per-row construction.

## 4. Validation

The module ships with a continuous-variable reference
(`PowerFlow.compute_reference_solution`) that minimises the same
residual using `scipy.optimize.minimize` (L-BFGS-B). It serves as the
"analytic" baseline used elsewhere in the package.

For the 4-bus case the reference reaches residual `≈ 4e-13` and
recovers

| bus | `|V|`   | θ (°)     |
|-----|---------|-----------|
| 0   | 1.0000  |   0.000   |
| 1   | 0.9070  |  -5.850   |
| 2   | 0.9196  |  -5.005   |
| 3   | 0.8964  |  -6.716   |

Cross-checking with the standard AC equations `S = V · conj(Y·V)`
reproduces the demanded P/Q within 1e-7 pu, confirming that the
rectangular-form residual and the bus-admittance assembly are correct.
The 9-bus case converges to the canonical Anderson & Fouad operating
point (|V|≈1.04, |V|≈1.01 at the two PV buses, ±9° angle spread).

## 5. Running the example

From the repository root:

```powershell
python examples/powerflow/powerflow_4bus.py --bus 4 --bits 6 --num-reads 200
```

The script performs, in order:

1. case loading from `engioptiqa/problems/powerflow/data/4-bus/`,
2. continuous reference solution (`compute_reference_solution`),
3. binary discretisation of mu/omega over `[-1.5, 1.5]`,
4. Amplify polynomial assembly (`generate_problem_formulation`),
5. QUBO construction (`transform_to_dwave` from the base `Problem`),
6. simulated-annealing sampling via D-Wave's `neal`,
7. per-sample decoding plus comparison against the reference.

Results land in `examples/powerflow/results/powerflow_4bus_sa/<timestamp>/`
together with the `log.txt` produced by `Problem.print_and_log`.

## 6. Known limitations and extension points

* **PV voltage-magnitude constraint.** Only the active-power mismatch
  is currently enforced at PV buses. Adding
  `(mu_i^2 + omega_i^2 - V_set_i^2)^2` as an extra penalty raises the
  polynomial degree to 8 before quadratisation; doable but expensive.
* **Discretisation resolution vs. accuracy.** With 6 bits over a
  `[-1.5, 1.5]` window the resolution is ≈ 0.048 pu per voltage
  component. The annealed solution will therefore at best match the
  reference to roughly that precision. Refinement requires more bits
  (quadratically more auxiliary binaries) or an outer adaptive-range
  loop using `RealNumber.update_range`.
* **Quadratisation memory.** The dense QUBO matrix is `O(n_aux^2)` and
  becomes infeasible above ~30 k auxiliaries. Switching to the sparse
  `dimod.BinaryQuadraticModel` constructor, passing the coefficient
  dictionary directly rather than the matrix, sidesteps this for the
  9-bus and 14-bus cases.
* **Hybrid samplers.** For the larger cases, D-Wave's
  `LeapHybridSampler` (already wired into `AnnealingSolverDWave`) is
  the practical route; this script defaults to simulated annealing so
  that it runs without a Leap account.

## 7. Files

* [`engioptiqa/problems/powerflow/powerflow_analysis.py`](../../../engioptiqa/problems/powerflow/powerflow_analysis.py) 
  `PowerFlow` problem class, `PowerFlowData` loader.
* [`engioptiqa/problems/powerflow/data/<case>/`](../../../engioptiqa/problems/powerflow/data/)
  tabular input for 4-, 9- and 14-bus cases.
* [`examples/powerflow/powerflow_4bus.py`](../../../examples/powerflow/powerflow_4bus.py)
  end-to-end runnable demonstration.
