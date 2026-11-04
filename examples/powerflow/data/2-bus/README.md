# Two-bus power-flow test case

Minimal case compatible with `PowerFlowData` in EngiOptiQA's power-flow feature branch.

- System base: **100 MVA**.
- Bus 0: **slack**, fixed voltage `V0 = 1 + 0j` p.u.
- Bus 1: **PQ**, load `Pd = 0.50` p.u. (50 MW), `Qd = 0.20` p.u. (20 MVAr), no local generation.
- Single line 0–1: series impedance `Z = 0.02 + 0.10j` p.u., no shunts or charging.
- Series admittance: `1/Z = 1.923076923076923 - 9.615384615384615j` p.u.
- The `g.csv` and `b.csv` sparse triplets describe the full 2×2 Y-bus, including both diagonal and off-diagonal elements.

Analytical high-voltage solution (with the included equations): `V1 ≈ 0.96678046 - 0.04600000j` p.u., `|V1| ≈ 0.96787420`, `angle ≈ -2.72411°`.

Run from the repository root after copying the provided files into the matching folders:

```bash
python examples/powerflow/powerflow_2bus.py --bits 2 --num-reads 200
```

**QAOA sizing:** 2 bits each for real and imaginary voltage at bus 1 means **4 logical qubits** for the direct PUBO, before any ancillas or compilation overhead. For 3 bits per component it is 6 logical qubits. Standard QAOA needs a phase-separator circuit for the PUBO; OpenJij here is a classical correctness/smoke test, not a QAOA implementation.

**Compatibility note:** `PowerFlow.generate_discretization()` in the inspected feature branch hardcodes separate voltage ranges `mu∈[0.8,1.2]` and `omega∈[-0.2,0.2]` in its method body. Bounds passed through `lower_lim`/`upper_lim` are overridden for these variables. The reference high-voltage solution lies inside those hardcoded bounds.
