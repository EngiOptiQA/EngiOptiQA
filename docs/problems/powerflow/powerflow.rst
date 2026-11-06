AC Power Flow
=============

**Contributors:**

* Zeynab Kaseb (z.kaseb@tudelft.nl)
* Matthias Moller (m.moller@tudelft.nl)

The EngiOptiQA power flow module formulates AC power flow in rectangular
voltage coordinates and converts the squared power balance mismatch into a
polynomial unconstrained binary optimization (PUBO) problem.

The example documented here solves **one PUBO with OpenJij simulated
annealing**. The implementation supports the 2-, 4-, 9-, and 14-bus datasets.

Paper references: https://doi.org/10.1038/s41598-024-73512-7

Background
----------

For a network with :math:`n` buses, let the complex bus-admittance matrix be
:math:`Y = G + \mathrm{j}B`. Bus voltages are expressed as

.. math::

   V_i = \mu_i + \mathrm{j}\omega_i.

The complex nodal power injection is

.. math::

   P_i + \mathrm{j}Q_i
   = V_i \overline{\sum_{j=1}^{n} Y_{ij}V_j}.

The supported bus-type convention is:

.. list-table:: Bus types
   :header-rows: 1
   :widths: 15 35 50

   * - Type
     - Specified quantities
     - Unknown quantities
   * - Slack (0)
     - Voltage (real and imaginary components)
     - Active and reactive injections
   * - PV (1)
     - Active-power injection and voltage-magnitude setpoint
     - Reactive-power injection and voltage angle
   * - PQ (2)
     - Active- and reactive-power injections
     - Real and imaginary voltage components

Writing the injection equations in rectangular coordinates gives

.. math::

   P_i &= \sum_{j=1}^{n}
      \left[G_{ij}(\mu_i\mu_j+\omega_i\omega_j)
      +B_{ij}(\omega_i\mu_j-\mu_i\omega_j)\right], \\
   Q_i &= \sum_{j=1}^{n}
      \left[G_{ij}(\omega_i\mu_j-\mu_i\omega_j)
      -B_{ij}(\mu_i\mu_j+\omega_i\omega_j)\right].

Rectangular coordinates keep these expressions polynomial in the voltage
components.

Objective
---------

The current implementation minimizes the normalized squared mismatch

.. math::

   \mathcal{L}(\mu,\omega)
   = \frac{1}{2(n-1)}
   \left[
      \sum_{i\in\mathrm{PQ}\cup\mathrm{PV}}
        (P_i+P_{d,i}-P_{g,i})^2
      +\sum_{i\in\mathrm{PQ}}
        (Q_i+Q_{d,i}-Q_{g,i})^2
   \right].

The slack-bus voltage is fixed at :math:`V_{\mathrm{slack}}=1+\mathrm{j}0`
p.u. A zero objective means all **enforced** power-balance equations are
satisfied. The active-power mismatch is enforced at PV and PQ buses, while the
reactive-power mismatch is enforced at PQ buses. **The PV-bus
voltage-magnitude equality is not yet independently enforced** by this
objective, even though a magnitude setpoint may be supplied in the case data.

One-shot binary discretization
------------------------------

Each non-slack bus has two unknown rectangular voltage components:
:math:`\mu_i` and :math:`\omega_i`. The example calls
``PowerFlow.generate_discretization()`` using EngiOptiQA's ``RealNumber``
representation, followed by ``PowerFlow.generate_problem_formulation()``.

For a uniformly weighted :math:`N`-bit representation over
:math:`[x_{\min},x_{\max}]`, the voltage levels can be written as

.. math::

   x = x_{\min}
       +\frac{x_{\max}-x_{\min}}{2^N-1}
       \sum_{\ell=0}^{N-1}2^{\ell}q_{\ell},
   \qquad q_{\ell}\in\{0,1\}.

The exact binary representation is determined by the selected EngiOptiQA
``RealNumber`` option. The default example choice is ``adaptive_range``.

With :math:`N` bits per component, the number of binary variables is

.. math::

   n_{\mathrm{binary}}=2(n-1)N.

For example, a 4-bus network with 6 bits per voltage component uses
:math:`2\times3\times6=36` binary variables.

The default component ranges in the updated implementation are:

* Real voltage :math:`\mu`: ``[0.8, 1.2]`` p.u.
* Imaginary voltage :math:`\omega`: ``[-0.2, 0.2]`` p.u.

Use ``--mu-lower``, ``--mu-upper``, ``--omega-lower`` and ``--omega-upper``
to change these bounds independently. The legacy ``--lower-lim`` and
``--upper-lim`` options set a common interval for **both** components unless
component-specific bounds are given. The defaults are not ``[-0.8, 1.2]``
for both components.

PUBO formulation and OpenJij
----------------------------

The power injections are quadratic in rectangular voltage coordinates.
Squaring their mismatches produces terms of degree up to four. EngiOptiQA
constructs the polynomial and its binary optimization model; the example
passes that model to ``AnnealingSolverOpenJij``.

.. code-block:: python

   problem.generate_discretization(
       n_qubits_per_var=bits,
       binary_representation="adaptive_range",
       mu_lower=0.8,
       mu_upper=1.2,
       omega_lower=-0.2,
       omega_upper=0.2,
   )
   problem.generate_problem_formulation()

   solver = AnnealingSolverOpenJij()
   solver.setup_solver()
   solver.solve_problem(problem, num_reads=200, seed=0)

The example uses the native PUBO formulation without explicitly introducing
QUBO quadratization variables.

Running the one-shot example
----------------------------

Run these commands from the EngiOptiQA repository root. The current example
is ``examples/powerflow/powerflow_oneshot_openjij.py``; the older
``powerflow_2bus.py`` and ``powerflow_4bus.py`` example names are no longer
used here.

A small 2-bus run, using the default 2 bits per voltage component:

.. code-block:: bash

   python -u examples/powerflow/powerflow_oneshot_openjij.py \
       --bus 2 --bits 2 --num-reads 200 --seed 0

A 4-bus case with finer discretization:

.. code-block:: bash

   python -u examples/powerflow/powerflow_oneshot_openjij.py \
       --bus 4 --bits 6 --num-reads 200 --seed 0

To specify separate real and imaginary voltage ranges:

.. code-block:: bash

   python -u examples/powerflow/powerflow_oneshot_openjij.py \
       --bus 2 --bits 3 --num-reads 200 \
       --mu-lower 0.90 --mu-upper 1.10 \
       --omega-lower -0.10 --omega-upper 0.10

The command-line default is ``--bits 2``. Available case sizes are 2, 4, 9
and 14 buses. The search space grows rapidly with the number of buses and
bits, so larger runs may require additional time and memory.

Reference solution and output
-----------------------------

By default, the example also calls
``PowerFlow.compute_reference_solution()`` to compute a continuous
optimization-based reference, using SciPy's L-BFGS-B algorithm. The reference
provides a baseline for comparing voltage magnitudes, phase angles and the
power-flow residual. It is a numerical minimizer, not a guarantee that every
physical bus constraint has been enforced.

To skip this extra calculation during a quick solver test, add
``--no-reference``. This does **not** change the OpenJij PUBO or annealing
configuration.

The example reports the decoded real and imaginary voltages, magnitudes,
angles and physical mismatch residual. It also writes ``voltages.csv`` to a
timestamped results directory of the form

.. code-block:: text

   examples/powerflow/results/powerflow_<bus>bus_openjij_oneshot/<timestamp>/

Known limitations
-----------------

* PV voltage-magnitude setpoints are not independently enforced in the
  current objective.
* Accuracy depends on the selected voltage bounds and binary resolution.
* Annealing results depend on the number of reads and the sampling process;
  they are not guaranteed to reach the global PUBO minimum.
* Increasing the number of buses or bits increases the binary optimization
  problem size and may significantly increase computational cost.

Relevant files
--------------

* ``engioptiqa/problems/powerflow/powerflow_analysis.py`` — power-flow
  equations, discretization, PUBO generation and voltage decoding.
* ``examples/powerflow/powerflow_oneshot_openjij.py`` — one-shot OpenJij
  example documented here.
* ``examples/powerflow/data/`` — example input datasets.
* ``engioptiqa/solvers/annealing_solvers/annealing_solver_openjij.py`` —
  OpenJij solver implementation.
