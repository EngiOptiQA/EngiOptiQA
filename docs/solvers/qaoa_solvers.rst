============================
QAOA Solvers
============================

The Quantum Approximate Optimization Algorithm (QAOA) solvers construct a cost
Hamiltonian from the binary problem formulation and returns a distribution of
candidate solutions.

Base Class
----------

.. currentmodule:: engioptiqa.solvers.vqa_solvers.qaoa.solver_base

.. autoclass:: QAOASolverBase
   :members: solve_problem

PennyLane Solver
----------------

.. currentmodule:: engioptiqa.solvers.vqa_solvers.qaoa.solver_pennylane

.. autoclass:: QAOASolverPennylane
   :members: solve_problem

``QAOASolverPennylane.solve_problem`` supports the following parameter modes:

- ``fixed`` uses a deterministic linear parameter schedule.
- ``linear_ramp`` optimizes two parameters that scale the linear schedule.
- ``optimize`` independently optimizes every QAOA beta and gamma parameter.

Use ``optimization_iterations`` to control either optimization mode. Set ``circuit="probs"`` for a
probability distribution or ``circuit="sample"`` with a positive ``shots`` value
for sampled frequencies.

AQT Solver
----------

.. currentmodule:: engioptiqa.solvers.vqa_solvers.qaoa.solver_aqt

.. autoclass:: QAOASolverAQT
   :members: solve_problem

The AQT solver uses PennyLane to export the QAOA ansatz to OpenQASM, then runs the
Qiskit circuit through ``qiskit-aqt-provider``'s ``AQTSampler``. It supports the
fixed parameter schedule and shot-based results.

Install offline AQT simulator support with:

.. code-block:: console

   pip install 'engioptiqa[aqt]'

By default, the solver obtains an AQT offline simulator backend. To run on hardware,
create a compatible backend in the calling application and provide it through the
``backend`` constructor argument.

``optimization_level`` configures AQTSampler's Qiskit transpilation level and
defaults to ``3``.
