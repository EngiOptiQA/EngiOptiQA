"""AC power-flow problem in rectangular voltage coordinates.

The bus voltages are split into real and imaginary parts,
``V_i = mu_i + j*omega_i``, which keeps the power-injection equations
polynomial (bilinear in the unknowns). After binary encoding of mu/omega
through :class:`engioptiqa.variables.real_number.RealNumber`, the squared
P/Q mismatch becomes a degree-4 polynomial in binary variables and is
handed off to Amplify as the QUBO source polynomial.
"""

from amplify import Model, Poly, sum_poly
from dimod.views.samples import SampleView
import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
from types import SimpleNamespace

from engioptiqa.problems.problem import Problem
from engioptiqa.variables.real_number import RealNumber


class PowerFlowData:
    """Per-case input data loaded from a flat directory.

    The directory layout follows a flattened MATPOWER-style case: one
    file per quantity. Bus types use the convention of the data files
    shipped with the package:

    ===== =======================
    type  meaning
    ===== =======================
    0     slack / reference bus
    1     PV (generator) bus
    2     PQ (load) bus
    ===== =======================
    """

    def __init__(self, path):
        path = os.path.join(path, '')

        # Demand and generation set-points (per-unit).
        self.pd_arr = np.loadtxt(path + 'pd.txt')
        self.qd_arr = np.loadtxt(path + 'qd.txt')
        self.pg_arr = np.loadtxt(path + 'pg.txt')
        self.qg_arr = np.loadtxt(path + 'qg.txt')

        buses = pd.read_csv(path + 'bus_type.csv')
        buses['index'] = buses['index'].astype(float).astype(int)
        buses['type'] = buses['type'].astype(float).astype(int)
        self.n_bus = len(buses)
        self.ref_bus = int(buses.loc[buses['type'] == 0, 'index'].iloc[0])
        self.pv_buses = buses.loc[buses['type'] == 1, 'index'].astype(int).tolist()
        self.pq_buses = buses.loc[buses['type'] == 2, 'index'].astype(int).tolist()

        self.g_matrix = self._sparse_to_dense(path + 'g.csv', self.n_bus)
        self.b_matrix = self._sparse_to_dense(path + 'b.csv', self.n_bus)

        # Operational bounds (kept on the object for later optimisation
        # use; not consumed by the residual objective itself).
        self.v_min = np.loadtxt(path + 'vmin.txt')
        self.v_max = np.loadtxt(path + 'vmax.txt')
        self.p_min = np.loadtxt(path + 'pmin.txt')
        self.p_max = np.loadtxt(path + 'pmax.txt')
        self.q_min = np.loadtxt(path + 'qmin.txt')
        self.q_max = np.loadtxt(path + 'qmax.txt')
        self.cost = np.loadtxt(path + 'cost.txt')

    @staticmethod
    def _sparse_to_dense(csv_file, n_bus):
        df = pd.read_csv(csv_file)
        i = df['source'].astype(float).astype(int).to_numpy()
        j = df['target'].astype(float).astype(int).to_numpy()
        w = df['weight'].astype(float).to_numpy()
        M = np.zeros((n_bus, n_bus))
        np.add.at(M, (i, j), w)
        return M


class PowerFlow(Problem):
    """AC power-flow residual as a polynomial QUBO source.

    The objective is the per-equation squared mismatch

    .. math::

        \\mathcal{L} = \\frac{1}{2(n-1)} \\left(
            \\sum_{i \\in \\mathrm{PQ} \\cup \\mathrm{PV}}
                (P_i + P_{d,i} - P_{g,i})^2
            + \\sum_{i \\in \\mathrm{PQ}}
                (Q_i + Q_{d,i} - Q_{g,i})^2 \\right),

    where

    .. math::

        P_i = \\sum_j G_{ij}(\\mu_i\\mu_j + \\omega_i\\omega_j)
              + B_{ij}(\\omega_i\\mu_j - \\mu_i\\omega_j),

    .. math::

        Q_i = \\sum_j G_{ij}(\\omega_i\\mu_j - \\mu_i\\omega_j)
              - B_{ij}(\\mu_i\\mu_j + \\omega_i\\omega_j).

    The slack bus is pinned at :math:`\\mu_\\mathrm{ref}=1`,
    :math:`\\omega_\\mathrm{ref}=0`. Only PV active power mismatch is
    enforced; the PV voltage-magnitude set point is *not* added as a
    penalty term (see ``docs/problems/powerflow/powerflow.md``).
    """

    def __init__(self, data, output_path=None):
        super().__init__(output_path)
        self.name = 'Power Flow'
        self.data = data
        self.print_and_log(self.name + '\n')

    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------
    def generate_discretization(self, n_qubits_per_var, binary_representation,
                                lower_lim=None, upper_lim=None):
        """Encode mu/omega at every non-reference bus as a RealNumber polynomial.

        At the slack bus the polynomials collapse to the pinned constants
        ``1.0`` and ``0.0``, which keeps the objective syntactically uniform.
        """
        self.initialize_discretization()
        self.n_qubits_per_var = n_qubits_per_var
        self.binary_representation = binary_representation
        self.real_number = RealNumber(
            n_qubits_per_var, binary_representation, lower_lim, upper_lim
        )

        mu_polys = [None] * self.data.n_bus
        omega_polys = [None] * self.data.n_bus
        for i in range(self.data.n_bus):
            if i == self.data.ref_bus:
                mu_polys[i] = 1.0
                omega_polys[i] = 0.0
                continue
            q_mu = self.variable_generator.array('Binary', n_qubits_per_var)
            q_omega = self.variable_generator.array('Binary', n_qubits_per_var)
            if binary_representation == 'adaptive_range':
                self.real_number.set_range(lower_lim, upper_lim)
            mu_polys[i] = self.real_number.evaluate(q_mu)
            omega_polys[i] = self.real_number.evaluate(q_omega)

        self.mu_polys = mu_polys
        self.omega_polys = omega_polys

    # ------------------------------------------------------------------
    # Polynomial assembly
    # ------------------------------------------------------------------
    def _injection_polys(self, mu, omega):
        n = self.data.n_bus
        G, B = self.data.g_matrix, self.data.b_matrix
        p_poly = []
        q_poly = []
        for i in range(n):
            p_poly.append(sum_poly(
                mu[i] * G[i, j] * mu[j] + omega[i] * G[i, j] * omega[j]
                + omega[i] * B[i, j] * mu[j] - mu[i] * B[i, j] * omega[j]
                for j in range(n)
            ))
            q_poly.append(sum_poly(
                omega[i] * G[i, j] * mu[j] - mu[i] * G[i, j] * omega[j]
                - mu[i] * B[i, j] * mu[j] - omega[i] * B[i, j] * omega[j]
                for j in range(n)
            ))
        return p_poly, q_poly

    def generate_problem_formulation(self):
        if not hasattr(self, 'mu_polys'):
            raise RuntimeError(
                'PowerFlow.generate_problem_formulation() requires '
                'generate_discretization() to have been called first.'
            )

        data = self.data
        p_i, q_i = self._injection_polys(self.mu_polys, self.omega_polys)

        active = sum_poly(
            (p_i[i] + data.pd_arr[i] - data.pg_arr[i]) ** 2
            for i in (data.pq_buses + data.pv_buses)
        )
        reactive = sum_poly(
            (q_i[i] + data.qd_arr[i] - data.qg_arr[i]) ** 2
            for i in data.pq_buses
        )
        n_eq = max(1, data.n_bus - 1)
        self.poly = (active + reactive) / (2 * n_eq)
        self.binary_model = Model(self.poly)

        output = f'Number of binary variables: {len(self.binary_model.get_variables())}\n'
        self.print_and_log(output)

    # ------------------------------------------------------------------
    # Continuous reference solution
    # ------------------------------------------------------------------
    def compute_reference_solution(self, tol=1e-12, max_iter=200):
        """Continuous-variable AC power-flow reference.

        Minimises the same residual the QUBO uses, but on real-valued
        :math:`(\\mu, \\omega)` without binary encoding. Provides the
        gold-standard against which the annealed solution is compared.
        """
        n = self.data.n_bus
        free = [i for i in range(n) if i != self.data.ref_bus]
        n_free = len(free)

        def residual(x):
            mu = np.ones(n)
            omega = np.zeros(n)
            mu[free] = x[:n_free]
            omega[free] = x[n_free:]
            return self.compute_residual(mu, omega)

        x0 = np.concatenate([np.ones(n_free), np.zeros(n_free)])
        result = minimize(residual, x0, method='L-BFGS-B', tol=tol,
                          options={'maxiter': max_iter, 'gtol': tol})
        if not result.success:
            raise RuntimeError(
                f'Reference solver did not converge: {result.message}'
            )

        mu = np.ones(n)
        omega = np.zeros(n)
        mu[free] = result.x[:n_free]
        omega[free] = result.x[n_free:]

        self.mu_reference = mu
        self.omega_reference = omega
        self.residual_reference = float(result.fun)

        self.print_and_log(
            f'Reference residual: {self.residual_reference:.3e}\n'
        )
        return mu, omega

    # ------------------------------------------------------------------
    # Residual evaluation and decoding
    # ------------------------------------------------------------------
    def compute_residual(self, mu, omega):
        """Evaluate the residual objective on numeric mu/omega arrays."""
        mu = np.asarray(mu, dtype=float)
        omega = np.asarray(omega, dtype=float)
        G = self.data.g_matrix
        B = self.data.b_matrix

        # Vectorised rectangular-form injection.
        P = mu * (G @ mu) + omega * (G @ omega) \
            + omega * (B @ mu) - mu * (B @ omega)
        Q = omega * (G @ mu) - mu * (G @ omega) \
            - mu * (B @ mu) - omega * (B @ omega)

        idx_p = list(self.data.pq_buses) + list(self.data.pv_buses)
        idx_q = list(self.data.pq_buses)
        term_p = np.sum((P[idx_p]
                         + self.data.pd_arr[idx_p]
                         - self.data.pg_arr[idx_p]) ** 2)
        term_q = np.sum((Q[idx_q]
                         + self.data.qd_arr[idx_q]
                         - self.data.qg_arr[idx_q]) ** 2)
        n_eq = max(1, self.data.n_bus - 1)
        return float((term_p + term_q) / (2 * n_eq))

    def decode_voltages(self, result):
        """Decode (mu, omega) numpy arrays from a solver result."""
        def decode_one(poly):
            if isinstance(poly, Poly):
                if type(result) is SampleView:
                    return self.decode_amplify_poly_with_bitstring(
                        poly, result._data)
                if type(result) is SimpleNamespace:
                    return self.decode_amplify_poly_with_bitstring(
                        poly, result.values)
                if isinstance(result, dict):
                    return self.decode_amplify_poly_with_bitstring(
                        poly, result)
                return poly.decode(result.values)
            return float(poly)

        mu = np.array([decode_one(p) for p in self.mu_polys])
        omega = np.array([decode_one(p) for p in self.omega_polys])
        return mu, omega

    # ------------------------------------------------------------------
    # Solution analysis
    # ------------------------------------------------------------------
    def analyze_results(self, results=None, result_max=5):
        if results is None and not hasattr(self, 'results'):
            raise Exception(
                'analyze_results called without results to analyse.'
            )
        if results is None:
            results = self.results

        solutions = []
        for i_result, result in enumerate(results):
            bit_array = self.get_bit_array(result)
            mu, omega = self.decode_voltages(result)
            res = self.compute_residual(mu, omega)
            entry = {
                'bit_array': bit_array,
                'mu': mu,
                'omega': omega,
                'voltage_magnitude': np.sqrt(mu ** 2 + omega ** 2),
                'voltage_angle': np.arctan2(omega, mu),
                'residual': res,
                'energy': self.get_energy(i_result),
                'frequency': self.get_frequency(i_result),
            }
            solutions.append(entry)

            if i_result < result_max:
                self._report_solution(i_result, entry)

        self._summarise_best(solutions)
        return solutions

    def _report_solution(self, i_result, entry):
        output = (
            f'Solution {i_result}\n'
            f"\tenergy = {entry['energy']:.6g}, "
            f"frequency = {entry['frequency']}\n"
            f"\tresidual = {entry['residual']:.3e}\n"
        )
        for i in range(self.data.n_bus):
            output += (
                f"\tBus {i}: |V| = {entry['voltage_magnitude'][i]:.4f}, "
                f"theta = {np.degrees(entry['voltage_angle'][i]):+7.3f} deg\n"
            )
        self.print_and_log(output)

    def _summarise_best(self, solutions):
        if not solutions:
            return
        best = min(solutions, key=lambda s: s['residual'])
        output = (
            f"Best annealed residual: {best['residual']:.3e}\n"
        )
        if hasattr(self, 'residual_reference'):
            output += (
                f"Reference residual    : {self.residual_reference:.3e}\n"
                f"|V| max abs error     : "
                f"{np.max(np.abs(best['voltage_magnitude'] - np.sqrt(self.mu_reference**2 + self.omega_reference**2))):.3e}\n"
                f"theta max abs error   : "
                f"{np.max(np.abs(np.arctan2(best['omega'], best['mu']) - np.arctan2(self.omega_reference, self.mu_reference))):.3e} rad\n"
            )
        self.print_and_log(output)
